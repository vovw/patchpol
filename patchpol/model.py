"""Patch Policy trunk: transformer over dense patch tokens, block-causal in time.

Input:  (B, T, P, D) frozen DINOv2 patch features  (T frames, P=256 patches, D=384)
Output: (B, T, D) readouts — the last patch token of each frame, which (thanks
        to the mask) has attended to everything up to and including that frame.

Run the causality unit tests:  uv run python -m patchpol.model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def block_causal_mask(T: int, P: int) -> torch.Tensor:
    """(T*P, T*P) bool matrix. mask[i, j] = True  <=>  token i MAY attend to j.

    Rule: frame(j) <= frame(i). Same frame -> bidirectional; past -> visible;
    future -> blocked. This one comparison is the paper's core contribution.
    """
    frame = torch.arange(T).repeat_interleave(P)  # [0,0,...,0, 1,1,...,1] (T*P,)
    return frame[None, :] <= frame[:, None]


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)  # one matmul makes q, k, v together
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # each (B, S, D)
        # split D into heads: (B, S, D) -> (B, heads, S, D/heads)
        q, k, v = (
            t.view(B, S, self.heads, -1).transpose(1, 2) for t in (q, k, v)
        )
        # fused softmax(q k^T / sqrt(d) + mask) v ; True in mask = "may attend"
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, S, D)  # merge heads back
        return self.proj(out)


class Block(nn.Module):
    """Pre-LN transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchTrunk(nn.Module):
    def __init__(
        self, dim: int = 384, depth: int = 8, heads: int = 6,
        T: int = 2, P: int = 256,
    ):
        super().__init__()
        self.T, self.P = T, P
        # one learned vector per position in the flattened T*P sequence
        # (paper: "learned 1D positional embedding"); GPT-style 0.02 init
        self.pos_emb = nn.Parameter(torch.randn(1, T * P, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        # buffer: moves with .to(device), saved out of state_dict (derivable)
        self.register_buffer("mask", block_causal_mask(T, P), persistent=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, T, P, D) -> (B, T, D) per-frame readouts."""
        B, T, P, D = feats.shape
        x = feats.flatten(1, 2) + self.pos_emb  # (B, T*P, D)
        for blk in self.blocks:
            x = blk(x, self.mask)
        x = self.norm(x)
        return x[:, P - 1 :: P]  # last patch token of each frame: (B, T, D)


if __name__ == "__main__":
    torch.manual_seed(0)
    trunk = PatchTrunk().eval()
    n_params = sum(p.numel() for p in trunk.parameters())
    print(f"trunk params: {n_params/1e6:.1f}M")

    feats = torch.randn(2, 2, 256, 384)
    with torch.inference_mode():
        out = trunk(feats)
        assert out.shape == (2, 2, 384), out.shape

        # --- causality tests: the mask IS the model, so prove it works ---
        # 1) perturb frame 1 -> frame 0's readout must NOT change (no time travel)
        f2 = feats.clone()
        f2[:, 1] += torch.randn_like(f2[:, 1])
        out2 = trunk(f2)
        assert torch.allclose(out[:, 0], out2[:, 0], atol=1e-5), "future leaked into past!"
        assert not torch.allclose(out[:, 1], out2[:, 1], atol=1e-3), "frame 1 ignored its own input?!"

        # 2) perturb frame 0 -> frame 1's readout MUST change (memory flows forward)
        f3 = feats.clone()
        f3[:, 0] += torch.randn_like(f3[:, 0])
        out3 = trunk(f3)
        assert not torch.allclose(out[:, 1], out3[:, 1], atol=1e-3), "past invisible to present?!"

        # 3) perturb ONE patch of frame 0 -> frame 0's readout (its LAST patch)
        #    must change: within-frame attention is bidirectional
        # (noise, NOT a constant — LayerNorm subtracts each token's mean, so a
        # uniform shift like +10 on every dim is invisible to the whole network)
        f4 = feats.clone()
        f4[:, 0, 0] += torch.randn(384) * 5
        out4 = trunk(f4)
        assert not torch.allclose(out[:, 0], out4[:, 0], atol=1e-3), "intra-frame attention broken?!"

    print("causality tests passed ✓  (no time travel, memory works, intra-frame is bidirectional)")
