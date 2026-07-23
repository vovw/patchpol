"""Hand-rolled DDPM action head + the full Patch Policy (trunk + head).

The idea in three sentences:
  FORWARD (fixed, no learning): take a clean action chunk a0, mix in Gaussian
  noise at strength t: a_t = sqrt(ab_t)*a0 + sqrt(1-ab_t)*eps. At t=T the chunk
  is pure noise. REVERSE (learned): a network looks at (a_t, t, obs) and
  predicts the eps that was mixed in; subtracting it step by step walks pure
  noise back into a valid action chunk *for this observation*.

Training is one line: MSE(predicted eps, true eps) at a random t.
Sampling runs the T-step walk. Quick self-test:  uv run python -m patchpol.diffusion
"""

import math

import torch
import torch.nn as nn

from patchpol.model import Block, PatchTrunk


# ---------------------------------------------------------------- schedule --
class DDPMSchedule:
    """Squared-cosine beta schedule (Nichol & Dhariwal) — the one Diffusion
    Policy uses. All tensors are precomputed lookup tables indexed by t."""

    def __init__(self, n_steps: int = 100, device: torch.device = "cpu"):
        self.n_steps = n_steps
        s = 0.008  # small offset so ab(0) ~ 1 without a singularity
        ts = torch.linspace(0, 1, n_steps + 1, device=device)
        f = torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]                              # ab(0)=1, decays to ~0
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        self.betas = betas                                # (T,)
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)    # (T,)
        prev = torch.cat([torch.ones(1, device=device), self.alpha_bar[:-1]])
        # posterior variance beta~_t = (1-ab_{t-1})/(1-ab_t) * beta_t
        self.post_var = betas * (1 - prev) / (1 - self.alpha_bar)

    def add_noise(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor):
        """q(x_t | x_0): one jump to any noise level. t: (B,) long."""
        ab = self.alpha_bar[t].view(-1, 1, 1)             # broadcast over (B,5,2)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    @torch.no_grad()
    def sample(self, denoiser, cond: torch.Tensor, shape) -> torch.Tensor:
        """Ancestral sampling: pure noise -> action chunk, conditioned on obs."""
        B = cond.shape[0]
        x = torch.randn(B, *shape, device=cond.device)
        for t in reversed(range(self.n_steps)):
            tt = torch.full((B,), t, device=cond.device, dtype=torch.long)
            eps = denoiser(x, tt, cond)
            # mean of p(x_{t-1} | x_t): remove the predicted noise
            x = (x - self.betas[t] / (1 - self.alpha_bar[t]).sqrt() * eps) \
                / self.alphas[t].sqrt()
            if t > 0:  # last step is deterministic
                x = x + self.post_var[t].sqrt() * torch.randn_like(x)
        return x


# ---------------------------------------------------------------- denoiser --
def sinusoidal_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Classic transformer position encoding, here for the *noise level* t."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None].float() * freqs[None, :]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)  # (B, dim)


class DiffusionHead(nn.Module):
    """Transformer denoiser (paper Table 11: 8 layers, 4 heads, 256 dim).

    Sequence = [obs token, time token, 5 noisy-action tokens], full attention
    (no causality here — denoising a chunk is not a temporal process).
    eps is read out at the action-token positions.
    """

    def __init__(self, act_dim: int = 2, horizon: int = 5, cond_dim: int = 384,
                 dim: int = 256, depth: int = 8, heads: int = 4):
        super().__init__()
        self.horizon = horizon
        self.act_in = nn.Linear(act_dim, dim)
        self.cond_in = nn.Linear(cond_dim, dim)
        self.time_in = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pos_emb = nn.Parameter(torch.randn(1, horizon + 2, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, act_dim)
        S = horizon + 2
        self.register_buffer("mask", torch.ones(S, S, dtype=torch.bool), persistent=False)

    def forward(self, noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        """(B,5,2) noisy chunk, (B,) t, (B,384) obs readout -> (B,5,2) eps."""
        toks = torch.cat([
            self.cond_in(cond)[:, None],                       # (B,1,dim)
            self.time_in(sinusoidal_emb(t, self.pos_emb.shape[-1]))[:, None],
            self.act_in(noisy),                                # (B,5,dim)
        ], dim=1) + self.pos_emb
        for blk in self.blocks:
            toks = blk(toks, self.mask)
        return self.out(self.norm(toks[:, 2:]))                # action positions only


# ------------------------------------------------------------ full policy --
class PatchPolicy(nn.Module):
    """Trunk over dense patch tokens -> readout -> DDPM head over action chunks."""

    def __init__(self, obs_horizon: int = 2, act_horizon: int = 5,
                 act_dim: int = 2, n_diffusion_steps: int = 100):
        super().__init__()
        self.act_horizon, self.act_dim = act_horizon, act_dim
        self.trunk = PatchTrunk(dim=384, depth=8, heads=6, T=obs_horizon, P=256)
        self.head = DiffusionHead(act_dim=act_dim, horizon=act_horizon, cond_dim=384)
        self.n_diffusion_steps = n_diffusion_steps
        self._sched = None  # built lazily on the right device

    def sched(self, device) -> DDPMSchedule:
        if self._sched is None or self._sched.betas.device != torch.device(device):
            self._sched = DDPMSchedule(self.n_diffusion_steps, device=device)
        return self._sched

    def loss(self, feats: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """feats (B,T,256,384), actions (B,5,2) in [-1,1] -> scalar MSE."""
        sched = self.sched(feats.device)
        cond = self.trunk(feats)[:, -1]  # last frame's readout has seen it all
        t = torch.randint(0, sched.n_steps, (len(actions),), device=feats.device)
        eps = torch.randn_like(actions)
        noisy = sched.add_noise(actions, eps, t)
        return nn.functional.mse_loss(self.head(noisy, t, cond), eps)

    @torch.no_grad()
    def act(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (B,T,256,384) -> (B,5,2) action chunk in [-1,1]."""
        cond = self.trunk(feats)[:, -1]
        a = self.sched(feats.device).sample(self.head, cond, (self.act_horizon, self.act_dim))
        return a.clamp(-1, 1)


# ------------------------------------------------------------------- EMA ---
class EMA:
    """Exponential moving average of weights (Diffusion Policy's warmup rule:
    decay ramps 0 -> ~0.9999 so early garbage weights wash out quickly)."""

    def __init__(self, model: nn.Module, power: float = 0.75, max_decay: float = 0.9999):
        self.power, self.max_decay = power, max_decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int):
        decay = min(self.max_decay, 1 - (1 + step) ** -self.power)
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].lerp_(v.detach(), 1 - decay)  # s = d*s + (1-d)*v
            else:
                self.shadow[k].copy_(v)


if __name__ == "__main__":
    torch.manual_seed(0)
    sched = DDPMSchedule(100)
    assert sched.alpha_bar[0] > 0.99 and sched.alpha_bar[-1] < 0.01, "schedule endpoints off"
    assert (sched.alpha_bar.diff() < 0).all(), "alpha_bar must decay monotonically"

    policy = PatchPolicy()
    n = sum(p.numel() for p in policy.parameters())
    feats, acts = torch.randn(4, 2, 256, 384), torch.rand(4, 5, 2) * 2 - 1
    loss = policy.loss(feats, acts)
    loss.backward()  # verify grads flow through trunk AND head
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    chunk = policy.act(feats[:2])
    assert chunk.shape == (2, 5, 2) and chunk.abs().max() <= 1
    print(f"params {n/1e6:.1f}M | loss {loss.item():.3f} | {len(grads)} tensors got grads")
    print("diffusion self-test passed ✓")
