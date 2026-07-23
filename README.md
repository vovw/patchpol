# Patch Policy

visuomotor policies simply need dense features.
replacing global-pooled features with these patch features

- dense representations outperform global features for control
- pretrained ViT features transfer to control
- spatial compression degrades control ??
- "highly efficient"

Reproduction of [Patch Policy: Efficient Embodied Control via Dense Visual
Representations](https://patch-policy.github.io/) (arXiv 2607.18236) on Push-T,
DINOv2 + Diffusion Policy head. **Paper target: 0.83 coverage** (Table 13),
100 rollouts.

## How it works

```
224x224 frame ──frozen DINOv2 ViT-S/14──▶ 256 patch tokens × 384d   (features.py)
2 frames of tokens ──block-causal transformer──▶ per-frame readout   (model.py)
readout ──DDPM denoiser (100 steps)──▶ 5-action chunk in [-1,1]      (diffusion.py)
```

The block-causal mask is the paper's core: bidirectional attention *within* a
frame's 256 tokens, causal *across* frames. `model.py` unit-tests this by
perturbation (no time travel / memory flows forward / intra-frame bidirectional).

## Setup (fresh machine)

```bash
uv sync
uv run python -m patchpol.prepare   # download 206 demos -> re-render @224 -> DINOv2 features (~5GB)
```

## Train

```bash
uv run python -m patchpol.train                # 50k steps, bs 256, lr 1e-4 (Table 11)
```

Checkpoints land in `checkpoints/` every 5k steps. Loss should fall from ~1.0
to well under 0.1. On a 4090 expect roughly 1–2 h.

## Eval

```bash
uv run python -m patchpol.eval --ckpt checkpoints/final.pt                  # 100 rollouts
uv run python -m patchpol.eval --ckpt checkpoints/final.pt --episodes 5 --video 2
```

Reports final coverage (paper's metric), max coverage, and success rate
(coverage > 0.95). Eval uses the **EMA** weights.

## Files

| file | what |
|---|---|
| `patchpol/dataset.py` | zarr -> (2-frame obs, 5-action chunk) windows, episode-boundary padding via index clipping |
| `patchpol/render224.py` | re-render all 25,650 frames at 224px from ground-truth sim state |
| `patchpol/features.py` | frozen DINOv2 ViT-S/14 -> (25650, 256, 384) fp16 patch features |
| `patchpol/model.py` | block-causal trunk (8L/6H/384d, 14.4M params) + causality unit tests |
| `patchpol/diffusion.py` | hand-rolled DDPM (squared-cosine schedule), transformer denoiser (8L/4H/256d), EMA |
| `patchpol/train.py` | AdamW, cosine LR + warmup, EMA tracking |
| `patchpol/eval.py` | gym-pusht rollouts, online DINOv2, coverage metrics, videos |
| `patchpol/prepare.py` | one-shot data prep for a fresh machine |

## Known deviations from the paper

- Paper doesn't specify the trunk size for the DP variant (Table 11's
  8L/4H/256 reads as the denoiser); we use 8L/6H/384 (no projection, so trunk
  width = DINOv2's 384).
- Diffusion specifics (schedule, 100 steps, EMA) follow the original Diffusion
  Policy since the paper doesn't state them.
- Conditioning uses the last frame's readout only; the paper reads out a chunk
  at every frame.
- `pymunk` must stay `<7` (gym-pusht uses the pymunk 6 collision API).
