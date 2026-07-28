# CLAUDE.md

Reproduction of Patch Policy (arXiv 2607.18236) on Push-T: frozen DINOv2
ViT-S/14 patch tokens → block-causal transformer trunk → DDPM action head.
Paper target 0.83 final coverage; this repro reached 0.772 on the original
data path. README.md has results, architecture notes, and the per-file map.

## Environment

- `uv` for everything: `uv sync`, `uv run python -m patchpol.<module>`.
- Python is pinned `>=3.12,<3.13`.
- **Never `uv add lerobot`.** Every lerobot release pins `numpy<2.3` and
  `torchvision<0.26`, which is unsatisfiable against this project's
  torch>=2.13 / numpy>=2.5 stack and would downgrade all of it. LeRobot v3
  datasets are read by our own minimal reader (`patchpol/lerobot_data.py`)
  using only huggingface-hub + pyarrow + imageio-ffmpeg.
- `pymunk` must stay `<7` (gym-pusht uses the pymunk 6 collision API).
- The dev box is a Mac (MPS); training, feature precompute, and full evals
  run on an RTX 4090 box. Don't launch heavy compute on the Mac unasked.

## Two data pipelines

Train-time features and eval-time featurization MUST match. The paths differ
in image fidelity, and DINOv2 features shift with it — mixing them degrades
coverage silently.

|            | original zarr (default)                   | lerobot hub                                    |
|------------|-------------------------------------------|------------------------------------------------|
| source     | `data/pusht/pusht_cchi_v7_replay.zarr`    | `lerobot/pusht` @ tag v3.0 → `data/lerobot/pusht` |
| images     | re-rendered at 224 from 5-D sim state (`render224.py`) | 96×96 av1 video, bicubic-upscaled to 224 (`observation.state` is agent pos only — re-rendering is impossible) |
| features   | `data/pusht/dino_vits14.zarr`             | `data/lerobot/pusht/dino_vits14.zarr`          |
| train flag | `--data zarr`                             | `--data lerobot`                               |
| eval flag  | (none)                                    | `--pixels96`                                   |

Same underlying demos either way: 206 episodes, 25,650 frames, 10 fps.
Actions are min/max-normalized to [-1,1]; the lerobot path reads min/max from
`meta/stats.json` (`[12,25]..[511,511]`, equal to the true data range). Both
values ride in every checkpoint as `act_min`/`act_max`, and eval denormalizes
from the checkpoint — so checkpoints stay self-describing across paths.

Expect the lerobot path to score somewhat below 0.772: upscaled 96px frames
give softer DINOv2 features than the re-rendered 224px originals.

## Commands (4090 box)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv sync

# lerobot path
uv run python -m patchpol.lerobot_data       # auto-downloads + self-checks the reader
uv run python -m patchpol.lerobot_features   # DINOv2 precompute -> ~5 GB zarr
uv run python -m patchpol.train --data lerobot --amp --batch-size 256   # 50k steps, ~3.5 h
uv run python -m patchpol.eval --ckpt checkpoints/final.pt --pixels96   # 100 rollouts

# original path
uv run python -m patchpol.prepare
uv run python -m patchpol.train --amp --batch-size 256
uv run python -m patchpol.eval --ckpt checkpoints/final.pt
```

## Gotchas

- `--amp` is mandatory at batch 256 on a 24 GB card: the block-causal
  `attn_mask` forces SDPA onto the math path, materializing the full
  (B, heads, 512, 512) score matrix — fp32 needs ~24.1 GB and OOMs; bf16
  peaks at 15.8 GB and is ~2.3× faster. No-autocast alternative:
  `--batch-size 128 --grad-accum 2` (~8 h).
- Eval must load the EMA weights (`ckpt["ema"]`) — eval.py already does.
- The features zarr is loaded fully into RAM (~5 GB fp16) at train start.
- DINOv2 loads via torch.hub (facebookresearch/dinov2); first run per machine
  needs internet.
- lerobot videos are av1-encoded; imageio-ffmpeg's bundled ffmpeg 7.x decodes
  them. If frames come back empty, check the ffmpeg build for av1 support.
- Self-tests: `uv run python -m patchpol.model` (block-causal mask
  perturbation tests), `uv run python -m patchpol.lerobot_data` (reader,
  frame-count, and episode-boundary windowing checks).
