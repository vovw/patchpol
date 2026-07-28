# Patch Policy

Visuomotor policies don't need a pooled feature vector — they need dense ones.

Reproduction of [Patch Policy: Efficient Embodied Control via Dense Visual
Representations](https://patch-policy.github.io/) (arXiv 2607.18236): frozen
DINOv2 patch tokens → block-causal transformer trunk → DDPM action head.
Push-T in sim first, then the same trunk driving a real SO-101 arm.

<table>
<tr>
<td><img src="assets/pusht_rollout.gif" width="320" alt="Push-T rollout"></td>
<td><img src="assets/so101_hardware.gif" width="320" alt="SO-101 pick and place"></td>
</tr>
<tr>
<td align="center"><b>Push-T (sim)</b><br>one <code>patchpol.eval</code> rollout</td>
<td align="center"><b>SO-101 (real)</b><br>2× speed · inference on a MacBook M3 Pro</td>
</tr>
</table>

Sim: **paper target 0.83 coverage, this repro gets 0.772** over 100 rollouts
([Results](#results)). Hardware: [Real robot](#real-robot-so-101).

## How it works

```
224×224 frame ──frozen DINOv2 ViT-S/14──▶ 256 patch tokens × 384d      features.py
2 frames of tokens ──block-causal transformer──▶ per-frame readout     model.py
readout ──DDPM denoiser (100 steps)──▶ 5-action chunk in [-1,1]        diffusion.py
```

The block-causal mask is the paper's core: bidirectional attention *within* a
frame's 256 tokens, causal *across* frames. It's one comparison in
`model.py:block_causal_mask`, unit-tested by perturbation (no time travel /
memory flows forward / intra-frame bidirectional):

```bash
uv run python -m patchpol.model        # causality tests
uv run python -m patchpol.diffusion    # schedule + sampler self-test
```

Nothing here is Push-T-specific — the trunk takes patch tokens and emits
readouts, so the same code drives a 2-D pusher and a 6-DoF arm.

## Two data paths

Same 206 demos either way (25,650 frames, 10 fps), different image fidelity.
**Train-time and eval-time featurization must match**: DINOv2 features shift
with input fidelity, so mixing the paths costs coverage silently.

|          | original zarr (default)                                | LeRobot hub                                     |
|----------|--------------------------------------------------------|-------------------------------------------------|
| source   | `pusht_cchi_v7_replay.zarr` (Diffusion Policy release)  | `lerobot/pusht` @ tag `v3.0`                    |
| images   | re-rendered at 224 from the 5-D sim state               | 96×96 AV1 video, bicubic-upscaled to 224        |
| features | `data/pusht/dino_vits14.zarr`                           | `data/lerobot/pusht/dino_vits14.zarr`           |
| train    | `--data zarr`                                           | `--data lerobot`                                |
| eval     | *(default)*                                             | `--pixels96`                                    |

The LeRobot path can't re-render: its `observation.state` is agent position
only (2-D), so the block pose the renderer needs isn't in the dataset. Expect
it to land somewhat below the zarr path's 0.772 — upscaled 96px frames give
softer patch features than native 224px renders.

## Setup

```bash
uv sync
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 24 GB cards
```

**Never `uv add lerobot`.** Every lerobot release pins `numpy<2.3` and
`torchvision<0.26`, unsatisfiable against this project's torch ≥2.13 /
numpy ≥2.5 stack — installing it downgrades the whole torch stack. LeRobot v3
datasets are read by our own `patchpol/lerobot_data.py` instead, on
huggingface-hub + pyarrow + imageio-ffmpeg alone.

## Train on LeRobot data

**1. Download and verify the dataset**

```bash
uv run python -m patchpol.lerobot_data
```

Pulls `lerobot/pusht` @ `v3.0` into `data/lerobot/pusht`, then self-checks the
reader: episode metadata against `meta/info.json`, a full decode of every
video, and the window-padding rules at episode boundaries.

```
meta ok: 206 eps, 25650 frames, fps 10
video ok: 25650 frames
```

The videos are AV1-encoded; imageio-ffmpeg's bundled ffmpeg 7.x decodes them
fine. If frames come back empty, your ffmpeg build lacks AV1 support.

**2. Precompute DINOv2 features** (writes ~5 GB)

```bash
uv run python -m patchpol.lerobot_features
```

Decodes every frame, bicubic-upscales 96 → 224, runs frozen DINOv2 ViT-S/14,
writes `(25650, 256, 384)` fp16 to `data/lerobot/pusht/dino_vits14.zarr`.
Wall clock is dominated by AV1 video decode on the CPU, not by the GPU.
First run on a machine needs internet — torch.hub fetches DINOv2.

Re-running step 1 now also exercises the dataset class:
`dataset ok: 25650 samples`.

**3. Train** (50k steps, ~3.5 h on a 4090 at ~3.9 it/s)

```bash
uv run python -m patchpol.train --data lerobot --amp --batch-size 256
```

Paper config (Table 11): lr 1e-4, weight decay 0, batch 256, obs horizon 2,
action horizon 5; cosine LR with 500 warmup steps, EMA tracked throughout.
Loss falls ~1.0 → ~0.003. Checkpoints land in `checkpoints/` every 5k steps,
plus `final.pt`.

Actions are min/max-normalized to [-1,1] from `meta/stats.json`
(`[12,25]`..`[511,511]`, equal to the true data range). Those bounds ride in
every checkpoint as `act_min`/`act_max` and eval denormalizes from the
checkpoint — so checkpoints stay self-describing across both data paths.

**4. Eval — with `--pixels96`** (100 rollouts)

```bash
uv run python -m patchpol.eval --ckpt checkpoints/final.pt --pixels96
```

`--pixels96` renders the env at 96 and upscales to 224, mirroring step 2.
Drop it and the policy gets sharper features than it trained on. Eval always
loads the **EMA** weights (`ckpt["ema"]`), and reports final coverage (the
paper's metric), max coverage, and success rate (coverage > 0.95).

Add `--episodes 5 --video 2` to save `rollout_0.mp4` / `rollout_1.mp4` — the
left-hand GIF up top is one of those.

## Train on the original zarr

```bash
uv run python -m patchpol.prepare                        # download -> re-render @224 -> features
uv run python -m patchpol.train --amp --batch-size 256
uv run python -m patchpol.eval --ckpt checkpoints/final.pt
```

`prepare.py` is idempotent — it skips any stage whose output already exists.
It fetches `pusht.zip` from the Diffusion Policy release, replays all 25,650
frames through the pusht renderer at 224 (`render224.py`), then featurizes
(`features.py`).

### `--amp` is not optional at batch 256 on a 24 GB card

The block-causal `attn_mask` forces `F.scaled_dot_product_attention` off the
flash kernel and onto the math path, which materializes the full
`(B, heads, 512, 512)` score matrix. fp32 at bs 256 needs ~24.1 GB and OOMs on
a 4090; bf16 autocast peaks at **15.8 GB** and is ~2.3× faster, with the same
loss curve. To avoid autocast, keep the paper's effective batch by
accumulating instead (~8 h):

```bash
uv run python -m patchpol.train --batch-size 128 --grad-accum 2
```

The features zarr is loaded fully into RAM (~5 GB fp16) at train start.

## Results

100 rollouts on `final.pt` (50k steps, EMA weights), original zarr path,
RTX 4090:

| metric | this repro | paper (Table 13) |
|---|---|---|
| **final coverage** | **0.772 ± 0.032** | **0.83** |
| max coverage | 0.821 ± 0.028 | — |
| success rate (>0.95) | 52% | — |

About 0.06 short. The informative number is the gap between *max* (0.821,
essentially at target) and *final* (0.772): the policy reliably drives the T
onto the goal but sometimes drifts back off before the 300-step limit, losing
coverage it had already earned. That's terminal-holding, not a failure to
learn the task — closing it is where the remaining 0.06 lives (longer
training, receding-horizon replanning, or the paper's per-frame chunk readout).

Coverage varies run to run: only the env seed is fixed (`env.reset(seed=ep)`),
while the DDPM sampler draws fresh noise every rollout.

Progress for reference, same checkpoint family early in training:

| checkpoint | final coverage | success rate |
|---|---|---|
| `step_5000.pt` (10 eps) | 0.596 | 40% |
| `final.pt` (100 eps) | 0.772 | 52% |

## Real robot (SO-101)

> **Scope:** the two-view training script is not in this repo — it lives on the
> training box, with a vendored reference copy inside the exported LeRobot
> plugin (`lerobot_plugin/patchpol/`). What follows is the recipe as it was
> run, not commands you can execute from this checkout.

The right-hand GIF is this trunk and DDPM head driving an SO-101 follower arm:
pick the cube off the table, drop it in the box. Inference runs on a MacBook
M3 Pro, ~400 ms per action chunk on MPS.
([original clip](https://x.com/k7agar/status/2081313497741447559))

What differs from Push-T:

- two camera views (`observation.images.front`, `observation.images.ee`) rather
  than one, so the trunk runs `views=2`
- actions are 6-D joint positions (`shoulder_pan`, `shoulder_lift`,
  `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`) rather than 2-D pusher
  targets
- data is a LeRobot v3 dataset you record yourself at 30 fps — here 16 episodes
  / 10,342 frames of "grab the cube"

Frozen DINOv2 at 224, the block-causal trunk, obs horizon 2, the 100-step
DDPM head, and min/max action normalization are all unchanged.

**1. Record demos.** Leader/follower teleop with two cameras, in a *separate*
uv project that has `lerobot` installed (see the never-add-lerobot note above):

```bash
uv run lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodemXXXX --robot.id=follow \
  --teleop.type=so101_leader  --teleop.port=/dev/tty.usbmodemYYYY --teleop.id=lead \
  --robot.cameras='{"ee":    {"type":"opencv","index_or_path":0,"width":1920,"height":1080,"fps":30},
                    "front": {"type":"opencv","index_or_path":1,"width":1920,"height":1080,"fps":30}}' \
  --dataset.repo_id=$HF_USER/cube --dataset.num_episodes=16 \
  --dataset.single_task="Grab the cube" --display_data=true
```

macOS reshuffles OpenCV camera indices between sessions — check wrist-vs-front
by eye before every recording *and* every rollout, or the two views arrive
swapped relative to training.

**2. Trim the idle prefixes.** This is the step that decides whether the
policy moves at all. The first trained policy held its home pose forever:
38.1% of recorded frames were idle (mean 3.2 s of dead time before motion
onset per episode), and at action horizon 5, **0.0%** of those idle frames had
any motion in their 5-step label — so "hold still" is the correct answer for
more than a third of the training set, and the policy learned it. ACT moves on
the same data only because its horizon-100 labels *do* contain the onset
(48.5% of idle frames). Fix: trim idle prefixes and tails to ~0.5 s before
onset, and raise the action horizon to ~50, executing ~25 before replanning.

**3. Featurize and train** the two-view trunk on the trimmed dataset —
DINOv2 features per view, `views=2`, 6-D actions, otherwise the same loop as
`patchpol/train.py`.

**4. Smoke-test offline, before the robot moves.** Feed recorded frames
through the exported checkpoint, build the 2-frame history exactly like a
training window, and compare predicted chunks against teleop ground truth per
joint; assert every prediction lands inside the recorded action range.

**5. Roll out.** The export uses LeRobot's pretrained-policy layout
(`config.json`, `model.safetensors`, pre/post-processor state), so a one-time
plugin install registers `patchpol` for LeRobot's policy discovery:

```bash
uv pip install -e path/to/lerobot_plugin
lerobot-rollout --policy.path=<hf-repo-or-local-dir> --robot.type=so101_follower ...
```

Do **not** pass `--policy.discover_packages_path` — discovery already imports
`lerobot_policy_*` packages, and the flag breaks argument parsing.

## Files

| file | what |
|---|---|
| `patchpol/model.py` | block-causal trunk (8L/6H/384d, 14.4M params) + causality unit tests |
| `patchpol/diffusion.py` | hand-rolled DDPM (squared-cosine schedule), transformer denoiser (8L/4H/256d), EMA |
| `patchpol/train.py` | AdamW, cosine LR + warmup, EMA tracking, `--data`, `--amp` (bf16), `--grad-accum` |
| `patchpol/eval.py` | gym-pusht rollouts, online DINOv2, coverage metrics, `--pixels96`, videos |
| `patchpol/features.py` | frozen DINOv2 ViT-S/14 → (25650, 256, 384) fp16 patch features |
| `patchpol/dataset.py` | zarr → (2-frame obs, 5-action chunk) windows, episode-boundary padding by index clipping |
| `patchpol/render224.py` | re-render all 25,650 frames at 224px from ground-truth sim state |
| `patchpol/prepare.py` | one-shot data prep for the zarr path on a fresh machine |
| `patchpol/lerobot_data.py` | minimal LeRobotDataset v3 reader (no `lerobot` package) + dataset class + self-checks |
| `patchpol/lerobot_features.py` | 96 → 224 upscale + DINOv2 precompute for the LeRobot path |

## Known deviations from the paper

- Paper doesn't specify the trunk size for the DP variant (Table 11's
  8L/4H/256 reads as the denoiser); we use 8L/6H/384 — no projection, so trunk
  width = DINOv2's 384.
- Diffusion specifics (squared-cosine schedule, 100 steps, EMA) follow the
  original Diffusion Policy, since the paper doesn't state them.
- Conditioning uses the last frame's readout only; the paper reads out a chunk
  at every frame. Most likely source of the terminal-drift gap in
  [Results](#results).
- Training runs in bf16 autocast (`--amp`); the paper doesn't state a precision.
- `pymunk` is held `<7` — gym-pusht uses the pymunk 6 collision API.
