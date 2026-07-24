"""Evaluate Patch Policy in gym-pusht.

Paper protocol: 100 rollouts, metric = coverage of the T-block over the
target. We report final coverage, max coverage over the episode, and
success rate (coverage > 0.95, at which point the env terminates).

  uv run python -m patchpol.eval --ckpt checkpoints/final.pt
  uv run python -m patchpol.eval --ckpt checkpoints/final.pt --episodes 5 --video 2
"""

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
from gym_pusht.envs import PushTEnv

from patchpol.diffusion import PatchPolicy
from patchpol.features import get_device, load_dino, preprocess

OBS_HORIZON = 2
MAX_STEPS = 300
RES = 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--video", type=int, default=0, help="save this many episode mp4s")
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    policy = PatchPolicy().to(device)
    policy.load_state_dict(ckpt["ema"])  # EMA weights, not the raw ones
    policy.eval()
    act_min = ckpt["act_min"].cpu().numpy()
    act_max = ckpt["act_max"].cpu().numpy()
    print(f"loaded {args.ckpt} (step {ckpt['step']}) on {device}")

    dino = load_dino(device)

    @torch.inference_mode()
    def frame_features(img_u8: np.ndarray) -> torch.Tensor:
        x = preprocess(img_u8[None], device)                     # (1,3,224,224)
        return dino.forward_features(x)["x_norm_patchtokens"][0]  # (256, 384)

    env = PushTEnv(obs_type="pixels", render_mode="rgb_array",
                   observation_width=RES, observation_height=RES)

    finals, maxes, successes = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)  # fixed seeds -> reproducible eval set
        # history starts as the first frame twice — same padding rule as training
        hist = [frame_features(obs)] * OBS_HORIZON
        frames, max_cov, cov, steps = [obs], 0.0, 0.0, 0

        while steps < MAX_STEPS:
            feats = torch.stack(hist)[None]                # (1, 2, 256, 384)
            with torch.inference_mode():
                chunk = policy.act(feats)[0].cpu().numpy()  # (5, 2) in [-1,1]
            chunk = (chunk + 1) / 2 * (act_max - act_min) + act_min  # denormalize

            done = False
            for a in chunk:  # execute the whole chunk, then replan
                obs, _, terminated, _, info = env.step(a.astype(np.float32))
                hist = [hist[-1], frame_features(obs)]
                cov = info["coverage"]
                max_cov = max(max_cov, cov)
                steps += 1
                if ep < args.video:
                    frames.append(obs)
                if terminated or steps >= MAX_STEPS:
                    done = terminated
                    break
            if done:
                break

        finals.append(cov)
        maxes.append(max_cov)
        successes.append(done)
        print(f"ep {ep:>3}: final {cov:.3f} | max {max_cov:.3f} | "
              f"{'SUCCESS' if done else 'timeout'} @ {steps} steps", flush=True)

        if ep < args.video:
            import imageio.v3 as iio
            iio.imwrite(f"rollout_{ep}.mp4", np.stack(frames), fps=30)
            print(f"  saved rollout_{ep}.mp4")

    print("\n==== results ====")
    print(f"episodes:       {args.episodes}")
    print(f"final coverage: {np.mean(finals):.3f} ± {np.std(finals)/np.sqrt(len(finals)):.3f}")
    print(f"max coverage:   {np.mean(maxes):.3f} ± {np.std(maxes)/np.sqrt(len(maxes)):.3f}")
    print(f"success rate:   {np.mean(successes):.2%}")
    print("(paper, DINOv2 + DP head: 0.83 coverage)")


if __name__ == "__main__":
    main()
