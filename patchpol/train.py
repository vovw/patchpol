"""Train Patch Policy (DP head) on Push-T.

Paper config (Table 11): lr 1e-4, weight decay 0, batch 256,
obs horizon 2, action horizon 5.

  uv run python -m patchpol.train                 # full run (default 50k steps)
  uv run python -m patchpol.train --steps 300     # smoke test

Checkpoints land in checkpoints/ — 'ema' weights are the ones to eval.
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from patchpol.dataset import PushTDataset
from patchpol.diffusion import EMA, PatchPolicy
from patchpol.features import OUT_ZARR, get_device


def cosine_lr(step: int, total: int, base: float, warmup: int = 500) -> float:
    if step < warmup:
        return base * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="checkpoints")
    ap.add_argument("--save-every", type=int, default=5_000)
    args = ap.parse_args()

    device = get_device()
    ds = PushTDataset(features_zarr=OUT_ZARR)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=2, persistent_workers=True,
    )

    policy = PatchPolicy().to(device)
    ema = EMA(policy)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    def save(name: str, step: int):
        torch.save(
            {
                "model": policy.state_dict(),
                "ema": ema.shadow,
                "act_min": torch.as_tensor(ds.act_min),  # eval must denormalize identically
                "act_max": torch.as_tensor(ds.act_max),
                "step": step,
            },
            out_dir / name,
        )

    n = sum(p.numel() for p in policy.parameters())
    print(f"device={device} | params={n/1e6:.1f}M | {args.steps} steps @ bs={args.batch_size}")

    step, t0, running = 0, time.time(), 0.0
    while step < args.steps:
        for batch in dl:  # re-enter the loader each epoch
            if step >= args.steps:
                break
            lr = cosine_lr(step, args.steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            loss = policy.loss(batch["obs"].to(device), batch["action"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(policy, step)

            running += loss.item()
            step += 1
            if step % 100 == 0:
                sps = step / (time.time() - t0)
                eta_min = (args.steps - step) / sps / 60
                print(f"step {step:>6}/{args.steps} | loss {running/100:.4f} | "
                      f"lr {lr:.2e} | {sps:.1f} it/s | eta {eta_min:.0f}m", flush=True)
                running = 0.0
            if step % args.save_every == 0:
                save(f"step_{step}.pt", step)

    save("final.pt", step)
    print(f"done. final checkpoint: {out_dir/'final.pt'}")


if __name__ == "__main__":
    main()
