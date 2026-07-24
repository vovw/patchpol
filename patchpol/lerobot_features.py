"""Precompute DINOv2 patch features for every lerobot/pusht frame.

The v3 videos are 96x96 but the paper feeds 224x224 to DINOv2. This dataset
has no full sim state (observation.state is agent pos only), so unlike
render224.py we can't replay frames through the renderer — we bicubic-upscale
96 -> 224 instead. Expect slightly softer features than the re-rendered zarr.

Run:  uv run python -m patchpol.lerobot_features
Writes: data/lerobot/pusht/dino_vits14.zarr  (25650, 256, 384) float16
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm

from patchpol.features import (
    DIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    N_PATCHES,
    get_device,
    load_dino,
)
from patchpol.lerobot_data import FEATS_ZARR, iter_video_frames, load_info

BATCH = 64
RES = 224


def batched(it, size: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            yield np.stack(buf)
            buf = []
    if buf:
        yield np.stack(buf)


def upscale_preprocess(frames_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B, 96, 96, 3) uint8 numpy -> (B, 3, 224, 224) float32 on device,
    bicubic-upscaled, [0,1]-scaled, ImageNet-normalized."""
    x = torch.from_numpy(frames_u8).to(device)
    x = x.permute(0, 3, 1, 2).float() / 255.0            # (B,3,96,96) in [0,1]
    x = F.interpolate(x, size=(RES, RES), mode="bicubic", antialias=True)
    x = x.clamp_(0, 1)                                   # bicubic overshoots
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def main():
    n = load_info()["total_frames"]
    out = zarr.open(
        FEATS_ZARR, "w", shape=(n, N_PATCHES, DIM), chunks=(256, N_PATCHES, DIM),
        dtype="f2",
    )

    device = get_device()
    model = load_dino(device)
    print(f"device={device}, frames={n}")

    written = 0
    with torch.inference_mode():
        for batch_u8 in tqdm(batched(iter_video_frames(), BATCH),
                             total=math.ceil(n / BATCH)):
            x = upscale_preprocess(batch_u8, device)
            feats = model.forward_features(x)["x_norm_patchtokens"]
            out[written : written + len(batch_u8)] = feats.cpu().half().numpy()
            written += len(batch_u8)

    assert written == n
    print(f"wrote {FEATS_ZARR}: {out.shape} float16")


if __name__ == "__main__":
    main()
