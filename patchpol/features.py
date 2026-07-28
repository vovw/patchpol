"""Precompute frozen DINOv2 ViT-S/14 patch features for every frame.

Reads:  data/pusht/img224.zarr        (25650, 224, 224, 3) uint8
Writes: data/pusht/dino_vits14.zarr   (25650, 256, 384) float16

Run:  uv run python -m patchpol.features   (or patchpol.prepare, which chains it)
"""

import numpy as np
import torch
import zarr
from tqdm import tqdm

IMG_ZARR = "data/pusht/img224.zarr"
OUT_ZARR = "data/pusht/dino_vits14.zarr"
BATCH = 64
N_PATCHES = 256  # (224/14)^2
DIM = 384        # ViT-S embed dim

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess(imgs_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B, 224, 224, 3) uint8 numpy  ->  (B, 3, 224, 224) float32 on device,
    scaled to [0,1] then ImageNet-normalized."""
    x = torch.from_numpy(imgs_u8).to(device)   # uint8 (B,224,224,3)
    x = x.permute(0, 3, 1, 2).float() / 255.0  # -> (B,3,224,224) in [0,1]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def load_dino(device: torch.device) -> torch.nn.Module:
    """Load frozen DINOv2 ViT-S/14 ready for inference."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()                    # dropout etc. -> inference behavior
    model.requires_grad_(False)     # freeze: this net is a fixed feature extractor
    return model.to(device)


def main():
    imgs = zarr.open(IMG_ZARR, "r")
    n = imgs.shape[0]
    out = zarr.open(
        OUT_ZARR, "w", shape=(n, N_PATCHES, DIM), chunks=(256, N_PATCHES, DIM),
        dtype="f2",
    )

    device = get_device()
    model = load_dino(device)
    print(f"device={device}, frames={n}")

    with torch.inference_mode():
        for i in tqdm(range(0, n, BATCH)):
            batch_u8 = imgs[i : i + BATCH]          # (b, 224, 224, 3) uint8
            x = preprocess(batch_u8, device)        # (b, 3, 224, 224) f32

            # forward -> dict; patch tokens are (b, 256, 384) on device.
            # Round trip device -> cpu -> fp16 -> numpy for zarr storage.
            feats = model.forward_features(x)["x_norm_patchtokens"]
            feats = feats.cpu().half().numpy()

            out[i : i + len(batch_u8)] = feats

    print(f"wrote {OUT_ZARR}: {out.shape} float16")


if __name__ == "__main__":
    main()
