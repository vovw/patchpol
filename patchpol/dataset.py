"""Push-T dataset: (2-frame obs window, 5-action chunk) samples.

Fill in the four TODOs, then run:  uv run python -m patchpol.dataset
"""

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

ZARR_PATH = "data/pusht/pusht_cchi_v7_replay.zarr"

OBS_HORIZON = 2   # frames of context the policy sees
ACT_HORIZON = 5   # actions the policy must predict


class PushTDataset(Dataset):
    def __init__(self, zarr_path: str = ZARR_PATH, features_zarr: str | None = None):
        """If features_zarr is given, 'obs' is DINOv2 features (T, 256, 384)
        instead of raw images. Features load fully into RAM (~5 GB fp16)."""
        root = zarr.open(zarr_path, "r")

        self.feats = None
        if features_zarr is not None:
            self.feats = zarr.open(features_zarr, "r")[:]  # (25650, 256, 384) f16

        self.actions = root["data/action"][:]          # (25650, 2) float32
        self.states = root["data/state"][:]            # (25650, 5) float32
        episode_ends = root["meta/episode_ends"][:]    # (206,) cumulative!

        self.imgs = root["data/img"]                   # (25650, 96, 96, 3)

        self.act_min = self.actions.min(axis=0)
        self.act_max = self.actions.max(axis=0)

        starts, ends = [], []
        prev = 0
        for end in episode_ends:
            n = end - prev
            starts.append(np.full(n, prev))
            ends.append(np.full(n, end))
            prev = end
        self.ep_start = np.concatenate(starts)
        self.ep_end = np.concatenate(ends)

    def normalize_action(self, a: np.ndarray) -> np.ndarray:
        return 2*(a - self.act_min) / (self.act_max - self.act_min) - 1

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, t: int) -> dict:
        lo, hi = self.ep_start[t], self.ep_end[t] - 1
        obs_idx = np.clip(np.arange(t - OBS_HORIZON + 1, t +1), lo, hi)
        act_idx = np.clip(np.arange(t, t + ACT_HORIZON), lo, hi)

        if self.feats is not None:
            obs = self.feats[obs_idx].astype(np.float32)  # (2, 256, 384)
        else:
            obs = np.ascontiguousarray(self.imgs[obs_idx])  # (2, 96, 96, 3)
        action = self.normalize_action(self.actions[act_idx])  # (5, 2)

        return {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(action.astype(np.float32)),
        }

