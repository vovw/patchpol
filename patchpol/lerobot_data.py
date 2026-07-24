"""Minimal LeRobotDataset v3 reader — no `lerobot` package needed.

On-disk layout (data/lerobot/pusht, hub repo lerobot/pusht @ tag v3.0):
  meta/info.json            fps, feature schema, path templates
  meta/episodes/…/*.parquet per-episode: row span in data files, time span in videos
  data/…/*.parquet          all episodes concatenated row-wise: action, state, index, …
  videos/<key>/…/*.mp4      frames, many episodes back-to-back per file (av1)

Same underlying data as pusht_cchi_v7_replay.zarr: 206 episodes, 25650 frames,
10 fps — but images are 96x96 video (no 5-D sim state, so no 224 re-render;
lerobot_features.py upscales instead).

Self-check:  uv run python -m patchpol.lerobot_data
"""

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from torch.utils.data import Dataset

ROOT = Path("data/lerobot/pusht")
FEATS_ZARR = "data/lerobot/pusht/dino_vits14.zarr"
VIDEO_KEY = "observation.image"

OBS_HORIZON = 2   # frames of context the policy sees
ACT_HORIZON = 5   # actions the policy must predict


def download(repo_id: str = "lerobot/pusht", root: Path = ROOT) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(repo_id, repo_type="dataset", revision="v3.0",
                          local_dir=str(root))
    )


def load_info(root: Path = ROOT) -> dict:
    return json.loads((root / "meta/info.json").read_text())


def load_stats(root: Path = ROOT) -> dict:
    return json.loads((root / "meta/stats.json").read_text())


def load_episodes(root: Path = ROOT) -> dict[str, np.ndarray]:
    """Per-episode metadata as flat arrays, sorted by episode_index."""
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    tables = [pq.read_table(f) for f in files]
    cols = {}
    for name in [
        "episode_index", "length", "dataset_from_index", "dataset_to_index",
        f"videos/{VIDEO_KEY}/chunk_index", f"videos/{VIDEO_KEY}/file_index",
        f"videos/{VIDEO_KEY}/from_timestamp", f"videos/{VIDEO_KEY}/to_timestamp",
    ]:
        cols[name] = np.concatenate([t.column(name).to_numpy() for t in tables])
    order = np.argsort(cols["episode_index"])
    return {k: v[order] for k, v in cols.items()}


def load_column(name: str, root: Path = ROOT) -> np.ndarray:
    """One tabular feature over all frames, in global `index` order.

    Relies on chunk-/file- names being zero-padded, so lexicographic file
    order == index order.
    """
    files = sorted((root / "data").rglob("*.parquet"))
    parts = [pq.read_table(f, columns=[name]).column(name).to_numpy() for f in files]
    col = np.concatenate(parts)
    if col.dtype == object:  # list-valued columns (e.g. action) -> (N, D)
        col = np.stack(col)
    return col


def iter_video_frames(root: Path = ROOT):
    """Yield every frame as (96, 96, 3) uint8, in global `index` order.

    Episodes are written back-to-back and in episode order within each mp4
    (frame t of an episode sits at round(from_timestamp * fps) + t), so each
    file decodes once, straight through. Files are walked in the order their
    first episode appears, which is global index order.
    """
    info = load_info(root)
    eps = load_episodes(root)
    fps = info["fps"]
    chunk = eps[f"videos/{VIDEO_KEY}/chunk_index"]
    file_ = eps[f"videos/{VIDEO_KEY}/file_index"]
    from_ts = eps[f"videos/{VIDEO_KEY}/from_timestamp"]
    lengths = eps["length"]

    groups: dict[tuple[int, int], list[int]] = {}
    for e in range(len(lengths)):
        groups.setdefault((int(chunk[e]), int(file_[e])), []).append(e)

    for (ci, fi), ep_ids in groups.items():
        path = root / info["video_path"].format(
            video_key=VIDEO_KEY, chunk_index=ci, file_index=fi
        )
        pos = 0
        for e in ep_ids:  # back-to-back layout is assumed above; verify it
            assert round(float(from_ts[e]) * fps) == pos, (path, e, from_ts[e], pos)
            pos += int(lengths[e])

        n = 0
        for frame in iio.imiter(str(path)):
            yield frame
            n += 1
        assert n == pos, f"{path}: decoded {n} frames, expected {pos}"


class LeRobotPushT(Dataset):
    """(2-frame obs window, 5-action chunk) samples — same contract as
    PushTDataset, so train.py only needs to swap the class. Exposes
    act_min / act_max (train.py stores them in checkpoints for eval).

    'obs' is precomputed DINOv2 features (T, 256, 384); run
    patchpol.lerobot_features first to build the zarr.
    """

    def __init__(self, root: Path = ROOT, features_zarr: str = FEATS_ZARR):
        self.feats = zarr.open(features_zarr, "r")[:]  # (25650, 256, 384) f16
        self.actions = load_column("action", root)     # (25650, 2) float32

        stats = load_stats(root)
        self.act_min = np.asarray(stats["action"]["min"], dtype=np.float32)
        self.act_max = np.asarray(stats["action"]["max"], dtype=np.float32)

        eps = load_episodes(root)
        self.ep_start = np.repeat(eps["dataset_from_index"], eps["length"])
        self.ep_end = np.repeat(eps["dataset_to_index"], eps["length"])

    def normalize_action(self, a: np.ndarray) -> np.ndarray:
        return 2 * (a - self.act_min) / (self.act_max - self.act_min) - 1

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, t: int) -> dict:
        lo, hi = self.ep_start[t], self.ep_end[t] - 1
        obs_idx = np.clip(np.arange(t - OBS_HORIZON + 1, t + 1), lo, hi)
        act_idx = np.clip(np.arange(t, t + ACT_HORIZON), lo, hi)

        obs = self.feats[obs_idx].astype(np.float32)            # (2, 256, 384)
        action = self.normalize_action(self.actions[act_idx])  # (5, 2)

        return {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(action.astype(np.float32)),
        }


if __name__ == "__main__":
    if not (ROOT / "meta/info.json").exists():
        print(f"downloading lerobot/pusht -> {ROOT}")
        download()
    info = load_info()
    eps = load_episodes()
    assert len(eps["episode_index"]) == info["total_episodes"]
    assert eps["length"].sum() == info["total_frames"] == 25650
    assert eps["dataset_to_index"][-1] == info["total_frames"]
    print(f"meta ok: {info['total_episodes']} eps, {info['total_frames']} frames, "
          f"fps {info['fps']}")

    it = iter_video_frames()
    first = next(it)
    assert first.shape == (96, 96, 3) and first.dtype == np.uint8
    n = 1 + sum(1 for _ in it)
    assert n == info["total_frames"], f"decoded {n} frames, expected {info['total_frames']}"
    print(f"video ok: {n} frames")

    if Path(FEATS_ZARR).exists():
        ds = LeRobotPushT()
        s = ds[100]
        assert s["obs"].shape == (OBS_HORIZON, 256, 384)
        assert s["action"].shape == (ACT_HORIZON, 2)
        assert s["action"].abs().max() <= 1.0

        ep0 = ds[0]         # episode start: obs window must repeat frame 0
        assert torch.equal(ep0["obs"][0], ep0["obs"][1])
        last = int(eps["dataset_to_index"][0]) - 1
        end = ds[last]      # episode end: action chunk must repeat last action
        assert torch.equal(end["action"][-1], end["action"][0])
        print(f"dataset ok: {len(ds)} samples")
    else:
        print(f"no {FEATS_ZARR} yet — run: uv run python -m patchpol.lerobot_features")
