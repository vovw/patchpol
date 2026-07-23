"""Re-render every dataset frame at 224x224 from ground-truth sim state.

The zarr stores 96x96 images, but the paper feeds 224x224 to DINOv2.
Rather than upsampling blurry 96px images, we replay each frame's
state = [agent_x, agent_y, block_x, block_y, block_angle] through the
gym-pusht renderer at 224x224.

Run:  uv run python -m patchpol.render224
Writes: data/pusht/img224.zarr  (25650, 224, 224, 3) uint8
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame

import numpy as np
import zarr
from gym_pusht.envs import PushTEnv
from tqdm import tqdm

SRC = "data/pusht/pusht_cchi_v7_replay.zarr"
DST = "data/pusht/img224.zarr"
RES = 224


def main():
    src = zarr.open(SRC, "r")
    states = src["data/state"][:]  # (N, 5)
    n = len(states)

    dst = zarr.open(
        DST, "w", shape=(n, RES, RES, 3), chunks=(64, RES, RES, 3), dtype="u1"
    )

    env = PushTEnv(
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=RES,
        observation_height=RES,
    )
    env.reset()

    buf = np.empty((64, RES, RES, 3), dtype=np.uint8)
    for i in tqdm(range(n), desc="rendering @224"):
        env._set_state(states[i])
        buf[i % 64] = env._render()
        if i % 64 == 63:
            dst[i - 63 : i + 1] = buf
    rem = n % 64
    if rem:
        dst[n - rem : n] = buf[:rem]

    print(f"wrote {DST}: {dst.shape} uint8")


if __name__ == "__main__":
    main()
