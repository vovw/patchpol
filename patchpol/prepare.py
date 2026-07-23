"""One-shot data preparation: download -> re-render @224 -> DINOv2 features.

Run this once on a fresh machine:  uv run python -m patchpol.prepare
Idempotent: skips any stage whose output already exists.
"""

import subprocess
import urllib.request
import zipfile
from pathlib import Path

from patchpol import features, render224

DATA_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"


def main():
    src = Path(render224.SRC)
    if not src.exists():
        Path("data").mkdir(exist_ok=True)
        zip_path = Path("data/pusht.zip")
        print(f"downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall("data")
        zip_path.unlink()
        print(f"extracted -> {src}")
    else:
        print(f"dataset ok: {src}")

    if not Path(render224.DST).exists():
        # subprocess: pygame's SDL_VIDEODRIVER env var must be set before import
        subprocess.run(["python", "-m", "patchpol.render224"], check=True)
    else:
        print(f"renders ok: {render224.DST}")

    if not Path(features.OUT_ZARR).exists():
        features.main()
    else:
        print(f"features ok: {features.OUT_ZARR}")

    print("data preparation complete — ready to train.")


if __name__ == "__main__":
    main()
