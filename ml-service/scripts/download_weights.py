"""Download a public pretrained ResNeXt50 + LSTM deepfake checkpoint into ./weights.

The checkpoints come from the GPL-3.0 project "Deepfake_detection_using_deep_learning" by Abhijit Jadhav
(https://github.com/abhijitjadhav1998/Deepfake_detection_using_deep_learning). They were trained on
FaceForensics++, Celeb-DF and DFDC face crops and are loaded here through the ``jadhav-lstm`` architecture
profile. Train your own GRU model with ``scripts/train.py`` for a checkpoint under your own licence.

Usage: python -m scripts.download_weights [--frames 20] [--list]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FOLDER_ID = "1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=20, help="sequence length encoded in the checkpoint name")
    ap.add_argument("--list", action="store_true", help="only list the files available in the Drive folder")
    args = ap.parse_args()
    try:
        import gdown
    except ImportError:
        print("pip install gdown   (it is in requirements-dev.txt)", file=sys.stderr)
        return 1
    import inspect
    kw = {"remaining_ok": True} if "remaining_ok" in inspect.signature(gdown.download_folder).parameters else {}
    files = gdown.download_folder(id=FOLDER_ID, skip_download=True, quiet=True, **kw) or []
    pts = [f for f in files if f.path.endswith(".pt")]
    if args.list or not pts:
        for f in pts:
            print(f.id, Path(f.path).name)
        return 0 if pts else 2
    match = [f for f in pts if f"_{args.frames}_frames" in f.path] or pts
    chosen = sorted(match, key=lambda f: f.path)[0]
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out = WEIGHTS_DIR / Path(chosen.path).name
    print("downloading", out.name)
    gdown.download(id=chosen.id, output=str(out), quiet=False)
    print("saved to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
