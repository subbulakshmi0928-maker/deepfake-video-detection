"""Generate small synthetic test clips (no real people) for pipeline tests and demos.

Usage: python -m scripts.make_sample_video OUT_DIR [--count 2] [--seconds 3]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_clip(path: Path, seconds: float = 3.0, fps: int = 24, size: tuple[int, int] = (320, 240), seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    w, h = size
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    base_hue = int(rng.integers(0, 180))
    n = int(seconds * fps)
    for i in range(n):
        t = i / max(n - 1, 1)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = (base_hue + np.linspace(0, 60, w)[None, :] + t * 40).astype(np.uint8) % 180
        hsv[..., 1] = 120
        hsv[..., 2] = 200
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # a cartoon "face" drifting across the frame
        cx = int(w * (0.3 + 0.4 * t)); cy = int(h * 0.5 + 15 * np.sin(t * 6.28))
        cv2.ellipse(frame, (cx, cy), (45, 58), 0, 0, 360, (150, 190, 235), -1)
        cv2.circle(frame, (cx - 16, cy - 12), 6, (30, 30, 30), -1)
        cv2.circle(frame, (cx + 16, cy - 12), 6, (30, 30, 30), -1)
        cv2.ellipse(frame, (cx, cy + 18), (18, 8 + int(6 * abs(np.sin(t * 9)))), 0, 0, 180, (40, 40, 120), 3)
        cv2.putText(frame, f"synthetic clip #{seed}  frame {i:03d}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()
    for i in range(args.count):
        p = make_clip(args.out_dir / f"sample_clip_{i + 1}.mp4", args.seconds, seed=i + 1)
        print("wrote", p, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
