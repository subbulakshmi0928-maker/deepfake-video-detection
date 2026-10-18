"""Frame extraction, face cropping and tensor preprocessing for uploaded videos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import IMAGENET_MEAN, IMAGENET_STD, ArchProfile

_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_DETECT_MAX_WIDTH = 640          # run the cascade on a downscaled copy; boxes are scaled back
_HEAD_SCAN_SECONDS = 3.0         # how far into the clip we look for the first face ("head" sampling)


class VideoError(ValueError):
    """Raised when a file cannot be decoded as a video or yields no frames."""


@dataclass
class ClipResult:
    tensor: torch.Tensor        # (T, 3, S, S)
    frames_total: int
    frames_used: int
    faces_detected: int
    fps: float
    duration_s: float
    start_frame: int


def _detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    scale = min(1.0, _DETECT_MAX_WIDTH / w)
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [(int(x / scale), int(y / scale), int(fw / scale), int(fh / scale)) for x, y, fw, fh in faces]


def _crop_face(frame: np.ndarray, padding: float) -> tuple[np.ndarray, bool]:
    """Crop the largest frontal face (with padding); fall back to a centre square when none is found."""
    faces = _detect_faces(frame)
    h, w = frame.shape[:2]
    if faces:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        px, py = int(fw * padding), int(fh * padding)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(w, x + fw + px), min(h, y + fh + py)
        return frame[y0:y1, x0:x1], True
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side], False


def _to_tensor(crop: np.ndarray, size: int, channel_order: str) -> torch.Tensor:
    img = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    if channel_order == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def _uniform_indices(total: int, wanted: int) -> list[int]:
    if total <= 0:
        return []
    if total <= wanted:
        return list(range(total))
    return np.linspace(0, total - 1, wanted).round().astype(int).tolist()


def _first_face_index(path: Path, max_scan: int) -> int:
    """Index of the first frame that contains a face (0 when none is found within ``max_scan`` frames)."""
    cap = cv2.VideoCapture(str(path))
    idx = 0
    try:
        while idx < max_scan:
            ok, frame = cap.read()
            if not ok:
                break
            if _detect_faces(frame):
                return idx
            idx += 1
    finally:
        cap.release()
    return 0


def load_clip(path: Path, sequence_length: int, image_size: int, profile: ArchProfile,
              face_padding: float = 0.15) -> ClipResult:
    """Decode ``path`` into a normalised (T, 3, S, S) tensor.

    * ``uniform`` sampling spreads T frames evenly over the whole clip (good temporal coverage).
    * ``head`` sampling takes T consecutive frames starting at the first frame with a visible face, which
      is how the public ResNeXt50 + LSTM checkpoints were trained.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("file could not be opened as a video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if profile.frame_sampling == "head":
        start = _first_face_index(path, int(max(fps, 24.0) * _HEAD_SCAN_SECONDS))
        wanted = set(range(start, start + sequence_length))
    else:
        start = 0
        wanted = set(_uniform_indices(total, sequence_length))
    last_wanted = max(wanted) if wanted else None

    frames: list[torch.Tensor] = []
    faces = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not wanted or idx in wanted:
            crop, has_face = _crop_face(frame, face_padding)
            faces += int(has_face)
            frames.append(_to_tensor(crop, image_size, profile.channel_order))
            if len(frames) >= sequence_length:
                break
        idx += 1
        if last_wanted is not None and idx > last_wanted:
            break
    cap.release()
    if not frames:
        raise VideoError("no decodable frames found in the video")
    # Pad short clips by repeating the last frame so the sequence length is always what the model expects.
    while len(frames) < sequence_length:
        frames.append(frames[-1])
    frames_total = total if total > 0 else idx
    duration = round(frames_total / fps, 2) if fps > 0 and frames_total > 0 else 0.0
    return ClipResult(torch.stack(frames), frames_total, len(frames), faces, fps, duration, start)
