"""Runtime settings for the ML service, all overridable through environment variables."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", ROOT / "weights"))

_FRAMES_IN_NAME = re.compile(r"_(\d+)_frames")


def _discover_weights() -> Path | None:
    """Return the checkpoint to load: MODEL_PATH if set, else the first *.pt in the weights dir."""
    explicit = os.getenv("MODEL_PATH")
    if explicit:
        return Path(explicit)
    if WEIGHTS_DIR.is_dir():
        candidates = sorted(WEIGHTS_DIR.glob("*.pt")) + sorted(WEIGHTS_DIR.glob("*.pth"))
        if candidates:
            return candidates[0]
    return None


def _default_sequence_length(weights: Path | None) -> int:
    """Public checkpoints encode their frame count in the file name (model_87_acc_20_frames_...)."""
    if weights is not None:
        m = _FRAMES_IN_NAME.search(weights.name)
        if m:
            return int(m.group(1))
    return 20


@dataclass
class Settings:
    weights_path: Path | None = field(default_factory=_discover_weights)
    sequence_length: int = 0
    image_size: int = int(os.getenv("IMAGE_SIZE", "112"))
    device: str = os.getenv("DEVICE", "auto")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "200"))
    face_padding: float = float(os.getenv("FACE_PADDING", "0.15"))
    allowed_extensions: tuple[str, ...] = ("mp4", "mov", "avi", "mkv", "webm")
    # Architecture profile: "auto" picks from the checkpoint; "gru" is this project's own architecture;
    # "jadhav-lstm" reproduces the public ResNeXt50 + LSTM checkpoints so their weights can be reused.
    arch_profile: str = os.getenv("ARCH_PROFILE", "auto")

    def __post_init__(self) -> None:
        env_len = os.getenv("SEQUENCE_LENGTH")
        self.sequence_length = int(env_len) if env_len else _default_sequence_length(self.weights_path)


settings = Settings()
