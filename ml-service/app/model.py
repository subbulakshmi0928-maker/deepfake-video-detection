"""ResNeXt50_32x4d + recurrent head for frame-sequence deepfake classification.

Class index convention (shared with the public checkpoints): 0 = FAKE, 1 = REAL.

Two architecture profiles are supported:

* ``gru``          - this project's architecture: ResNeXt50 features per frame, a batch-first GRU over the
                     sequence, mean pooling over time, then a linear classifier.
* ``jadhav-lstm``  - bit-compatible with the widely shared ResNeXt50 + LSTM checkpoints
                     (``model_87_acc_20_frames_final_data.pt`` and friends). Those were trained with an LSTM
                     created as ``nn.LSTM(2048, 2048, 1, False)`` - i.e. ``bias=False`` and ``batch_first=False`` -
                     so every frame is effectively scored on its own. We keep that behaviour so the weights
                     stay valid, but average the per-frame logits instead of using only the last frame, which is
                     a strictly more stable estimate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

log = logging.getLogger("deepfake.model")

LABELS = ("FAKE", "REAL")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ArchProfile:
    name: str
    rnn_type: str          # "gru" | "lstm"
    rnn_bias: bool
    batch_first: bool
    pool: str              # "mean" (pool features over time) | "mean_logits" (score frames, average logits)
    channel_order: str     # "rgb" | "bgr"  (both the GRU model and the public checkpoints expect RGB)
    frame_sampling: str    # "uniform" (spread over the clip) | "head" (consecutive, from the first face)


PROFILES = {
    "gru": ArchProfile("gru", "gru", True, True, "mean", "rgb", "uniform"),
    "jadhav-lstm": ArchProfile("jadhav-lstm", "lstm", False, False, "mean_logits", "rgb", "head"),
}


class DeepfakeDetector(nn.Module):
    """ResNeXt50_32x4d backbone -> per-frame 2048-d features -> recurrent layer -> 2-way classifier."""

    def __init__(self, profile: ArchProfile, num_classes: int = 2, latent_dim: int = 2048,
                 hidden_dim: int = 2048, rnn_layers: int = 1, dropout: float = 0.4,
                 pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.profile = profile
        weights = torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = torchvision.models.resnext50_32x4d(weights=weights)
        self.model = nn.Sequential(*list(backbone.children())[:-2])   # conv feature map (B*T, 2048, h, w)
        rnn_cls = nn.GRU if profile.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(latent_dim, hidden_dim, rnn_layers, bias=profile.rnn_bias,
                           batch_first=profile.batch_first)
        self.dp = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) -> logits (B, num_classes)."""
        b, t, c, h, w = x.shape
        fmap = self.model(x.reshape(b * t, c, h, w))
        feats = self.avgpool(fmap).reshape(b, t, -1)
        out, _ = self.rnn(feats)
        if self.profile.pool == "mean_logits":
            return self.linear1(self.dp(out)).mean(dim=1)
        return self.linear1(self.dp(out.mean(dim=1)))


def _remap_keys(state: dict) -> dict:
    """Accept checkpoints whose recurrent layer was called ``lstm`` or ``gru``."""
    out = {}
    for k, v in state.items():
        for old in ("lstm.", "gru."):
            if k.startswith(old):
                k = "rnn." + k[len(old):]
                break
        out[k] = v
    return out


def detect_profile(state: dict) -> str:
    keys = set(state)
    if "rnn.weight_ih_l0" in keys or "gru.weight_ih_l0" in keys:
        return "gru"
    if "lstm.weight_ih_l0" in keys and "lstm.bias_ih_l0" not in keys:
        return "jadhav-lstm"
    return "gru"


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")   # MPS is deliberately not auto-selected: recurrent kernels are flaky there


class LoadedModel:
    """A ready-to-serve model plus the metadata the API reports."""

    def __init__(self, weights_path: Path | None, arch_profile: str, device: torch.device) -> None:
        self.device = device
        self.weights_path = weights_path
        state = None
        if weights_path is not None and weights_path.is_file():
            # weights_only=True refuses pickled code, so a downloaded checkpoint cannot execute anything.
            raw = torch.load(weights_path, map_location="cpu", weights_only=True)
            state = _remap_keys(raw.get("state_dict", raw) if isinstance(raw, dict) else raw)
            if arch_profile == "auto":
                arch_profile = raw.get("arch_profile", detect_profile(raw)) if isinstance(raw, dict) else "gru"
        elif arch_profile == "auto":
            arch_profile = "gru"
        self.profile = PROFILES[arch_profile]
        self.model = DeepfakeDetector(self.profile, pretrained_backbone=state is None)
        self.trained = False
        if state is not None:
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                log.warning("checkpoint key mismatch - missing=%s unexpected=%s", missing, unexpected)
            self.trained = not missing
            log.info("loaded weights %s (profile=%s)", weights_path.name, self.profile.name)
        else:
            log.warning("no trained weights found - running with an ImageNet backbone and an untrained head; "
                        "predictions are for pipeline demonstration only")
        self.model.eval().to(device)

    @torch.inference_mode()
    def predict(self, clip: torch.Tensor) -> dict:
        """clip: (T, C, H, W) float tensor already normalised. Returns label + probabilities."""
        logits = self.model(clip.unsqueeze(0).to(self.device))
        probs = torch.softmax(logits, dim=1)[0].cpu()
        idx = int(torch.argmax(probs))
        return {
            "label": LABELS[idx],
            "confidence": round(float(probs[idx]), 4),
            "fake_probability": round(float(probs[0]), 4),
            "real_probability": round(float(probs[1]), 4),
        }

    def info(self) -> dict:
        return {
            "backbone": "resnext50_32x4d",
            "recurrent": self.profile.rnn_type,
            "profile": self.profile.name,
            "weights": self.weights_path.name if self.weights_path else None,
            "trained": self.trained,
            "device": str(self.device),
        }
