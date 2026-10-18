"""Train the ResNeXt50 + GRU detector on your own face-video dataset.

Expected layout (videos in any of mp4/mov/avi/mkv/webm):
    DATA/train/real/*.mp4   DATA/train/fake/*.mp4
    DATA/val/real/*.mp4     DATA/val/fake/*.mp4

Each video is decoded once, face-cropped, resized and cached as a tensor under DATA/.cache so later epochs
are fast. The best checkpoint (by validation accuracy) is written with the metadata the service needs.

Usage:
    python -m scripts.train --data DATA --epochs 10 --seq-len 20 --out weights/deepfake_gru_20_frames.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.model import PROFILES, DeepfakeDetector, resolve_device
from app.video import VideoError, load_clip

EXTS = ("mp4", "mov", "avi", "mkv", "webm")
PROFILE = PROFILES["gru"]


class VideoClips(Dataset):
    def __init__(self, root: Path, seq_len: int, img_size: int, cache: Path) -> None:
        self.items: list[tuple[Path, int]] = []
        for label, name in ((1, "real"), (0, "fake")):          # 0 = FAKE, 1 = REAL
            for ext in EXTS:
                self.items += [(p, label) for p in sorted((root / name).glob(f"*.{ext}"))]
        if not self.items:
            raise SystemExit(f"no videos found under {root}/real and {root}/fake")
        self.seq_len, self.img_size, self.cache = seq_len, img_size, cache
        cache.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        key = hashlib.md5(f"{path.resolve()}|{path.stat().st_mtime}|{self.seq_len}|{self.img_size}".encode()).hexdigest()
        cached = self.cache / f"{key}.pt"
        if cached.exists():
            clip = torch.load(cached, weights_only=True)
        else:
            try:
                clip = load_clip(path, self.seq_len, self.img_size, PROFILE).tensor
            except VideoError:
                clip = torch.zeros(self.seq_len, 3, self.img_size, self.img_size)
            torch.save(clip, cached)
        return clip, label


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += float(loss) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("weights/deepfake_gru_20_frames.pt"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--freeze-backbone", action="store_true", help="train only the GRU + classifier")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    cache = args.data / ".cache"
    train_ds = VideoClips(args.data / "train", args.seq_len, args.img_size, cache)
    val_ds = VideoClips(args.data / "val", args.seq_len, args.img_size, cache)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = DeepfakeDetector(PROFILE, pretrained_backbone=True).to(device)
    if args.freeze_backbone:
        for p in model.model.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} params={sum(p.numel() for p in params):,}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_dl, device, criterion, optimizer)
        vl, va = run_epoch(model, val_dl, device, criterion)
        print(f"epoch {epoch:02d}  train loss {tl:.4f} acc {ta:.3f} | val loss {vl:.4f} acc {va:.3f} | {time.time() - t0:.0f}s")
        if va >= best:
            best = va
            torch.save({"state_dict": model.state_dict(), "arch_profile": "gru", "sequence_length": args.seq_len,
                        "image_size": args.img_size, "val_accuracy": va, "epoch": epoch}, args.out)
            args.out.with_suffix(".json").write_text(json.dumps({"arch_profile": "gru", "sequence_length": args.seq_len,
                                                                 "image_size": args.img_size, "val_accuracy": va}, indent=2))
            print(f"  saved {args.out} (val acc {va:.3f})")
    print(f"best val acc {best:.3f}")


if __name__ == "__main__":
    main()
