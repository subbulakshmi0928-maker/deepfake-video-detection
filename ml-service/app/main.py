"""FastAPI entry point: /health, /model and /predict."""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .model import LoadedModel, resolve_device
from .video import VideoError, load_clip

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("deepfake.api")

app = FastAPI(title="Deepfake Video Detection - ML Service", version="1.0.0",
              description="ResNeXt50_32x4d + recurrent head that classifies a video as REAL or FAKE.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model: LoadedModel | None = None


def get_model() -> LoadedModel:
    global _model
    if _model is None:
        _model = LoadedModel(settings.weights_path, settings.arch_profile, resolve_device(settings.device))
    return _model


@app.on_event("startup")
def _warm_up() -> None:
    get_model()


@app.get("/health")
def health() -> dict:
    m = get_model()
    return {"status": "ok", "model_loaded": True, "trained": m.trained, "device": str(m.device),
            "weights": m.weights_path.name if m.weights_path else None,
            "sequence_length": settings.sequence_length, "image_size": settings.image_size}


@app.get("/model")
def model_info() -> dict:
    return {**get_model().info(), "sequence_length": settings.sequence_length,
            "image_size": settings.image_size, "class_index": {"0": "FAKE", "1": "REAL"}}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    name = file.filename or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in settings.allowed_extensions:
        raise HTTPException(400, f"unsupported file type '.{ext}'; allowed: {', '.join(settings.allowed_extensions)}")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="deepfake-") as tmp:
        target = Path(tmp) / f"clip.{ext}"
        size = 0
        with target.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"file exceeds {settings.max_upload_mb} MB")
                out.write(chunk)
        m = get_model()
        try:
            clip = load_clip(target, settings.sequence_length, settings.image_size, m.profile, settings.face_padding)
        except VideoError as e:
            raise HTTPException(400, str(e)) from e
        result = m.predict(clip.tensor)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info("%s -> %s (%.3f) frames=%d faces=%d %dms", name, result["label"], result["confidence"],
             clip.frames_used, clip.faces_detected, elapsed_ms)
    return {
        **result,
        "frames_analyzed": clip.frames_used,
        "frames_total": clip.frames_total,
        "faces_detected": clip.faces_detected,
        "fps": round(clip.fps, 2),
        "duration_s": clip.duration_s,
        "processing_ms": elapsed_ms,
        "file_name": name,
        "size_bytes": size,
        "model": m.info(),
    }
