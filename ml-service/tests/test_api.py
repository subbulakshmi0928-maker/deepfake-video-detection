import os
from pathlib import Path

os.environ.setdefault("SEQUENCE_LENGTH", "6")   # keep the test clip small and fast
os.environ.setdefault("DEVICE", "cpu")

import pytest
from fastapi.testclient import TestClient

from app.main import app
from scripts.make_sample_video import make_clip


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    return make_clip(tmp_path_factory.mktemp("clips") / "clip.mp4", seconds=1.0, seed=7)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["sequence_length"] == 6


def test_model_info(client):
    body = client.get("/model").json()
    assert body["backbone"] == "resnext50_32x4d"
    assert body["recurrent"] in ("gru", "lstm")


def test_predict_returns_verdict(client, clip):
    with clip.open("rb") as f:
        r = client.post("/predict", files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] in ("REAL", "FAKE")
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(body["fake_probability"] + body["real_probability"] - 1.0) < 1e-3
    assert body["frames_analyzed"] == 6
    assert body["processing_ms"] >= 0
    assert body["model"]["backbone"] == "resnext50_32x4d"


def test_rejects_unsupported_extension(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_rejects_undecodable_video(client):
    r = client.post("/predict", files={"file": ("broken.mp4", b"definitely not a video", "video/mp4")})
    assert r.status_code == 400
