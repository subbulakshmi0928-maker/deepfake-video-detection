# Architecture

```
 Browser (static UI)            Spring Boot backend (:8080)                 ML microservice (:8000)
 ┌───────────────────┐  POST    ┌──────────────────────────────┐  POST       ┌──────────────────────────────┐
 │ index.html/app.js │ ───────▶ │ AnalysisController           │ ──────────▶ │ FastAPI /predict             │
 │ upload + history  │ /api/.. │  └ DetectionService          │ multipart   │  ├ frame sampling (OpenCV)   │
 └───────────────────┘          │     ├ StorageService (disk)  │             │  ├ Haar face crop + resize   │
                                │     ├ MlClient (RestClient)  │ ◀────────── │  ├ ResNeXt50_32x4d features  │
                                │     └ VideoAnalysisRepository│  JSON       │  └ GRU / LSTM head → softmax │
                                └──────────────┬───────────────┘             └──────────────────────────────┘
                                               │ JPA
                                      ┌────────▼────────┐
                                      │ PostgreSQL      │  video_analysis: verdict, confidence,
                                      │ deepfake_db     │  frames, faces, timing, file metadata
                                      └─────────────────┘
```

## Request flow

1. The browser posts the video as `multipart/form-data` to `POST /api/analyses`.
2. `DetectionService` validates the extension, stores the file under a UUID name and calls `MlClient.predict`.
3. The ML service picks `SEQUENCE_LENGTH` frames (consecutive from the first detected face for the public LSTM
   checkpoints; spread evenly over the clip for the GRU model), crops the largest frontal face per frame with 15 %
   padding (centre square fallback), resizes to 112×112 RGB, normalises with ImageNet statistics and runs the
   ResNeXt50 + recurrent model. Softmax over the two classes gives `fake_probability` / `real_probability`.
4. The backend persists a `video_analysis` row (or an `ERROR` row if the ML call failed) and returns it.
5. The UI renders the verdict card and refreshes the history table and stats tiles.

## Model

* Backbone: `torchvision.models.resnext50_32x4d`, ImageNet-pretrained, final pooling/fc removed → 2048-d per frame.
* Head: GRU (hidden 2048) over the frame sequence, mean-pooled, dropout 0.4, linear → 2 classes (0 = FAKE, 1 = REAL).
* `jadhav-lstm` profile reproduces the public ResNeXt50 + LSTM checkpoints so they can be loaded unchanged.
