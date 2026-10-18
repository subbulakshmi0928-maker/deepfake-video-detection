-- PostgreSQL 9.5 compatible (no IDENTITY columns). Idempotent: safe to run on every startup.
CREATE TABLE IF NOT EXISTS video_analysis (
    id                 BIGSERIAL PRIMARY KEY,
    original_file_name VARCHAR(255)     NOT NULL,
    stored_file_name   VARCHAR(255)     NOT NULL,
    content_type       VARCHAR(100),
    size_bytes         BIGINT           NOT NULL,
    verdict            VARCHAR(16)      NOT NULL,
    confidence         DOUBLE PRECISION,
    fake_probability   DOUBLE PRECISION,
    frames_analyzed    INTEGER,
    faces_detected     INTEGER,
    processing_ms      BIGINT,
    model_name         VARCHAR(120),
    error_message      VARCHAR(500),
    created_at         TIMESTAMP        NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_video_analysis_created_at ON video_analysis (created_at DESC);
