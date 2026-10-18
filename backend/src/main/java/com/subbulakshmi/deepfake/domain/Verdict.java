package com.subbulakshmi.deepfake.domain;

/**
 * Outcome of a video analysis. {@link #ERROR} marks rows whose ML call failed.
 */
public enum Verdict {
    REAL,
    FAKE,
    ERROR
}
