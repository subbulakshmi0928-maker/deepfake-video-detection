package com.subbulakshmi.deepfake.ml;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * JSON body returned by {@code POST /predict} on the ML service. The service uses snake_case,
 * hence the explicit {@link JsonProperty} names; unknown extra fields are ignored.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record MlPrediction(
        @JsonProperty("label") String label,
        @JsonProperty("confidence") double confidence,
        @JsonProperty("fake_probability") double fakeProbability,
        @JsonProperty("real_probability") double realProbability,
        @JsonProperty("frames_analyzed") int framesAnalyzed,
        @JsonProperty("faces_detected") int facesDetected,
        @JsonProperty("processing_ms") long processingMs,
        @JsonProperty("model") MlModelInfo model) {
}
