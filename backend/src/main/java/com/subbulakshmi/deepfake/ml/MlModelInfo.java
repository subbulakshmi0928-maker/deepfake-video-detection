package com.subbulakshmi.deepfake.ml;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * Model metadata embedded in every ML prediction ({@code "model": {...}}).
 *
 * @param backbone  CNN backbone name, e.g. {@code resnext50_32x4d}
 * @param recurrent recurrent head type, e.g. {@code lstm}
 * @param weights   weights file name, may be {@code null} when the service runs untrained
 * @param trained   whether real trained weights were loaded
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record MlModelInfo(String backbone, String recurrent, String weights, boolean trained) {
}
