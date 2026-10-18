package com.subbulakshmi.deepfake.web.dto;

import java.time.Instant;

/**
 * Uniform JSON error body produced by the global exception handler.
 *
 * @param status  HTTP status code
 * @param error   HTTP reason phrase, e.g. {@code Bad Request}
 * @param message human-readable detail
 */
public record ApiError(Instant timestamp, int status, String error, String message) {
}
