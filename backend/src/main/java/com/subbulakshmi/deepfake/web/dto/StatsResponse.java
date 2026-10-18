package com.subbulakshmi.deepfake.web.dto;

/**
 * Aggregate counters for {@code GET /api/analyses/stats}.
 *
 * @param avgConfidence mean confidence over non-error rows, {@code null} when there are none
 */
public record StatsResponse(long total, long fake, long real, long errors, Double avgConfidence) {
}
