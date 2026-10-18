package com.subbulakshmi.deepfake.web.dto;

import com.subbulakshmi.deepfake.domain.VideoAnalysis;

import java.time.Instant;

/**
 * Public representation of one {@link VideoAnalysis}.
 */
public record AnalysisResponse(
        Long id,
        String fileName,
        long sizeBytes,
        String verdict,
        Double confidence,
        Double fakeProbability,
        Integer framesAnalyzed,
        Integer facesDetected,
        Long processingMs,
        String modelName,
        String errorMessage,
        Instant createdAt) {

    /** Maps an entity to its API form. */
    public static AnalysisResponse from(VideoAnalysis a) {
        return new AnalysisResponse(
                a.getId(),
                a.getOriginalFileName(),
                a.getSizeBytes(),
                a.getVerdict() == null ? null : a.getVerdict().name(),
                a.getConfidence(),
                a.getFakeProbability(),
                a.getFramesAnalyzed(),
                a.getFacesDetected(),
                a.getProcessingMs(),
                a.getModelName(),
                a.getErrorMessage(),
                a.getCreatedAt());
    }
}
