package com.subbulakshmi.deepfake.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.Instant;

/**
 * One analysed upload: file metadata plus the verdict returned by the ML service.
 * Maps to the {@code video_analysis} table created by {@code schema.sql}.
 */
@Entity
@Table(name = "video_analysis")
public class VideoAnalysis {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "original_file_name", nullable = false, length = 255)
    private String originalFileName;

    @Column(name = "stored_file_name", nullable = false, length = 255)
    private String storedFileName;

    @Column(name = "content_type", length = 100)
    private String contentType;

    @Column(name = "size_bytes", nullable = false)
    private long sizeBytes;

    @Enumerated(EnumType.STRING)
    @Column(name = "verdict", nullable = false, length = 16)
    private Verdict verdict;

    @Column(name = "confidence")
    private Double confidence;

    @Column(name = "fake_probability")
    private Double fakeProbability;

    @Column(name = "frames_analyzed")
    private Integer framesAnalyzed;

    @Column(name = "faces_detected")
    private Integer facesDetected;

    @Column(name = "processing_ms")
    private Long processingMs;

    @Column(name = "model_name", length = 120)
    private String modelName;

    @Column(name = "error_message", length = 500)
    private String errorMessage;

    /**
     * Stored in a plain {@code TIMESTAMP} column (PostgreSQL 9.5 friendly); Hibernate normalises
     * the value to UTC via {@code hibernate.jdbc.time_zone=UTC}.
     */
    @JdbcTypeCode(SqlTypes.TIMESTAMP)
    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    public Long getId() {
        return id;
    }

    public String getOriginalFileName() {
        return originalFileName;
    }

    public void setOriginalFileName(String originalFileName) {
        this.originalFileName = originalFileName;
    }

    public String getStoredFileName() {
        return storedFileName;
    }

    public void setStoredFileName(String storedFileName) {
        this.storedFileName = storedFileName;
    }

    public String getContentType() {
        return contentType;
    }

    public void setContentType(String contentType) {
        this.contentType = contentType;
    }

    public long getSizeBytes() {
        return sizeBytes;
    }

    public void setSizeBytes(long sizeBytes) {
        this.sizeBytes = sizeBytes;
    }

    public Verdict getVerdict() {
        return verdict;
    }

    public void setVerdict(Verdict verdict) {
        this.verdict = verdict;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public Double getFakeProbability() {
        return fakeProbability;
    }

    public void setFakeProbability(Double fakeProbability) {
        this.fakeProbability = fakeProbability;
    }

    public Integer getFramesAnalyzed() {
        return framesAnalyzed;
    }

    public void setFramesAnalyzed(Integer framesAnalyzed) {
        this.framesAnalyzed = framesAnalyzed;
    }

    public Integer getFacesDetected() {
        return facesDetected;
    }

    public void setFacesDetected(Integer facesDetected) {
        this.facesDetected = facesDetected;
    }

    public Long getProcessingMs() {
        return processingMs;
    }

    public void setProcessingMs(Long processingMs) {
        this.processingMs = processingMs;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }
}
