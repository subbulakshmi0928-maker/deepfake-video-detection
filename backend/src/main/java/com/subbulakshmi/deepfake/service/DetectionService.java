package com.subbulakshmi.deepfake.service;

import com.subbulakshmi.deepfake.domain.Verdict;
import com.subbulakshmi.deepfake.domain.VideoAnalysis;
import com.subbulakshmi.deepfake.domain.VideoAnalysisRepository;
import com.subbulakshmi.deepfake.exception.InvalidVideoException;
import com.subbulakshmi.deepfake.exception.NotFoundException;
import com.subbulakshmi.deepfake.ml.MlClient;
import com.subbulakshmi.deepfake.ml.MlModelInfo;
import com.subbulakshmi.deepfake.ml.MlPrediction;
import com.subbulakshmi.deepfake.ml.MlServiceException;
import com.subbulakshmi.deepfake.storage.StorageService;
import com.subbulakshmi.deepfake.web.dto.AnalysisResponse;
import com.subbulakshmi.deepfake.web.dto.PagedResponse;
import com.subbulakshmi.deepfake.web.dto.StatsResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Path;
import java.time.Instant;
import java.util.Locale;
import java.util.Set;

/**
 * Orchestrates an analysis: validates the upload, stores it, asks the ML service for a verdict and
 * records the outcome (including failures) in the database.
 */
@Service
public class DetectionService {

    private static final Logger log = LoggerFactory.getLogger(DetectionService.class);

    static final Set<String> ALLOWED_EXTENSIONS = Set.of("mp4", "mov", "avi", "mkv", "webm");
    static final int MAX_PAGE_SIZE = 100;
    static final int DEFAULT_PAGE_SIZE = 20;

    private final VideoAnalysisRepository repository;
    private final StorageService storageService;
    private final MlClient mlClient;

    public DetectionService(VideoAnalysisRepository repository, StorageService storageService, MlClient mlClient) {
        this.repository = repository;
        this.storageService = storageService;
        this.mlClient = mlClient;
    }

    /**
     * Full pipeline for one upload. Deliberately not transactional: the ML call can take minutes and
     * must not hold a database connection.
     *
     * @throws InvalidVideoException when the file is empty or has an unsupported extension
     * @throws MlServiceException    when the ML service fails; an {@code ERROR} row is persisted first
     */
    public AnalysisResponse analyze(MultipartFile file) {
        String originalName = sanitizeFileName(file == null ? null : file.getOriginalFilename());
        String extension = validate(file, originalName);

        Path stored = storageService.store(file, extension);

        VideoAnalysis entity = new VideoAnalysis();
        entity.setOriginalFileName(originalName);
        entity.setStoredFileName(stored.getFileName().toString());
        entity.setContentType(truncate(file.getContentType(), 100));
        entity.setSizeBytes(file.getSize());
        entity.setCreatedAt(Instant.now());

        try {
            MlPrediction prediction = mlClient.predict(stored, originalName, file.getContentType());
            applyPrediction(entity, prediction);
        } catch (MlServiceException e) {
            log.warn("ML analysis failed for {}: {}", originalName, e.getMessage());
            entity.setVerdict(Verdict.ERROR);
            entity.setErrorMessage(truncate(e.getMessage(), 500));
            repository.save(entity);
            throw e;
        }

        VideoAnalysis saved = repository.save(entity);
        log.info("Analysis {} for {}: {} (confidence {})", saved.getId(), originalName,
                saved.getVerdict(), saved.getConfidence());
        return AnalysisResponse.from(saved);
    }

    /** Newest-first page of analyses. Page index and size are clamped to sane bounds. */
    @Transactional(readOnly = true)
    public PagedResponse<AnalysisResponse> history(int page, int size) {
        int safePage = Math.max(page, 0);
        int safeSize = size <= 0 ? DEFAULT_PAGE_SIZE : Math.min(size, MAX_PAGE_SIZE);
        Page<VideoAnalysis> result = repository.findAllByOrderByCreatedAtDesc(PageRequest.of(safePage, safeSize));
        return new PagedResponse<>(
                result.getContent().stream().map(AnalysisResponse::from).toList(),
                safePage, safeSize, result.getTotalElements());
    }

    /** @throws NotFoundException when no row has that id */
    @Transactional(readOnly = true)
    public AnalysisResponse get(long id) {
        return AnalysisResponse.from(find(id));
    }

    /** Removes the database row and the stored video (best effort for the file). */
    public void delete(long id) {
        VideoAnalysis entity = find(id);
        repository.delete(entity);
        try {
            storageService.delete(entity.getStoredFileName());
        } catch (RuntimeException e) {
            log.warn("Deleted analysis {} but could not remove file {}: {}", id, entity.getStoredFileName(),
                    e.getMessage());
        }
    }

    @Transactional(readOnly = true)
    public StatsResponse stats() {
        return new StatsResponse(
                repository.count(),
                repository.countByVerdict(Verdict.FAKE),
                repository.countByVerdict(Verdict.REAL),
                repository.countByVerdict(Verdict.ERROR),
                repository.averageConfidenceExcludingErrors());
    }

    private VideoAnalysis find(long id) {
        return repository.findById(id)
                .orElseThrow(() -> new NotFoundException("Analysis " + id + " not found"));
    }

    /** Returns the lower-case extension when the upload is acceptable. */
    private static String validate(MultipartFile file, String originalName) {
        if (file == null || file.isEmpty()) {
            throw new InvalidVideoException("Uploaded file is empty");
        }
        String extension = StringUtils.getFilenameExtension(originalName);
        if (extension == null) {
            throw new InvalidVideoException("File name has no extension; allowed: " + allowedList());
        }
        String lower = extension.toLowerCase(Locale.ROOT);
        if (!ALLOWED_EXTENSIONS.contains(lower)) {
            throw new InvalidVideoException("Unsupported file type '." + lower + "'; allowed: " + allowedList());
        }
        return lower;
    }

    private static void applyPrediction(VideoAnalysis entity, MlPrediction p) {
        entity.setVerdict(toVerdict(p.label()));
        entity.setConfidence(p.confidence());
        entity.setFakeProbability(p.fakeProbability());
        entity.setFramesAnalyzed(p.framesAnalyzed());
        entity.setFacesDetected(p.facesDetected());
        entity.setProcessingMs(p.processingMs());
        entity.setModelName(truncate(modelName(p.model()), 120));
        entity.setErrorMessage(null);
    }

    private static Verdict toVerdict(String label) {
        if (label == null) {
            throw new MlServiceException("ML service response has no label");
        }
        return switch (label.trim().toUpperCase(Locale.ROOT)) {
            case "FAKE" -> Verdict.FAKE;
            case "REAL" -> Verdict.REAL;
            default -> throw new MlServiceException("Unexpected label from ML service: " + label);
        };
    }

    /** e.g. {@code resnext50_32x4d+lstm · model_87_acc_20_frames_final_data.pt}, suffixed with {@code (untrained)} when no weights were loaded. */
    private static String modelName(MlModelInfo model) {
        if (model == null) {
            return null;
        }
        StringBuilder sb = new StringBuilder();
        if (StringUtils.hasText(model.backbone())) {
            sb.append(model.backbone());
        }
        if (StringUtils.hasText(model.recurrent())) {
            if (sb.length() > 0) {
                sb.append('+');
            }
            sb.append(model.recurrent());
        }
        if (StringUtils.hasText(model.weights())) {
            sb.append(sb.length() > 0 ? " \u00b7 " : "").append(model.weights());
        }
        if (!model.trained()) {
            sb.append(sb.length() > 0 ? " (untrained)" : "untrained");
        }
        return sb.length() == 0 ? null : sb.toString();
    }

    /** Strips any directory component and clamps to the column width. */
    private static String sanitizeFileName(String name) {
        if (!StringUtils.hasText(name)) {
            return "upload";
        }
        String cleaned = StringUtils.getFilename(StringUtils.cleanPath(name));
        if (!StringUtils.hasText(cleaned)) {
            return "upload";
        }
        return truncate(cleaned, 255);
    }

    private static String truncate(String s, int max) {
        if (s == null) {
            return null;
        }
        return s.length() <= max ? s : s.substring(0, max);
    }

    private static String allowedList() {
        return String.join(", ", ALLOWED_EXTENSIONS.stream().sorted().toList());
    }
}
