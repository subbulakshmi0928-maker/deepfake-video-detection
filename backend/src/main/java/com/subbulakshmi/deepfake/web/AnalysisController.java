package com.subbulakshmi.deepfake.web;

import com.subbulakshmi.deepfake.ml.MlClient;
import com.subbulakshmi.deepfake.service.DetectionService;
import com.subbulakshmi.deepfake.web.dto.AnalysisResponse;
import com.subbulakshmi.deepfake.web.dto.PagedResponse;
import com.subbulakshmi.deepfake.web.dto.StatsResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * JSON REST API under {@code /api}: upload a video for analysis, browse history, stats and health.
 */
@RestController
@RequestMapping(path = "/api", produces = MediaType.APPLICATION_JSON_VALUE)
public class AnalysisController {

    private final DetectionService detectionService;
    private final MlClient mlClient;

    public AnalysisController(DetectionService detectionService, MlClient mlClient) {
        this.detectionService = detectionService;
        this.mlClient = mlClient;
    }

    /** Uploads a video (multipart field {@code file}) and returns the persisted analysis. */
    @PostMapping(path = "/analyses", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @ResponseStatus(HttpStatus.CREATED)
    public AnalysisResponse analyze(@RequestParam("file") MultipartFile file) {
        return detectionService.analyze(file);
    }

    @GetMapping("/analyses")
    public PagedResponse<AnalysisResponse> history(
            @RequestParam(name = "page", defaultValue = "0") int page,
            @RequestParam(name = "size", defaultValue = "20") int size) {
        return detectionService.history(page, size);
    }

    /** Declared before {@code /analyses/{id}} and the id is digits-only, so "stats" never binds as an id. */
    @GetMapping("/analyses/stats")
    public StatsResponse stats() {
        return detectionService.stats();
    }

    @GetMapping("/analyses/{id:\\d+}")
    public AnalysisResponse get(@PathVariable("id") long id) {
        return detectionService.get(id);
    }

    @DeleteMapping("/analyses/{id:\\d+}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@PathVariable("id") long id) {
        detectionService.delete(id);
    }

    /** Backend liveness plus whatever the ML service reports on its own {@code /health}. */
    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("backend", "ok");
        body.put("mlService", mlClient.health());
        return body;
    }
}
