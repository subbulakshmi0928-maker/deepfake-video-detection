package com.subbulakshmi.deepfake.ml;

import com.subbulakshmi.deepfake.config.MlProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.ClientHttpRequestFactory;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.util.StringUtils;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * HTTP client for the Python FastAPI ML microservice, built on Spring's {@link RestClient}.
 * <p>
 * Two clients are kept: one with the long read timeout for {@code /predict} (inference can take
 * minutes for large clips) and one with a short timeout for {@code /health}, so a health probe never
 * blocks for the full inference timeout.
 */
@Component
public class MlClient {

    private static final Logger log = LoggerFactory.getLogger(MlClient.class);
    private static final int MAX_ERROR_SNIPPET = 300;

    private final RestClient predictClient;
    private final RestClient healthClient;

    @Autowired
    public MlClient(RestClient.Builder builder, MlProperties properties) {
        this(builder.clone()
                        .baseUrl(properties.baseUrl())
                        .requestFactory(requestFactory(properties.connectTimeoutMs(), properties.readTimeoutMs()))
                        .build(),
                builder.clone()
                        .baseUrl(properties.baseUrl())
                        .requestFactory(requestFactory(properties.connectTimeoutMs(), properties.connectTimeoutMs()))
                        .build());
        log.info("ML client configured for {} (connect {} ms, read {} ms)",
                properties.baseUrl(), properties.connectTimeoutMs(), properties.readTimeoutMs());
    }

    /** Constructor for tests: inject fully built clients (e.g. bound to a MockRestServiceServer). */
    MlClient(RestClient predictClient, RestClient healthClient) {
        this.predictClient = predictClient;
        this.healthClient = healthClient;
    }

    private static ClientHttpRequestFactory requestFactory(int connectTimeoutMs, int readTimeoutMs) {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(connectTimeoutMs);
        factory.setReadTimeout(readTimeoutMs);
        return factory;
    }

    /**
     * Uploads the stored video to {@code POST {baseUrl}/predict} as multipart field {@code file}.
     *
     * @param file         path of the stored video on disk
     * @param originalName file name to present to the ML service (its extension check uses it)
     * @param contentType  MIME type reported by the uploader, may be {@code null}
     * @return the parsed prediction
     * @throws MlServiceException on any transport, status or parsing failure
     */
    public MlPrediction predict(Path file, String originalName, String contentType) {
        String partName = StringUtils.hasText(originalName) ? originalName : file.getFileName().toString();
        Resource resource = new FileSystemResource(file) {
            @Override
            public String getFilename() {
                return partName;
            }
        };
        HttpHeaders partHeaders = new HttpHeaders();
        partHeaders.setContentType(parseMediaType(contentType));

        MultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
        parts.add("file", new HttpEntity<>(resource, partHeaders));

        try {
            MlPrediction prediction = predictClient.post()
                    .uri("/predict")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(parts)
                    .retrieve()
                    .body(MlPrediction.class);
            if (prediction == null) {
                throw new MlServiceException("ML service returned an empty response");
            }
            return prediction;
        } catch (RestClientResponseException e) {
            throw new MlServiceException("ML service returned HTTP " + e.getStatusCode().value()
                    + ": " + snippet(e.getResponseBodyAsString()), e);
        } catch (ResourceAccessException e) {
            throw new MlServiceException("ML service unreachable: " + rootMessage(e), e);
        } catch (RestClientException e) {
            throw new MlServiceException("Invalid response from ML service: " + rootMessage(e), e);
        }
    }

    /**
     * Calls {@code GET {baseUrl}/health}. Never throws: on failure returns
     * {@code {"status":"down","error":...}}.
     */
    public Map<String, Object> health() {
        try {
            Map<String, Object> body = healthClient.get()
                    .uri("/health")
                    .retrieve()
                    .body(new ParameterizedTypeReference<Map<String, Object>>() {
                    });
            if (body == null) {
                return Map.of("status", "down", "error", "empty response");
            }
            return new LinkedHashMap<>(body);
        } catch (Exception e) {
            log.warn("ML health check failed: {}", rootMessage(e));
            return Map.of("status", "down", "error", rootMessage(e));
        }
    }

    private static MediaType parseMediaType(String contentType) {
        if (!StringUtils.hasText(contentType)) {
            return MediaType.APPLICATION_OCTET_STREAM;
        }
        try {
            return MediaType.parseMediaType(contentType);
        } catch (IllegalArgumentException e) {
            return MediaType.APPLICATION_OCTET_STREAM;
        }
    }

    private static String rootMessage(Throwable t) {
        Throwable cur = t;
        while (cur.getCause() != null && cur.getCause() != cur) {
            cur = cur.getCause();
        }
        String msg = cur.getMessage();
        if (!StringUtils.hasText(msg)) {
            msg = cur.getClass().getSimpleName();
        }
        return snippet(msg);
    }

    private static String snippet(String s) {
        if (s == null) {
            return "";
        }
        String trimmed = s.strip();
        return trimmed.length() > MAX_ERROR_SNIPPET ? trimmed.substring(0, MAX_ERROR_SNIPPET) + "..." : trimmed;
    }
}
