package com.subbulakshmi.deepfake.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Connection settings for the Python ML microservice, bound from {@code app.ml.*}.
 *
 * @param baseUrl          root URL of the ML service, e.g. {@code http://localhost:8000}
 * @param connectTimeoutMs TCP connect timeout in milliseconds
 * @param readTimeoutMs    read timeout in milliseconds for {@code /predict} (video inference can be slow)
 */
@ConfigurationProperties(prefix = "app.ml")
public record MlProperties(
        @DefaultValue("http://localhost:8000") String baseUrl,
        @DefaultValue("5000") int connectTimeoutMs,
        @DefaultValue("180000") int readTimeoutMs) {
}
