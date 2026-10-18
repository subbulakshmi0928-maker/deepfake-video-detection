package com.subbulakshmi.deepfake;

import com.subbulakshmi.deepfake.config.MlProperties;
import com.subbulakshmi.deepfake.config.StorageProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

/**
 * Entry point of the Deepfake Video Detection backend.
 * <p>
 * Accepts video uploads over REST, delegates the actual classification to the Python ML
 * microservice and persists every analysis in PostgreSQL.
 */
@SpringBootApplication
@EnableConfigurationProperties({MlProperties.class, StorageProperties.class})
public class DeepfakeBackendApplication {

    public static void main(String[] args) {
        SpringApplication.run(DeepfakeBackendApplication.class, args);
    }
}
