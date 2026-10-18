package com.subbulakshmi.deepfake.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Local file storage settings, bound from {@code app.storage.*}.
 *
 * @param dir directory where uploaded videos are kept; created on startup if missing
 */
@ConfigurationProperties(prefix = "app.storage")
public record StorageProperties(@DefaultValue("./uploads") String dir) {
}
