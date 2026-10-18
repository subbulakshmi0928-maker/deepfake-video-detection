package com.subbulakshmi.deepfake.storage;

import com.subbulakshmi.deepfake.config.StorageProperties;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.UUID;

/**
 * Keeps uploaded videos on the local filesystem under {@code app.storage.dir}. Files are renamed to
 * a random UUID (extension preserved) so user-supplied names never touch the disk.
 */
@Service
public class StorageService {

    private static final Logger log = LoggerFactory.getLogger(StorageService.class);

    private final Path root;

    public StorageService(StorageProperties properties) {
        this.root = Paths.get(properties.dir()).toAbsolutePath().normalize();
    }

    @PostConstruct
    void init() {
        try {
            Files.createDirectories(root);
            log.info("Upload storage directory: {}", root);
        } catch (IOException e) {
            throw new UncheckedIOException("Could not create storage directory " + root, e);
        }
    }

    /** Root directory, absolute and normalised. */
    public Path root() {
        return root;
    }

    /**
     * Copies the upload into the storage directory under a fresh UUID-based name.
     *
     * @param file      the multipart upload
     * @param extension lower-case extension without the dot, e.g. {@code mp4}; may be empty
     * @return absolute path of the stored file
     */
    public Path store(MultipartFile file, String extension) {
        String ext = StringUtils.hasText(extension) ? "." + extension.toLowerCase(Locale.ROOT) : "";
        Path target = root.resolve(UUID.randomUUID() + ext);
        try (InputStream in = file.getInputStream()) {
            Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new UncheckedIOException("Could not store upload " + file.getOriginalFilename(), e);
        }
        return target;
    }

    /** Resolves a stored name to its absolute path, refusing anything that escapes the root. */
    public Path resolve(String storedName) {
        Path path = root.resolve(storedName).normalize();
        if (!path.startsWith(root) || path.equals(root)) {
            throw new IllegalArgumentException("Illegal stored file name: " + storedName);
        }
        return path;
    }

    /**
     * Deletes a stored file if present.
     *
     * @return {@code true} when a file was actually removed
     */
    public boolean delete(String storedName) {
        if (!StringUtils.hasText(storedName)) {
            return false;
        }
        try {
            return Files.deleteIfExists(resolve(storedName));
        } catch (IOException e) {
            throw new UncheckedIOException("Could not delete stored file " + storedName, e);
        }
    }
}
