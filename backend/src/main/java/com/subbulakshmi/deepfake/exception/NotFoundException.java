package com.subbulakshmi.deepfake.exception;

/**
 * Thrown when a requested analysis does not exist. Mapped to HTTP 404.
 */
public class NotFoundException extends RuntimeException {

    public NotFoundException(String message) {
        super(message);
    }
}
