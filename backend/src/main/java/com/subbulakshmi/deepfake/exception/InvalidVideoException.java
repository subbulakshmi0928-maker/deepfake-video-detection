package com.subbulakshmi.deepfake.exception;

/**
 * Thrown when an upload is rejected before reaching the ML service (empty file, unsupported
 * extension, ...). Mapped to HTTP 400.
 */
public class InvalidVideoException extends RuntimeException {

    public InvalidVideoException(String message) {
        super(message);
    }
}
