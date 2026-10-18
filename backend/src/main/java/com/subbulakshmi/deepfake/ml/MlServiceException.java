package com.subbulakshmi.deepfake.ml;

/**
 * Raised when the ML microservice cannot be reached, times out, answers with an error status or
 * returns an unparseable body. Mapped to HTTP 502 by the API layer.
 */
public class MlServiceException extends RuntimeException {

    public MlServiceException(String message) {
        super(message);
    }

    public MlServiceException(String message, Throwable cause) {
        super(message, cause);
    }
}
