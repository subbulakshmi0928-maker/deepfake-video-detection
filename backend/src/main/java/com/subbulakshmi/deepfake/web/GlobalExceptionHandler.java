package com.subbulakshmi.deepfake.web;

import com.subbulakshmi.deepfake.exception.InvalidVideoException;
import com.subbulakshmi.deepfake.exception.NotFoundException;
import com.subbulakshmi.deepfake.ml.MlServiceException;
import com.subbulakshmi.deepfake.web.dto.ApiError;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StringUtils;
import org.springframework.web.ErrorResponse;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.multipart.MultipartException;

import java.time.Instant;

/**
 * Translates exceptions into a uniform JSON body
 * {@code { "timestamp", "status", "error", "message" }}.
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    @ExceptionHandler(InvalidVideoException.class)
    public ResponseEntity<ApiError> invalidVideo(InvalidVideoException ex) {
        return build(HttpStatus.BAD_REQUEST, ex.getMessage());
    }

    @ExceptionHandler(NotFoundException.class)
    public ResponseEntity<ApiError> notFound(NotFoundException ex) {
        return build(HttpStatus.NOT_FOUND, ex.getMessage());
    }

    @ExceptionHandler(MlServiceException.class)
    public ResponseEntity<ApiError> mlFailure(MlServiceException ex) {
        return build(HttpStatus.BAD_GATEWAY, ex.getMessage());
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<ApiError> tooLarge(MaxUploadSizeExceededException ex) {
        return build(HttpStatus.PAYLOAD_TOO_LARGE, "Upload exceeds the maximum allowed size");
    }

    /** Malformed multipart bodies (missing boundary, aborted stream, ...). */
    @ExceptionHandler(MultipartException.class)
    public ResponseEntity<ApiError> badMultipart(MultipartException ex) {
        return build(HttpStatus.BAD_REQUEST, "Invalid multipart request: " + rootMessage(ex));
    }

    /**
     * Fallback. Spring MVC's own exceptions (missing part, unknown route, wrong method, ...) implement
     * {@link ErrorResponse} and keep their natural status; anything else becomes a 500.
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiError> other(Exception ex) {
        if (ex instanceof ErrorResponse er) {
            HttpStatusCode status = er.getStatusCode();
            String detail = er.getBody() != null ? er.getBody().getDetail() : null;
            return build(status, StringUtils.hasText(detail) ? detail : ex.getMessage());
        }
        log.error("Unhandled exception", ex);
        return build(HttpStatus.INTERNAL_SERVER_ERROR, "Unexpected server error: " + rootMessage(ex));
    }

    private static ResponseEntity<ApiError> build(HttpStatusCode status, String message) {
        HttpStatus resolved = HttpStatus.resolve(status.value());
        String reason = resolved != null ? resolved.getReasonPhrase() : "Error";
        ApiError body = new ApiError(Instant.now(), status.value(), reason,
                StringUtils.hasText(message) ? message : reason);
        return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(body);
    }

    private static String rootMessage(Throwable t) {
        Throwable cur = t;
        while (cur.getCause() != null && cur.getCause() != cur) {
            cur = cur.getCause();
        }
        return StringUtils.hasText(cur.getMessage()) ? cur.getMessage() : cur.getClass().getSimpleName();
    }
}
