package com.subbulakshmi.deepfake.web.dto;

import java.util.List;

/**
 * Page envelope used by {@code GET /api/analyses}.
 *
 * @param items the rows of this page
 * @param page  zero-based page index
 * @param size  requested page size
 * @param total total number of rows across all pages
 */
public record PagedResponse<T>(List<T> items, int page, int size, long total) {
}
