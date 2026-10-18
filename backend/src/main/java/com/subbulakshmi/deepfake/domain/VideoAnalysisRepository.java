package com.subbulakshmi.deepfake.domain;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

/**
 * Spring Data repository for {@link VideoAnalysis} rows.
 */
public interface VideoAnalysisRepository extends JpaRepository<VideoAnalysis, Long> {

    Page<VideoAnalysis> findAllByOrderByCreatedAtDesc(Pageable pageable);

    long countByVerdict(Verdict verdict);

    /**
     * Average confidence over successful analyses only (rows with verdict {@code ERROR} are
     * excluded). Returns {@code null} when there is nothing to average.
     */
    @Query("select avg(v.confidence) from VideoAnalysis v "
            + "where v.verdict <> com.subbulakshmi.deepfake.domain.Verdict.ERROR "
            + "and v.confidence is not null")
    Double averageConfidenceExcludingErrors();
}
