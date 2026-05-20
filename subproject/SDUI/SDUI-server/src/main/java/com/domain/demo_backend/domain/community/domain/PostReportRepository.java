package com.domain.demo_backend.domain.community.domain;

import org.springframework.data.jpa.repository.JpaRepository;

public interface PostReportRepository extends JpaRepository<PostReport, Long> {

    boolean existsByPost_PostIdAndReporter_UserSqno(Long postId, Long reporterSqno);
}
