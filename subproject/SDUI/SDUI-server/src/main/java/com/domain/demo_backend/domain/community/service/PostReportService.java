package com.domain.demo_backend.domain.community.service;

import com.domain.demo_backend.domain.community.domain.*;
import com.domain.demo_backend.domain.community.dto.ReportRequest;
import com.domain.demo_backend.domain.user.domain.User;
import com.domain.demo_backend.domain.user.domain.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class PostReportService {

    private final PostReportRepository reportRepository;
    private final CommunityPostRepository postRepository;
    private final UserRepository userRepository;

    @Transactional
    public void reportPost(Long postId, Long reporterSqno, ReportRequest request) {
        if (reportRepository.existsByPost_PostIdAndReporter_UserSqno(postId, reporterSqno)) {
            throw new IllegalArgumentException("이미 신고한 게시글입니다.");
        }

        CommunityPost post = postRepository.findById(postId)
                .orElseThrow(() -> new IllegalArgumentException("게시글을 찾을 수 없습니다."));
        User reporter = userRepository.findById(reporterSqno)
                .orElseThrow(() -> new IllegalArgumentException("사용자를 찾을 수 없습니다."));

        PostReport report = PostReport.builder()
                .post(post)
                .reporter(reporter)
                .reasonCode(request.getReasonCode())
                .detailText(request.getDetailText())
                .build();

        reportRepository.save(report);
        post.setReportCount(post.getReportCount() + 1);
    }
}
