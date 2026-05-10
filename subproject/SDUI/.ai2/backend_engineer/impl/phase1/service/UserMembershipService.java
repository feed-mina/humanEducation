// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/membership/service/UserMembershipService.java
package com.domain.demo_backend.domain.membership.service;

import com.domain.demo_backend.domain.membership.domain.Membership;
import com.domain.demo_backend.domain.membership.domain.MembershipRepository;
import com.domain.demo_backend.domain.membership.domain.UserMembership;
import com.domain.demo_backend.domain.membership.domain.UserMembershipRepository;
import com.domain.demo_backend.domain.membership.dto.UserMembershipRequest;
import com.domain.demo_backend.domain.membership.dto.UserMembershipResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
public class UserMembershipService {

    private final UserMembershipRepository userMembershipRepository;
    private final MembershipRepository membershipRepository;

    /**
     * 현재 활성 멤버십 조회 (없으면 null 반환)
     */
    @Transactional(readOnly = true)
    public UserMembershipResponse findCurrent(Long userId) {
        return userMembershipRepository
                .findActiveByUserId(userId, LocalDateTime.now())
                .map(UserMembershipResponse::new)
                .orElse(null);
    }

    /**
     * 멤버십 부여: 기존 활성 멤버십은 cancelled 처리 후 새로 생성
     */
    @Transactional
    public UserMembershipResponse grant(UserMembershipRequest req) {
        Membership membership = membershipRepository.findById(req.getMembershipId())
                .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 멤버십입니다: " + req.getMembershipId()));

        LocalDateTime now = LocalDateTime.now();
        LocalDateTime startedAt = req.getStartedAt() != null ? req.getStartedAt() : now;
        LocalDateTime expiresAt = startedAt.plusDays(membership.getDurationDays());

        // 기존 활성 멤버십 취소
        userMembershipRepository.cancelActiveByUserId(req.getUserId(), now);

        UserMembership userMembership = UserMembership.builder()
                .userId(req.getUserId())
                .membership(membership)
                .startedAt(startedAt)
                .expiresAt(expiresAt)
                .status("active")
                .grantedBy(req.getGrantedBy() != null ? req.getGrantedBy() : "purchase")
                .build();

        log.info("멤버십 부여 - userId={}, membership={}, expires={}",
                req.getUserId(), membership.getName(), expiresAt);

        return new UserMembershipResponse(userMembershipRepository.save(userMembership));
    }

    /**
     * 멤버십 권한 체크 (Phase 4 게이팅용)
     */
    @Transactional(readOnly = true)
    public boolean canConverse(Long userId) {
        return userMembershipRepository.findActiveByUserId(userId, LocalDateTime.now())
                .map(um -> um.getMembership().isCanConverse())
                .orElse(false);
    }

    @Transactional(readOnly = true)
    public boolean canLearn(Long userId) {
        return userMembershipRepository.findActiveByUserId(userId, LocalDateTime.now())
                .map(um -> um.getMembership().isCanLearn())
                .orElse(false);
    }

    @Transactional(readOnly = true)
    public boolean canAnalyze(Long userId) {
        return userMembershipRepository.findActiveByUserId(userId, LocalDateTime.now())
                .map(um -> um.getMembership().isCanAnalyze())
                .orElse(false);
    }
}
