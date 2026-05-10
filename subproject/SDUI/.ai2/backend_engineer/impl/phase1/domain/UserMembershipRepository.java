// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/membership/domain/UserMembershipRepository.java
package com.domain.demo_backend.domain.membership.domain;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.LocalDateTime;
import java.util.Optional;

public interface UserMembershipRepository extends JpaRepository<UserMembership, Long> {

    /**
     * 사용자의 현재 활성 멤버십 조회 (status=active AND expires_at > now)
     * Phase 1 단순 조건 → JPQL 사용 (Phase 3+ 동적 필터 필요 시 QueryDSL 도입)
     */
    @Query("SELECT um FROM UserMembership um JOIN FETCH um.membership " +
           "WHERE um.userId = :userId AND um.status = 'active' AND um.expiresAt > :now " +
           "ORDER BY um.createdAt DESC")
    Optional<UserMembership> findActiveByUserId(
            @Param("userId") Long userId,
            @Param("now") LocalDateTime now
    );

    /**
     * 기존 활성 멤버십 취소 (새 멤버십 부여 전 처리)
     */
    @Modifying
    @Query("UPDATE UserMembership um SET um.status = 'cancelled', um.updatedAt = :now " +
           "WHERE um.userId = :userId AND um.status = 'active'")
    void cancelActiveByUserId(
            @Param("userId") Long userId,
            @Param("now") LocalDateTime now
    );
}
