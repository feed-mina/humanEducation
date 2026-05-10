// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/membership/controller/UserMembershipController.java
package com.domain.demo_backend.domain.membership.controller;

import com.domain.demo_backend.domain.membership.dto.UserMembershipRequest;
import com.domain.demo_backend.domain.membership.dto.UserMembershipResponse;
import com.domain.demo_backend.domain.membership.service.UserMembershipService;
import com.domain.demo_backend.global.common.response.ApiResponse;
import com.domain.demo_backend.global.security.CustomUserDetails;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@Slf4j
@RestController
@RequestMapping("/api/v1/user-memberships")
@RequiredArgsConstructor
public class UserMembershipController {

    private final UserMembershipService userMembershipService;

    /**
     * GET /api/v1/user-memberships/current
     * 현재 로그인 사용자의 활성 멤버십 조회
     * 멤버십 없으면 data: null 반환
     */
    @GetMapping("/current")
    public ResponseEntity<ApiResponse<UserMembershipResponse>> getCurrent(
            @AuthenticationPrincipal CustomUserDetails userDetails) {

        UserMembershipResponse current = userMembershipService.findCurrent(userDetails.getUserSqno());
        return ResponseEntity.ok(ApiResponse.success(current));
    }

    /**
     * POST /api/v1/user-memberships
     * 사용자 멤버십 부여 (어드민 또는 결제 완료 후 서버 내부 호출)
     * 기존 활성 멤버십은 자동 취소 후 새로 생성
     */
    @PostMapping
    public ResponseEntity<ApiResponse<UserMembershipResponse>> grant(
            @RequestBody UserMembershipRequest req) {

        log.info("멤버십 부여 요청 - userId={}, membershipId={}", req.getUserId(), req.getMembershipId());
        UserMembershipResponse created = userMembershipService.grant(req);
        return ResponseEntity.status(HttpStatus.CREATED).body(ApiResponse.success(created));
    }
}
