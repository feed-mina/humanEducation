// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/membership/controller/MembershipController.java
package com.domain.demo_backend.domain.membership.controller;

import com.domain.demo_backend.domain.membership.dto.MembershipRequest;
import com.domain.demo_backend.domain.membership.dto.MembershipResponse;
import com.domain.demo_backend.domain.membership.service.MembershipService;
import com.domain.demo_backend.global.common.response.ApiResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/api/v1/memberships")
@RequiredArgsConstructor
public class MembershipController {

    private final MembershipService membershipService;

    /**
     * GET /api/v1/memberships
     * 전체 멤버십 플랜 조회 (공개)
     */
    @GetMapping
    public ResponseEntity<ApiResponse<List<MembershipResponse>>> findAll() {
        return ResponseEntity.ok(ApiResponse.success(membershipService.findAll()));
    }

    /**
     * POST /api/v1/memberships
     * 멤버십 플랜 생성 (어드민)
     */
    @PostMapping
    public ResponseEntity<ApiResponse<MembershipResponse>> create(@RequestBody MembershipRequest req) {
        MembershipResponse created = membershipService.create(req);
        return ResponseEntity.status(HttpStatus.CREATED).body(ApiResponse.success(created));
    }

    /**
     * DELETE /api/v1/memberships/{id}
     * 멤버십 플랜 삭제 (어드민)
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        membershipService.delete(id);
        return ResponseEntity.noContent().build();
    }
}
