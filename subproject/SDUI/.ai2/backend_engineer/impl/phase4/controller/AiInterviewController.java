// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/controller/AiInterviewController.java
// Phase 4 버전: 멤버십 canLearn 권한 게이팅 추가 (Phase 3 버전 교체)
package com.domain.demo_backend.domain.ai.controller;

import com.domain.demo_backend.domain.ai.dto.InterviewAnswerRequest;
import com.domain.demo_backend.domain.ai.dto.InterviewStartRequest;
import com.domain.demo_backend.domain.ai.service.InterviewService;
import com.domain.demo_backend.domain.membership.service.UserMembershipService;
import com.domain.demo_backend.global.security.CustomUserDetails;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.concurrent.Executor;

@Slf4j
@RestController
@RequestMapping("/api/ai/interview")
public class AiInterviewController {

    private final InterviewService interviewService;
    private final UserMembershipService userMembershipService;
    private final Executor sseExecutor;

    public AiInterviewController(InterviewService interviewService,
                                  UserMembershipService userMembershipService,
                                  @Qualifier("sseExecutor") Executor sseExecutor) {
        this.interviewService = interviewService;
        this.userMembershipService = userMembershipService;
        this.sseExecutor = sseExecutor;
    }

    /**
     * POST /api/ai/interview/start
     * 멤버십 canLearn 권한 필요 (베이직 이상)
     */
    @PostMapping("/start")
    public SseEmitter startInterview(
            @RequestBody InterviewStartRequest req,
            @AuthenticationPrincipal CustomUserDetails userDetails) {

        Long userId = userDetails.getUserSqno();

        if (!userMembershipService.canLearn(userId)) {
            return membershipErrorEmitter("면접 기능은 베이직 이상 멤버십이 필요합니다.");
        }

        log.info("면접 시작 요청 - userId={}, language={}", userId, req.getLanguage());
        SseEmitter emitter = new SseEmitter(180_000L);
        sseExecutor.execute(() -> interviewService.startInterview(req, userId, emitter));
        return emitter;
    }

    /**
     * POST /api/ai/interview/answer
     * 멤버십 canLearn 권한 필요 (베이직 이상)
     */
    @PostMapping("/answer")
    public SseEmitter submitAnswer(
            @RequestBody InterviewAnswerRequest req,
            @AuthenticationPrincipal CustomUserDetails userDetails) {

        Long userId = userDetails.getUserSqno();

        if (!userMembershipService.canLearn(userId)) {
            return membershipErrorEmitter("면접 기능은 베이직 이상 멤버십이 필요합니다.");
        }

        log.info("면접 답변 제출 - userId={}, sessionId={}", userId, req.getSessionId());
        SseEmitter emitter = new SseEmitter(180_000L);
        sseExecutor.execute(() -> interviewService.continueInterview(req, emitter));
        return emitter;
    }

    private SseEmitter membershipErrorEmitter(String message) {
        SseEmitter emitter = new SseEmitter();
        try {
            emitter.send(SseEmitter.event()
                    .data("{\"error\":\"" + message + "\"}"));
            emitter.complete();
        } catch (Exception e) {
            emitter.completeWithError(e);
        }
        return emitter;
    }
}
