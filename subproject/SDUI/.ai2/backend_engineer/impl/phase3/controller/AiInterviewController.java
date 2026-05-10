// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/controller/AiInterviewController.java
// Phase 3 버전: Redis 세션 기반 (Phase 2 버전 교체)
package com.domain.demo_backend.domain.ai.controller;

import com.domain.demo_backend.domain.ai.dto.InterviewAnswerRequest;
import com.domain.demo_backend.domain.ai.dto.InterviewStartRequest;
import com.domain.demo_backend.domain.ai.service.InterviewService;
import com.domain.demo_backend.global.security.CustomUserDetails;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.concurrent.Executor;

/**
 * SSE 스트림 포맷:
 *
 * [start] 첫 이벤트: data: {"sessionId":"uuid"}
 *         이후 이벤트: data: {"chunk":"..."}
 *         종료: data: [DONE]
 *
 * [answer] data: {"chunk":"..."}  ...  data: [DONE]
 */
@Slf4j
@RestController
@RequestMapping("/api/ai/interview")
public class AiInterviewController {

    private final InterviewService interviewService;
    private final Executor sseExecutor;

    public AiInterviewController(InterviewService interviewService,
                                  @Qualifier("sseExecutor") Executor sseExecutor) {
        this.interviewService = interviewService;
        this.sseExecutor = sseExecutor;
    }

    /**
     * POST /api/ai/interview/start
     * 이력서 기반 면접 시작
     * 첫 SSE 이벤트: {"sessionId":"..."}  → 클라이언트가 저장해야 함
     */
    @PostMapping("/start")
    public SseEmitter startInterview(
            @RequestBody InterviewStartRequest req,
            @AuthenticationPrincipal CustomUserDetails userDetails) {

        Long userId = userDetails.getUserSqno();
        log.info("면접 시작 요청 - userId={}, language={}", userId, req.getLanguage());

        SseEmitter emitter = new SseEmitter(180_000L);
        sseExecutor.execute(() -> interviewService.startInterview(req, userId, emitter));
        return emitter;
    }

    /**
     * POST /api/ai/interview/answer
     * 답변 제출 → 후속 면접 질문 스트리밍
     * Body: { "sessionId": "uuid", "answerText": "..." }
     */
    @PostMapping("/answer")
    public SseEmitter submitAnswer(
            @RequestBody InterviewAnswerRequest req,
            @AuthenticationPrincipal CustomUserDetails userDetails) {

        log.info("면접 답변 제출 - userId={}, sessionId={}",
                userDetails.getUserSqno(), req.getSessionId());

        SseEmitter emitter = new SseEmitter(180_000L);
        sseExecutor.execute(() -> interviewService.continueInterview(req, emitter));
        return emitter;
    }
}
