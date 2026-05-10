// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/controller/AiInterviewController.java
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
     * 이력서 기반 첫 면접 질문 SSE 스트리밍
     */
    @PostMapping("/start")
    public SseEmitter startInterview(
            @RequestBody InterviewStartRequest req,
            @AuthenticationPrincipal CustomUserDetails userDetails) {

        log.info("면접 시작 요청 - userId={}, language={}", userDetails.getUserSqno(), req.getLanguage());

        SseEmitter emitter = new SseEmitter(180_000L);
        sseExecutor.execute(() -> interviewService.startInterview(req, emitter));
        return emitter;
    }

    /**
     * POST /api/ai/interview/answer
     * 답변 제출 → 후속 면접 질문 SSE 스트리밍
     * 클라이언트가 대화 이력(history)을 유지해 전달
     */
    @PostMapping("/answer")
    public SseEmitter submitAnswer(
            @RequestBody InterviewAnswerRequest req,
            @AuthenticationPrincipal CustomUserDetails userDetails) {

        log.info("면접 답변 제출 - userId={}, historySize={}",
                userDetails.getUserSqno(),
                req.getHistory() != null ? req.getHistory().size() : 0);

        SseEmitter emitter = new SseEmitter(180_000L);
        sseExecutor.execute(() -> interviewService.continueInterview(req, emitter));
        return emitter;
    }
}
