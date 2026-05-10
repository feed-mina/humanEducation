// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/controller/AiChatController.java
// Phase 4 버전: 멤버십 canConverse 권한 게이팅 추가 (Phase 2 버전 교체)
package com.domain.demo_backend.domain.ai.controller;

import com.domain.demo_backend.domain.ai.dto.ChatRequest;
import com.domain.demo_backend.domain.ai.service.ChatService;
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
@RequestMapping("/api/ai")
public class AiChatController {

    private final ChatService chatService;
    private final UserMembershipService userMembershipService;
    private final Executor sseExecutor;

    public AiChatController(ChatService chatService,
                             UserMembershipService userMembershipService,
                             @Qualifier("sseExecutor") Executor sseExecutor) {
        this.chatService = chatService;
        this.userMembershipService = userMembershipService;
        this.sseExecutor = sseExecutor;
    }

    /**
     * POST /api/ai/chat/stream
     * 멤버십 canConverse 권한 필요 (프리미엄 이상)
     */
    @PostMapping("/chat/stream")
    public SseEmitter streamChat(
            @RequestBody ChatRequest req,
            @AuthenticationPrincipal CustomUserDetails userDetails) {

        Long userId = userDetails.getUserSqno();

        if (!userMembershipService.canConverse(userId)) {
            SseEmitter emitter = new SseEmitter();
            try {
                emitter.send(SseEmitter.event()
                        .data("{\"error\":\"음성 대화 기능은 프리미엄 멤버십이 필요합니다.\"}"));
                emitter.complete();
            } catch (Exception e) {
                emitter.completeWithError(e);
            }
            return emitter;
        }

        log.info("채팅 스트리밍 요청 - userId={}", userId);
        SseEmitter emitter = new SseEmitter(180_000L);
        sseExecutor.execute(() -> chatService.stream(req, userId, emitter));
        return emitter;
    }
}
