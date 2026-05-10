// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/service/ChatService.java
package com.domain.demo_backend.domain.ai.service;

import com.domain.demo_backend.domain.ai.client.OpenAiClient;
import com.domain.demo_backend.domain.ai.dto.ChatRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class ChatService {

    private final OpenAiClient openAiClient;

    public void stream(ChatRequest req, Long userId, SseEmitter emitter) {
        try {
            List<Map<String, String>> messages = req.getMessages().stream()
                    .map(m -> Map.of("role", m.getRole(), "content", m.getContent()))
                    .collect(Collectors.toList());

            openAiClient.streamChat(
                    messages,
                    chunk -> {
                        try {
                            emitter.send(SseEmitter.event()
                                    .data("{\"chunk\":\"" + escapeJson(chunk) + "\"}"));
                        } catch (Exception e) {
                            log.warn("SSE 청크 전송 실패 (userId={})", userId, e);
                        }
                    },
                    () -> {
                        try {
                            emitter.send(SseEmitter.event().data("[DONE]"));
                            emitter.complete();
                        } catch (Exception e) {
                            log.warn("SSE 완료 전송 실패 (userId={})", userId, e);
                        }
                    }
            );
        } catch (Exception e) {
            log.error("채팅 스트리밍 실패 (userId={})", userId, e);
            emitter.completeWithError(e);
        }
    }

    private String escapeJson(String text) {
        return text.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
