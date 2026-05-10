// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/service/InterviewService.java
// Phase 3 버전: Redis 세션 기반 (Phase 2의 클라이언트 이력 전달 방식에서 교체)
package com.domain.demo_backend.domain.ai.service;

import com.domain.demo_backend.domain.ai.client.OpenAiClient;
import com.domain.demo_backend.domain.ai.dto.ChatMessage;
import com.domain.demo_backend.domain.ai.dto.InterviewAnswerRequest;
import com.domain.demo_backend.domain.ai.dto.InterviewStartRequest;
import com.domain.demo_backend.domain.ai.session.InterviewSession;
import com.domain.demo_backend.domain.ai.session.InterviewSessionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class InterviewService {

    private static final String SYSTEM_PROMPT_KO = """
            당신은 경력 면접관입니다. 아래 이력서를 바탕으로 면접을 진행하세요.
            - 질문은 1개씩 차례로 합니다.
            - 구체적인 경험과 역량을 검증하는 질문을 합니다.
            - 이전 답변을 참조해 후속 질문을 자연스럽게 이어가세요.
            """;

    private static final String SYSTEM_PROMPT_EN = """
            You are an experienced interviewer. Conduct an interview based on the resume below.
            - Ask one question at a time.
            - Focus on verifying specific experience and competencies.
            - Follow up naturally based on previous answers.
            """;

    private final OpenAiClient openAiClient;
    private final InterviewSessionService sessionService;

    /**
     * 면접 시작
     * 1. Redis 세션 생성 → sessionId 발급
     * 2. SSE 첫 이벤트로 sessionId 전송
     * 3. 첫 면접 질문 스트리밍
     * 4. 어시스턴트 응답 전체를 세션 이력에 저장
     */
    public void startInterview(InterviewStartRequest req, Long userId, SseEmitter emitter) {
        String sessionId = sessionService.createSession(userId, req.getLanguage(), req.getResumeText());

        // 첫 SSE 이벤트: sessionId 전달
        try {
            emitter.send(SseEmitter.event().data("{\"sessionId\":\"" + sessionId + "\"}"));
        } catch (Exception e) {
            log.warn("sessionId 전송 실패", e);
            emitter.completeWithError(e);
            return;
        }

        String systemPrompt = buildSystemPrompt(req.getLanguage(), req.getResumeText());
        String startMessage = "ko".equals(req.getLanguage()) ? "면접을 시작해주세요." : "Please start the interview.";

        List<Map<String, String>> messages = List.of(
                Map.of("role", "system", "content", systemPrompt),
                Map.of("role", "user", "content", startMessage)
        );

        streamAndSave(messages, emitter, sessionId, null, "면접 시작");
    }

    /**
     * 면접 답변 제출
     * 1. Redis에서 세션 조회 (이력 포함)
     * 2. 시스템 프롬프트 + 누적 이력 + 새 답변으로 메시지 목록 구성
     * 3. 후속 질문 스트리밍
     * 4. 사용자 답변 + 어시스턴트 응답 세션 이력에 저장
     */
    public void continueInterview(InterviewAnswerRequest req, SseEmitter emitter) {
        InterviewSession session = sessionService.getSession(req.getSessionId());

        String systemPrompt = buildSystemPrompt(session.getLanguage(), session.getResumeText());

        List<Map<String, String>> messages = new ArrayList<>();
        messages.add(Map.of("role", "system", "content", systemPrompt));

        // 누적 이력 추가
        for (ChatMessage msg : session.getHistory()) {
            messages.add(Map.of("role", msg.getRole(), "content", msg.getContent()));
        }

        // 최신 답변 추가
        messages.add(Map.of("role", "user", "content", req.getAnswerText()));

        streamAndSave(messages, emitter, req.getSessionId(), req.getAnswerText(), "면접 답변");
    }

    /**
     * SSE 스트리밍 + 완료 후 이력 저장
     * @param userAnswer null이면 startInterview (사용자 메시지 없음)
     */
    private void streamAndSave(List<Map<String, String>> messages, SseEmitter emitter,
                                String sessionId, String userAnswer, String context) {
        StringBuilder fullResponse = new StringBuilder();
        try {
            openAiClient.streamChat(
                    messages,
                    chunk -> {
                        fullResponse.append(chunk);
                        try {
                            emitter.send(SseEmitter.event()
                                    .data("{\"chunk\":\"" + escapeJson(chunk) + "\"}"));
                        } catch (Exception e) {
                            log.warn("SSE 청크 전송 실패 ({})", context, e);
                        }
                    },
                    () -> {
                        try {
                            emitter.send(SseEmitter.event().data("[DONE]"));
                            emitter.complete();
                        } catch (Exception e) {
                            log.warn("SSE 완료 전송 실패 ({})", context, e);
                        }
                        // 이력 저장 (스트리밍 완료 후)
                        List<ChatMessage> toSave = new ArrayList<>();
                        if (userAnswer != null) {
                            toSave.add(new ChatMessage("user", userAnswer));
                        }
                        toSave.add(new ChatMessage("assistant", fullResponse.toString()));
                        sessionService.appendHistory(sessionId, toSave);
                    }
            );
        } catch (Exception e) {
            log.error("인터뷰 스트리밍 실패 ({})", context, e);
            emitter.completeWithError(e);
        }
    }

    private String buildSystemPrompt(String language, String resumeText) {
        String base = "ko".equals(language) ? SYSTEM_PROMPT_KO : SYSTEM_PROMPT_EN;
        String label = "ko".equals(language) ? "\n\n이력서:\n" : "\n\nResume:\n";
        return base + label + (resumeText != null ? resumeText : "");
    }

    private String escapeJson(String text) {
        return text.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
