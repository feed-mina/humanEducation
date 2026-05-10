// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/service/InterviewService.java
package com.domain.demo_backend.domain.ai.service;

import com.domain.demo_backend.domain.ai.client.OpenAiClient;
import com.domain.demo_backend.domain.ai.dto.InterviewAnswerRequest;
import com.domain.demo_backend.domain.ai.dto.InterviewStartRequest;
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

    /**
     * 면접 시작: 이력서 → 시스템 프롬프트 구성 → 첫 질문 SSE 스트리밍
     */
    public void startInterview(InterviewStartRequest req, SseEmitter emitter) {
        String systemPrompt = buildSystemPrompt(req.getLanguage(), req.getResumeText());
        String startMessage = "ko".equals(req.getLanguage())
                ? "면접을 시작해주세요."
                : "Please start the interview.";

        List<Map<String, String>> messages = List.of(
                Map.of("role", "system", "content", systemPrompt),
                Map.of("role", "user", "content", startMessage)
        );

        streamToEmitter(messages, emitter, "면접 시작");
    }

    /**
     * 면접 답변 제출: 대화 이력 + 새 답변 → 후속 질문 SSE 스트리밍
     * 대화 이력은 클라이언트가 유지하여 전달 (서버 세션 불필요)
     */
    public void continueInterview(InterviewAnswerRequest req, SseEmitter emitter) {
        String systemPrompt = buildSystemPrompt(req.getLanguage(), req.getResumeText());

        List<Map<String, String>> messages = new ArrayList<>();
        messages.add(Map.of("role", "system", "content", systemPrompt));

        // 이전 대화 이력 추가
        if (req.getHistory() != null) {
            for (var msg : req.getHistory()) {
                messages.add(Map.of("role", msg.getRole(), "content", msg.getContent()));
            }
        }

        // 최신 답변 추가
        messages.add(Map.of("role", "user", "content", req.getAnswerText()));

        streamToEmitter(messages, emitter, "면접 답변");
    }

    private String buildSystemPrompt(String language, String resumeText) {
        String base = "ko".equals(language) ? SYSTEM_PROMPT_KO : SYSTEM_PROMPT_EN;
        String resumeLabel = "ko".equals(language) ? "\n\n이력서:\n" : "\n\nResume:\n";
        return base + resumeLabel + (resumeText != null ? resumeText : "");
    }

    private void streamToEmitter(List<Map<String, String>> messages, SseEmitter emitter, String context) {
        try {
            openAiClient.streamChat(
                    messages,
                    chunk -> {
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
                    }
            );
        } catch (Exception e) {
            log.error("인터뷰 스트리밍 실패 ({})", context, e);
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
