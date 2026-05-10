// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/dto/InterviewAnswerRequest.java
// Phase 3 버전: sessionId 기반 (Phase 2의 history 전달 방식에서 교체)
package com.domain.demo_backend.domain.ai.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class InterviewAnswerRequest {
    private String sessionId;   // Redis 세션 ID (startInterview SSE의 첫 이벤트에서 수신)
    private String answerText;  // 사용자 답변
}
