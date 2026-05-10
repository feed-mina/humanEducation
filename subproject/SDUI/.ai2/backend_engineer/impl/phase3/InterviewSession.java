// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/session/InterviewSession.java
package com.domain.demo_backend.domain.ai.session;

import com.domain.demo_backend.domain.ai.dto.ChatMessage;
import lombok.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Redis에 저장되는 면접 세션 객체
 * Key: "interview:{sessionId}"
 * TTL: 30분 (InterviewSessionService에서 관리)
 * 직렬화: GenericJackson2JsonRedisSerializer (redisObjectTemplate)
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class InterviewSession {

    private String sessionId;
    private Long userId;
    private String language;
    private String resumeText;

    /**
     * 누적 대화 이력 (system 제외, user/assistant 교대)
     * [{role:"assistant", content:"첫 질문..."}, {role:"user", content:"답변..."}, ...]
     */
    @Builder.Default
    private List<ChatMessage> history = new ArrayList<>();
}
