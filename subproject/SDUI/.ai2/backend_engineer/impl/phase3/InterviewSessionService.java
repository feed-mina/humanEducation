// DESTINATION: SDUI-server/src/main/java/com/domain/demo_backend/domain/ai/session/InterviewSessionService.java
package com.domain.demo_backend.domain.ai.session;

import com.domain.demo_backend.domain.ai.dto.ChatMessage;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
public class InterviewSessionService {

    private static final String KEY_PREFIX = "interview:";
    private static final long SESSION_TTL_SECONDS = 1800L; // 30분

    private final RedisTemplate<String, Object> redisTemplate;
    private final ObjectMapper objectMapper;

    // @Qualifier + Lombok @RequiredArgsConstructor 미지원 → 명시적 생성자
    public InterviewSessionService(
            @Qualifier("redisObjectTemplate") RedisTemplate<String, Object> redisTemplate,
            ObjectMapper objectMapper) {
        this.redisTemplate = redisTemplate;
        this.objectMapper = objectMapper;
    }

    /**
     * 신규 세션 생성 후 sessionId 반환
     */
    public String createSession(Long userId, String language, String resumeText) {
        String sessionId = UUID.randomUUID().toString();
        InterviewSession session = InterviewSession.builder()
                .sessionId(sessionId)
                .userId(userId)
                .language(language)
                .resumeText(resumeText)
                .build();
        save(session);
        log.info("면접 세션 생성 - sessionId={}, userId={}", sessionId, userId);
        return sessionId;
    }

    /**
     * 세션 조회 (없으면 IllegalArgumentException)
     */
    public InterviewSession getSession(String sessionId) {
        Object raw = redisTemplate.opsForValue().get(KEY_PREFIX + sessionId);
        if (raw == null) {
            throw new IllegalArgumentException("면접 세션을 찾을 수 없습니다 (만료 또는 미존재): " + sessionId);
        }
        return objectMapper.convertValue(raw, InterviewSession.class);
    }

    /**
     * 대화 이력 추가 후 TTL 갱신
     */
    public void appendHistory(String sessionId, List<ChatMessage> newMessages) {
        InterviewSession session = getSession(sessionId);
        session.getHistory().addAll(newMessages);
        save(session);
    }

    /**
     * 세션 삭제 (면접 종료 시 optional)
     */
    public void deleteSession(String sessionId) {
        redisTemplate.delete(KEY_PREFIX + sessionId);
        log.info("면접 세션 삭제 - sessionId={}", sessionId);
    }

    private void save(InterviewSession session) {
        redisTemplate.opsForValue().set(
                KEY_PREFIX + session.getSessionId(),
                session,
                SESSION_TTL_SECONDS,
                TimeUnit.SECONDS
        );
    }
}
