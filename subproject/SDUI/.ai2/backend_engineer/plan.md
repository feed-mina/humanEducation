# Backend Engineer Plan: SDUI 기반 멤버십 + AI 파이프라인 구현

> 작성일: 2026-03-11
> 최종 수정: 2026-03-13
> 근거: architect/plan.md + backend_engineer/research.md
> 상태: **Phase 2 구현 완료 / Phase 1(멤버십) 미구현**

---

## 구현 순서 개요

**Phase 2 먼저** (AI 파이프라인 검증) → **Phase 1** (멤버십 도메인) → **Phase 3** (AI 면접관) → **Phase 4** (통합 테스트)

---

## Phase 2: AI 파이프라인 구현

### 신규 파일 목록 (SDUI-server 내)

```
SDUI-server/src/main/java/com/domain/demo_backend/
├── domain/ai/
│   ├── client/
│   │   └── OpenAiClient.java
│   ├── dto/
│   │   ├── SttRequest.java       (MultipartFile 래핑)
│   │   ├── SttResponse.java      (text 필드)
│   │   ├── ChatRequest.java      (messages, language)
│   │   ├── ChatMessage.java      (role, content)
│   │   ├── InterviewStartRequest.java  (resumeText, language)
│   │   └── InterviewAnswerRequest.java (sessionId, answerText)
│   ├── service/
│   │   ├── SttService.java
│   │   ├── ChatService.java
│   │   └── InterviewService.java
│   └── controller/
│       ├── AiSttController.java
│       ├── AiChatController.java
│       └── AiInterviewController.java
└── global/config/
    └── AsyncConfig.java
```

### API 스펙

#### POST /api/ai/stt
```
Content-Type: multipart/form-data
Body: audio (MultipartFile)

200 OK
{ "data": { "text": "Hello, how are you?" } }
```

#### POST /api/ai/chat (SSE Streaming)
```
Content-Type: application/json
Body: { "messages": [...], "language": "en" }
Authorization: Bearer {jwt}

200 OK — text/event-stream
data: {"chunk": "Hello"}
data: {"chunk": " there"}
data: [DONE]
```

#### POST /api/ai/interview/start (SSE Streaming)
```json
// Request
{ "resumeText": "...", "language": "ko" }

// SSE Stream (면접 질문 텍스트 스트리밍)
data: {"chunk": "안녕하세요. 이력서를 보니"}
data: {"chunk": " 스프링부트 경험이 있으시군요. "}
data: [DONE]
```

#### POST /api/ai/interview/answer (SSE Streaming)
```json
// Request
{ "sessionId": "abc123", "answerText": "네, 3년 경험이 있습니다." }

// SSE Stream (다음 질문 스트리밍)
data: {"chunk": "그렇군요. 그렇다면"}
data: [DONE]
```

### 핵심 코드 스니펫

#### `OpenAiClient.java`
> ⚠️ Spring Boot 3.1.4 = Spring 6.0.x → `RestClient` 없음 (Spring 6.1+ 필요)
> STT: `RestTemplate` (Spring 내장, multipart 지원) / 스트리밍: `java.net.http.HttpClient` (Java 17 내장, 추가 의존성 없음)

```java
@Component
public class OpenAiClient {
    // STT: Spring RestTemplate (multipart/form-data 기본 지원)
    private final RestTemplate restTemplate = new RestTemplate();
    // Streaming: Java 17 내장 HttpClient
    private final HttpClient httpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(30)).build();
    private final ObjectMapper objectMapper = new ObjectMapper();

    // STT: multipart/form-data → Whisper
    public String transcribe(MultipartFile audio, String language) throws IOException {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        headers.set("Authorization", "Bearer " + apiKey);
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new ByteArrayResource(audio.getBytes()) {
            @Override public String getFilename() { return audio.getOriginalFilename(); }
        });
        body.add("model", whisperModel);
        body.add("language", language);
        ResponseEntity<Map> response = restTemplate.postForEntity(
            OPENAI_BASE_URL + "/audio/transcriptions",
            new HttpEntity<>(body, headers), Map.class
        );
        return response.getBody().get("text").toString();
    }

    // Chat: GPT-4o SSE 스트리밍 (java.net.http.HttpClient)
    public void streamChat(List<Map<String,String>> messages, Consumer<String> onChunk, Runnable onComplete) throws Exception {
        String jsonBody = objectMapper.writeValueAsString(
            Map.of("model", model, "messages", messages, "stream", true)
        );
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(OPENAI_BASE_URL + "/chat/completions"))
            .header("Authorization", "Bearer " + apiKey)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(jsonBody)).build();
        HttpResponse<InputStream> response = httpClient.send(request, HttpResponse.BodyHandlers.ofInputStream());
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.body()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("data: ") && !line.equals("data: [DONE]")) {
                    String chunk = extractChunk(line.substring(6));
                    if (chunk != null) onChunk.accept(chunk);
                }
            }
        }
        onComplete.run();
    }
}
```

#### `AiChatController.java`
```java
@RestController
@RequestMapping("/api/ai")
public class AiChatController {

    private final ChatService chatService;
    private final Executor executor;

    @PostMapping(value = "/chat", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter chat(
            @RequestBody ChatRequest req,
            @AuthenticationPrincipal UserDetails user) {
        SseEmitter emitter = new SseEmitter(30_000L);
        executor.execute(() -> chatService.stream(req, user.getUsername(), emitter));
        return emitter;
    }
}
```

#### `ChatService.java` (SSE 스트리밍)
```java
@Service
public class ChatService {

    public void stream(ChatRequest req, String userId, SseEmitter emitter) {
        try {
            openAiClient.streamChat(req.getMessages(),
                chunk -> emitter.send(SseEmitter.event().data("{\"chunk\":\"" + chunk + "\"}")),
                () -> {
                    emitter.send(SseEmitter.event().data("[DONE]"));
                    emitter.complete();
                });
        } catch (Exception e) {
            emitter.completeWithError(e);
        }
    }
}
```

#### `AsyncConfig.java`
```java
@Configuration
@EnableAsync
public class AsyncConfig {
    @Bean(name = "sseExecutor")
    public Executor sseExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(10);
        executor.setMaxPoolSize(50);
        executor.setQueueCapacity(100);
        executor.setThreadNamePrefix("sse-");
        executor.initialize();
        return executor;
    }
}
```

#### `InterviewService.java` (프롬프트 엔지니어링)
```java
@Service
public class InterviewService {

    private static final String INTERVIEW_SYSTEM_PROMPT_KO = """
        당신은 경력 면접관입니다. 아래 이력서를 바탕으로 면접을 진행하세요.
        - 질문은 1개씩 차례로 합니다.
        - 구체적인 경험과 역량을 검증하는 질문을 합니다.
        - 이전 답변을 참조해 후속 질문을 자연스럽게 이어가세요.
        """;

    private static final String INTERVIEW_SYSTEM_PROMPT_EN = """
        You are an experienced interviewer. Conduct an interview based on the resume below.
        - Ask one question at a time.
        - Focus on verifying specific experience and competencies.
        - Follow up naturally based on previous answers.
        """;

    public void startInterview(InterviewStartRequest req, SseEmitter emitter) {
        String systemPrompt = "ko".equals(req.getLanguage())
            ? INTERVIEW_SYSTEM_PROMPT_KO : INTERVIEW_SYSTEM_PROMPT_EN;
        List<ChatMessage> messages = List.of(
            new ChatMessage("system", systemPrompt + "\n\n이력서:\n" + req.getResumeText()),
            new ChatMessage("user", "면접을 시작해주세요.")
        );
        // streamChat 호출
    }
}
```

---

## Phase 1: 멤버십 도메인 구현

### Flyway 마이그레이션 파일

> ⚠️ 현재 프로젝트 Flyway 최신 버전: V21 → V22, V23 사용

```
SDUI-server/src/main/resources/db/migration/
├── V22__create_memberships.sql
└── V23__create_user_memberships.sql
```

#### `V22__create_memberships.sql`
```sql
CREATE TABLE memberships (
    id            BIGSERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL UNIQUE,
    can_learn     BOOLEAN NOT NULL DEFAULT FALSE,
    can_converse  BOOLEAN NOT NULL DEFAULT FALSE,
    can_analyze   BOOLEAN NOT NULL DEFAULT FALSE,
    duration_days INTEGER NOT NULL,
    price_cents   INTEGER NOT NULL,
    description   TEXT,
    created_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

INSERT INTO memberships (name, can_learn, can_converse, can_analyze, duration_days, price_cents, description)
VALUES
  ('베이직',   TRUE,  FALSE, FALSE, 30, 129000, 'AI 학습 기능만 이용 가능'),
  ('프리미엄', TRUE,  TRUE,  TRUE,  30, 219000, 'AI 학습 + 음성 대화 + 분석 이용 가능');
```

#### `V23__create_user_memberships.sql`
```sql
CREATE TABLE user_memberships (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT NOT NULL,
    membership_id BIGINT NOT NULL REFERENCES memberships(id),
    started_at    TIMESTAMP NOT NULL,
    expires_at    TIMESTAMP NOT NULL,
    status        VARCHAR(20) NOT NULL DEFAULT 'active',
    granted_by    VARCHAR(20) NOT NULL DEFAULT 'purchase',
    created_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_user_memberships_user_id ON user_memberships(user_id);
CREATE INDEX idx_user_memberships_user_status ON user_memberships(user_id, status);
CREATE INDEX idx_user_memberships_expires_at ON user_memberships(expires_at);
```

### 신규 Java 파일 목록

```
SDUI-server/src/main/java/com/domain/demo_backend/
└── domain/membership/
    ├── entity/
    │   ├── Membership.java
    │   └── UserMembership.java
    ├── repository/
    │   ├── MembershipRepository.java
    │   └── UserMembershipRepository.java
    ├── service/
    │   ├── MembershipService.java
    │   └── UserMembershipService.java
    ├── controller/
    │   ├── MembershipController.java
    │   └── UserMembershipController.java
    └── dto/
        ├── MembershipRequest.java
        ├── MembershipResponse.java
        ├── UserMembershipRequest.java
        └── UserMembershipResponse.java
```

### API 스펙

#### GET /api/v1/memberships
```json
// 200 OK
{
  "data": [
    { "id": 1, "name": "베이직", "canLearn": true, "canConverse": false, "canAnalyze": false, "durationDays": 30, "priceCents": 129000 }
  ]
}
```

#### POST /api/v1/memberships (어드민)
```json
// Request
{ "name": "프리미엄", "canLearn": true, "canConverse": true, "canAnalyze": true, "durationDays": 30, "priceCents": 219000 }
// 201 Created
{ "data": { Membership } }
// 422
{ "error": { "code": "VALIDATION_ERROR", "message": "..." } }
```

#### DELETE /api/v1/memberships/{id}
```
204 No Content
404 Not Found: { "error": { "code": "NOT_FOUND", "message": "Membership not found." } }
```

#### GET /api/v1/user-memberships/current
```json
// Authorization: Bearer {jwt}
// 200 OK (활성 멤버십 있음)
{ "data": { "id": 1, "userId": 1, "membership": {...}, "startedAt": "...", "expiresAt": "...", "status": "active", "grantedBy": "purchase" } }
// 200 OK (없음)
{ "data": null }
```

#### POST /api/v1/user-memberships
```json
// Request
{ "userId": 1, "membershipId": 1, "startedAt": "2026-03-11T00:00:00Z" }
// 201 Created
{ "data": { UserMembership } }
```

### 핵심 코드 스니펫

#### `UserMembership.java` (비즈니스 로직 포함)
```java
@Entity
@Table(name = "user_memberships")
public class UserMembership {
    // ...

    public boolean isActive() {
        return "active".equals(status) && expiresAt.isAfter(LocalDateTime.now());
    }

    // JPA Named Query 또는 Repository에서 처리
    // SELECT * FROM user_memberships WHERE user_id = ? AND status = 'active' AND expires_at > NOW()
}
```

#### `UserMembershipRepository.java`
```java
public interface UserMembershipRepository extends JpaRepository<UserMembership, Long> {

    @Query("SELECT um FROM UserMembership um WHERE um.userId = :userId " +
           "AND um.status = 'active' AND um.expiresAt > :now " +
           "ORDER BY um.createdAt DESC")
    Optional<UserMembership> findActiveByUserId(
        @Param("userId") Long userId,
        @Param("now") LocalDateTime now
    );
}
```

---

## Phase 3: AI 면접관 (Phase 2 이후)

Phase 2 AI 파이프라인에 interview 세션 관리 추가:
- `V13__create_interview_sessions.sql` (optional: sessionId → conversationHistory 저장)
- `InterviewSessionService.java` — 세션별 대화 이력 관리 (Redis 활용)
- `AiInterviewController.java` — 이미 Phase 2에서 생성

---

## Phase 4: 멤버십 권한 게이팅

AI 엔드포인트에 멤버십 권한 체크 추가:
```java
// AiChatController.java에 추가
@PreAuthorize("@membershipService.canConverse(authentication.name)")
@PostMapping(value = "/chat", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public SseEmitter chat(...) { ... }
```

---

## 아키텍처 결정 사항

### QueryDSL 도입 시기

| 단계 | 쿼리 방식 | 근거 |
|------|----------|------|
| Phase 1 (멤버십 기본 CRUD) | JPQL (`@Query`) | 조건이 단순하므로 충분 (`status = 'active' AND expires_at > NOW()`) |
| Phase 2 (AI 엔드포인트) | 쿼리 없음 or JPQL | AI 도메인 DB 조회 없음 |
| Phase 3+ (어드민 검색/필터) | **QueryDSL 도입** | 동적 조건 조합 (기간/상태/멤버십 타입 복합 필터) |

**결정**: Phase 1은 JPQL로 시작. 어드민 멤버십 관리 화면처럼 **동적 필터** 필요 시점에 QueryDSL 추가.

추가 의존성 (Phase 3 진입 시):
```groovy
implementation 'com.querydsl:querydsl-jpa:5.0.0:jakarta'
annotationProcessor 'com.querydsl:querydsl-apt:5.0.0:jakarta'
annotationProcessor 'jakarta.annotation:jakarta.annotation-api'
annotationProcessor 'jakarta.persistence:jakarta.persistence-api'
```

### WebSocket 기각 → SSE(SseEmitter) 채택

| 항목 | SseEmitter ✅ | WebSocket ❌ |
|------|-------------|------------|
| AI 스트리밍 방향 | 서버→클라이언트 단방향 (충분) | 양방향 (오버스펙) |
| JWT 인증 | HTTP Bearer 헤더 그대로 | 핸드쉐이크 시 URL 파라미터 or 별도 처리 필요 |
| 구현 복잡도 | 낮음 (SseEmitter + Executor) | 높음 (STOMP or 순수 WS 프로토콜 설계) |
| 기존 인프라 | Spring MVC 그대로 유지 | 별도 WS 핸들러 추가 필요 |

**근거**: 현재 AI 채팅 패턴은 `클라이언트 → POST → 서버`, `서버 → SSE 스트림 → 클라이언트`의 요청-스트림 구조로 **단방향이면 충분**. SDUI-server에 `WebSocketConfig` 이미 존재하므로 추후 실시간 기능 필요 시 추가 비용 낮음.

---

## application.yml 추가 설정

```yaml
# SDUI-server/src/main/resources/application.yml 또는 application-secret.yml
openai:
  api-key: ${OPENAI_API_KEY}
  model: gpt-4o
  whisper-model: whisper-1
  timeout-seconds: 30

spring:
  async:
    executor:
      core-pool-size: 10
      max-pool-size: 50
```

---

## TODO (구현 체크리스트)

### Phase 2 (AI 파이프라인) ✅ 완료
- [x] `AsyncConfig.java` — ThreadPoolTaskExecutor 설정
- [x] `OpenAiClient.java` — Java HttpClient 기반 STT/Chat 호출
- [x] DTO: SttResponse, ChatRequest, ChatMessage, InterviewStartRequest, InterviewAnswerRequest
- [x] `SttService.java` — Whisper 호출
- [x] `ChatService.java` — GPT-4o SSE 스트리밍
- [x] `InterviewService.java` — 프롬프트 엔지니어링 + SSE
- [x] `AiSttController.java` — POST /api/ai/stt
- [x] `AiChatController.java` — POST /api/ai/chat (SSE)
- [x] `AiInterviewController.java` — POST /api/ai/interview/start, /answer (SSE)
- [ ] `application.yml`에 `openai.api-key` 설정 ← **확인 필요**
- [ ] JUnit 테스트: AiSttControllerTest, AiChatControllerTest (MockRestServiceServer)

### Phase 1 (멤버십 도메인) ❌ 미구현
> ⚠️ 실제 Flyway 최신 버전 V25 → **V28, V29 사용** (plan.md의 V22/V23은 이미 사용됨)
- [ ] `V28__create_memberships.sql` (기존 V22 파일은 impl/phase1에 준비됨, 번호 수정 필요)
- [ ] `V29__create_user_memberships.sql`
- [ ] `Membership.java` (JPA Entity)
- [ ] `UserMembership.java` (JPA Entity + isActive() 메서드)
- [ ] `MembershipRepository.java`
- [ ] `UserMembershipRepository.java` (findActiveByUserId JPQL)
- [ ] `MembershipService.java` (CRUD)
- [ ] `UserMembershipService.java` (create, cancel, findCurrent)
- [ ] `MembershipController.java` — GET/POST/DELETE /api/v1/memberships
- [ ] `UserMembershipController.java` — GET current, POST, DELETE /api/v1/user-memberships
- [ ] DTO 클래스 (Request/Response)
- [ ] Flyway migrate: `./gradlew bootRun` 시 자동 실행
- [ ] JUnit 테스트: MembershipControllerTest, UserMembershipControllerTest

### Phase 3 (AI 면접관)
- [ ] `V13__create_interview_sessions.sql` (선택)
- [ ] Redis 기반 InterviewSession 캐시 (선택)

### Phase 4 (멤버십 게이팅)
- [ ] `MembershipAuthorizationService.java` — canConverse(), canLearn(), canAnalyze()
- [ ] AI 컨트롤러에 권한 체크 추가
