# Backend Engineer Research: SDUI-server AI + 멤버십 도메인 분석

> 작성일: 2026-03-11
> 상태: Phase 2 구현 대기

---

## 1. 기존 SDUI-server 구조 분석

### 1.1 현재 패키지 구조

```
SDUI-server/src/main/java/com/domain/demo_backend/
├── domain/
│   ├── ui/         # UiController, UiService, UiMetadata (기존)
│   ├── query/      # QueryMasterService, Redis 캐시 (기존)
│   └── user/       # AuthController, KakaoController (기존)
└── global/
    ├── security/   # JwtAuthenticationFilter, JwtUtil, SecurityConfig (기존)
    ├── config/     # RedisConfig, WebConfig (CORS), WebSocketConfig (기존)
    └── exception/  # GlobalExceptionHandler (있다면)
```

### 1.2 신규 추가 패키지

```
SDUI-server/src/main/java/com/domain/demo_backend/
├── domain/
│   ├── ai/         # Phase 2: STT, Chat, Interview (신규)
│   └── membership/ # Phase 1: Membership, UserMembership (신규)
└── global/
    └── config/
        └── AsyncConfig.java  # Phase 2: SseEmitter용 ThreadPool (신규)
```

---

## 2. 기존 코드 재사용 분석

### 2.1 재사용 가능한 기존 코드

| 항목 | 파일/패키지 | 재사용 방법 |
|------|-----------|------------|
| JWT 인증 | `global/security/JwtAuthenticationFilter` | AI 엔드포인트 동일 적용 |
| Redis | `global/config/RedisConfig` | 면접 세션 캐시에 활용 가능 |
| CORS 설정 | `global/config/WebConfig` | 기존 설정 그대로 (`/api/ai/**` 포함됨) |
| GlobalExceptionHandler | `global/exception/` | AI 오류도 동일 형식으로 처리 |
| Flyway | `resources/db/migration/` | V11, V12 추가 (기존 V1~V10 이후) |

### 2.2 인증 방식

- **기존 SDUI**: JWT Bearer Token (`Authorization` 헤더)
- **AI 엔드포인트**: 동일한 JWT 인증 유지 (X-User-Id 헤더 불필요)
- `@AuthenticationPrincipal UserDetails user`로 현재 사용자 추출

---

## 3. 기술 스택 분석

### 3.1 Spring Boot 버전 확인 필요 항목

| 기능 | 필요 버전 | 비고 |
|------|---------|------|
| `RestClient` | Spring Boot 3.2+ (Spring 6.1+) | OpenAI API 호출에 사용 |
| `SseEmitter` | Spring 4.2+ | SSE 스트리밍 |
| `@EnableAsync` | Spring 3+ | AsyncConfig |

SDUI-server가 Spring Boot 3.x 기반이므로 `RestClient` 사용 가능.

### 3.2 OpenAI API 연동 방식

```
옵션 비교:
1. openai-java SDK (theokanning) → 구버전, 유지보수 불안정 ❌
2. openai-java (sashirestela) → 최신이나 외부 의존성 ❌
3. Spring RestClient (내장) → 라이브러리 의존 없음, 버전 이슈 없음 ✅
```

**결정**: `RestClient`로 OpenAI HTTP API 직접 호출.
- STT: `POST https://api.openai.com/v1/audio/transcriptions` (multipart/form-data)
- Chat: `POST https://api.openai.com/v1/chat/completions` (stream: true)

### 3.3 SSE Streaming 구현

```java
// Spring MVC SseEmitter 방식 (WebFlux 전환 불필요)
SseEmitter emitter = new SseEmitter(30_000L);
executor.execute(() -> service.stream(request, emitter));
return emitter;
// emitter.send() → 클라이언트에 청크 전송
// emitter.complete() → 스트림 종료
// emitter.completeWithError(e) → 오류 종료
```

**주의**: SseEmitter는 별도 스레드에서 실행해야 함 → `AsyncConfig` 필수.

### 3.4 OpenAI Streaming 응답 파싱

OpenAI Chat Completions streaming 응답 형식:
```
data: {"id":"...","choices":[{"delta":{"content":"Hello"},...}]}
data: {"id":"...","choices":[{"delta":{"content":" there"},...}]}
data: [DONE]
```

RestClient로 InputStream을 직접 읽어 라인 단위 파싱:
```java
restClient.post()
    .uri("/chat/completions")
    .body(requestBody)
    .exchange((request, response) -> {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(response.getBody()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("data: ") && !line.equals("data: [DONE]")) {
                    String json = line.substring(6);
                    String chunk = extractContent(json); // choices[0].delta.content
                    onChunk.accept(chunk);
                }
            }
            onComplete.run();
        }
        return null;
    });
```

---

## 4. 의존성 추가 필요 항목

### build.gradle 추가
```groovy
// PDF 파싱 (이력서 PDF 업로드 지원)
implementation 'org.apache.pdfbox:pdfbox:3.0.2'

// Multipart 파일 처리는 Spring Boot 기본 지원 (별도 의존성 불필요)
// RestClient도 Spring Boot 내장 (별도 의존성 불필요)
```

### application.yml 추가
```yaml
openai:
  api-key: ${OPENAI_API_KEY}  # 환경변수 또는 application-secret.yml
  model: gpt-4o
  whisper-model: whisper-1

spring:
  servlet:
    multipart:
      max-file-size: 25MB       # Whisper API 최대 25MB
      max-request-size: 26MB
```

---

## 5. DB 스키마 설계

### 5.1 memberships 테이블 (V11)

| 컬럼 | 타입 | 제약 |
|------|------|------|
| id | BIGSERIAL | PK |
| name | VARCHAR(100) | NOT NULL, UNIQUE |
| can_learn | BOOLEAN | NOT NULL, DEFAULT FALSE |
| can_converse | BOOLEAN | NOT NULL, DEFAULT FALSE |
| can_analyze | BOOLEAN | NOT NULL, DEFAULT FALSE |
| duration_days | INTEGER | NOT NULL |
| price_cents | INTEGER | NOT NULL |
| description | TEXT | nullable |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() |
| updated_at | TIMESTAMP | NOT NULL, DEFAULT NOW() |

### 5.2 user_memberships 테이블 (V12)

| 컬럼 | 타입 | 제약 |
|------|------|------|
| id | BIGSERIAL | PK |
| user_id | BIGINT | NOT NULL, indexed |
| membership_id | BIGINT | NOT NULL, FK → memberships.id |
| started_at | TIMESTAMP | NOT NULL |
| expires_at | TIMESTAMP | NOT NULL |
| status | VARCHAR(20) | NOT NULL, DEFAULT 'active' |
| granted_by | VARCHAR(20) | NOT NULL, DEFAULT 'purchase' |
| created_at/updated_at | TIMESTAMP | NOT NULL |

인덱스:
- `(user_id)` — 사용자별 조회
- `(user_id, status)` — 활성 멤버십 조회
- `(expires_at)` — 만료 배치 처리 (미래 확장)

---

## 6. 멤버십 비즈니스 로직

### 6.1 활성 멤버십 조회
```java
// 실시간 만료 체크 (Background Job 미사용)
// expires_at > NOW() AND status = 'active'
Optional<UserMembership> findActiveByUserId(Long userId, LocalDateTime now);
```

### 6.2 권한 체크 (Phase 4에서 AI 엔드포인트에 추가)
```java
public boolean canConverse(String username) {
    Long userId = userRepository.findByEmail(username).getId();
    return userMembershipRepository
        .findActiveByUserId(userId, LocalDateTime.now())
        .map(um -> um.getMembership().isCanConverse())
        .orElse(false);
}
```

---

## 7. 테스트 전략

### 7.1 JUnit 5 테스트 구조
```
SDUI-server/src/test/java/com/domain/demo_backend/
├── domain/ai/
│   ├── AiSttControllerTest.java    (MockMvc + MockRestServiceServer)
│   └── AiChatControllerTest.java   (SseEmitter 스트리밍 검증)
└── domain/membership/
    ├── MembershipControllerTest.java
    └── UserMembershipControllerTest.java
```

### 7.2 OpenAI API 모킹
```java
// MockRestServiceServer로 RestClient 모킹
MockRestServiceServer mockServer = MockRestServiceServer.createServer(restTemplate);
mockServer.expect(requestTo("/audio/transcriptions"))
    .andRespond(withSuccess("{\"text\":\"Hello\"}", MediaType.APPLICATION_JSON));
```

---

## 8. pronounce-api 역할 재정의

| 기능 | 기존 | 새 계획 |
|------|------|---------|
| STT (한국어) | Flask /upload-audio (Google STT) | ❌ OpenAI Whisper로 대체 |
| TTS (일본어) | FastAPI /tts_blob (Google Cloud TTS) | ❌ AI 응답은 텍스트만 표시 |
| 발음 채점 | speechRecognition.py (Levenshtein) | ✅ 유지 (영어 대화 모드) |
| 번역 | /translate* | ❌ 불필요 |
| 형태소 분석 | /analyze | ❌ 불필요 |

**주의**: `pronounce-api/app/main.py:28`에 하드코딩된 Google Cloud credentials 경로 수정 필요 (발음 채점 모드 사용 시).

---

## 9. 리스크 및 주의사항

| 항목 | 심각도 | 대응 |
|------|--------|------|
| OpenAI API 레이턴시 | 높음 | SSE Streaming으로 체감 지연 감소 |
| SseEmitter 타임아웃 | 중간 | 30초 설정 + 클라이언트 retry |
| Flyway label_text NOT NULL | 중간 | V11/V12는 ai/membership만 → 해당 없음 |
| RestClient InputStream 파싱 | 중간 | OpenAI SSE 형식 정확히 파싱 필요 |
| OPENAI_API_KEY 보안 | 높음 | application-secret.yml (git 미추적) |
| Whisper 25MB 파일 제한 | 낮음 | multipart 설정 + 클라이언트 안내 |

---

## 10. 분석 히스토리

| 날짜 | 분석 내용 | 결론 |
|------|-----------|------|
| 2026-03-11 | 기존 Rails 계획 → Spring Boot SDUI 전환 분석 | SDUI-server domain 추가 방식으로 통합 가능 |
| 2026-03-11 | OpenAI SDK vs RestClient 비교 | RestClient 채택 (의존성 최소화) |
| 2026-03-11 | SseEmitter vs WebFlux 비교 | SseEmitter 채택 (기존 MVC 유지) |
| 2026-03-11 | pronounce-api 역할 재정의 | 발음 채점만 유지, STT/TTS → OpenAI |
