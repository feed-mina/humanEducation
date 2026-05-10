# Phase 2 패치 가이드

Phase 2 AI 파이프라인 구현을 위해 **기존 파일**에 적용해야 할 변경사항입니다.
새 파일들(`domain/ai/` 패키지 전체)은 `.ai2/backend_engineer/impl/phase2/` 에 준비되어 있습니다.

---

## 1. SecurityConfig.java 수정

**파일**: `SDUI-server/src/main/java/com/domain/demo_backend/global/security/SecurityConfig.java`

`anyRequest().denyAll()` 이전에 `/api/ai/**` 엔드포인트를 추가합니다.

```java
// 기존 (anyRequest 바로 위에 위치)
.requestMatchers(HttpMethod.GET,  "/api/ui/**").permitAll()
.requestMatchers(HttpMethod.POST, "/api/auth/**").permitAll()
// ... 기타 기존 규칙 ...

// ✅ 추가: AI API (인증 필요)
.requestMatchers(HttpMethod.POST, "/api/ai/**").authenticated()

.anyRequest().denyAll()
```

> **주의**: `denyAll()` 바로 위에 삽입해야 합니다. 순서가 중요합니다.

---

## 2. application.yml 수정

**파일**: `SDUI-server/src/main/resources/application.yml` (또는 `application-secret.yml`)

```yaml
openai:
  api-key: ${OPENAI_API_KEY}       # 환경변수로 주입 (절대 하드코딩 금지)
  model: gpt-4o-mini                # 기본 채팅 모델
  whisper-model: whisper-1          # STT 모델
```

> **환경변수 설정**:
> - 로컬: `.env` 또는 IDE Run Configuration에 `OPENAI_API_KEY=sk-...` 추가
> - AWS: EC2 환경변수 또는 Secrets Manager에서 주입
> - **절대 application.yml에 실제 키 값을 커밋하지 마세요**

---

## 3. 새 파일 배치 위치 요약

```
SDUI-server/src/main/java/com/domain/demo_backend/
├── domain/ai/
│   ├── client/
│   │   └── OpenAiClient.java          ← impl/phase2/OpenAiClient.java
│   ├── controller/
│   │   ├── AiSttController.java       ← impl/phase2/controller/AiSttController.java
│   │   ├── AiChatController.java      ← impl/phase2/controller/AiChatController.java
│   │   └── AiInterviewController.java ← impl/phase2/controller/AiInterviewController.java
│   ├── dto/
│   │   ├── ChatMessage.java           ← impl/phase2/dto/ChatMessage.java
│   │   ├── ChatRequest.java           ← impl/phase2/dto/ChatRequest.java
│   │   ├── SttResponse.java           ← impl/phase2/dto/SttResponse.java
│   │   ├── InterviewStartRequest.java ← impl/phase2/dto/InterviewStartRequest.java
│   │   └── InterviewAnswerRequest.java← impl/phase2/dto/InterviewAnswerRequest.java
│   └── service/
│       ├── SttService.java            ← impl/phase2/service/SttService.java
│       ├── ChatService.java           ← impl/phase2/service/ChatService.java
│       └── InterviewService.java      ← impl/phase2/service/InterviewService.java
└── global/config/
    └── AsyncConfig.java               ← impl/phase2/AsyncConfig.java
```

---

## 4. API 엔드포인트 요약

| 메서드 | 경로 | 설명 | 응답 타입 |
|--------|------|------|-----------|
| `POST` | `/api/ai/stt` | 음성 → 텍스트 변환 | `ApiResponse<SttResponse>` |
| `POST` | `/api/ai/chat/stream` | 자유 대화 스트리밍 | `SseEmitter` (text/event-stream) |
| `POST` | `/api/ai/interview/start` | 이력서 기반 첫 면접 질문 | `SseEmitter` (text/event-stream) |
| `POST` | `/api/ai/interview/answer` | 답변 제출 → 후속 질문 | `SseEmitter` (text/event-stream) |

**SSE 스트림 포맷**:
```
data: {"chunk":"안녕하세요, "}
data: {"chunk":"자기소개를 "}
data: {"chunk":"해주세요."}
data: [DONE]
```

---

## 5. 의존성 확인

Phase 2는 **추가 의존성 없음**:
- `RestTemplate`: Spring Web에 포함 (기존 사용 중)
- `java.net.http.HttpClient`: Java 17 내장
- `ObjectMapper`: Jackson (기존 사용 중)
- `SseEmitter`: Spring Web MVC에 포함
