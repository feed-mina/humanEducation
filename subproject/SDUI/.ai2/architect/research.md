# Architect Research: SDUI 기반 AI 튜터 + AI 면접관 설계 분석

> 작성일: 2026-03-11
> 최종 수정: 2026-03-11
> 상태: 완료 → plan.md 작성 완료

---

## 1. 요구사항 분석

### 1.1 원천 아이디어 (링글 과제)

- 멤버십 기반 AI 영어 튜터 (학습/대화/분석 권한 조합)
- AI 대화: 마이크 녹음 → STT → LLM → TTS 파이프라인
- 어드민 UI: 멤버십 종류 CRUD

### 1.2 SDUI 프로젝트에 맞게 확장된 요구사항

| 기능 | 원안 (링글) | SDUI 확장 |
|------|------------|-----------|
| 백엔드 | Rails 7 | Spring Boot (기존 SDUI-server) |
| 인증 | X-User-Id 헤더 (임시) | 기존 JWT 인증 그대로 |
| 어드민 UI | 별도 React 페이지 | SDUI 메타데이터 (ui_metadata) |
| AI 응답 표시 | TTS 음성 재생 | **채팅창 텍스트 스트리밍** (사용자 요청) |
| 언어 | 영어 전용 | **영어 + 한국어** (2개 모드) |
| 신규 기능 | 없음 | **AI 면접관** (이력서 → 질문 생성 → 음성 답변) |

### 1.3 사용자 목적

- SDUI 아키텍처 확장성 어필
- AI 파이프라인 (STT→LLM→TTS) 실전 구현 역량
- **직접 사용**: 면접 준비 시 이력서 기반 AI 면접 연습

---

## 2. 기존 SDUI 코드베이스 분석

### 2.1 재사용 가능한 기존 코드

| 항목 | 파일/패키지 | 재사용 방법 |
|------|------------|------------|
| JWT 인증 | `global/security` | AI 엔드포인트에 동일하게 적용 |
| Redis 캐시 | `global/config/RedisConfig` | 면접 세션 캐시에 활용 가능 |
| SDUI 엔진 | `components/DynamicEngine` | 커스텀 컴포넌트 componentMap 등록 |
| usePageHook | `hook/usePageHook.tsx` | AI 관련 action_type 추가 |
| Flyway 마이그레이션 | `resources/db/migration/` | V11~V13 추가 |
| CORS 설정 | `global/config/WebConfig` | 기존 설정 그대로 |
| 액션 라우팅 | `useBusinessActions.tsx` | AI_CHAT_START, AI_INTERVIEW_START 추가 |

### 2.2 pronounce-api 재사용 분석

| 기능 | 파일 | 재사용 여부 | 비고 |
|------|------|------------|------|
| 발음 점수 채점 | `speechRecognition.py` (Levenshtein) | ✅ 유지 | 영어 대화 모드 발음 피드백 |
| TTS (일본어) | `app/main.py /tts_blob` | 언어 변경 가능하나 미사용 | OpenAI TTS로 대체 |
| 한국어 STT | `server.py /upload-audio` | 미사용 | OpenAI Whisper로 대체 |
| 번역 | `app/main.py /translate*` | ❌ 불필요 | |
| 형태소 분석 | `app/main.py /analyze` | ❌ 불필요 | |

**주의**: `app/main.py:28`에 하드코딩된 Google Cloud credentials 경로 수정 필요.

---

## 3. AI 파이프라인 기술 검토

### 3.1 OpenAI API 선택 근거

| API | 용도 | 비고 |
|-----|------|------|
| `whisper-1` | STT (음성→텍스트) | 영어/한국어 모두 지원 |
| `gpt-4o` | 대화 + 이력서 분석 + 질문 생성 | Streaming 지원 |
| OpenAI TTS | (사용 안 함) | AI 응답은 텍스트만 표시 |

### 3.2 Spring Boot SSE 구현 방식

```java
// SseEmitter 방식 (Spring MVC)
// - 기존 SDUI-server가 Spring MVC 사용 → WebFlux 전환 불필요
// - 비동기 처리: @Async + ExecutorService
@PostMapping(value = "/api/ai/chat", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public SseEmitter chat(@RequestBody ChatRequest req, @AuthenticationPrincipal UserDetails user) {
    SseEmitter emitter = new SseEmitter(30_000L);
    executor.execute(() -> chatService.stream(req, emitter));
    return emitter;
}
```

### 3.3 OpenAI Java SDK 옵션

| 옵션 | 장단점 |
|------|--------|
| `com.theokanning.openai-gpt3-java` | 구버전, 유지보수 불안정 |
| `io.github.sashirestela.openai-java` | 최신, streaming 지원 |
| **순수 HTTP (WebClient/RestClient)** | 라이브러리 의존 없음, 버전 이슈 없음, 권장 |

**결정**: Spring Boot 내장 `RestClient` (Spring 6.1+) 로 OpenAI HTTP API 직접 호출.
SDUI-server가 Spring Boot 3.x 기반이므로 `RestClient` 사용 가능.

### 3.4 SSE Streaming 소비 (프론트엔드)

```typescript
// EventSource는 GET 전용 → POST body 전송 불가
// fetch + ReadableStream 사용 (기존 metadata-project 방식과 동일)
const response = await fetch('/api/ai/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${jwt}` },
  body: JSON.stringify(payload),
});
const reader = response.body!.getReader();
```

---

## 4. SDUI 아키텍처 적용 방식

### 4.1 커스텀 컴포넌트 등록 패턴

```typescript
// 기존 패턴 (componentMap.tsx)
INPUT: InputComponent,
BUTTON: ButtonComponent,

// 신규 추가
AI_CHAT: AIChatComponent,        // 대화 컴포넌트
AI_INTERVIEW: AIInterviewComponent, // 면접 컴포넌트
```

### 4.2 ui_metadata 구조 (AI_ENGLISH_CHAT_PAGE 예시)

```sql
-- AI 채팅 페이지: 단일 커스텀 컴포넌트로 구성
INSERT INTO ui_metadata (screen_id, component_type, label_text, css_class, allowed_roles)
VALUES
  ('AI_ENGLISH_CHAT_PAGE', 'AI_CHAT', '', 'ai-chat-container', 'ROLE_USER'),
```

- `component_type = 'AI_CHAT'`: DynamicEngine이 `AIChatComponent` 렌더링
- `css_class`: 레이아웃 스타일
- `allowed_roles = 'ROLE_USER'`: 멤버십 추가 후 `can_converse` 체크로 교체

### 4.3 액션 핸들러 패턴 (usePageHook → useBusinessActions)

```typescript
// useBusinessActions.tsx 추가 예정
case 'AI_CHAT_SUBMIT':    // 답변완료 버튼 액션
case 'AI_INTERVIEW_START': // 면접 시작 액션
case 'AI_RESUME_UPLOAD':   // 이력서 업로드 액션
```

---

## 5. 리스크 목록

| 리스크 | 심각도 | 대응 방안 |
|--------|--------|-----------|
| OpenAI API 레이턴시 | 높음 | SSE Streaming으로 체감 지연 감소 |
| AudioContext 브라우저 제한 (최대 6개) | 중간 | useRef 단일 인스턴스 패턴 |
| Spring Boot SseEmitter 타임아웃 | 중간 | 30초 설정 + 클라이언트 retry |
| 마이크 권한 거부 | 낮음 | 에러 UI + 텍스트 입력 대체 모드 |
| Flyway 마이그레이션 label_text NOT NULL | 중간 | V11 DDL에 DEFAULT '' 명시 (기존 버그 경험) |
| 이력서 내용 길이 | 중간 | GPT-4o 컨텍스트 128k → 문제없음, 요약 옵션 고려 |

---

## 6. 의존성 추가 필요 항목

### SDUI-server (build.gradle)
```groovy
// OpenAI API HTTP 호출 (RestClient 내장이므로 별도 SDK 불필요)
// PDF 파싱 (이력서 PDF 업로드 지원 시)
implementation 'org.apache.pdfbox:pdfbox:3.0.2'
// 멀티파트 오디오 파일 처리는 Spring Boot 기본 지원
```

### metadata-project (package.json)
```json
// 오디오 Waveform 시각화
// Web Audio API는 브라우저 내장 → 별도 라이브러리 불필요
// wavesurfer.js 고려 가능 (선택)
```

---

## 7. 분석 히스토리

| 날짜 | 분석 내용 | 결론 |
|------|-----------|------|
| 2026-03-11 | 링글 과제 → SDUI 전환 가능성 분석 | 프론트 기술스택 일치, 백엔드 Rails → Spring Boot 전환 필요 |
| 2026-03-11 | pronounce-api 재사용 분석 | TTS/STT → OpenAI 통합, 발음 채점만 유지 |
| 2026-03-11 | 사용자 요구사항 구체화 | 텍스트 표시 방식, 영어+한국어, AI 면접관 추가, Phase 2 우선 |
