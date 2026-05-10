# Architect Plan: SDUI 기반 AI 튜터 + AI 면접관 통합 구현 계획

> 작성일: 2026-03-11
> 최종 수정: 2026-03-13
> 상태: **Phase 2 백엔드 구현 완료 / 프론트엔드 연결 진행 중**
> 기반: 링글 과제 아이디어 → SDUI 아키텍처로 전환

---

## 핵심 설계 결정

| 항목 | 결정 | 이유 |
|------|------|------|
| 백엔드 | Spring Boot (SDUI-server 확장) | Rails 신규 프로젝트 불필요, 기존 JWT 인증 재사용 |
| 프론트엔드 | metadata-project (SDUI 엔진 + 커스텀 컴포넌트) | 별도 프로젝트 불필요 |
| 인증 | 기존 JWT (X-User-Id 헤더 불필요) | SDUI-server에 이미 구현됨 |
| DB | 기존 PostgreSQL + Flyway 마이그레이션 | Docker sdui-db 컨테이너 그대로 사용 |
| AI 응답 표시 | **채팅창 텍스트 스트리밍** (TTS 음성 재생 없음) | 직접 사용 목적, 가독성 우선 |
| 언어 모드 | 영어 대화 + 한국어 대화 (2개) | 학습 + 실생활 모두 커버 |
| 어드민 UI | SDUI 메타데이터로 구성 | SDUI 확장성 어필 |
| STT/TTS | pronounce-api FastAPI → Spring Boot OpenAI 통합 | OpenAI Whisper+GPT-4o 단일 API |
| 구현 순서 | Phase 2 (AI 파이프라인) 먼저 → Phase 1 (멤버십) | AI 파이프라인 검증 우선 |
| 차별화 기능 | AI 면접관 (이력서 → 질문 생성 → 음성 답변 → 텍스트 표시) | 직접 사용 목적 |

---

## 1. 프로젝트 폴더 구조 (SDUI 기반)

```
SDUI/
├── SDUI-server/
│   └── src/main/java/com/domain/demo_backend/
│       └── domain/
│           ├── membership/              # 신규 추가
│           │   ├── MembershipController.java
│           │   ├── MembershipService.java
│           │   ├── UserMembershipController.java
│           │   ├── UserMembershipService.java
│           │   ├── MembershipRepository.java
│           │   ├── UserMembershipRepository.java
│           │   ├── Membership.java      (Entity)
│           │   └── UserMembership.java  (Entity)
│           └── ai/                      # 신규 추가
│               ├── SttController.java
│               ├── ChatController.java  (SSE Streaming)
│               ├── InterviewController.java
│               ├── SttService.java      (OpenAI Whisper)
│               ├── ChatService.java     (OpenAI GPT-4o)
│               └── InterviewService.java (이력서 분석 + 질문 생성)
│
├── metadata-project/
│   └── components/
│       ├── fields/                      # 신규 커스텀 컴포넌트
│       │   ├── AIChatComponent.tsx      # 영어/한국어 AI 대화
│       │   └── AIInterviewComponent.tsx # AI 면접관
│       ├── constants/
│       │   └── componentMap.tsx         # AI_CHAT, AI_INTERVIEW 등록
│       └── DynamicEngine/hook/
│           └── useBusinessActions.tsx   # AI 관련 액션 핸들러 추가
│
├── pronounce-api/                       # 유지 (발음 채점 기능)
│   └── app/main.py                     # 역할 축소: 발음 점수 API만 유지
│
└── SDUI-server/src/main/resources/
    └── db/migration/
        ├── V11__add_memberships.sql
        ├── V12__add_user_memberships.sql
        └── V13__add_interview_sessions.sql (optional)
```

---

## 2. DB 스키마

### 2.1 ERD

```
users (기존) ──< user_memberships >── memberships
                       │
                       └──< interview_sessions (optional)
```

### 2.2 신규 테이블 DDL

```sql
-- V11: 멤버십 종류 정의
CREATE TABLE memberships (
  id            BIGSERIAL PRIMARY KEY,
  name          VARCHAR(100) NOT NULL UNIQUE,
  can_learn     BOOLEAN NOT NULL DEFAULT false,
  can_converse  BOOLEAN NOT NULL DEFAULT false,
  can_analyze   BOOLEAN NOT NULL DEFAULT false,
  duration_days INTEGER NOT NULL,
  price_cents   INTEGER NOT NULL DEFAULT 0,
  description   TEXT,
  label_text    VARCHAR(255) NOT NULL DEFAULT '',  -- SDUI NOT NULL 제약 준수
  created_at    TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

-- V12: 유저 멤버십 인스턴스
CREATE TABLE user_memberships (
  id             BIGSERIAL PRIMARY KEY,
  user_id        BIGINT NOT NULL REFERENCES users(id),
  membership_id  BIGINT NOT NULL REFERENCES memberships(id),
  started_at     TIMESTAMP NOT NULL,
  expires_at     TIMESTAMP NOT NULL,
  status         VARCHAR(20) NOT NULL DEFAULT 'active',
  granted_by     VARCHAR(20) NOT NULL DEFAULT 'purchase',
  created_at     TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at     TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_user_memberships_user_id ON user_memberships(user_id);
CREATE INDEX idx_user_memberships_user_status ON user_memberships(user_id, status);
CREATE INDEX idx_user_memberships_expires_at ON user_memberships(expires_at);

-- V13 (optional): AI 면접 세션
CREATE TABLE interview_sessions (
  id                 BIGSERIAL PRIMARY KEY,
  user_id            BIGINT NOT NULL REFERENCES users(id),
  resume_content     TEXT NOT NULL,
  generated_questions JSONB NOT NULL DEFAULT '[]',
  messages           JSONB NOT NULL DEFAULT '[]',
  language           VARCHAR(10) NOT NULL DEFAULT 'ko',
  created_at         TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at         TIMESTAMP NOT NULL DEFAULT NOW()
);
```

### 2.3 시드 데이터 (Flyway 내 INSERT)

```sql
-- V11 하단에 포함
INSERT INTO memberships (name, can_learn, can_converse, can_analyze, duration_days, price_cents, description, label_text) VALUES
  ('베이직', true, false, false, 30, 129000, 'AI 학습 기능만 이용 가능한 기본 멤버십', '베이직'),
  ('프리미엄', true, true, true, 60, 219000, 'AI 학습 + 대화 + 분석 모두 이용 가능', '프리미엄');
```

---

## 3. API 계약

### 3.1 공통 규칙

- Base URL: `/api` (기존 SDUI 프록시 그대로)
- 인증: **기존 JWT Bearer 토큰** (X-User-Id 헤더 불필요)
- 성공: `{ "data": {...} }`
- 실패: `{ "error": { "code": "...", "message": "..." } }`

### 3.2 멤버십 API

```
GET  /api/memberships                    → 전체 멤버십 종류 목록
POST /api/memberships                    → 멤버십 종류 생성 (어드민)
DELETE /api/memberships/{id}             → 멤버십 종류 삭제 (어드민)

GET  /api/user-memberships/current       → 현재 유저 활성 멤버십 조회
POST /api/user-memberships               → 어드민 강제 부여
DELETE /api/user-memberships/{id}        → 멤버십 취소
```

### 3.3 AI 파이프라인 API

```
# STT: 오디오 → 텍스트
POST /api/ai/stt
Content-Type: multipart/form-data
Body: { "audio": <File (webm/wav)> }
Response: { "data": { "text": "Hello, I'm..." } }
필요 권한: can_converse = true

# Chat: 텍스트 → AI 응답 (SSE Streaming)
POST /api/ai/chat
Body: {
  "message": "Tell me about yourself",
  "language": "en" | "ko",
  "conversationHistory": [{"role": "user"|"assistant", "content": "..."}]
}
Response: text/event-stream
  data: {"chunk": "Hello"}
  data: {"chunk": "!"}
  data: {"done": true, "fullText": "Hello!"}

# AI 면접관: 이력서 분석 + 질문 생성
POST /api/ai/interview/start
Body: { "resumeContent": "...", "language": "ko" | "en" }
Response: {
  "data": {
    "sessionId": "uuid",
    "firstQuestion": "자기소개를 해주세요.",
    "questions": ["q1", "q2", "q3"]
  }
}

# AI 면접관: 답변 → 꼬리 질문 생성 (SSE Streaming)
POST /api/ai/interview/answer
Body: {
  "sessionId": "uuid",
  "userAnswer": "저는 3년차 백엔드 개발자입니다...",
  "currentQuestion": "자기소개를 해주세요.",
  "conversationHistory": [...]
}
Response: text/event-stream (꼬리 질문 스트리밍)
```

---

## 4. AI 파이프라인 설계

### 4.1 AI 대화 화면 UX 흐름 (텍스트 표시 방식)

```
[화면 진입]
    ↓ JWT로 멤버십 권한 체크
[AI 첫 메시지 텍스트로 채팅창에 스트리밍 표시]
    ↓
[마이크 버튼 클릭 → MediaRecorder 시작 + Waveform 시각화]
    ↓
[답변완료 버튼 → 오디오 Blob]
    ↓ POST /api/ai/stt (OpenAI Whisper)
[유저 텍스트 채팅창에 표시]
    ↓ POST /api/ai/chat (SSE Streaming)
[AI 응답 텍스트 채팅창에 타이핑 애니메이션으로 스트리밍 표시]
    ↓ (TTS 음성 재생 없음 — 텍스트만)
[다음 차례 반복]
```

### 4.2 AI 면접관 UX 흐름

```
[AI_INTERVIEW_PAGE 진입]
    ↓
[이력서 텍스트 입력 또는 PDF 업로드 영역]
    ↓ POST /api/ai/interview/start
[GPT-4o가 이력서 분석 → 면접 질문 3~5개 생성]
    ↓
[첫 질문 채팅창에 표시 (텍스트)]
    ↓
[마이크 버튼 → 음성 답변 → STT 변환]
    ↓
[유저 답변 채팅창에 표시]
    ↓ POST /api/ai/interview/answer (SSE Streaming)
[꼬리 질문 채팅창에 타이핑 애니메이션으로 표시]
    ↓
[반복 → 세션 종료 → 피드백 리포트 (optional)]
```

### 4.3 Spring Boot SSE 구현 방식

```java
// ChatController.java
@PostMapping(value = "/api/ai/chat", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public SseEmitter chat(@RequestBody ChatRequest request) {
    SseEmitter emitter = new SseEmitter(30_000L); // 30초 타임아웃
    chatService.streamResponse(request, emitter);
    return emitter;
}
```

### 4.4 지연 시간 최적화

| 단계 | 예상 지연 | 최적화 |
|------|-----------|--------|
| STT (Whisper) | 1~3초 | 오디오 크기 최소화 |
| LLM (GPT-4o) | 2~5초 | **SSE Streaming** → 첫 토큰 즉시 표시 |
| TTS | 없음 | 텍스트 표시로 대체 → 이 단계 제거 |
| **합계** | **3~8초** | 스트리밍으로 체감 2초 이내 |

---

## 5. SDUI 화면 구성 (ui_metadata)

| screen_id | 구성 방식 | 설명 |
|-----------|-----------|------|
| `MAIN_PAGE` (기존) | 수정 | 멤버십 현황 카드 추가 |
| `MEMBERSHIP_SHOP_PAGE` | SDUI 메타데이터 신규 | 멤버십 목록 + 구매 버튼 |
| `ADMIN_MEMBERSHIP_PAGE` | SDUI 메타데이터 신규 | 멤버십 종류 관리 (어드민) |
| `AI_ENGLISH_CHAT_PAGE` | 커스텀 컴포넌트 | 영어 AI 대화 |
| `AI_KOREAN_CHAT_PAGE` | 커스텀 컴포넌트 | 한국어 AI 대화 |
| `AI_INTERVIEW_PAGE` | 커스텀 컴포넌트 | 이력서 업로드 → AI 면접관 |

### componentMap 추가 항목

```typescript
// components/constants/componentMap.tsx 추가
AI_CHAT: AIChatComponent,         // 영어/한국어 대화 (language prop으로 분기)
AI_INTERVIEW: AIInterviewComponent, // AI 면접관
```

---

## 6. FastAPI 역할 재정의

```
pronounce-api (FastAPI, 포트 8001)
  유지 기능:
    - POST /pronunciation-score  → 발음 점수 채점 (Levenshtein)
    - 영어 대화 모드에서 사용자가 원할 때 발음 피드백 제공

  제거 기능:
    - /tts_blob, /tts_only → Spring Boot에서 OpenAI TTS로 대체 (단, AI 응답은 텍스트만)
    - /translate* → 불필요
    - /analyze → 불필요

Spring Boot (포트 8080) 담당:
  - OpenAI Whisper STT
  - OpenAI GPT-4o Chat (SSE Streaming)
  - 이력서 분석 + 면접 질문 생성
```

---

## 7. 구현 순서 (Phase 2 우선)

### Phase 2: AI 파이프라인 검증 (최우선)

```
[x] SDUI-server: RestClient/HttpClient 방식으로 OpenAI API 직접 호출 구현
[x] SttService.java (OpenAI Whisper 연동)
[x] ChatService.java (GPT-4o SSE Streaming)
[x] SttController.java + ChatController.java + InterviewController.java
[x] metadata-project: AIChatComponent.tsx (마이크 + Waveform + 채팅창)
[ ] componentMap에 AI_CHAT 등록 ← **다음 단계**
[ ] ui_metadata: AI_ENGLISH_CHAT_PAGE, AI_KOREAN_CHAT_PAGE 삽입 (V26부터 적용)
[ ] 동작 검증 (영어 대화 → 한국어 대화)
```

### Phase 1: 멤버십 도메인 (Phase 2 검증 후)

```
[ ] Flyway V11: memberships 테이블 + 시드
[ ] Flyway V12: user_memberships 테이블
[ ] Membership.java, UserMembership.java (Entity)
[ ] MembershipController, UserMembershipController
[ ] 멤버십 권한 체크 인터셉터 (AI 엔드포인트 보호)
[ ] ui_metadata: MEMBERSHIP_SHOP_PAGE, ADMIN_MEMBERSHIP_PAGE
```

### Phase 3: AI 면접관 (차별화 기능)

```
[ ] InterviewService.java (이력서 파싱 + 질문 생성)
[ ] InterviewController.java (SSE Streaming)
[ ] Flyway V13: interview_sessions 테이블 (optional)
[ ] AIInterviewComponent.tsx (이력서 입력 + 대화)
[ ] componentMap에 AI_INTERVIEW 등록
[ ] ui_metadata: AI_INTERVIEW_PAGE 삽입
```

### Phase 4: 통합 + 테스트

```
[ ] JUnit 5: 서비스 단위 테스트 (OpenAI mock)
[ ] Jest + MSW: 프론트엔드 컴포넌트 테스트
[ ] Playwright E2E: 대화 플로우 테스트
[ ] README.md 업데이트
```

---

## 8. 환경변수 추가 목록

```bash
# SDUI-server application.yml 또는 .env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_WHISPER_MODEL=whisper-1
```

---

## 9. 트레이드오프

| 결정 | 선택 | 이유 |
|------|------|------|
| AI 응답 표시 | 텍스트 채팅창 | 직접 사용 목적, TTS 레이턴시 제거 |
| 구현 순서 | Phase 2 우선 | AI 파이프라인 조기 검증으로 리스크 감소 |
| 언어 분리 방식 | language 파라미터 | 컴포넌트 재사용, API 단일화 |
| 이력서 저장 | DB 저장 optional | 우선 세션 메모리로 검증 후 결정 |
| pronounce-api | 발음 채점만 유지 | STT/TTS는 OpenAI로 통합, 발음 피드백은 차별화 포인트 |
