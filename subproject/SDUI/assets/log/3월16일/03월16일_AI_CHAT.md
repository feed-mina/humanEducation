// [메모] 
3월 16일 폴더에 캡쳐이미지 참고

1.  일본어 페이지가 Title 은 JAPAN으로 되어있는데 AI_ENGLISH 와 같음
 AI_INTERVIEW 도 마찬가지 .ai2 폴더에 적힌  AI_INTERVIEW 계획처럼 바꿔야 합니다.

 2. 프롬포트로 들어가는 하드코딩된 부분을 query_master 나 ui_metadata로 전환

 3. AI_JAPAN, AI_INTERVIEW 한 기능 마칠때마다 스프링부트 유닛테스트, Next 유닛테스트, E2E통합테스트 항상 같이 포함
 
 4. 현재 FastAPI 를 활용하는 부분이 어떻게 되는지 
 AI_CHAT, AI_JAPAN 사람이 말하는 발음 정확도를 평가하는 시스템을 만들고 싶습니다.

5. 현재 로컬에서 테스트한 부분이 migration db sql을 사용했는데 aws적용할때 migration 버전을 정리해서 적용해야 합니다. (아직 배포하기 전이라서 AI_CHAT, AI_Interview 관련하여 migration을 하기 전입니다. 현재 파일은 feature_ 브랜ㄹ치로 분리, 배포에 영향가는 브랜치는 lab_claude나 main입니다.) 
하지만 우선 로컬에서 테스트(로컬db, 로컬 redis 사용) > 로컬 도커 테스트 > AWS 적용으로 넘어가고 싶습니다. 필요하다면 현재 로컬 db의 SDUI_TB 데이터베이스를 삭제하고 시작해도 됩니다.

6. API 호출 로직 분리 (Dependency Inversion): 훅 내부에 axios와 특정 API 엔드포인트가 하드코딩되어 있어 다른 프로젝트에서 재사용하거나 테스트하기 어렵습니다. API 통신을 전담하는 별도의 서비스 모듈로 분리합니다.

7. 복잡한 콜백 함수 분리: useAudioRecorder의 onAudioReady 콜백 내부에 STT 요청, 번역, 상태 업데이트 등 너무 많은 로직이 집중되어 있습니다. 이를 독립적인 함수로 분리하여 가독성과 응집도를 높입니다.

8. 응답 데이터 파싱 로직 개선: 스트리밍 응답에서 JSON을 추출하는 로직이 lastIndexOf에 의존하고 있어 불안정합니다. 정규식과 타입 가드를 사용하여 더 안정적으로 개선합니다.

9. 역할과 책임 분리 (SRP): 목표 달성 체크와 같은 부가적인 비즈니스 로직을 별도의 작은 훅으로 분리하여 useAIChatLogic이 핵심 기능에만 집중하도록 만듭니다.

---

## 실행 계획 (2026-03-16)

> 작성 기준: 위 9개 항목을 의존성·영향 범위 순으로 재정렬
> 규칙: 각 Phase 완료 후 **Spring Boot 유닛테스트 + Next 유닛테스트 + E2E 통합테스트** 필수 실행 (항목 3)

---

### Phase 0 — 현황 파악 (사전 조사)

**목표**: 코드 수정 전, 실제 문제 지점을 확인한다.

| 조사 항목 | 확인 방법 | 관련 항목 |
|-----------|-----------|-----------|
| AI_JAPANESE_CHAT_PAGE / AI_INTERVIEW_PAGE 가 AI_ENGLISH와 동일하게 보이는 원인 | 브라우저 DevTools → Elements 탭에서 적용된 CSS 클래스 확인 (`.ai-japanese-theme`, `.ai-interview-theme` 존재 여부) | 1 |
| `AIInterviewComponent.tsx`가 렌더링되고 있는지 확인 | `componentMap.tsx`에서 `AI_INTERVIEW` 키 등록 여부 확인 | 1 |
| FastAPI (`pronounce-api`) 현재 동작 상태 | `pronounce-api/app/main.py` 코드 읽기 + 포트 8001 응답 테스트 | 4 |
| 현재 하드코딩된 시스템 프롬프트 위치 목록화 | `AIInterviewComponent.tsx`, `AIChatComponentV2.tsx` 등 읽기 | 2 |
| 로컬 DB SDUI_TB Flyway 히스토리 상태 | `SELECT version, description, success FROM flyway_schema_history ORDER BY installed_rank;` | 5 |

---

### Phase 1 — `useAIChatLogic` 리팩토링 (항목 6·7·8·9)

**목표**: 모든 AI 컴포넌트가 공유하는 핵심 훅을 안정화하여, 이후 UI 수정·프롬프트 외부화의 기반을 마련한다.

#### 1-A. API 서비스 모듈 분리 (항목 6)

- **파일 생성**: `metadata-project/services/aiService.ts`
- **이동 대상**: `useAIChatLogic.ts` 내 `api.post(sttEndpoint, ...)`, `api.post(translateEndpoint, ...)` 호출 로직
- **결과**: 훅은 `aiService.stt(blob, language)`, `aiService.translate(text, target)` 만 호출
- **장점**: 테스트 시 서비스만 모킹하면 됨, 엔드포인트 변경 시 서비스 파일 1곳만 수정

```
metadata-project/
├── services/
│   ├── axios.ts        (기존)
│   └── aiService.ts    (신규 — STT·번역·채팅 API 호출 전담)
```

#### 1-B. `onAudioReady` 콜백 분리 (항목 7)

- **분리할 함수**:
  - `handleSTT(blob, sttLanguage): Promise<string>` — STT 요청만
  - `handleTranslateIfNeeded(transcript, mode, language): Promise<string>` — 번역 조건 판단 + 번역 요청
  - `buildUserMessage(transcript, blob): ChatMessage` — 메시지 객체 생성
- `onAudioReady` 콜백은 위 함수들을 순서대로 호출하는 오케스트레이터 역할만 유지

#### 1-C. JSON 파싱 안정화 (항목 8)

현재 `handleDone`의 `lastIndexOf('{')` / `lastIndexOf('}')` 방식을 아래로 교체:

```typescript
// 개선 방향: 정규식으로 첫 번째 완전한 JSON 객체 추출
const jsonMatch = content.match(/\{[\s\S]*?\}/);
// 타입 가드 추가
function isAIResponse(obj: unknown): obj is { en?: string; ko?: string } {
    return typeof obj === 'object' && obj !== null && ('en' in obj || 'ko' in obj);
}
```

#### 1-D. 목표 달성 훅 분리 (항목 9)

- **파일 생성**: `metadata-project/lib/hooks/useGoalTracker.ts`
- **이동 대상**: `useAIChatLogic.ts` 내 `userMessageCount` 계산 + `hasTriggeredGoalRef` + 목표 달성 `useEffect`
- **인터페이스**:
  ```typescript
  useGoalTracker({ messages, goalCount: 10, onGoalAchieved })
  // 반환: { userMessageCount, isGoalAchieved }
  ```
- `useAIChatLogic`은 `useGoalTracker`를 호출만 하도록 단순화

**Phase 1 완료 기준**
- [ ] `npm run test` — 기존 useAIChatLogic 테스트 전부 통과
- [ ] `npx jest tests/hooks/useAIChatLogic.test.ts` — 신규 분리 로직 테스트 추가 + 통과
- [ ] `npx playwright test tests/e2e/AIChatV2.test.ts` — E2E 회귀 없음

---

### Phase 2 — AI_INTERVIEW / AI_JAPANESE UI 차별화 (항목 1)

**목표**: 각 페이지가 고유한 테마로 렌더링되도록 수정한다.

#### 2-A. AI_INTERVIEW 확인 및 수정

원인별 수정 방향:

| 원인 | 수정 위치 | 수정 내용 |
|------|-----------|-----------|
| `componentMap`에 `AI_INTERVIEW` 미등록 | `componentMap.tsx` | `AI_INTERVIEW: AIInterviewComponent` 추가 (미등록 시) |
| `ui_metadata`의 `css_class`에 `ai-interview-theme` 누락 | `V33__add_ai_interview_page.sql` 또는 로컬 DB 직접 수정 | 최상위 GROUP에 `css_class = 'ai-interview-theme'` 설정 |
| `AI_INTERVIEW.css` import 누락 | `app/styles/index.css` | `@import './AI_INTERVIEW.css';` 추가 확인 |

#### 2-B. AI_JAPANESE_CHAT 확인 및 수정

| 원인 | 수정 위치 | 수정 내용 |
|------|-----------|-----------|
| `ui_metadata`의 `css_class`에 `ai-japanese-theme` 누락 | `V35__add_ai_japanese_chat_page.sql` 또는 로컬 DB | 최상위 GROUP에 `css_class = 'ai-japanese-theme'` 설정 |
| `AI_JAPANESE.css` import 누락 | `app/styles/index.css` | `@import './AI_JAPANESE.css';` 추가 확인 |

**Phase 2 완료 기준**
- [ ] `http://localhost:3000/view/AI_INTERVIEW_PAGE` → Slate Navy 테마 확인
- [ ] `http://localhost:3000/view/AI_JAPANESE_CHAT_PAGE` → Cherry Blossom 핑크 테마 확인
- [ ] `npx jest tests/components/AIInterviewComponent.test.tsx` 통과
- [ ] `npx jest tests/components/AIJapaneseChat.test.tsx` 통과
- [ ] `npx playwright test tests/e2e/AIInterview.test.ts tests/e2e/AIJapaneseChat.test.ts` 통과

---

### Phase 3 — 하드코딩 시스템 프롬프트 DB 전환 (항목 2)

**목표**: 컴포넌트 내 하드코딩된 프롬프트를 `ui_metadata.system_prompt_template` 컬럼으로 이전한다.
(V34 마이그레이션으로 컬럼은 이미 추가됨, `UiMetadata.java` + `UiResponseDto.java` 이미 반영 완료)

#### 3-A. 현재 하드코딩 위치 목록화

```
AIInterviewComponent.tsx     — 면접관 시스템 프롬프트 (영어/한국어 조건 분기)
AIChatComponentV2.tsx        — 영어 튜터 시스템 프롬프트
(AIChatComponentV2 재사용) 일본어 프롬프트 — V35 migration DATA_SOURCE로 주입 여부 확인
```

#### 3-B. DB 전환 방식

1. `V35` 또는 신규 `V36` SQL에서 `DATA_SOURCE` 컴포넌트의 `system_prompt_template` 컬럼에 프롬프트 저장
2. 프론트엔드에서 `pageData.systemPrompt` (SDUI 엔진이 바인딩)를 `useAIChatLogic`의 `systemPrompt` prop으로 전달
3. 컴포넌트 내 하드코딩 문자열 제거

**Phase 3 완료 기준**
- [ ] 각 페이지 컴포넌트 내 시스템 프롬프트 상수 제거
- [ ] DB에서 프롬프트 변경 시 서버 재시작 없이 반영 확인 (Redis TTL 고려)
- [ ] `npx jest` — 전체 프론트 테스트 통과
- [ ] `./gradlew test` — 백엔드 테스트 통과

---

### Phase 4 — FastAPI 발음 정확도 시스템 (항목 4)

**목표**: AI_CHAT(영어), AI_JAPANESE(일본어) 화면에 사용자 발음 정확도 피드백을 추가한다.

#### 4-A. 현황 파악 (Phase 0에서 조사 결과 반영)

- `pronounce-api/app/main.py` 현재 엔드포인트 목록 확인
- `POST /pronunciation-score` 존재 여부 + 입력 형식 확인 (사용자 텍스트 vs. 정답 텍스트 비교 방식)

#### 4-B. 설계 방향

```
사용자 음성 → Whisper STT → transcript
                            ↓
                   Spring Boot → POST http://localhost:8001/pronunciation-score
                                 { "spoken": transcript, "expected": AI가 낸 문장 }
                            ↓
                   점수 + 피드백 → 프론트엔드 표시
```

- **Spring Boot 신규 엔드포인트**: `POST /api/ai/pronunciation` (FastAPI 호출 프록시)
- **프론트**: STT 완료 후 선택적으로 발음 점수 요청 (버튼 또는 자동)
- **표시**: ConversationPanel 내 사용자 메시지 아래 점수 배지 (0~100)

#### 4-C. 구현 파일 목록

| 파일 | 역할 |
|------|------|
| `pronounce-api/app/main.py` | `/pronunciation-score` 엔드포인트 확인/보강 |
| `SDUI-server/.../ai/controller/AiPronunciationController.java` | FastAPI 호출 프록시 |
| `SDUI-server/.../ai/service/PronunciationService.java` | RestTemplate으로 FastAPI 호출 |
| `services/aiService.ts` | `checkPronunciation(spoken, expected)` 추가 |
| `components/fields/ai/ConversationPanelV2.tsx` | 점수 배지 UI 추가 |

**Phase 4 완료 기준**
- [ ] FastAPI 포트 8001 정상 응답 확인
- [ ] `./gradlew test --tests "...AiPronunciationControllerTest"` 통과
- [ ] `npx jest tests/components/AIChatComponentV2.test.tsx` — 발음 점수 표시 테스트 추가 + 통과
- [ ] E2E: 발음 점수 배지 렌더링 확인

---

### Phase 5 — Migration 정리 및 배포 전략 (항목 5)

**목표**: 로컬 → 로컬 Docker → AWS 순서로 안전하게 마이그레이션을 적용한다.

#### 5-A. 현재 Migration 버전 정리

| 버전 | 내용 | 상태 |
|------|------|------|
| V1–V10 | 기존 SDUI 기능 | main 브랜치 적용 완료 |
| V28–V32 | AI 멤버십 + 영어 채팅 V2 | feature 브랜치 (로컬만) |
| V33 | AI_INTERVIEW_PAGE | feature 브랜치 (로컬만) |
| V34 | system_prompt_template 컬럼 DDL | feature 브랜치 (로컬만) |
| V35 | AI_JAPANESE_CHAT_PAGE 데이터 | feature 브랜치 (로컬만) |
| V36 | enable_japanese_chat | feature 브랜치 (로컬만) |

#### 5-B. 로컬 DB 클린 스타트 절차 (필요 시)

```sql
-- PostgreSQL 로컬 SDUI_TB 완전 초기화
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
-- Spring Boot 재시작 → Flyway V1부터 순차 실행
```

#### 5-C. 단계별 검증 체크리스트

**Step 1 — 로컬 DB (현재)**
- [ ] `./gradlew bootRun` 후 Flyway V1~V36 전부 `success=true` 확인
- [ ] `http://localhost:3000` 전체 화면 동작 확인
- [ ] `npm run test` + `./gradlew test` 전부 통과

**Step 2 — 로컬 Docker**
- [ ] `docker-compose up -d` → `sdui-db` 컨테이너 기준으로 동일 마이그레이션 실행
- [ ] `docker logs sdui-db` 오류 없음 확인
- [ ] 동일 테스트 반복

**Step 3 — AWS 적용 (마지막)**
- [ ] lab_claude 또는 main 브랜치 머지 전, Migration SQL 최종 검토
- [ ] RDS 백업 스냅샷 생성 후 적용
- [ ] 배포 후 헬스체크 + E2E 스모크 테스트

---

### 전체 실행 순서 요약

```
Phase 0 (현황 파악)
    ↓
Phase 1 (useAIChatLogic 리팩토링 — 항목 6·7·8·9)
    ↓  [테스트]
Phase 2 (UI 차별화 — 항목 1)
    ↓  [테스트]
Phase 3 (프롬프트 DB 전환 — 항목 2)
    ↓  [테스트]
Phase 4 (발음 채점 시스템 — 항목 4)
    ↓  [테스트]
Phase 5 (Migration 정리 및 배포 — 항목 5)
```

> **우선순위 근거**:
> Phase 1 리팩토링을 먼저 해야 Phase 2~4의 코드 변경이 깔끔하게 적용된다.
> Phase 3(프롬프트 외부화)은 Phase 2(UI 구분) 완료 후 각 페이지의 프롬프트 위치가 확정된 뒤 진행해야 누락이 없다.
> Phase 4(발음 채점)는 FastAPI 의존성이 있어 독립적으로 병렬 진행 가능하나, Phase 1 완료 후 `aiService.ts`에 추가하는 것이 자연스럽다.
> Phase 5(배포)는 기능이 모두 안정된 마지막 단계.



// [메모] AI-INTERVIEW 기능 추가 (.ai2폴더 아래 문서참고)

 Phase 3: AI 면접관 (Phase 2 이후)

Phase 2 AI 파이프라인에 interview 세션 관리 추가:

- `V13__create_interview_sessions.sql` (optional: sessionId → conversationHistory 저장)
- `InterviewSessionService.java` — 세션별 대화 이력 관리 (Redis 활용)
- `AiInterviewController.java` — 이미 Phase 2에서 생성


 AI 파이프라인 API

```
 STT: 오디오 → 텍스트
POST /api/ai/stt
Content-Type: multipart/form-data
Body: { "audio": <File (webm/wav)> }
Response: { "data": { "text": "Hello, I'm..." } }
필요 권한: can_converse = true

 Chat: 텍스트 → AI 응답 (SSE Streaming)
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

 AI 면접관: 이력서 분석 + 질문 생성
POST /api/ai/interview/start
Body: { "resumeContent": "...", "language": "ko" | "en" }
Response: {
  "data": {
    "sessionId": "uuid",
    "firstQuestion": "자기소개를 해주세요.",
    "questions": ["q1", "q2", "q3"]
  }
}

 AI 면접관: 답변 → 꼬리 질문 생성 (SSE Streaming)
POST /api/ai/interview/answer
Body: {
  "sessionId": "uuid",
  "userAnswer": "저는 3년차 백엔드 개발자입니다...",
  "currentQuestion": "자기소개를 해주세요.",
  "conversationHistory": [...]
}
Response: text/event-stream (꼬리 질문 스트리밍)
```

 4.2 AI 면접관 UX 흐름

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

현재까지 진행된 작업
✅ 완료된 작업
Phase 1 — useAIChatLogic 리팩토링 (이전 세션)

aiService.ts 생성: API 호출 로직 분리
useGoalTracker.ts 생성: 목표 추적 로직 분리
useAIChatLogic.ts 리팩토링: 상태 관리 단순화
관련 테스트 통과
Phase 2-A — screenMap.ts 등록 (이전 세션)

AI_INTERVIEW_PAGE, AI_JAPANESE_CHAT_PAGE screenMap 등록 완료
Phase 2-B/C — CSS 시각적 분리 (이번 세션)

AI_INTERVIEW.css: Corporate Navy 테마 완성
헤더: 다크 네이비 배경 (#1E293B)
푸터: 딥 네이비 배경
마이크 버튼: 직사각형(rectangular) 포멀 스타일
메시지 버블: 왼쪽 포인트 보더 + 네이비 색상
AI_JAPANESE.css: Cherry Blossom 테마 완성
인트로: 사쿠라 그라디언트 배경
헤더: 벚꽃 핑크
마이크 버튼: 체리 그라디언트
메시지 버블: 핑크 왼쪽 보더 + 그라디언트 유저 버블
테스트 결과 (Phase 2 부분)


PASS tests/components/AIInterviewComponent.test.tsx
PASS tests/components/AIJapaneseChat.test.tsx
⏳ 남은 작업
Phase	내용	상태
Phase 2	전체 테스트 스위트 실행	중단됨 (재실행 필요)
Phase 3-A	AIInterviewComponent 풀 스펙 업그레이드	미착수
Phase 3-B	시스템 프롬프트 DB 외부화	미착수
Phase 4	FastAPI 발음 채점	미착수
Phase 5	마이그레이션 정리	미착수
AI-INTERVIEW 기능 추가 반영 계획
.ai2 폴더 문서에서 확인한 풀 스펙 기준으로:

현재 상태 (간소화 버전)

[AIInterviewComponent] → useAIChatLogic (generic) → POST /api/chat/stream
                                                    ↑ 하드코딩된 프롬프트
Phase 3 완료 후 목표 상태

[AI_INTERVIEW_PAGE 진입]
    ↓
[이력서 입력 단계] ← ResumeUploader (텍스트/PDF/이미지 탭)
    ↓
POST /api/ai/interview/start → { sessionId, firstQuestion }
    ↓
[ConversationPanel — 면접 진행]
    ↓
POST /api/ai/interview/answer (SSE 스트리밍)
구체적 변경 파일
AIInterviewComponent.tsx

ResumeUploader 단계 추가 (텍스트/PDF/이미지)
useAIChatLogic → 인터뷰 전용 훅 (useInterviewLogic)으로 교체
/api/ai/interview/start + /api/ai/interview/answer 엔드포인트 사용
sessionId 상태 관리 추가
useInterviewLogic.ts (신규 생성)

startInterview(resumeText) → sessionId 획득
answerInterview(sessionId, answer) → SSE 응답
V34 SQL (system_prompt_template 외부화)

ui_metadata에 system_prompt_template 컬럼 값 추가
하드코딩된 프롬프트 제거


// [메모] AI-INTERVIEW 
한국어만 지원, AI면접이므로 이미지나 pdf나 텍스트를 분석한 맥락을 면접관이 알고 있어야 하고 
우선 면접관이 "자기소개 해주세요"로 시작해서 꼬리질문을 하는 방식으로 진행된다.  
따라서 AI-INTERVIEW는 마이크가 하나여야만 한다. 
(취소 기능, stop기능은 동일)

// [메모] AI-JAPANESE
AI-mode 는 일본어만 지원, 사람은 Generic mic 이나 일본어 말하는 버튼 두개 (AI-mode, Human-mode가 AI-ENGISH와 동일한 방식)

---

## 이번 세션 작업 현황 (2026-03-16, 계속)

### ✅ 이번 세션에서 완료된 작업

#### Phase 3-B — 시스템 프롬프트 DB 외부화 (백엔드 반영)

| 파일 | 변경 내용 |
|------|-----------|
| `SDUI-server/.../domain/ui/domain/UiMetadata.java` | `systemPromptTemplate` 필드 추가 (`@Column(name="system_prompt_template")`) |
| `SDUI-server/.../domain/ui/dto/UiResponseDto.java` | 동일 필드 DTO 반영 + `buildOverrides()` 매핑 |
| `./gradlew compileJava` | BUILD SUCCESSFUL 확인 |

#### Phase 4 — FastAPI 발음 채점 시스템 (백엔드·타입·서비스 계층 완료)

| 파일 | 변경 내용 |
|------|-----------|
| `pronounce-api/app/main.py` | `from difflib import SequenceMatcher` 추가, `PronunciationRequest` 모델, `POST /pronunciation-score` 엔드포인트 구현 |
| `SDUI-server/.../ai/dto/PronunciationRequest.java` | 신규 (spoken, expected, language) |
| `SDUI-server/.../ai/dto/PronunciationResponse.java` | 신규 (score: int, feedback: String) |
| `SDUI-server/.../ai/service/PronunciationService.java` | 신규 (RestTemplate → FastAPI 프록시, `${fastapi.url:http://localhost:8001}`) |
| `SDUI-server/.../ai/controller/AiPronunciationController.java` | 신규 (`POST /api/ai/pronunciation`) |
| `metadata-project/lib/types/ai.ts` | `ChatMessage`에 4개 발음 필드 추가 (`pronunciationScore`, `pronunciationSpoken`, `pronunciationExpected`, `pronunciationFeedback`) |
| `metadata-project/services/aiService.ts` | `checkPronunciation(spoken, expected, language)` 메서드 추가, `PRONUNCIATION_ENDPOINT` export |

---

### ⚠️ [메모] 기반 수정 필요 사항

#### AI-INTERVIEW 스펙 수정
현재 `AIInterviewComponent.tsx`는 `useAIChatLogic` (generic) 기반으로 구현된 간소화 버전.
[메모]에 따라 아래 스펙으로 변경 필요:

| 항목 | 현재 (간소화) | 목표 스펙 |
|------|--------------|----------|
| 언어 | 영어/한국어 조건 분기 | **한국어 전용** |
| 마이크 버튼 | 두 개 (AI-mode, Human-mode) | **하나** (취소/stop 기능은 유지) |
| 시작 방식 | 즉시 채팅 시작 | **이력서 입력(텍스트/PDF/이미지) → 면접 시작** |
| API | `POST /api/ai/v2/chat/stream` | `POST /api/ai/interview/start` + `/api/ai/interview/answer` (SSE) |
| 훅 | `useAIChatLogic` | `useInterviewLogic` (신규 분리 필요) |

#### AI-JAPANESE 스펙 확인
[메모]에 따라 현재 구현이 맞는지 확인 필요:

| 항목 | 확인 사항 |
|------|-----------|
| 마이크 버튼 수 | **두 개** (Generic mic / 일본어 전용 버튼) — AI-ENGLISH와 동일 방식 |
| AI-mode | 일본어만 지원 (언어 고정) |
| Human-mode | 동일한 방식 유지 |

---

### ⏳ 남은 작업 (우선순위 순)

| Phase | 작업 | 상태 | 비고 |
|-------|------|------|------|
| Phase 4 | `useAIChatLogic.ts` — `aiService.checkPronunciation` 호출 추가 | **중단됨** | 사용자가 중단 요청, 방향 확인 후 재개 필요 |
| Phase 4 | `ConversationPanelV2.tsx` — 점수 배지 UI (score + spoken vs expected 비교 표시) | 미착수 | |
| Phase 4 | CSS — 점수 배지 스타일 (0~44: 빨강, 45~64: 노랑, 65~84: 초록, 85+: 파랑) | 미착수 | |
| Phase 4 | `AiPronunciationControllerTest.java` (Spring Boot 유닛테스트) | 미착수 | |
| Phase 4 | 프론트엔드 테스트 업데이트 (`AIChatComponentV2.test.tsx`) | 미착수 | |
| Phase 4 | `./gradlew compileJava` (Phase 4 백엔드 파일 포함) | 완료 | exit code 0 확인됨 |
| Phase 3-A | `AIInterviewComponent.tsx` 풀 스펙 업그레이드 | 미착수 | 마이크 1개, 한국어 전용, 이력서 입력 단계 추가 |
| Phase 3-A | `useInterviewLogic.ts` 신규 생성 | 미착수 | |
| Phase 3-A | `AiInterviewController.java` — `/start` + `/answer` 엔드포인트 확인/완성 | 미착수 | |
| Phase 3-B | 프론트엔드 — 하드코딩 시스템 프롬프트 제거, `pageData.systemPromptTemplate` 연동 | 미착수 | |
| Phase 5 | 마이그레이션 정리 (V33~V36) + 로컬 DB 클린 스타트 검증 | 미착수 | |

---

### 다음 세션에서 확인할 것

1. **`useAIChatLogic.ts` 발음 채점 연동 방식**: 자동(STT 직후 자동 호출) vs 수동(버튼) 중 어떤 방식 원하는지 확인
-> 자동(STT 직후 자동 호출)
2. **Phase 3-A vs Phase 4 우선순위**: AI-INTERVIEW 풀 스펙 먼저 vs 발음 채점 UI 먼저 어떤 것 먼저 진행할지 확인
-> AI-INTERVIEW 풀 스펙 먼저
3. **AI-JAPANESE 마이크 UI**: 현재 두 개 버튼이 올바르게 동작하는지 브라우저 확인

// [메모] AI-JAPANESE 페이지 브라우저 테스트
0316_일본어테스트_1, 0316_일본어테스트_2, 0316_일본어테스트_3을 업로드 했습니다. AI_JAPANESE_CHAT_PAGE 에서 일본어 인식이 안됩니다.

css는 SDUI_AI_ENGLISH_CHAT_PAGE 와 동일하게 AI_INTERVIEW 와 AI_JAPANESE_CHAT_PAGE 에 적용이 필요합니다.

---

## 2026-03-16 오후 세션 — AI-INTERVIEW 풀 스펙 + Phase 4 발음 채점 완료

### 현재 상황 (중단점)

`git stash`로 변경사항이 임시 저장된 상태입니다. 변경사항을 복원하려면 `git stash pop`이 필요합니다.

### 완료된 작업

| 항목 | 파일 | 내용 |
|------|------|------|
| A-1 DTO @Setter | `InterviewStartRequest.java`, `InterviewAnswerRequest.java` | @Setter 어노테이션 추가 → 통합 테스트 컴파일 가능 |
| A-2 V37 migration | `V37__update_interview_config_to_korean.sql` | query_master 한국어 전환 (language='ko', 버튼 레이블 모두 한국어) |
| A-3 CSS import | `app/globals.css` | AI_INTERVIEW.css, AI_JAPANESE.css import 추가 |
| A-4 componentMap | `componentMap.tsx` | AI_INTERVIEW 키 등록 → 렌더링 버그 수정 |
| A-5 한국어 기본값 | `AIInterviewComponent.tsx`, `AIInterviewIntro.tsx` | language 기본값 'ko', 버튼 레이블 한국어 |
| A-6 테스트 수정 | `AIInterviewComponent.test.tsx` | '면접 시작하기' 텍스트로 수정 + 테스트 2개 추가 (3개 PASS) |
| A-6 E2E 수정 | `AIInterview.test.ts` | 버튼 텍스트 한국어 수정 + textarea 입력 추가 (버튼 disabled 버그 수정) |
| A-7 빌드 검증 | gradle compileJava | exit code 0 ✅ |
| B-1 발음 채점 | `useAIChatLogic.ts` | aiService.checkPronunciation STT 직후 자동 호출, originalTranscript 캡처, wasTranslated 조건 체크 |
| B-2 배지 UI | `ConversationPanelV2.tsx` | getScoreLevel 함수 + pronunciation-badge 렌더링 |
| B-3 CSS 스타일 | `AI_CHAT_V2.css` | 점수 배지 스타일 (4단계 색상: 파랑/초록/노랑/빨강) |

### 테스트 결과 (git stash 하기 직전)

- 내 변경 후: **2 FAIL, 6 PASS** (8 total)
  - FAIL: `rendering_optimization.test.tsx` (5개), `auth_security.test.tsx` (5개)
  - 이 10개 실패는 **기존 실패** — git stash로 확인 완료
- git stash 적용 후 (변경 전 상태): **5 FAIL, 6 PASS** (11 total) — 더 많이 실패
  - 추가 실패: e2e/AIInterview, e2e/AIJapaneseChat, e2e/AIChatV2 (영어 텍스트 체크 때문)

### git stash pop 설명

`git stash`는 현재 작업 중인 변경사항(커밋 안 된 것)을 임시 저장소(stash)에 보관하고 작업 트리를 깨끗하게 되돌립니다. 검증 목적으로 사용했습니다.

`git stash pop`은 stash에 보관된 변경사항을 다시 꺼내서 작업 트리에 복원합니다. 실행하면 B-1~B-3 변경사항이 모두 돌아옵니다.

### 다음 작업

1. `git stash pop` → 변경사항 복원
2. `npm run test` 전체 통과 확인 (10개 기존 실패 제외)
3. 백엔드 통합 테스트: `./gradlew test --tests "*.AiInterviewIntegrationTest"`



// [메모] AI-JAPANESE, AI_ENGLISH_CHAT_PAGE 테스트

0316_일본어테스트_pronunciation.png, 0316_일본어테스트_pronunciation2.png,
0316_영어테스트.png 참고

---

## Phase 4 — FastAPI 발음 채점 서버 기동 및 아키텍처

### 500 에러 원인 및 해결

| 항목 | 내용 |
|------|------|
| **에러** | `Connection refused: connect` to `http://localhost:8001/pronunciation-score` |
| **원인** | FastAPI 서버(`pronounce-api/app/main.py`)가 실행되지 않은 상태 |
| **해결** | `cd pronounce-api && uvicorn app.main:app --port 8001 --reload` |

> **주의**: `main.py` 353~354행에 HuggingFace ML 번역 모델이 모듈 레벨 로드 → 첫 기동 시 수 분 소요 가능 (모델 미캐시 상태)

---

### 로컬 전체 기동 순서

```bash
# 터미널 1 — DB + Redis
docker-compose up -d

# 터미널 2 — Spring Boot
cd SDUI-server && ./gradlew bootRun

# 터미널 3 — FastAPI (발음 채점)
cd pronounce-api && uvicorn app.main:app --port 8001 --reload

# 터미널 4 — Next.js
cd metadata-project && npm run dev
```

---

### FastAPI ↔ Spring Boot 연결 구조

```
[Browser]
   │  JWT (HttpOnly Cookie)
   ▼
[Next.js :3000]
   │  /api/* → proxy
   ▼
[Spring Boot :8080]   ← JWT 검증은 여기서만 (JwtAuthenticationFilter)
   │  RestTemplate 내부 호출 (인증 헤더 없음)
   │  POST http://localhost:8001/pronunciation-score
   ▼
[FastAPI :8001]        ← 순수 채점 서비스, JWT/인증 로직 없음
```

**핵심**: FastAPI는 Spring Boot의 내부 마이크로서비스 역할.
외부에서 직접 접근하지 않으므로 별도 인증 불필요.

---

### JWT 처리 방식

| 레이어 | JWT 처리 |
|--------|----------|
| **Next.js** | JWT를 쿠키로 전달만 함 (검증 없음) |
| **Spring Boot** | `JwtAuthenticationFilter`에서 토큰 파싱 + 검증 → `SecurityContext` 설정 |
| **AiPronunciationController** | `@AuthenticationPrincipal CustomUserDetails` → 인증된 사용자만 `/api/ai/pronunciation` 접근 가능 |
| **FastAPI** | JWT 처리 없음 — Spring Boot가 인증을 마친 후 내부 호출 |

- **FastAPI URL 설정**: `PronunciationService.java` → `@Value("${fastapi.url:http://localhost:8001}")`
- **보안 경계**: Spring Boot가 게이트웨이 역할, FastAPI는 내부망에만 노출

---

### AWS 배포 시 적용 방법

#### 현재 문제점

`docker-compose.yml`에 FastAPI 서비스가 없음 → AWS 배포 시 발음 채점 기능 동작 안 함

#### 권장 방식: docker-compose에 FastAPI 서비스 추가

`docker-compose.yml`에 아래 서비스 추가 필요:

```yaml
  pronounce-api:
    build:
      context: ./pronounce-api
      dockerfile: Dockerfile       # 신규 생성 필요
    container_name: sdui-pronounce
    # ports: - "8001:8001"         # 외부 노출 금지 — 내부 통신만 사용
    networks:
      - sdui-network               # Spring Boot와 같은 네트워크
```

`app` 서비스에 FastAPI URL 환경변수 추가:

```yaml
  app:
    environment:
      - FASTAPI_URL=http://pronounce-api:8001   # Docker 내부 네트워크 주소
```

`application.yml`의 `fastapi.url`은 이미 환경변수 대응:
```yaml
# PronunciationService.java
@Value("${fastapi.url:http://localhost:8001}")  # 로컬 기본값, AWS에서 환경변수 오버라이드
```

#### FastAPI Dockerfile 생성 필요 (현재 없음)

`pronounce-api/` 안에 Dockerfile이 없어 Docker 빌드 불가. 생성 시 주의사항:
- `main.py` 29행: `GOOGLE_APPLICATION_CREDENTIALS` 경로 하드코딩 → 환경변수로 교체 필요
- ML 번역 모델(`transformers pipeline`) → `/pronunciation-score`만 사용 시 불필요한 의존성 (경량화 고려)

#### 단계별 배포 체크리스트

```
Step 1 — 로컬 직접 기동 테스트 (현재)
  ☐ uvicorn app.main:app --port 8001 --reload
  ☐ POST http://localhost:8001/pronunciation-score 응답 확인
  ☐ 브라우저에서 발음 점수 배지 렌더링 확인

Step 2 — 로컬 Docker 테스트 (다음)
  ☐ pronounce-api/Dockerfile 생성
  ☐ docker-compose.yml에 pronounce-api 서비스 추가
  ☐ docker-compose up -d → 전체 연동 확인

Step 3 — AWS EC2 배포 (마지막, feature→main 머지 후)
  ☐ EC2 보안그룹: 8001 포트 외부 노출 금지
  ☐ FASTAPI_URL=http://pronounce-api:8001 환경변수 설정
  ☐ GOOGLE_APPLICATION_CREDENTIALS 시크릿 관리 (AWS Secrets Manager 권장)
  ☐ Migration V33~V37 순차 적용 (RDS 백업 스냅샷 후)
```

---

### 부수 버그: 일본어 페이지에서 `language: "ko"` 전송

- `AIChatComponentV2.tsx:28`: `meta.target_language(없음) || data?.language(미수신) || fallback('ko')`
- `actionType = 'AI_CHAT_JA'` → `includes('EN')` 미충족 → `'ko'` 폴백
- 현재 채점 로직(`SequenceMatcher`)은 `language` 필드를 미사용 → 점수 자체에 영향 없음

---

요청 URL
http://localhost:3000/api/ai/pronunciation
요청 메서드
POST
상태 코드
500 Internal Server Error
원격 주소
[::1]:3000
리퍼러 정책
strict-origin-when-cross-origin
{
    "spoken": "Konnichiwa",
    "expected": "こんにちは",
    "language": "ko"
}
{
    "status": "error",
    "message": "서버 오류가 발생했습니다",
    "error": "IllegalStateException: 발음 채점에 실패했습니다: I/O error on POST request for \"http://localhost:8001/pronunciation-score\": Connection refused: connect",
    "timestamp": "2026-03-16T12:29:31.1813505",
    "path": "/api/ai/pronunciation"
}
http://localhost:3000/api/ai/pronunciation

{
    "spoken": "How to speak Korean",
    "expected": "It sounds like you might be watching a movie in a hotel, but speaking in Korean. Are you watching an English movie and practicing your English at the same time? What movie are you currently watching?",
    "language": "en"
}
{
    "status": "error",
    "message": "서버 오류가 발생했습니다",
    "error": "IllegalStateException: 발음 채점에 실패했습니다: I/O error on POST request for \"http://localhost:8001/pronunciation-score\": Connection refused: connect",
    "timestamp": "2026-03-16T12:36:01.0497804",
    "path": "/api/ai/pronunciation"
}

---

## Phase 5 — FastAPI RBAC + AI Interview 마무리 (2026-03-16)

### FastAPI RBAC — 내부 API Key 방식

**배경**: FastAPI `/pronunciation-score`는 인증 없이 외부에서 직접 접근 가능한 상태였음.

**방식**: `X-Internal-Api-Key` 공유 비밀키 헤더

```
[Browser] → JWT → [Next.js]
                 → proxy
              [Spring Boot] ← JWT 검증 (JwtAuthenticationFilter)
                 │  X-Internal-Api-Key: {shared-secret}  ← 신규 추가
                 ▼
              [FastAPI]  ← 키 일치 여부만 확인 (401 반환)
```

#### 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `pronounce-api/.env` | `INTERNAL_API_KEY=sdui-internal-dev-key` 추가 (gitignore 등록됨) |
| `pronounce-api/app/main.py` | `Header, HTTPException` import + `INTERNAL_API_KEY` 변수 + `/pronunciation-score` 헤더 검증 |
| `PronunciationService.java` | `@Value("${fastapi.internal-api-key:...}")` + `headers.set("X-Internal-Api-Key", internalApiKey)` |
| `application-local.yml` | `fastapi.url` + `fastapi.internal-api-key` 명시 |
| `application-test.yml` | `fastapi.internal-api-key: test-internal-key` 추가 |

#### FastAPI 핵심 변경 (main.py)

```python
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "sdui-internal-dev-key")

@app.post("/pronunciation-score")
async def pronunciation_score(
    req: PronunciationRequest,
    x_internal_api_key: str = Header(None, alias="X-Internal-Api-Key")
):
    if x_internal_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # 기존 채점 로직 유지
```

> **AWS 배포 시**: `INTERNAL_API_KEY`는 환경변수 또는 AWS Secrets Manager로 관리.
> 기본값(`sdui-internal-dev-key`)은 로컬 전용.

---

### AI Interview 마무리

#### V38 마이그레이션
`V38__update_interview_label_to_korean.sql`:
```sql
UPDATE ui_metadata
SET label_text = 'AI 면접관'
WHERE component_id = 'ai_interview_field'
  AND screen_id = 'AI_INTERVIEW_PAGE';
```
→ `'AI Interview Specialist'` → `'AI 면접관'` 한국어 전환

#### E2E 테스트 수정
`tests/e2e/AIInterview.test.ts`:
- `/AI Interview Specialist/` → `/AI 면접관/` (2곳)

---

### 프론트엔드 테스트 결과

```
PASS  tests/components/AIInterviewComponent.test.tsx  ← 3/3 통과
PASS  tests/hooks/useAIChatLogic.test.ts
PASS  tests/components/AIJapaneseChat.test.tsx
PASS  tests/components/AIChatComponentV2.test.tsx
PASS  tests/api_duplicated.test.tsx
PASS  tests/components/TimeSelect.test.tsx
FAIL  tests/rendering_optimization.test.tsx  ← 기존 실패 (변경과 무관)
FAIL  tests/integration/auth_security.test.tsx  ← 기존 실패 (변경과 무관)
```

> 실패한 2개는 이번 변경 이전부터 존재하던 테스트. 렌더링 최적화 카운트 불일치.

---

### 백엔드 통합 테스트

```bash
cd SDUI-server && ./gradlew clean test --tests "*.AiInterviewIntegrationTest"
```
> 직접 실행 필요 (Gradle 캐시 이슈로 `clean` 옵션 필수)

---

## 2026-03-16 저녁 세션 — FastAPI 환경 정비 + 엔드포인트 정리

### 테스트 실패 원인 분석 (기존 실패 2개)

#### `rendering_optimization.test.tsx` — 렌더 3회 발생

| 렌더 | 원인 |
|------|------|
| #1 | 컴포넌트 초기 마운트 |
| #2 | `useDeviceType()` → `isDesktop` 상태 변경 |
| #3 | **`AuthProvider.checkLoginStatus()`** → `finally { setIsLoading(false) }` (true→false) → Context 리렌더 |

- `isLoading`이 `useState(true)`로 시작 → `finally`에서 항상 `false`로 변경 → Context 업데이트 → DynamicEngine 리렌더
- 테스트: `expect(engineRenderCount).toBe(2)` → Received: 3 → FAIL
- **처리 방침**: 현재 브랜치 완료 후 별도 수정 예정

#### `auth_security.test.tsx` — `response.data = ""`

- **원인**: `jest.config.js`의 `customExportConditions: ['node', 'node-addons']`
- jsdom 환경인데 MSW가 Node 어댑터로 로드 → axios XHR 인터셉션 불완전 → 응답 본문 빈 문자열
- **처리 방침**: 현재 브랜치 완료 후 별도 수정 예정

---

### FastAPI 환경 문제 해결

#### 1. `requirements.txt` UTF-16 인코딩 문제

- **증상**: `pip install -r requirements.txt` 실행 시 패키지명 인식 불가
- **원인**: 파일이 UTF-16으로 저장 → 각 문자 사이에 null 바이트(`\0`) 삽입됨
- **해결**: UTF-8로 재작성

#### 2. numpy 빌드 실패

- **증상**: `numpy==2.2.5` 설치 시 Meson 빌드 시스템이 C 컴파일러 탐색 → `Unknown compiler(s)` 에러
- **원인**: `torch`/`transformers`의 간접 의존성. 로컬에 Visual Studio Build Tools 없음
- **해결**: torch, transformers, numpy 제거 결정 (아래 엔드포인트 정리 참고)

---

### FastAPI 엔드포인트 정리

사용자 결정 (3가지 선택):
- **번역**: googletrans + HuggingFace API 2종 유지 (로컬 torch 모델 제거)
- **TTS**: Google Cloud TTS + gTTS 모두 유지
- **/analyze**: 유지 (한국어 형태소 분석, konlpy)

#### requirements.txt — 제거된 패키지 (15개)

`torch`, `transformers`, `numpy`, `tokenizers`, `safetensors`, `huggingface-hub`,
`regex`, `tqdm`, `sympy`, `mpmath`, `networkx`, `filelock`, `fsspec`, `sacremoses`, `sentencepiece`

→ 70개 → 55개, 설치 용량 약 2.5GB 감소

#### main.py 최종 엔드포인트

| 엔드포인트 | 용도 | 변경 |
|---|---|---|
| `GET /` | 헬스체크 | 유지 |
| `POST /pronunciation-score` | 발음 점수 (SDUI 통합, RBAC 적용) | 유지 |
| `POST /analyze` | 한국어 형태소 분석 (konlpy) | 유지 |
| `POST /translate` | 번역 — googletrans (무료) | 유지 |
| `POST /translate1` | 번역 — HuggingFace API | 유지 |
| `POST /tts_blob` | Google Cloud TTS streaming | 유지 |
| `POST /tts_only` | Google Cloud TTS → 파일 URL | 유지 |
| `POST /tts_gtts` | gTTS TTS streaming (무료) | **신규 추가** |
| `POST /translate_and_tts` | 번역 + TTS 복합 | 유지 |
| `GET /test` | 브라우저 테스트 페이지 | 유지 |
| `POST /translate2` | 로컬 torch 모델 번역 | **제거** |
| `POST /translate_only` | /translate 중복 | **제거** |
| `POST /tts_only_test` | /tts_blob 중복 테스트 | **제거** |

#### main.py 추가 수정

- `GOOGLE_APPLICATION_CREDENTIALS` 하드코딩 경로 제거 → `.env` + `load_dotenv()`로 자동 로드
- `from transformers import pipeline` import 제거

---

### FastAPI 서버 기동 명령 (업데이트)

```bash
cd pronounce-api
source venv/Scripts/activate
venv/Scripts/pip.exe install -r requirements.txt   # 최초 1회
uvicorn app.main:app --port 8001 --reload
```

---

### 남은 작업 업데이트

| Phase | 작업 | 상태 |
|-------|------|------|
| FastAPI | `pip install -r requirements.txt` 설치 완료 확인 | **진행 필요** |
| FastAPI | `uvicorn app.main:app --port 8001 --reload` 기동 확인 | **진행 필요** |
| FastAPI | `pronounce-api/Dockerfile` 생성 (Docker 배포 준비) | 미착수 |
| Phase 3-A | `AIInterviewComponent.tsx` 풀 스펙 (이력서 입력, 세션 관리) | 미착수 |
| Phase 3-B | 하드코딩 시스템 프롬프트 → `pageData.systemPromptTemplate` 연동 | 미착수 |
| Phase 5 | Migration V33~V38 정리 + 로컬 Docker 검증 | 미착수 |
| 테스트 | `rendering_optimization`, `auth_security` 실패 원인 수정 | 별도 처리 예정 |



// [메모] AI_JAPANESE_CHAT_PAGE, AI_INTERVIEW_PAGE 테스트

AI_ENGLISH_CHAT_PAGE 에서 
General  Mic 버튼을 누를때 작동하지 않는다. 
오히려 한국어로 말하기 버튼 누를때 prononcprononcation이ation이 작동한다. 
하지만 한국어로 말하기 할때는 prononcation이 작동하면 안된다. 
(왜냐? 한국인이 영어표현이 어려워서 한국어로 말하기로 대화를 대신한거이긴 때문, 영어 발음은 General Mic일때 영어로 말하기 때문에 )


AI_JAPANESE_CHAT_PAGE 에서 
General Mic 일때 화면에 나타나는 text는 일본어야 한다. 
지금은 영어로 나온다.

한국어로 말하기 버튼 누를때 일본어로 번역되어서 일본어가 나와야한다. 현재는 한국어로 말하기 버튼 누르고 한국말로 말하면 한국어가 나온다. 

AI_ENGLISH_CHAT_PAGE 와 같이 prononcation이 General Mic에서 적용이 안되고 한국어로 말하기  에서 적용이 되고 있다. 

AI_INTERVIEW_PAGE 
1. 이미지 업로드와 pdf 업로드 기능이 없다. 현재는 텍스트 입력 만 가능하다. 
2. 면접 시작하기 누르면 AI_ENGLISH_CHAT_PAGE 와 똑같이 보인다. AI-mode는 글씨도 안보인다. 

* AI_면접관_1, AI_면접관_2, 0316_일본어테스트_저녁 캡처이미지를 참고해주세요.

---

## 2026-03-16 야간 세션 — AI Japanese Chat 버그 수정

### 발견된 버그 3가지

모든 버그의 근본 원인: `AIChatComponentV2`에서 `language = 'ko'` (잘못된 fallback)

```
targetLanguage = meta.target_language || data?.language ||
  (actionType.includes('EN') ? 'en' : 'ko')  // ← AI_CHAT_JA는 'ko' fallback
```

| 증상 | 원인 |
|------|------|
| General Mic → "こんばんは"가 "곤방와"로 출력 | `sttLanguage = 'ko'` → Whisper 한국어 모드 |
| 피드백 "Good evening" (영어) | `evaluateExpression(language='ko')` → `langName = "English"` |
| 한국어로 말하기 → "안녕하세요" 그대로 표시 | `wasTranslated = isKorean && ('ko'==='en'||'ko'==='ja')` = false |

### 수정 내용

#### 1. `AIChatComponentV2.tsx` — JA 액션타입 fallback 추가

```typescript
// Before
(actionType.includes('EN') ? 'en' : 'ko')
// After
(actionType.includes('EN') ? 'en' : actionType.includes('JA') ? 'ja' : 'ko')
```

+ AudioRecorder에 `language={targetLanguage}` prop 전달

#### 2. `AudioRecorder.tsx` — General Mic이 타겟 언어 전달

```typescript
// Before: onStart('en') 하드코딩
// After: onStart(language || 'en') — Japanese chat에서 'ja' 전달
// interface: onStart 타입 'en' | 'ko' → string으로 확장
```

#### 3. `OpenAiClientV2.java` — 피드백 언어 분리

```java
// Before: feedback + idealExpression 모두 langName("Japanese") 언어로
// After: feedbackLangName = 'ja' ? "Korean" : "English"
//        idealExpression은 targetLangName("Japanese") 그대로
```

**결과**: 일본어 채팅 → 피드백 한국어 + 추천 표현 일본어 ✅

### DB 변경 없음

기존 `action_type = 'AI_CHAT_JA'` 활용 → 마이그레이션 불필요

### 남은 작업

| Phase | 작업 | 상태 |
|-------|------|------|
| FastAPI | `pronounce-api/Dockerfile` 생성 (Docker 배포 준비) | 미착수 |
| Phase 3-A | `AIInterviewComponent.tsx` 풀 스펙 (이력서 입력, 세션 관리) | 미착수 |
| Phase 3-B | 하드코딩 시스템 프롬프트 → `pageData.systemPromptTemplate` 연동 | 미착수 |
| Phase 5 | Migration V33~V38 정리 + 로컬 Docker 검증 | 미착수 |
| 테스트 | `rendering_optimization`, `auth_security` 실패 원인 수정 | 별도 처리 예정 |


// [메모] AI_JAPANESE_CHAT_PAGE AI번역
AI_JAPANESE_CHAT_PAGE에서 AI번역이 될때도 있고 안될때도 있습니다. 

AI_JAPANESE_CHAT_PAGE_1900.png 를 참고해주세요


테스트 결과: 8 PASS, 2 FAIL (기존 pre-existing 실패). 새 코드 변경 무관.

완료된 작업 요약
이번 세션에서 추가 수정된 내용
1. useAIChatLogic.ts — parsed.ja 키 지원

AI가 {"ja": "日本語", "ko": "..."} 형태로 반환 시 content = parsed.ja 처리
기존: parsed.en || parsed.ko → 신규: parsed.en || parsed.ko || parsed.ja
2. V39__strengthen_japanese_chat_prompt.sql — 시스템 프롬프트 강화

AI가 plain text 반환하는 근본 원인 해소
JSON 예시 2개 추가 + NEVER 키워드로 지시 강화
유저 데이터에서 확인된 현황
항목	상태
General Mic → こんにちは STT 정상 인식	✅ 이전 fix 적용됨
한국어 말하기 → 일본어 번역 ("韓国語のボタンを押しました")	✅ 정상
한국어 모드에서 pronunciation 스킵	✅ 정상
번역 간헐적 실패	✅ 이번 세션 fix
pronunciation 피드백이 일본어 (とても自然な挨拶です)	⚠️ Java 서버 재시작 필요 (코드 fix는 적용됨)
pronunciation 피드백 일본어 → 한국어 전환 방법
Java fix는 이미 OpenAiClientV2.java:223-224 에 적용됨. 서버 재빌드 후 반영됩니다:


cd SDUI-server && ./gradlew bootRun
남은 작업 (메모에서)
AI_INTERVIEW_PAGE: 이미지/PDF 업로드 기능 없음, UI가 English Chat과 동일하게 보임
rendering_optimization, auth_security 테스트 실패 — 별도 브랜치 처리 예정


// [메모] AI_JAPANESE_CHAT_PAGE 추가 기능 구현

스프링부트를 다시 시작했습니다.
AI_JAPANESE_CHAT_PAGE_1930.png 를 참고해주세요.

1. 여전히 AI번역이 보이지 않는 경우가 있습니다.
http://localhost:3000/api/ai/v2/chat/stream
[
    {
        "role": "system",
        "content": "You are a friendly and professional Japanese tutor. Help the user improve their Japanese through natural conversation.\r\n\r\nCRITICAL RULE: You MUST respond with ONLY a single JSON object. No text before or after the JSON.\r\nFormat: {\"en\": \"<your Japanese response in kanji/kana>\", \"ko\": \"<Korean translation>\"}\r\n\r\nExamples:\r\n{\"en\": \"そうですね！とても面白いですね。次は何を話しましょうか？\", \"ko\": \"그렇군요! 매우 흥미롭네요. 다음에는 무엇을 이야기할까요?\"}\r\n{\"en\": \"日本語がお上手ですね！どのくらい勉強していますか？\", \"ko\": \"일본어를 잘 하시네요! 얼마나 공부하셨나요?\"}\r\n\r\nNEVER include any explanation, markdown, or extra text outside the JSON object."
    },
    {
        "role": "user",
        "content": "はじめまして。 ミン・エリンと申します。",
        "audioUrl": "blob:http://localhost:3000/f2b15793-1070-4546-8500-118ccf7b8f73",
        "pronunciationScore": 100,
        "pronunciationFeedback": "아주 자연스럽고 잘 표현했습니다.",
        "pronunciationSpoken": "はじめまして。 ミン・エリンと申します。",
        "pronunciationIdeal": "はじめまして。ミン・エリンと申します。"
    },
    {
        "role": "assistant",
        "content": "はじめまして、ミン・エリンさん！どうぞよろしくお願いします。趣味は何ですか？",
        "translation": "안녕하세요, 민 에린님! 잘 부탁드립니다. 취미는 무엇인가요?"
    },
    {
        "role": "user",
        "content": "バックエンドエンジニアを目指しています。",
        "audioUrl": "blob:http://localhost:3000/f8608acc-aa36-4a5f-8540-5b0f1cfda6fb",
        "pronunciationScore": 100,
        "pronunciationFeedback": "아주 자연스럽고 잘 표현했습니다.",
        "pronunciationSpoken": "バックエンドエンジニアを目指しています。",
        "pronunciationIdeal": "バックエンドエンジニアを目指しています。"
    },
    {
        "role": "assistant",
        "content": "バックエンドエンジニアを目指しているんですね！どのプログラミング言語を勉強していますか？"
    },
    {
        "role": "user",
        "content": "韓国語のボタンを押した。今回はどうなるかな？",
        "audioUrl": "blob:http://localhost:3000/04867ed2-1a4f-45fe-a31a-476b95cf4dda"
    },
    {
        "role": "assistant",
        "content": "今回は韓国語の翻訳を見てみましょう！楽しく会話を続けましょう。最近気になることはありますか？"
    },
    {
        "role": "user",
        "content": "静かにキングアイを整理することが好きです。",
        "audioUrl": "blob:http://localhost:3000/60971f31-6439-4e2a-b6af-f529f960a8ad",
        "pronunciationScore": 70,
        "pronunciationFeedback": "문장이 자연스럽지 않아서 의미가 불분명합니다.",
        "pronunciationSpoken": "静かにキングアイを整理することが好きです。",
        "pronunciationIdeal": "静かにキングアイを整理するのが好きです。"
    },
    {
        "role": "assistant",
        "content": "静かにキングアイを整理するのが好きなんですね！それはリラックスできそうです。どんな方法で整理していますか？"
    },
    {
        "role": "user",
        "content": "ご無沙汰。",
        "audioUrl": "blob:http://localhost:3000/8a1c56e9-31c6-458b-af83-6a6cbca2d71a",
        "pronunciationScore": 90,
        "pronunciationFeedback": "자연스러운 표현이지만, 좀 더 친근하게 말할 수 있습니다.",
        "pronunciationSpoken": "ご無沙汰。",
        "pronunciationIdeal": "ご無沙汰しています。"
    }
]
2. 일본어 페이지는 사람이 말한 내용 옆에 일본어 STT된 부분을
보이면 좋겠습니다. 예시캡처를 참고해주세요. 


1. AI번역 간헐적 실패 — 근본 원인 수정
useAIChatLogic.ts — sendToAI

이전엔 히스토리에서 assistant 메시지를 plain Japanese로 보냈기 때문에 AI가 이전 응답 패턴(plain text)을 따라했습니다.

이제 translation이 있는 assistant 메시지는 {"en": "...", "ko": "..."} 형태로 재구성해서 전송:


[assistant: {"en": "はじめまして...", "ko": "안녕하세요..."}]  ← AI가 JSON 패턴 학습
[user: バックエンド...]
→ AI가 "나는 항상 JSON으로 답한다"는 패턴을 유지

2. 한국어로 말하기 → 원본 한국어 표시
ai.ts: originalText?: string 추가
useAIChatLogic.ts: 번역 모드(wasTranslated = true)일 때 originalText: originalTranscript 저장
ConversationPanelV2.tsx: 사용자 버블에 원본 한국어 표시 (🇰🇷 안녕하세요)
AI_CHAT_V2.css: .ai-original-text 스타일 추가
결과: 한국어로 말하기 버튼 → 버블에 번역된 일본어(main) + 🇰🇷 원본 한국어(sub) 표시
