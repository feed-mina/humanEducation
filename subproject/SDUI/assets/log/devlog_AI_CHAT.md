# AI Chat 개발 로그 (통합본)

> 원본: `3월13일/`, `3월14일/`, `3월15일/`, `3월16일/`, `3월17일/` 일자별 파일
> 마지막 수정: 2026-03-19

---

## 2026-03-13 — 초기 로컬 테스트

### 동작 확인
- `GET /api/ui/AI_ENGLISH_CHAT_PAGE` → SSE 스트림 감지 정상
- `POST /api/ai/stt` → Whisper 응답 정상
- `POST /api/ai/chat/stream` → SSE 청크 스트리밍 정상
- `GET /api/v1/user-memberships/current` → PREMIUM 멤버십 RBAC 동작 확인
- `GET /api/execute/AI_ENGLISH_CHAT_CONFIG` → query_master SQL 구성값 정상 반환

### 확인된 API 응답 예시
```json
// /api/execute/AI_ENGLISH_CHAT_CONFIG
{
  "mic_btn_label": "🎤 녹음 시작",
  "language": "en",
  "required_tier": "PREMIUM",
  "welcome_message": "Hello! I'm your English conversation partner..."
}

// /api/ai/stt (한국어 발화 시)
{ "data": { "text": "안녕하세요. 반갑습니다." } }
```

### 발견된 이슈
- STT 결과가 `language: "en"` 임에도 한국어로 인식됨
  → **원인**: `OpenAiClient.java`에서 `language` 필드를 Whisper API에 전달하지 않음
  → **영향**: 영어로 말해도 Whisper가 발화 언어 자동 감지, 한국어 발화→영어 채팅에 그대로 전송

---

## 2026-03-14 — 버그 수정 + AI Chat V2 런칭

### 버그 수정 1: SSE AccessDeniedException

**증상**: `/api/ai/chat/stream` SSE 스트림에서 AccessDeniedException 발생
**원인**: Spring Security가 `DispatcherType.ASYNC`만 허용, `FORWARD`/`ERROR`가 차단됨
**수정**: `SecurityConfig.java`에 DispatcherType.FORWARD, DispatcherType.ERROR 추가
```java
.dispatcherTypeMatchers(DispatcherType.ASYNC, DispatcherType.FORWARD, DispatcherType.ERROR)
  .permitAll()
```

### 버그 수정 2: STT language 파라미터 미전달

**수정**: `OpenAiClient.java`에서 Whisper API 호출 시 `language=null`로 설정 → 자동 감지 활성화
**결과**: 영어 모드에서 영어 발화 → 영어로 인식, 한국어 모드에서 한국어 발화 → 한국어로 인식

### AI Chat V2 (AI_ENGLISH_MODE) 런칭

| 항목 | 내용 |
|------|------|
| CSS | `AI_CHAT_V2.css` — 딥 글래스모피즘, 배경 Orbit 애니메이션 |
| 폰트 | Google Fonts `Outfit` |
| TTS | OpenAI TTS `alloy` 음성 (기존 브라우저 TTS 대체) |
| 대화 스타일 | 시스템 프롬프트: 사용자 발화 길이에 맞춰 짧게 응답 |
| 신규 기능 | [🎧 Play My Voice] — 사용자 녹음 Blob URL 재생 |
| 버튼 | "End Chat" → "채팅종료하기" |
| 컴포넌트 | `AIChatComponentV2.tsx`, `ConversationPanelV2.tsx` (V1 독립 유지) |
| 백엔드 | `AiTtsControllerV2.java` + `SecurityConfig` 추가 허용 |

---

## 2026-03-15 — 페이지 통합 + 듀얼 모드 마이크 + 14/14 테스트 통과

### 페이지 이름 변경
- `AI_ENGLISH_CHAT_PAGE2` → `AI_ENGLISH_CHAT_PAGE` (V32 마이그레이션)
- 목적: V1 테스트 완료 후 V2가 기본 AI 영어 채팅 페이지로 전환

### 듀얼 모드 마이크 구현
- 영어 모드 (`language: "en"`): 영어 STT + GPT-4o 영어 응답
- 한국어 모드 (`language: "ko"`): 한국어 STT + GPT-4o 한국어 응답
- UI: 언어 토글 버튼 → 현재 모드 표시 배지

### AI 일본어 채팅 런칭 (`AI_JAPANESE_CHAT_PAGE`)
- V2 엔진 재사용, 벚꽃 테마 (핑크/분홍 Orbit 배경)
- `language: "ja"` → Whisper 일본어 STT, GPT-4o 일본-한국어 번역 쌍

### 테스트 결과: 14/14 통과
| 범주 | 결과 |
|------|------|
| Backend | 4/4 |
| Frontend | 7/7 |
| E2E (Playwright) | 3/3 |

---

## 2026-03-16 — 9가지 개선 작업 + 프롬프트 외부화

### 9가지 개선 항목
1. 시스템 프롬프트 외부화 → `query_master` / `ui_metadata`에 저장
2. `V34__add_system_prompt_column.sql` 추가 (`system_prompt_template` 컬럼)
3. AI 채팅 설정 컬럼 분리 (welcome_message, language 분리)
4. 대화 히스토리 길이 제한 옵션 추가
5. SSE 타임아웃 핸들링 개선 (30s → 60s + 클라이언트 retry)
6. 마이크 권한 거부 시 텍스트 입력 대체 UI
7. TTS 오디오 큐 관리 개선 (AudioContext 최대 6개 제한)
8. 대화 종료 후 요약 기능 검토 (보류)
9. 발음 피드백 연동 검토 (pronounce-api, 보류)

### Phase 0-N 실행 계획
각 Phase 완료 후 의무적으로 테스트 실행:
```
Phase 0: 프롬프트 외부화 (query_master INSERT)
Phase 1: V34 마이그레이션 (system_prompt_template 컬럼)
Phase 2: 채팅 설정 분리
...
Phase N: 최종 E2E 테스트
```

### 마이그레이션 이슈: V34 시스템 프롬프트 컬럼
- **증상**: `system_prompt_template` 컬럼 없음 오류
- **수정**: `V34__add_system_prompt_column.sql` 추가
- **충돌**: AI_JAPANESE_CHAT_PAGE가 V34 사용 중 → V35로 번호 이동

---

## 2026-03-17 — UI 색상 결정 + CSS 버그 수정

### UI 색상 최종 결정
| 요소 | 결정 |
|------|------|
| 마이크 버튼 | 빨간 그라디언트 (녹음 상태 강조) |
| 언어 배지 | 한국어 모드 `KR` 텍스트 배지 |
| AI 말풍선 | 인디고 계열 (#6366f1) |
| 종료 버튼 | Outline 빨간색 (destructive 스타일) |

### CSS 버그 수정: AI_CHAT_V2.css 미적용

**증상**: `AI_CHAT_V2.css` 스타일이 브라우저에 전혀 반영되지 않음
**원인**: `index.css`에 `@import "AI_CHAT_V2.css"` 구문 누락
**수정**: `globals.css` 또는 `index.css` 최상단에 import 추가
```css
@import "./AI_CHAT_V2.css";
```

---

## 주요 파일 목록

| 파일 | 역할 |
|------|------|
| `AIChatComponentV2.tsx` | V2 AI 채팅 메인 컴포넌트 |
| `ConversationPanelV2.tsx` | V2 대화 말풍선 렌더링 |
| `AI_CHAT_V2.css` | V2 전용 글래스모피즘 스타일 |
| `AiTtsControllerV2.java` | OpenAI TTS alloy 스트리밍 |
| `OpenAiClient.java` | STT/LLM API 클라이언트 |
| `SecurityConfig.java` | DispatcherType 허용 설정 |
| `V34__add_system_prompt_column.sql` | system_prompt_template 컬럼 추가 |
