# Frontend Engineer Plan: AI 파이프라인 + 멤버십 화면 구현

> 작성일: 2026-03-11
> 최종 수정: 2026-03-13
> 근거: architect/plan.md + frontend_engineer/research.md
> 상태: **Phase 1 컴포넌트 구현 완료 / componentMap 등록 및 DB SQL 미완료**

---

## 사용자 확정 결정사항

| 항목 | 결정 |
|------|------|
| 멤버십 없을 때 처리 | 업그레이드 모달 표시 → MEMBERSHIP_SHOP_PAGE 이동 |
| 구현 순서 | AI 채팅 먼저 → AI 면접관 → 멤버십 화면 |
| 이력서 입력 방식 | 텍스트 붙여넣기 + PDF 업로드 + **이미지 업로드** |
| AI 이미지 분석 | GPT-4o Vision API (이미지 → 이력서 내용 직접 분석) |
| **메타데이터 활용** | **버튼 라벨·환영 메시지·언어 모드·필요 멤버십 모두 DB에서 관리** |

---

## SDUI 메타데이터 최대 활용 전략

### SDUI 엔진 데이터 흐름 (실제 코드 기반)

```
ui_metadata (DB)
  └── DATA_SOURCE 행 (component_type='DATA_SOURCE', action_type='AUTO_FETCH', data_sql_key='ai_en_config')
  └── AI_CHAT 행 (ref_data_id='ai_en_config')

usePageMetadata
  └── DATA_SOURCE 감지 → GET /api/execute/ai_en_config
  └── 결과 → pageData['ai_en_config'] = { mic_btn_label, submit_btn_label, welcome_message, language, ... }

DynamicEngine
  └── getComponentData(node) → node.ref_data_id='ai_en_config' → data = pageData['ai_en_config']

AIChatComponent (props)
  ├── meta.labelText      → 화면 제목 ("AI 영어 대화")
  ├── meta.cssClass       → 컨테이너 CSS 클래스 ("ai-chat-en")
  ├── meta.actionType     → 언어/모드 ("AI_CHAT_EN" | "AI_CHAT_KO")
  ├── meta.placeholder    → 전사 텍스트 placeholder
  ├── meta.isReadonly     → 상호작용 비활성화 여부
  └── data (= pageData['ai_en_config'])
        ├── data.mic_btn_label      → "🎤 녹음 시작"
        ├── data.submit_btn_label   → "답변완료"
        ├── data.end_btn_label      → "대화 종료"
        ├── data.welcome_message    → "Hello! I'm your English conversation partner."
        ├── data.language           → "en"
        └── data.required_tier      → "PREMIUM" (모달 표시용 멤버십 이름)
```

### query_master SQL 설계

```sql
-- sql_key: 'ai_english_chat_config'
SELECT
  '🎤 녹음 시작'   AS mic_btn_label,
  '답변완료'        AS submit_btn_label,
  '대화 종료'       AS end_btn_label,
  'Hello! I''m your English conversation partner. What would you like to practice today?' AS welcome_message,
  'en'              AS language,
  'PREMIUM'         AS required_tier,
  '음성 대화 기능은 프리미엄 멤버십이 필요합니다.' AS upgrade_message;

-- sql_key: 'ai_korean_chat_config'
SELECT
  '🎤 녹음 시작'   AS mic_btn_label,
  '답변완료'        AS submit_btn_label,
  '대화 종료'       AS end_btn_label,
  '안녕하세요! 한국어 대화 연습을 도와드리겠습니다. 무엇을 연습하고 싶으신가요?' AS welcome_message,
  'ko'              AS language,
  'PREMIUM'         AS required_tier,
  '음성 대화 기능은 프리미엄 멤버십이 필요합니다.' AS upgrade_message;

-- sql_key: 'ai_interview_config'
SELECT
  '이력서를 붙여넣거나 파일을 업로드하세요...' AS resume_placeholder,
  '면접 시작'   AS start_btn_label,
  '답변 완료'   AS answer_btn_label,
  '🎤 답변 녹음' AS mic_btn_label,
  '면접 종료'   AS end_btn_label,
  'ko'          AS language,
  'BASIC'       AS required_tier,
  '면접 기능은 베이직 이상 멤버십이 필요합니다.' AS upgrade_message;
```

### ui_metadata 행 설계 (AI_ENGLISH_CHAT_PAGE 예시)

```sql
-- 행 1: 데이터 소스 (query_master 자동 호출)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, action_type, data_sql_key, sort_order)
VALUES
  ('AI_ENGLISH_CHAT_PAGE', 'ai_en_config', '', 'DATA_SOURCE', 'AUTO_FETCH', 'ai_english_chat_config', 0);

-- 행 2: AI 채팅 컴포넌트 (ref_data_id로 위 데이터 소스 참조)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, action_type, ref_data_id, css_class, allowed_roles, sort_order)
VALUES
  ('AI_ENGLISH_CHAT_PAGE', 'ai_en_chat', 'AI 영어 대화', 'AI_CHAT', 'AI_CHAT_EN', 'ai_en_config', 'ai-chat-en', 'ROLE_USER', 1);
```

### 컴포넌트 내 메타데이터 읽기 패턴

```typescript
// AIChatComponent.tsx 내부
const language = data?.language ?? (meta.actionType?.includes('EN') ? 'en' : 'ko');
const title = meta.labelText || meta.label_text || 'AI 대화';
const containerClass = meta.cssClass || meta.css_class || '';
const isDisabled = meta.isReadonly === true || meta.is_readonly === true;
const transcriptPlaceholder = meta.placeholder ?? '음성을 인식하면 텍스트가 표시됩니다.';

// query_master 데이터 (data prop)
const micBtnLabel = data?.mic_btn_label ?? '🎤 녹음 시작';
const submitBtnLabel = data?.submit_btn_label ?? '답변완료';
const endBtnLabel = data?.end_btn_label ?? '대화 종료';
const welcomeMessage = data?.welcome_message ?? '';
const requiredTier = data?.required_tier ?? 'PREMIUM';
const upgradeMessage = data?.upgrade_message ?? '이 기능은 멤버십이 필요합니다.';
```

### 하드코딩 vs 메타데이터 분류표

| 항목 | 방식 | 근거 |
|------|------|------|
| 화면 제목 | `meta.labelText` | DB 변경만으로 수정 가능 |
| 버튼 라벨 | `data.mic_btn_label` 등 (query_master) | 다국어/A-B 테스트 가능 |
| 환영 메시지 | `data.welcome_message` (query_master) | 기획자가 직접 변경 가능 |
| 언어 모드 | `meta.actionType` → 코드에서 파싱 | UI 구조 변경 없이 언어 전환 |
| 필요 멤버십 이름 | `data.required_tier` (query_master) | 플랜명 변경 시 코드 불필요 |
| 업그레이드 메시지 | `data.upgrade_message` (query_master) | 마케팅 문구 코드 밖에서 관리 |
| SSE 파싱 로직 | 컴포넌트 코드 (하드코딩 유지) | 프로토콜은 변경 빈도 낮음 |
| API 엔드포인트 경로 | 컴포넌트 코드 (하드코딩 유지) | 백엔드 계약 사항, 프론트 단독 변경 불가 |

---

## 화면 목록 (5개)

| screen_id | 방식 | 설명 |
|-----------|------|------|
| `AI_ENGLISH_CHAT_PAGE` | 커스텀 컴포넌트 | 영어 AI 대화 |
| `AI_KOREAN_CHAT_PAGE` | 커스텀 컴포넌트 | 한국어 AI 대화 |
| `AI_INTERVIEW_PAGE` | 커스텀 컴포넌트 | AI 면접관 (이미지/PDF/텍스트 이력서) |
| `MEMBERSHIP_SHOP_PAGE` | SDUI 메타데이터 | 멤버십 목록 + 구매 버튼 |
| `ADMIN_MEMBERSHIP_PAGE` | SDUI 메타데이터 | 어드민 멤버십 관리 |

---

## Phase 1: AI 채팅 (`AI_ENGLISH_CHAT_PAGE`, `AI_KOREAN_CHAT_PAGE`)

### 컴포넌트 구조

```
AIChatComponent.tsx          ← componentMap에 'AI_CHAT'으로 등록
  ├── MembershipGate.tsx      ← canConverse 체크 → 업그레이드 모달
  ├── ConversationPanel.tsx   ← 메시지 목록 + SSE 스트리밍
  ├── AudioRecorder.tsx       ← MediaRecorder + 마이크 권한 처리
  └── Waveform.tsx            ← AudioContext 기반 실시간 시각화
```

### 화면 레이아웃

```
┌────────────────────────────────────────┐
│  🤖 AI English Conversation            │  ← label_text from SDUI metadata
├────────────────────────────────────────┤
│                                        │
│  [AI]  Hello! I'm your English...  💬  │
│  [나]  I've been working at...     👤  │  ← ConversationPanel
│  [AI]  Great! Can you tell me...   💬  │    메시지 버블
│        ████ (스트리밍 중...)           │    스트리밍 시 마지막 줄 타이핑 애니메이션
│                                        │
├────────────────────────────────────────┤
│  ●  🎤  [==~~==~~==~~==~~]  [답변완료]│  ← AudioRecorder + Waveform
└────────────────────────────────────────┘
```

### 핵심 로직 흐름

```
1. 진입 시: GET /api/v1/user-memberships/current
   → canConverse=false → MembershipUpgradeModal 표시
   → canConverse=true  → AI 첫 메시지 자동 전송 (POST /api/ai/chat/stream)

2. 마이크 클릭 → MediaRecorder 시작 + AudioContext Waveform 시작
3. 답변완료 클릭 → MediaRecorder 중지 → Blob → POST /api/ai/stt
4. STT 응답 → 채팅창에 [나] 메시지 추가
5. POST /api/ai/chat/stream (SSE) → [AI] 메시지 스트리밍
   → SSE error 키 감지 시: MembershipUpgradeModal 표시
```

### 타입 정의

```typescript
// lib/types/ai.ts
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  language: 'en' | 'ko';
}

export type RecordingState = 'idle' | 'recording' | 'processing';
export type ConversationState = 'idle' | 'ai_speaking' | 'user_turn' | 'processing';
```

---

## Phase 2: AI 면접관 (`AI_INTERVIEW_PAGE`)

### 컴포넌트 구조

```
AIInterviewComponent.tsx     ← componentMap에 'AI_INTERVIEW'으로 등록
  ├── MembershipGate.tsx      ← canLearn 체크 (재사용)
  ├── ResumeUploader.tsx      ← 3가지 입력 방식 탭
  │     ├── 텍스트 붙여넣기   ← TextArea
  │     ├── PDF 업로드        ← input[type=file] accept=".pdf" + PDF.js 텍스트 추출
  │     └── 이미지 업로드     ← input[type=file] accept="image/*" → base64 변환 → Vision API
  ├── ConversationPanel.tsx   ← 재사용 (Phase 1과 동일)
  └── AudioRecorder.tsx       ← 재사용 (Phase 1과 동일)
```

### 이력서 입력 방식별 API 처리

| 입력 방식 | 프론트 처리 | 백엔드 처리 |
|-----------|-------------|-------------|
| 텍스트 | 그대로 전달 | `req.resumeText` 사용 |
| PDF | PDF.js로 텍스트 추출 → 텍스트로 전달 | `req.resumeText` 사용 |
| 이미지 | base64 인코딩 → `resumeImageBase64` 필드로 전달 | GPT-4o Vision API (⚠️ 백엔드 추가 필요) |

> ⚠️ 이미지 업로드는 백엔드 `InterviewService.startInterview()` 수정 필요:
> - `InterviewStartRequest`에 `resumeImageBase64` 필드 추가
> - GPT-4o 메시지에 image_url content type 추가

### 화면 레이아웃

```
[면접 시작 전]
┌────────────────────────────────────────┐
│  📋 AI 면접관                          │
├────────────────────────────────────────┤
│  [텍스트] [PDF] [이미지]  ← 탭 선택   │  ← ResumeUploader
│  ┌──────────────────────┐              │
│  │ 이력서를 붙여넣거나  │              │
│  │ 파일을 업로드하세요  │              │
│  └──────────────────────┘              │
│                      [면접 시작]       │
└────────────────────────────────────────┘

[면접 진행 중]
┌────────────────────────────────────────┐
│  📋 AI 면접관                          │
├────────────────────────────────────────┤
│  [AI]  자기소개 부탁드립니다.      💬  │
│  [나]  저는 3년차 백엔드...        👤  │
│  [AI]  구체적인 프로젝트 경험...   💬  │
├────────────────────────────────────────┤
│  🎤  [답변 녹음]                       │
└────────────────────────────────────────┘
```

---

## Phase 3: 멤버십 화면 (SDUI 메타데이터 전용)

### MEMBERSHIP_SHOP_PAGE

```sql
-- 순수 SDUI 컴포넌트 (커스텀 컴포넌트 불필요)
-- TEXT: 멤버십 이름 + 가격 표시
-- BUTTON: 구매하기 (action_type = 'MEMBERSHIP_PURCHASE')
-- IMAGE: 멤버십 배지/아이콘
```

### ADMIN_MEMBERSHIP_PAGE → USER_LIST 확장으로 대체 (사용자 확정)

- **별도 ADMIN_MEMBERSHIP_PAGE 화면 생성하지 않음**
- 기존 USER_LIST 페이지의 `AdminUserTable` + `useAdminUsers`에 멤버십 관리 기능 추가
- 변경 내용:
  - 백엔드: `GET /api/admin/users` 응답에 `membershipTier` 필드 추가
  - `AdminUserTable`: "현재 멤버십" 컬럼 + 멤버십 할당 드롭다운 + 변경 버튼 추가
  - `useAdminUsers`: `handleMembershipChange` 핸들러 추가 (`POST /api/v1/user-memberships`)
- V26 SQL: MEMBERSHIP_SHOP_PAGE 메타데이터만 (ADMIN 화면 SQL 불필요)

---

## 공유 컴포넌트: MembershipGate + MembershipUpgradeModal

```typescript
// SSE error 감지 시 또는 진입 권한 체크 시 공통 사용
interface MembershipUpgradeModalProps {
  isOpen: boolean;
  message: string;           // SSE의 error 메시지 or 직접 입력
  onClose: () => void;
  onUpgrade: () => void;     // MEMBERSHIP_SHOP_PAGE 이동
}
```

---

## 변경 파일 목록

### 신규 파일 (`metadata-project/components/fields/ai/`)

| 파일 | 설명 |
|------|------|
| `ConversationPanel.tsx` | 대화 메시지 목록 + SSE 스트리밍 |
| `AudioRecorder.tsx` | MediaRecorder + 마이크 권한 |
| `Waveform.tsx` | AudioContext 실시간 시각화 |
| `MembershipGate.tsx` | 멤버십 체크 + 업그레이드 모달 |
| `ResumeUploader.tsx` | 텍스트/PDF/이미지 입력 (Phase 2) |

### 신규 파일 (`metadata-project/components/fields/`)

| 파일 | 설명 |
|------|------|
| `AIChatComponent.tsx` | AI 대화 메인 컴포넌트 |
| `AIInterviewComponent.tsx` | AI 면접관 메인 컴포넌트 |

### 신규 파일 (`metadata-project/lib/`)

| 파일 | 설명 |
|------|------|
| `types/ai.ts` | AI 관련 타입 정의 |
| `hooks/useSSEStream.ts` | SSE fetch + ReadableStream 훅 |
| `hooks/useAudioRecorder.ts` | MediaRecorder 훅 |

### 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `components/constants/componentMap.tsx` | `AI_CHAT`, `AI_INTERVIEW` 등록 |
| `components/constants/screenMap.ts` | AI/멤버십 화면 URL 등록 |
| `app/view/[...slug]/page.tsx` | `PROTECTED_SCREENS`에 AI 화면 추가 |

### DB SQL (`.ai2/` 내 작성 후 Flyway 적용)

| 파일 | 설명 |
|------|------|
| `V26__ai_chat_pages.sql` | AI_ENGLISH_CHAT_PAGE, AI_KOREAN_CHAT_PAGE 메타데이터 ← 실제 버전 V26 |
| `V27__ai_interview_page.sql` | AI_INTERVIEW_PAGE 메타데이터 |
| `V28__create_memberships.sql` | memberships 테이블 생성 |
| `V29__create_user_memberships.sql` | user_memberships 테이블 생성 |
| `V30__membership_pages.sql` | MEMBERSHIP_SHOP_PAGE 메타데이터 |

---

## 의존성 추가 필요 (사전 승인 필요)

| 패키지 | 용도 | 필요 Phase |
|--------|------|-----------|
| `pdfjs-dist` | PDF → 텍스트 추출 | Phase 2 (이력서 PDF 업로드) |

---

## 구현 순서 (TODO)

### Phase 1: AI 채팅

```
[x] 1.  lib/types/ai.ts — ChatMessage, ChatRequest, ConversationState, AIChatConfig 타입
[x] 2.  lib/hooks/useSSEStream.ts — fetch + ReadableStream SSE 소비 훅
[x] 3.  lib/hooks/useAudioRecorder.ts — MediaRecorder + Blob 생성
[x] 4.  components/fields/ai/Waveform.tsx — AudioContext 시각화
[x] 5.  components/fields/ai/ConversationPanel.tsx — 메시지 목록 + 스트리밍
[x] 6.  components/fields/ai/AudioRecorder.tsx — 마이크 UI
[x] 7.  components/fields/ai/MembershipUpgradeModal.tsx — 업그레이드 모달
[x] 8.  components/fields/AIChatComponent.tsx — meta + data로 완전 구성
[ ] 9.  componentMap.tsx 등록 (AI_CHAT) ← **다음 단계**
[ ] 10. screenMap.ts 등록 (/ai/english, /ai/korean) ← **다음 단계**
[ ] 11. V26__ai_chat_pages.sql — ui_metadata + query_master (실제 버전 V26 사용)
[ ] 12. npm run test
```

### Phase 2: AI 면접관

```
[ ] 13. lib/types/ai.ts — InterviewStartRequest, ResumeInputType, AIInterviewConfig 타입 추가
[ ] 14. components/fields/ai/ResumeUploader.tsx — 텍스트/PDF/이미지 입력 (탭 UI)
[ ] 15. components/fields/AIInterviewComponent.tsx — meta + data로 완전 구성
[ ] 16. componentMap.tsx 등록 (AI_INTERVIEW)
[ ] 17. screenMap.ts 등록 (/ai/interview)
[ ] 18. V25__ai_interview_page.sql — ui_metadata + query_master (ai_interview_config)
[ ] 19. ⚠️ 백엔드 패치: InterviewStartRequest에 resumeImageBase64 필드 추가 (이미지 업로드)
[ ] 20. npm run test
```

### Phase 3: 멤버십 화면

```
[ ] 21. V26__membership_pages.sql — MEMBERSHIP_SHOP_PAGE 메타데이터만
[ ] 22. screenMap.ts 등록 (/membership)
[ ] 23. useAdminUsers.ts 확장 — membershipTier 필드 + handleMembershipChange 핸들러
[ ] 24. AdminUserTable.tsx 확장 — "현재 멤버십" 컬럼 + 멤버십 할당 UI
[ ] 25. ⚠️ 백엔드 패치: GET /api/admin/users 응답에 membershipTier 추가
[ ] 26. npm run test
```

---

## 트레이드오프

| 결정 | 선택 | 이유 |
|------|------|------|
| SSE 소비 방식 | fetch + ReadableStream (EventSource 아님) | POST body + JWT 헤더 지원 필수 |
| PDF 처리 위치 | 클라이언트 (pdfjs-dist) | 텍스트 추출 후 기존 API 그대로 사용 가능 |
| 이미지 처리 방식 | base64 → 백엔드 Vision API 위임 | 클라이언트에서 OCR 금지 |
| AudioContext 관리 | useRef 단일 인스턴스 | 브라우저 최대 6개 제한 |
| 멤버십 체크 위치 | 컴포넌트 진입 시 + SSE error 수신 시 | 이중 방어 |
