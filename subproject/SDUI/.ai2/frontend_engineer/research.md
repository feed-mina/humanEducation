# Frontend Engineer Research: SDUI 기반 AI 파이프라인 + 멤버십 구현 분석

> 작성일: 2026-03-11
> 상태: Phase 2 구현 대기

---

## 1. 기존 metadata-project 구조 분석

### 1.1 현재 프로젝트 경로 및 기술스택

- 경로: `SDUI/metadata-project/`
- Next.js App Router (TypeScript)
- React 19, Tailwind CSS v4
- 테스트: Jest + React Testing Library + MSW + Playwright

### 1.2 SDUI 엔진 핵심 파일

| 파일 | 역할 |
|------|------|
| `components/DynamicEngine/DynamicEngine.tsx` | 메타데이터 트리 순회 → 컴포넌트 렌더링 |
| `components/constants/componentMap.tsx` | `component_type` 문자열 → React 컴포넌트 매핑 |
| `components/DynamicEngine/hook/usePageHook.tsx` | action_type → useBusinessActions 또는 useUserActions 디스패치 |
| `components/providers/MetadataProvider.tsx` | `/api/ui/{screenId}` 메타데이터 fetch |
| `app/view/[...slug]/page.tsx` | 모든 화면 처리 (CommonPage) |

### 1.3 재사용 가능한 기존 패턴

| 패턴 | 위치 | 재사용 방법 |
|------|------|------------|
| componentMap 등록 | `componentMap.tsx` | `AI_CHAT`, `AI_INTERVIEW` 추가 |
| useBusinessActions | `hook/useBusinessActions.tsx` | `AI_CHAT_SUBMIT`, `AI_INTERVIEW_START` 추가 |
| API 호출 패턴 | `lib/api.ts` 또는 각 훅 | `/api/ai/**` 엔드포인트 추가 |
| JWT 인증 | 기존 auth context | `Authorization: Bearer {jwt}` 헤더 포함 |
| SDUI 화면 등록 | `ui_metadata` + `screenMap.ts` | 신규 screenId 등록 |

---

## 2. 신규 컴포넌트 설계

### 2.1 추가할 컴포넌트 목록

```
metadata-project/components/fields/
├── AIChatComponent.tsx        # AI 영어/한국어 대화 컴포넌트
└── AIInterviewComponent.tsx   # AI 면접관 컴포넌트
```

### 2.2 AIChatComponent 구조

```typescript
// 내부 구성:
// - ConversationPanel (대화 메시지 목록 + SSE 스트리밍 표시)
// - AudioRecorder (마이크 녹음 + 답변완료 버튼)
// - Waveform (AudioContext 기반 실시간 시각화)

type Props = {
  meta: UiMetadata;    // SDUI 메타데이터 (language, css_class 등)
  pageData?: Record<string, unknown>;
};
```

**AI 대화 UX 흐름:**
```
1. 화면 진입 → JWT로 멤버십 권한 체크
2. AI 첫 메시지 자동 표시 (텍스트)
3. 마이크 버튼 클릭 → MediaRecorder 시작 + Waveform 시각화
4. VAD 무음 감지 or 답변완료 버튼 → 녹음 중지
5. POST /api/ai/stt → 유저 텍스트 획득 → 채팅창에 표시
6. POST /api/ai/chat (SSE) → AI 응답 텍스트 청크 단위로 채팅창에 스트리밍
7. [TTS 없음 — 텍스트만 표시]
8. 사이클 반복
```

### 2.3 AIInterviewComponent 구조

```typescript
// 내부 구성:
// - ResumeUploader (PDF 텍스트 추출 or 직접 붙여넣기)
// - InterviewPanel (면접 Q&A 텍스트 표시)
// - AudioRecorder (마이크 녹음 — AIChatComponent와 공유)

type Props = {
  meta: UiMetadata;
  pageData?: Record<string, unknown>;
};
```

**AI 면접관 UX 흐름:**
```
1. 이력서 입력 (PDF 업로드 or 텍스트 붙여넣기)
2. "면접 시작" 버튼 → POST /api/ai/interview/start (SSE)
   → AI 첫 질문이 채팅창에 스트리밍 표시
3. 마이크로 답변 녹음 → POST /api/ai/stt → 텍스트 변환
4. POST /api/ai/interview/answer (SSE) → AI 후속 질문 스트리밍
5. 사이클 반복
```

---

## 3. SSE 스트리밍 소비 패턴

### 3.1 EventSource 대신 fetch + ReadableStream

```typescript
// EventSource는 GET 전용 + 커스텀 헤더 불가
// → fetch + ReadableStream 사용 (POST body + JWT 헤더 지원)

async function streamChat(
  messages: ChatMessage[],
  onChunk: (chunk: string) => void,
  onComplete: () => void
) {
  const response = await fetch('/api/ai/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${getJwt()}`,
    },
    body: JSON.stringify({ messages, language: 'en' }),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const lines = decoder.decode(value).split('\n');
    for (const line of lines) {
      if (line.startsWith('data: ') && !line.includes('[DONE]')) {
        const data = JSON.parse(line.slice(6));
        if (data.chunk) onChunk(data.chunk);
      }
    }
  }
  onComplete();
}
```

### 3.2 컴포넌트 언마운트 시 cleanup (메모리 누수 방지)
```typescript
useEffect(() => {
  const controller = new AbortController();
  // fetch에 signal: controller.signal 전달
  return () => controller.abort(); // cleanup
}, []);
```

---

## 4. Web Audio API 패턴

### 4.1 AudioContext 단일 인스턴스 (브라우저 최대 6개 제한)
```typescript
const audioContextRef = useRef<AudioContext | null>(null);

function getAudioContext() {
  if (!audioContextRef.current) {
    audioContextRef.current = new AudioContext();
  }
  return audioContextRef.current;
}

// 컴포넌트 언마운트 시
useEffect(() => {
  return () => {
    audioContextRef.current?.close();
    audioContextRef.current = null;
  };
}, []);
```

### 4.2 MediaRecorder → STT 흐름
```typescript
const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
const chunks: Blob[] = [];

mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
mediaRecorder.onstop = async () => {
  const audioBlob = new Blob(chunks, { type: 'audio/webm' });
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');

  const res = await fetch('/api/ai/stt', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${getJwt()}` },
    body: formData,
  });
  const { data } = await res.json();
  // data.text → 채팅창에 표시
};
```

---

## 5. componentMap 등록 방법

### 5.1 `componentMap.tsx` 수정
```typescript
// 기존 패턴 (componentMap.tsx)
import AIChatComponent from '../fields/AIChatComponent';
import AIInterviewComponent from '../fields/AIInterviewComponent';

export const componentMap = {
  // ... 기존 컴포넌트들 ...
  AI_CHAT: AIChatComponent,
  AI_INTERVIEW: AIInterviewComponent,
};
```

### 5.2 `ui_metadata` DB 등록
```sql
-- AI 영어 대화 화면
INSERT INTO ui_metadata (screen_id, component_type, label_text, css_class, allowed_roles)
VALUES ('AI_ENGLISH_CHAT_PAGE', 'AI_CHAT', '', 'ai-chat-container', 'ROLE_USER');

-- AI 한국어 대화 화면
INSERT INTO ui_metadata (screen_id, component_type, label_text, css_class, allowed_roles)
VALUES ('AI_KOREAN_CHAT_PAGE', 'AI_CHAT', '', 'ai-chat-container', 'ROLE_USER');

-- AI 면접관 화면
INSERT INTO ui_metadata (screen_id, component_type, label_text, css_class, allowed_roles)
VALUES ('AI_INTERVIEW_PAGE', 'AI_INTERVIEW', '', 'ai-interview-container', 'ROLE_USER');
```

### 5.3 `screenMap.ts` 추가
```typescript
export const SCREEN_MAP: Record<string, string> = {
  // ... 기존 ...
  'ai/english': 'AI_ENGLISH_CHAT_PAGE',
  'ai/korean':  'AI_KOREAN_CHAT_PAGE',
  'ai/interview': 'AI_INTERVIEW_PAGE',
};
```

---

## 6. useBusinessActions 추가 액션

```typescript
// hook/useBusinessActions.tsx에 추가
case 'AI_CHAT_SUBMIT':
  // 답변완료 버튼 → 녹음 중지 + STT 호출 트리거
  break;

case 'AI_INTERVIEW_START':
  // 면접 시작 → 이력서 텍스트 전송 + SSE 스트림 시작
  break;

case 'AI_RESUME_UPLOAD':
  // PDF 이력서 업로드 처리
  break;
```

---

## 7. 멤버십 컴포넌트 (Phase 1)

### 7.1 SDUI 어드민 화면으로 구현

멤버십 관리는 기존 SDUI 패턴 그대로 활용:
- `ADMIN_MEMBERSHIP_PAGE` — ui_metadata로 구성된 어드민 화면
- 기존 INPUT, BUTTON, TEXT 컴포넌트로 충분히 구현 가능
- 별도 커스텀 컴포넌트 불필요

### 7.2 멤버십 현황 표시

사용자 홈 화면에 현재 멤버십 상태 표시:
- `GET /api/v1/user-memberships/current` 호출
- 기존 TEXT, BUTTON 컴포넌트로 표시 (SDUI 메타데이터)

---

## 8. 버전 및 환경 정보

| 항목 | 버전 | 비고 |
|------|------|------|
| Next.js | 현재 설치 버전 | App Router 사용 |
| React | 19.x | Server Components 주의 |
| Tailwind CSS | v4 | `@import "tailwindcss"` 방식 |
| Jest | 30.x | |
| MSW | v2 | `http` (not `rest`) API |
| TypeScript | 5.x | strict mode |

---

## 9. 설계 결정 사항

### 9.1 AI 응답 표시 방식
- **결정**: TTS 없음 — 텍스트 채팅창에만 표시
- **이유**: 사용자 명시적 요청 ("채팅창에 텍스트로 보이기")
- **영향**: TTS 관련 코드(useTTS, AudioContext 재생) 불필요

### 9.2 언어 모드 분리
- 영어 전용 대화 모드: `AI_ENGLISH_CHAT_PAGE`
- 한국어 전용 대화 모드: `AI_KOREAN_CHAT_PAGE`
- `language` 파라미터를 API 요청에 포함 (`"en"` | `"ko"`)

### 9.3 컴포넌트 위치
- `components/fields/` — 기존 SDUI fields와 동일한 위치
- DynamicEngine의 componentMap에 등록하면 자동으로 렌더링

---

## 10. 구현 순서 (Phase 2)

```
1. lib/types.ts — Membership, UserMembership, ChatMessage, InterviewSession 타입 추가
2. AIChatComponent.tsx — 기본 UI 구조 (메시지 목록 + 입력 영역)
3. AudioRecorder (내부 또는 분리 컴포넌트) — MediaRecorder + Waveform
4. SSE 스트리밍 훅/함수 — fetch + ReadableStream 패턴
5. STT 호출 연결 — audio → text → 채팅창
6. AIInterviewComponent.tsx — 이력서 입력 + 면접 Q&A 패널
7. componentMap.tsx 등록
8. ui_metadata SQL (Flyway V13 또는 seed)
9. screenMap.ts 등록
10. useBusinessActions 추가
11. 테스트: Jest (컴포넌트) + MSW (API 모킹)
```

---

## 11. 분석 히스토리

| 날짜 | 분석 내용 | 결론 |
|------|-----------|------|
| 2026-03-11 | 링글 과제 ringle-frontend → metadata-project SDUI 전환 | componentMap 커스텀 컴포넌트 방식으로 통합 가능 |
| 2026-03-11 | SSE 소비 방식 결정 (EventSource vs fetch) | fetch + ReadableStream 채택 (POST + JWT 지원) |
| 2026-03-11 | AI 응답 방식 결정 | TTS 없음, 텍스트 채팅창만 표시 |
| 2026-03-11 | 언어 모드 결정 | 영어/한국어 별도 screenId로 분리 |
