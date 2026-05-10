// [메모] UI 고도화


3월 17일 폴더의 캡쳐이미지를 참고해주세요.
C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI\assets\log\3월16일 폴더에 
AI_CHAT_1, AI_CHAT_2, AI_CHAT_3, AI_CHAT_4, AI_CHAT_5 이미지를 참고해주세요. 

Asset으로 icon을 외부로 뺀 모듈이 이전에 있었습니다. 그것을 다시 가져오시고 사용해주세요. 
우선 UI고도화를 이런식으로 진행 후에 

후순위로 AI_INTERVIEW 수정 작업을 하겠습니다. 


// [메모] UI 미반영 및 추가 반영건

AI-CHAT-변경필요 이미지를 참고해주세요

v2 , SDUI v2 글씨 지웁니다.
UI가 예전과 같습니다.

// [메모] UI  추가 반영건
AI-CHAT-반영희망 이미지를 참고해주세요.
 UI 미반영 및 추가 반영건은 반영이 잘 되었습니다.

 과거에 AI-CHAT-반영희망 이미지처럼 구현했던 적이 있었습니다. 과거에 사용한 css나 모듈이 있는지 찾고 있다면 사용해주세요. 

 AI_CHAT UI 수정 후에 후순위로 AI_INTERVIEW 수정 작업을 하겠습니다.

---

// [세션 기록] 2026-03-17 — AI_CHAT UI 고도화 + CSS 버그 수정

## 완료된 작업

### 1. UI 디자인 결정 (6개 질문 응답)
| 항목 | 결정 |
|------|------|
| General Mic 버튼 색상 | 진한 빨강/로즈 그라디언트 `linear-gradient(135deg, #E53935, #B71C1C)` |
| Korean 버튼 표시 | "KR"(크게) + "KOREAN"(작게) 텍스트 배지, 플래그 이모지 제거 |
| 인트로 마이크 아이콘 | `🎤` 이모지 → `MicIcon` SVG (`color="#6366F1"`) |
| 유저 버블 스타일 | 인디고 그라디언트 유지 (현재 그대로) |
| AI 버블 스타일 | `#EEF2FF` (연한 인디고 틴트) |
| 채팅종료 버튼 | 아웃라인 레드 (border/color: `#EF4444`, background: transparent) |

### 2. 코드 수정 내역

#### `AI_CHAT_V2.css`
- `.ai-mic-btn-circle`: `#E8EAF6` → 빨강 그라디언트, shadow 변경
- `.ko-mic-circle` 추가: 흰 배경 + 네이비 테두리 (outlined)
- `.ai-ko-flag` → `.ai-ko-kr`: KR 텍스트 배지 스타일
- `.assistant-bubble .ai-message-body`: `var(--v2-white)` → `#EEF2FF`
- `.ai-session-end-btn`: 아웃라인 레드로 변경

#### `AudioRecorder.tsx`
- `<MicIcon />` → `<MicIcon color="white" />` (빨간 배경에 흰 아이콘)
- `<span className="ai-ko-flag">🇰🇷</span>` → `<span className="ai-ko-kr">KR</span>`

#### `AIChatIntro.tsx`
- `import MicIcon` 추가
- `🎤` 이모지 → `<MicIcon width="24px" height="24px" color="#6366F1" />`

#### `AIChatComponentV2.tsx`
- V2 타이틀 제거: `rawTitle.replace(/\s*[Vv]2\s*$/, '').trim()`
- JA 액션타입 fallback 추가: `actionType.includes('JA') ? 'ja' : 'ko'`
- `language` prop을 `ConversationPanelV2`, `AudioRecorder`에 전달

#### `AIChatHeader.tsx`
- "Live · SDUI V2" 태그 제거
- 게이지 바 UI 유지

---

## CSS 미적용 버그 수정 (핵심)

### 원인
`AI_CHAT_V2.css` 내부 4번 라인에 있던:
```css
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
```
이 Tailwind CSS v4 (Lightning CSS) 환경에서 로컬 파일 내 외부 URL import를 처리할 때 **파일 전체를 무효화**시킴.
CSS 스펙 상 `@import` 는 모든 규칙보다 먼저 와야 하는데, `globals.css`가 `AI_CHAT.css`를 먼저 인라인 처리한 후 `AI_CHAT_V2.css`를 처리하면서 규칙 순서 위반 발생.

### 수정
- `AI_CHAT_V2.css` 에서 Google Fonts `@import url()` 제거
- `globals.css` 최상단으로 이동 (모든 import 중 1번째 라인)

```css
/* globals.css */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
@import "tailwindcss";
@import "./styles/AI_CHAT.css";
@import "./styles/AI_CHAT_V2.css";
...
```

### 해결 방법 (서버 재시작 필수)
```bash
cd metadata-project
rm -rf .next
npm run dev
```

---

## 미완료 작업 (후순위)
- AI_INTERVIEW UI 개선: 한국어 텍스트, 이미지/PDF/텍스트 입력 영역, 버튼 SDUI 필드 분리
- FastAPI Dockerfile 생성
- Migration V33~V39 커밋 정리
- 테스트 실패 수정 (rendering_optimization, auth_security) — 별도 브랜치

---

// [세션 기록] 2026-03-17 (2차) — CSS 미적용 근본 원인 해결 + JA Voice 기능 추가

## CSS 완전 미적용 근본 원인 수정

### 문제
`layout.tsx`가 `globals.css`가 아닌 `./styles/index.css`를 import하고 있었음.
AI CSS 4개 파일은 `globals.css`에만 있었기 때문에 앱에 전혀 로드되지 않음.

```
layout.tsx → ./styles/index.css (AI CSS 미포함)
globals.css → 앱 어디에도 import되지 않음 → AI 스타일 전혀 미적용
```

### 수정
`metadata-project/app/styles/index.css`에 AI CSS 파일 4개 추가:
```css
@import "./AI_CHAT.css";
@import "./AI_CHAT_V2.css";
@import "./AI_INTERVIEW.css";
@import "./AI_JAPANESE.css";
```

---

## 신규 기능: Play JA Voice 버튼

### 목적
일본어 채팅(AI_JAPANESE_CHAT_PAGE)에서 한국어 모드로 말했을 때,
번역된 일본어 텍스트를 TTS로 들을 수 있는 버튼 추가.

### 수정 파일: `ConversationPanelV2.tsx`
- `playingJaIndex` state + `jaAudioRef` 추가
- `handlePlayJA(text, index)` 함수 추가 — `/api/ai/v2/tts` TTS 호출
- 유저 버블 액션바에 버튼 추가:
  - 조건: `language === 'ja'` && `msg.originalText` 존재 (KR 모드 메시지만)
  - 재생 텍스트: `msg.content` (일본어 번역본)
  - `Play My Voice`와 독립적으로 재생/정지 관리

---

## 신규 기능: 추천 표현 듣기 버튼

### 목적
발음 평가 배지의 "추천 표현" 텍스트 옆에 TTS 듣기 버튼 추가.

### 수정 파일: `ConversationPanelV2.tsx`
- `playingIdealIndex` state + `idealAudioRef` 추가
- `handlePlayIdeal(text, index)` 함수 추가
- `msg.pronunciationIdeal` 옆에 `🔊 듣기 / Stop` 버튼 추가

### 수정 파일: `AI_CHAT_V2.css`
```css
.pronunciation-expected { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.pronunciation-listen-btn { padding: 2px 8px; font-size: 0.75rem; gap: 3px; }
```

---

## 신규 기능: Play EN Voice 버튼

### 목적
영어 채팅(AI_ENGLISH_CHAT_PAGE)에서 한국어 모드로 말했을 때,
번역된 영어 텍스트를 TTS로 들을 수 있는 버튼 추가. (Play JA Voice와 동일한 패턴)

### 수정 파일: `ConversationPanelV2.tsx`
- `playingEnIndex` state + `enAudioRef` 추가
- `handlePlayEN(text, index)` 함수 추가 — `/api/ai/v2/tts` TTS 호출
- 유저 버블 액션바에 버튼 추가:
  - 조건: `language === 'en'` && `msg.originalText` 존재 (KR 모드 메시지만)
  - 재생 텍스트: `msg.content` (영어 번역본)
  - `Play JA Voice` 버튼 바로 아래에 위치
