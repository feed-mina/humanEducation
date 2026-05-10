# AI 면접(AI-Interview) 버전 개발 보고서 (2026-03-15)

기존 AI 영어 채팅 V2의 강력한 엔진과 컴포넌트를 재활용하여, 전문적인 면접 준비가 가능한 **AI 면접(AI-Interview)** 버전을 새롭게 구축하였습니다. 기존 파일은 전혀 수정하지 않고 독립적인 컴포넌트와 스타일로 구현되었습니다.

## 🚀 주요 특징 (Key Features)

### 1. 전문적인 비주얼 테마 (Professional UI)
- **전용 스타일시트**: `AI_INTERVIEW.css`를 생성하여 신뢰감을 주는 **슬레이트 네이비(Slate Navy)** 테마를 적용했습니다.
- **아이콘 교체**: 친근한 로봇 대신 정장과 서류를 연상시키는 전문적인 **면접 아이콘([InterviewIcon.tsx](file:///c:/Users/Samsung/Documents/Development/Personal_Projects/2026/SDUI/metadata-project/components/assets/icons/ai/InterviewIcon.tsx))**을 적용했습니다.
- **포멀한 디자인**: 버튼의 굴곡을 줄여(8px radius) 더 날카롭고 격식 있는 느낌을 주었습니다.

### 2. 전문 면접관 페르소나 (Recruiter Persona)
- **시스템 프롬프트 강화**: AI가 시니어 전문 면접관으로 동작하도록 설정했습니다.
- **질의응답 구조화**: 사용자의 답변을 평가하고, 한 번에 하나의 구조화된 질문을 던지는 면접 프로세스를 구현했습니다.
- **격식 있는 톤**: 대화의 톤을 포멀하게 유지하여 실제 면접과 유사한 환경을 제공합니다.

### 3. 독립적/안정적 아키텍처 (Standalone Architecture)
- **코드 재사용**: `useAIChatLogic.ts`, `ConversationPanelV2.tsx`, `AudioRecorder.tsx`를 기능 블록으로 그대로 재사용하여 안정성을 확보했습니다.
- **충돌 방지**: `.ai-interview-theme`라는 전용 클래스 스코프를 사용하여 기존 채팅 테마와 스타일이 겹치지 않도록 설계했습니다.

---

## 📝 워크스루 (Walkthrough)

### 1. 구현 상세
- **인트로 화면**: [AIInterviewIntro.tsx](file:///c:/Users/Samsung/Documents/Development/Personal_Projects/2026/SDUI/metadata-project/components/fields/ai/AIInterviewIntro.tsx)를 통해 "전문 AI 채용 담당자 평가"라는 문구와 공식 면접 아이콘을 노출합니다.
- **메인 컴포넌트**: [AIInterviewComponent.tsx](file:///c:/Users/Samsung/Documents/Development/Personal_Projects/2026/SDUI/metadata-project/components/fields/AIInterviewComponent.tsx)가 전체 면접 흐름을 관리하며 전용 프롬프트를 주입합니다.

### 2. 검증 결과
- **로직 호환성 테스트**: 기존 `useAIChatLogic` 테스트를 신규 페르소나 환경에서 실행하여 100% 통과(4/4)를 확인했습니다.
- **UI 일관성**: 다크 모드 및 반응형 환경에서도 면접 테마의 색상과 폰트가 일관되게 적용됨을 확인했습니다.

## 🔗 접속 경로 (Access Path)

새롭게 구축된 AI 면접 화면은 아래 경로를 통해 로컬 환경에서 즉시 확인하실 수 있습니다.

- **URL**: `http://localhost:3000/view/AI_INTERVIEW_PAGE`
- **참고**: 화면이 정상적으로 출력되지 않을 경우, 백엔드 서버를 재시작하여 `V33` 데이터베이스 마이그레이션이 적용되었는지 확인해 주세요.

---

이로써 동일한 SDUI 기반 위에서 목적이 다른 두 가지 프리미엄 AI 서비스(영어 채팅, AI 면접)를 성공적으로 구현하였습니다.

---

## 2026-03-16 추가 작업 — Phase 3-A: 이력서 파일 업로드 기능

### 작업 개요

기존 텍스트 직접 입력만 지원하던 이력서 입력 방식을 **텍스트 / 이미지 / PDF** 3가지 탭으로 확장하였습니다.

### 수정된 파일

| 파일 | 변경 내용 |
|------|-----------|
| `InterviewStartRequest.java` | `resumeImageBase64`, `resumePdfBase64` 필드 추가 |
| `OpenAiClient.java` | `streamChatObjects(List<Map<String, Object>>, ...)` 메서드 추가 — Vision API용 멀티모달 지원 |
| `InterviewService.java` | `startInterview` 3-branch 분기, `startInterviewWithVision`, `startInterviewWithText`, `buildSystemPromptBase`, `streamToEmitterObjects` 추가 |
| `useInterviewLogic.ts` | `resumeImageBase64?`, `resumePdfBase64?` prop 추가, `startInterview`에서 전달 |
| `AIInterviewComponent.tsx` | `resumeInputType` / `resumeImageBase64` / `resumePdfBase64` state 추가, `AIInterviewIntro`에 모든 prop 전달 |
| `AIInterviewIntro.tsx` | 전면 재작성 — 탭 UI (텍스트/이미지/PDF) + FileReader 기반 base64 변환 + 이미지 미리보기/제거 버튼 |
| `AI_INTERVIEW.css` | 업로드 UI 스타일 13개 클래스 추가 |

### 백엔드 분기 로직 (InterviewService)

```
이미지 base64 존재  →  Vision API (GPT-4o multimodal)
PDF base64 존재     →  Vision API (동일 경로, 현재는 임시)
텍스트만 존재       →  텍스트 프롬프트 경로
```

우선순위: **이미지 > PDF > 텍스트**

### 프론트엔드 탭 구성

| 탭 | 입력 방식 | 시작 조건 |
|----|-----------|-----------|
| 📝 텍스트 | `<textarea>` 직접 입력 | 텍스트 1자 이상 |
| 🖼️ 이미지 | JPG/PNG/WEBP 파일 선택 → base64 변환 | 이미지 파일 선택됨 |
| 📄 PDF | PDF 파일 선택 → base64 변환 | PDF 파일 선택됨 |

---

## 2026-03-16 야간 — Phase A-B: 클라우드 스토리지 아키텍처 결정

### 결정된 아키텍처

기존 base64 인라인 전송 방식의 한계(PDF Vision API 미지원, 대용량 HTTP 전송)를 해결하기 위해 클라우드 스토리지 기반 아키텍처로 전환을 결정하였습니다.

| 항목 | 결정 |
|------|------|
| 스토리지 | AWS S3 (추후 Cloudflare R2 전환 예정) |
| 파일 보존 기간 | 7일 후 S3 Lifecycle 자동 삭제 |
| 파일 수 제한 | 사용자당 1개 (최신 덮어쓰기) |
| 최대 파일 크기 | 5MB |
| 보안 | S3 SSE(서버 측 암호화) 적용 |
| PDF 처리 | Google Document AI OCR → 텍스트 추출 → GPT-4o 텍스트 프롬프트 (이중 파이프라인) |
| 이미지 처리 | S3 → Presigned URL → GPT-4o Vision API |

### 클라우드 인프라 설정 완료

**AWS S3**
- 버킷명: `sdui-273354627025-ap-northeast-2-an`
- 리전: `ap-northeast-2` (서울)

**GCP Document AI**
- 프로세서 이름: SDUI
- 프로세서 ID: `6ed87cfefab39a91`
- 프로세서 유형: Document OCR
- 리전: `us`
- 프로젝트 ID: `kdeliver` / 프로젝트 번호: `88124038673`
- 서비스 계정: `document-ai-server@kdeliver.iam.gserviceaccount.com`
- 서비스 계정 키: `assets/kdeliver-358f601d765c.json`

### 목표 아키텍처 흐름

```
[프론트엔드]
  파일 선택
    └─ POST /api/ai/interview/resume/upload  (multipart)
         └─ S3 업로드 → S3 key 반환

  면접 시작 클릭
    └─ POST /api/ai/interview/start  { resumeFileKey, language }
         ├─ 이미지: S3 presigned URL → GPT-4o Vision API → SSE 스트리밍
         └─ PDF:   S3 다운로드 → Google Document AI → 텍스트 → GPT-4o → SSE 스트리밍
```

### 남은 구현 작업

| Phase | 작업 | 상태 |
|-------|------|------|
| A | `build.gradle` AWS SDK v2 의존성 추가 | **진행 필요** |
| A | `S3Config.java` / `S3Service.java` 작성 | **진행 필요** |
| A | `POST /api/ai/interview/resume/upload` 엔드포인트 구현 | **진행 필요** |
| A | `AIInterviewIntro.tsx` — FileReader → 업로드 API 호출로 교체 | **진행 필요** |
| B | `build.gradle` Google Document AI 의존성 추가 | **진행 필요** |
| B | `GoogleDocumentAiClient.java` 작성 | **진행 필요** |
| B | `InterviewService.java` PDF 경로 → Document AI 파이프라인으로 교체 | **진행 필요** |
| C | Migration V39 — `interview_resume` 테이블 생성 | **진행 필요** |
