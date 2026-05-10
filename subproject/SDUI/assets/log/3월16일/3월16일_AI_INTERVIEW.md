# AI_INTERVIEW_PAGE 개발 로그 (2026-03-16)

## 이번 세션 작업 — 이미지/PDF 업로드 기능 추가 (Phase 3-A)

### 작업 배경

AI_INTERVIEW_PAGE에 텍스트 외에 이미지(JPG/PNG)와 PDF 이력서 업로드 기능이 없었음.
이번 세션에서 이력서 입력을 텍스트/이미지/PDF 탭으로 분리하고,
백엔드는 OpenAI Vision API를 통해 이미지·PDF도 처리하도록 수정.

참고 이미지: `AI_면접관_1.png`, `AI_면접관_2.png`, `AI_INTERVIEW_2000.png`

---

## 수정된 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `SDUI-server/.../ai/dto/InterviewStartRequest.java` | `resumeImageBase64`, `resumePdfBase64` 필드 추가 |
| `SDUI-server/.../ai/client/OpenAiClient.java` | `streamChatObjects(List<Map<String, Object>>, ...)` 메서드 추가 — Vision API용 |
| `SDUI-server/.../ai/service/InterviewService.java` | `startInterview` 3-branch 분기, `startInterviewWithVision`, `startInterviewWithText`, `buildSystemPromptBase`, `streamToEmitterObjects` 추가 |
| `metadata-project/lib/hooks/useInterviewLogic.ts` | `resumeImageBase64?`, `resumePdfBase64?` prop 추가, `startInterview`에서 전달, `_mode` unused param 수정 |
| `metadata-project/components/fields/AIInterviewComponent.tsx` | `AIInterviewConfig` 타입 캐스팅, 3개 input 상태 추가, `AIInterviewIntro`에 모든 prop 전달 |
| `metadata-project/components/fields/ai/AIInterviewIntro.tsx` | 전면 재작성 — 탭 UI(텍스트/이미지/PDF) + FileReader 기반 base64 변환 + 미리보기/제거 버튼 |

---

## 백엔드 설계

### InterviewStartRequest.java

```java
public class InterviewStartRequest {
    private String resumeText;          // 텍스트 직접 입력
    private String resumeImageBase64;   // 이미지 — data:image/...;base64,...
    private String resumePdfBase64;     // PDF — data:application/pdf;base64,...
    private String language;            // "en" | "ko"
}
```

### InterviewService.java — 3-branch startInterview

```java
public void startInterview(InterviewStartRequest req, SseEmitter emitter) {
    boolean hasImage = req.getResumeImageBase64() != null && !req.getResumeImageBase64().isBlank();
    boolean hasPdf   = req.getResumePdfBase64()   != null && !req.getResumePdfBase64().isBlank();

    if (hasImage) {
        startInterviewWithVision(req.getLanguage(), req.getResumeImageBase64(), emitter);
    } else if (hasPdf) {
        startInterviewWithVision(req.getLanguage(), req.getResumePdfBase64(), emitter);
    } else {
        startInterviewWithText(req.getLanguage(), req.getResumeText(), emitter);
    }
}
```

우선순위: **이미지 > PDF > 텍스트**

### Vision 모드 — startInterviewWithVision

```java
private void startInterviewWithVision(String language, String imageDataUrl, SseEmitter emitter) {
    String systemPrompt = buildSystemPromptBase(language);
    String promptText = "ko".equals(language)
            ? "이 이력서를 분석하고 면접관으로서 첫 질문을 해주세요."
            : "Analyze this resume and ask your first interview question.";

    List<Map<String, Object>> messages = List.of(
            Map.of("role", "system", "content", (Object) systemPrompt),
            Map.of("role", "user", "content", (Object) List.of(
                    Map.of("type", "text", "text", promptText),
                    Map.of("type", "image_url", "image_url", Map.of("url", imageDataUrl))
            ))
    );
    streamToEmitterObjects(messages, emitter, "면접 시작(Vision)");
}
```

> **PDF도 동일 경로**: `data:application/pdf;base64,...` URL을 `image_url`로 전달 → GPT-4o Vision API가 처리

### OpenAiClient.java — Vision API 지원 추가

기존 `streamChat(List<Map<String, String>>)`는 `content`가 문자열인 경우만 처리.
Vision API는 `content`가 배열(`List<Map<String,Object>>`) → 별도 메서드 추가.

```java
// 신규: Vision 지원 (content가 String 또는 List인 경우)
public void streamChatObjects(
        List<Map<String, Object>> messages,
        Consumer<String> onChunk,
        Runnable onComplete) throws Exception { ... }
```

---

## 프론트엔드 설계

### AIInterviewIntro.tsx — 탭 기반 입력 UI

```typescript
const INPUT_TABS = [
    { type: 'text',  label: '📝 텍스트' },
    { type: 'image', label: '🖼️ 이미지' },
    { type: 'pdf',   label: '📄 PDF' },
];
```

FileReader로 파일 → base64 변환 (추가 npm 패키지 없음):

```typescript
const handleImageFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => onImageChange(reader.result as string);
    reader.readAsDataURL(file);
    e.target.value = '';  // 동일 파일 재선택 허용
};
```

탭별 시작 가능 조건:

```typescript
const isStartable = (() => {
    if (isLoading) return false;
    if (resumeInputType === 'text')  return resumeText.trim().length > 0;
    if (resumeInputType === 'image') return resumeImageBase64 !== null;
    if (resumeInputType === 'pdf')   return resumePdfBase64 !== null;
    return false;
})();
```

### UI 구성 요약

| 탭 | UI |
|----|----|
| 텍스트 | `<textarea>` — 이력서 텍스트 붙여넣기 |
| 이미지 | 파일 선택 버튼 → 이미지 미리보기 + 제거 버튼 |
| PDF | 파일 선택 버튼 → "📄 PDF 파일 선택됨" + 제거 버튼 |

### AIInterviewComponent.tsx — 상태 추가

```typescript
const interviewData = data as (AIInterviewConfig | undefined);

const [resumeInputType, setResumeInputType] = useState<ResumeInputType>('text');
const [resumeImageBase64, setResumeImageBase64] = useState<string | null>(null);
const [resumePdfBase64, setResumePdfBase64] = useState<string | null>(null);
```

### useInterviewLogic.ts — 인터페이스 확장

```typescript
interface UseInterviewLogicProps {
    language: string;
    resumeText: string;
    resumeImageBase64?: string | null;  // 신규
    resumePdfBase64?: string | null;    // 신규
    onError?: (msg: string) => void;
}

// startInterview에서 전달
await stream('/api/ai/interview/start', {
    resumeText,
    resumeImageBase64: resumeImageBase64 || undefined,
    resumePdfBase64: resumePdfBase64 || undefined,
    language,
});
```

---

## 새로 추가된 CSS 클래스 (AI_INTERVIEW.css 스타일 추가 필요)

| 클래스 | 용도 |
|--------|------|
| `.ai-resume-tab-group` | 탭 버튼 그룹 컨테이너 |
| `.ai-resume-tab` | 개별 탭 버튼 |
| `.ai-resume-tab.active` | 선택된 탭 강조 |
| `.ai-resume-upload-area` | 업로드 영역 컨테이너 |
| `.ai-resume-upload-btn` | 파일 선택 트리거 버튼 |
| `.ai-upload-icon` | 업로드 버튼 내 이모지/아이콘 |
| `.ai-upload-hint` | 지원 포맷 안내 텍스트 |
| `.ai-resume-preview` | 이미지 미리보기 래퍼 |
| `.ai-resume-preview-img` | 이미지 미리보기 `<img>` |
| `.ai-resume-remove-btn` | 파일 제거 버튼 |
| `.ai-resume-pdf-selected` | PDF 선택 완료 표시 영역 |
| `.ai-pdf-icon` | PDF 아이콘 이모지 |
| `.ai-pdf-name` | "PDF 파일 선택됨" 텍스트 |

---

## 수정 중 발견·해결된 이슈

| 이슈 | 원인 | 해결 |
|------|------|------|
| `resume_placeholder` 타입 오류 | `data`가 `AIChatConfig` → `AIInterviewConfig` 전용 필드 없음 | `data as (AIInterviewConfig \| undefined)` 캐스팅 |
| `React` 미사용 import 경고 | JSX transform 적용 후 `React` 불필요 | `import { useRef, ChangeEvent } from 'react'`로 교체 |
| InterviewService 초기 수정 거부 | 이미지·텍스트 2-branch만 작성 → PDF 누락 | 3-branch로 전면 재설계 |
| `continueInterview` — 이미지/PDF 모드 컨텍스트 | `buildSystemPrompt(resumeText=null)` → base 프롬프트 반환 | AI가 대화 이력(클라이언트 유지)을 통해 컨텍스트 파악 |

---

## 남은 작업

| 항목 | 상태 |
|------|------|
| `AI_INTERVIEW.css` — 업로드 UI 스타일 추가 (위 클래스 목록) | **미착수** |
| 면접 시작 후 UI 차별화 (Issue 2: English Chat과 동일하게 보이는 문제) | 미착수 |
| AI-mode 레이블 가시성 개선 (Issue 3: 글씨 안 보임) | 미착수 |
| `pronounce-api/Dockerfile` 생성 | 미착수 |
| Phase 3-B — 하드코딩 시스템 프롬프트 → `pageData.systemPromptTemplate` 연동 | 미착수 |
| Phase 5 — Migration V33~V38 정리 + 로컬 Docker 검증 | 미착수 |
| `rendering_optimization`, `auth_security` 테스트 실패 수정 | 별도 브랜치 처리 예정 |


// [메모] 이미지와 PDF 업로드 구현방안 

1. Vision API 와 이미지, PDF 처리하는데 내용 인식을 하는데 있어 더 인식률이 높은 API를 추천/비교해주세요.

2. 이미지/PDF 등 처리할때 DB를 어떻게 관리하는지 생각해야합니다.

3. AWS s3이나 다른 클라우드를 도입하는게 좋을지 판단해주세요.

구현방안을 생각하는데 있어서 필요한 객관식 질문 6개와 주관식 질문 2개를 주세요.

---

## 2026-03-17 세션 — Cloud Storage(S3 + Google Document AI) 연동 완료

### 배경

기존 이력서 업로드 방식(FileReader base64 → HTTP 본문 직접 전송)의 두 가지 문제 해결:
1. PDF를 GPT-4o Vision API의 `image_url`로 전송 시 인식 불가 (Vision API는 PDF 미지원)
2. 대용량 파일이 HTTP 본문에 포함 → 전송 성능 저하, Gateway 크기 제한

**해결책**: S3 선업로드 → fileKey 반환 → 인터뷰 시작 시 fileKey만 전달

---

### 완료된 작업

#### .gitignore 크리덴셜 보호

```
assets/cloud-info.txt
assets/*.csv
assets/*.json
assets/kdeliver-358f601d765c.json
```

#### build.gradle — 의존성 추가

```groovy
implementation platform('software.amazon.awssdk:bom:2.25.0')
implementation 'software.amazon.awssdk:s3'
implementation 'software.amazon.awssdk:auth'
implementation 'com.google.cloud:google-cloud-documentai:2.36.0'
```

#### 신규 파일

| 파일 | 역할 |
|------|------|
| `global/config/S3Config.java` | S3Client + S3Presigner Bean 등록 |
| `domain/ai/service/S3Service.java` | upload / presignedUrl / downloadBytes / detectFileType |
| `domain/ai/client/GoogleDocumentAiClient.java` | PDF → OCR → 텍스트 |
| `V40__create_interview_resume_table.sql` | interview_resume 업로드 이력 테이블 |

#### 수정된 파일

| 파일 | 변경 내용 |
|------|-----------|
| `application-local.yml` | cloud.aws + cloud.gcp 설정 추가 (크리덴셜은 환경변수) |
| `InterviewStartRequest.java` | base64 필드 제거 → `resumeFileKey` 추가 |
| `AiInterviewController.java` | `POST /api/ai/interview/resume/upload` 엔드포인트 추가 |
| `InterviewService.java` | fileKey 기반 3-branch (Vision/PDF OCR/텍스트) |
| `AIInterviewIntro.tsx` | FileReader → API 업로드 방식 전환, uploading/uploadError 상태, Object URL 미리보기 |
| `AIInterviewComponent.tsx` | base64 state 2개 → `resumeFileKey` 1개 |
| `useInterviewLogic.ts` | interface + startInterview 페이로드 `resumeFileKey` 사용 |

#### 이력서 처리 파이프라인 (최종)

```
[이미지 업로드] → S3 → fileKey → generatePresignedUrl(15분) → GPT-4o Vision
[PDF 업로드]   → S3 → fileKey → downloadBytes → Document AI OCR → GPT-4o 텍스트
[텍스트 입력]  → resumeText → GPT-4o 직접 전달
```

---

### 환경변수 설정 (서버 기동 전 필요)

```bash
export AWS_ACCESS_KEY=AKIAT7JJUS7ISVCB5TFE
export AWS_SECRET_KEY=<secret>
export GCP_CREDENTIALS_PATH=assets/kdeliver-358f601d765c.json
```

---

### 남은 작업 (Phase 3-B 완료 후 업데이트됨)

| Phase | 작업 | 상태 |
|-------|------|------|
| Phase 3-B | 하드코딩 시스템 프롬프트 → `pageData.systemPromptTemplate` 연동 | **완료** |
| 테스트 | AI Interview 통합 테스트 (AiInterviewIntegrationTest) 4/4 통과 | **완료** |
| FastAPI | `pronounce-api/Dockerfile` 생성 | 미착수 |
| Phase 5 | Migration V33~V40 정리 + 로컬 Docker 검증 | 미착수 |
| 테스트 | `rendering_optimization`, `auth_security` 실패 수정 | 별도 브랜치 |

---

## 2026-03-17 — Phase 3-B 완료 + 백엔드 테스트 환경 정비

### Phase 3-B: systemPromptTemplate 연동

DB `ui_metadata.system_prompt_template` → 프론트 `meta.system_prompt_template` → 백엔드 `InterviewService.buildSystemPromptBase()` 전체 경로 완성.

#### 수정 파일

| 파일 | 변경 내용 |
|------|-----------|
| `InterviewStartRequest.java` | `systemPromptTemplate` 필드 추가 |
| `InterviewAnswerRequest.java` | `systemPromptTemplate` 필드 추가 |
| `InterviewService.java` | `buildSystemPromptBase(language, template)` — template null 시 상수 fallback |
| `useInterviewLogic.ts` | `systemPromptTemplate` 인터페이스 추가, SSE 페이로드에 포함 |
| `AIInterviewComponent.tsx` | `systemPromptTemplate: meta.system_prompt_template` 훅에 전달 |

#### Fallback 로직

```java
private String buildSystemPromptBase(String language, String template) {
    if (template != null && !template.isBlank()) return template;
    return "ko".equals(language) ? SYSTEM_PROMPT_KO : SYSTEM_PROMPT_EN;
}
```

---

### 백엔드 테스트 환경 정비

#### 1. `build.gradle` 의존성 수정

```groovy
// Before (존재하지 않는 아티팩트)
implementation 'com.google.cloud:google-cloud-documentai:2.36.0'
// After
implementation 'com.google.cloud:google-cloud-document-ai:2.91.0'
```

아티팩트 ID가 `documentai` → `document-ai` (하이픈 포함). Java 패키지명(`com.google.cloud.documentai.v1.*`)은 동일.

#### 2. `application-test.yml` stub 프로퍼티 추가

```yaml
cloud:
  aws:
    credentials:
      access-key: test-access-key
      secret-key: test-secret-key
    s3:
      region: ap-northeast-2
      bucket: test-bucket
  gcp:
    document-ai:
      project-id: test-project-id
      location: us
      processor-id: test-processor-id
      credentials-path: /tmp/test-credentials.json
```

#### 3. `AiInterviewIntegrationTest.java` — Mock 추가

```java
@MockBean
private GoogleDocumentAiClient googleDocumentAiClient;

@MockBean
@Qualifier("sseExecutor")
private Executor sseExecutor;

@BeforeEach
void setUp() {
    // sseExecutor를 동기 실행으로 → verify()가 즉시 동작
    doAnswer(inv -> { ((Runnable) inv.getArgument(0)).run(); return null; })
            .when(sseExecutor).execute(any());
}
```

**원인**: 컨트롤러가 `sseExecutor.execute(() -> interviewService.startInterview(...))` 비동기 실행 → verify() 타이밍 문제. 동기 Mock으로 해결.

#### 4. `AiJapaneseChatMetadataTest.java` — GoogleDocumentAiClient Mock 추가

```java
@MockBean
private GoogleDocumentAiClient googleDocumentAiClient;
```

#### 테스트 결과

| 테스트 | 결과 |
|--------|------|
| `AiInterviewIntegrationTest` 4개 | ✅ PASS |
| `AiJapaneseChatMetadataTest` 1개 | ✅ PASS |
| 전체 백엔드 (`./gradlew test`) | ✅ BUILD SUCCESSFUL |
| 프론트엔드 (`npm run test`) | ✅ 8/10 스위트 통과 (2개 기존 실패는 별도 처리 예정) |