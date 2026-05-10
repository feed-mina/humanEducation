# AI 면접관 개발 로그 (통합본)

> 원본: `3월15일/AI-Interview.md`, `3월16일/3월16일_AI_INTERVIEW.md`, `3월17일/03월17일AI_INTERVIEW.md`
> 마지막 수정: 2026-03-19

---

## 2026-03-15 — AI 면접관 V1 런칭

### 설계 방향
- AI 영어 채팅 (`useAIChatLogic` 훅) 재사용
- 슬레이트 네이비 테마 (전문적 면접 분위기)
- 시니어 면접관 페르소나 시스템 프롬프트
- 화면: `AI_INTERVIEW_PAGE` (screenId)

### V1 기본 플로우
```
이력서 텍스트 입력 → GPT-4o 질문 생성 → 사용자 음성 답변 (STT)
→ AI 꼬리질문 (SSE 스트리밍) → 대화 종료 → 피드백 요약
```

### 컴포넌트 구조
| 컴포넌트 | 역할 |
|----------|------|
| `AIInterviewComponent.tsx` | 면접 메인 컴포넌트 |
| `AIInterviewIntro.tsx` | 이력서 입력 + 면접 시작 탭 UI |
| `useAIChatLogic.tsx` | AI 채팅 로직 재사용 (공유 훅) |

---

## 2026-03-16 — Phase 3-A: 이미지/PDF 업로드

### 신규 기능: 이력서 업로드 탭
- **텍스트 탭**: 이력서 직접 입력 (기존 V1)
- **이미지 탭**: 이력서 이미지 업로드 → OpenAI Vision API 분석
- **PDF 탭**: PDF 업로드 → 텍스트 추출 → GPT 분석

### 백엔드 변경사항

#### `InterviewStartRequest.java` 필드 추가
```java
String resumeText;    // 텍스트 탭
String imageBase64;   // 이미지 탭 (Base64 인코딩)
String pdfText;       // PDF 탭 (추출된 텍스트)
String inputType;     // "text" | "image" | "pdf"
```

#### `OpenAiClient.java` — Vision API 지원
```java
// streamChatObjects() 메서드 — 이미지 포함 멀티모달 요청
List<Object> contentParts = new ArrayList<>();
contentParts.add(Map.of("type", "text", "text", prompt));
contentParts.add(Map.of("type", "image_url", "image_url",
    Map.of("url", "data:image/jpeg;base64," + imageBase64)));
```

#### `InterviewService.java` — 3분기 처리
```java
if ("image".equals(inputType)) {
    // Vision API로 이력서 분석
} else if ("pdf".equals(inputType)) {
    // 추출된 텍스트로 분석
} else {
    // 텍스트 직접 분석
}
```

### 프론트엔드: `AIInterviewIntro.tsx` 탭 UI
- 탭 1 (텍스트): textarea 입력
- 탭 2 (이미지): `<input type="file" accept="image/*">` → Base64 변환
- 탭 3 (PDF): `<input type="file" accept=".pdf">` → 서버 업로드

---

## 2026-03-17 — S3/PDF 이슈 발생

### 이슈 1: S3 업로드 500 오류

**증상**: PDF 업로드 시 `MaxUploadSizeExceededException` 500 반환
**원인**: Spring Boot 기본 파일 업로드 크기 제한 (1MB)
**수정 필요**:
```yaml
# application.yml
spring:
  servlet:
    multipart:
      max-file-size: 10MB
      max-request-size: 10MB
```

### 이슈 2: S3 업로드 403 오류

**증상**: AWS S3 PutObject 403 Forbidden
**원인**: IAM 역할에 S3 접근 권한 미부여
**수정 필요**: EC2 IAM 역할에 `s3:PutObject` 정책 추가

### 이슈 3: PDF 업로드 Connection Abort

**증상**: 대용량 PDF (>5MB) 업로드 시 연결 중단
**원인**: Nginx 프록시 타임아웃 (기본 60s)
**수정 필요**:
```nginx
proxy_read_timeout 120s;
client_max_body_size 20m;
```

### Google Document AI OCR — 미구현 (보류)

이력서 이미지에서 텍스트 추출을 위한 OCR 기능:
- **계획**: Google Document AI API 연동
- **현황**: 구현 보류 (OpenAI Vision으로 임시 대체)
- **이유**: API 비용 및 설정 복잡도

---

## 현재 상태 및 미해결 이슈

| 기능 | 상태 | 비고 |
|------|------|------|
| V1 텍스트 면접 | ✅ 완료 | 기본 SSE 스트리밍 동작 |
| 이미지 업로드 Vision | ✅ 완료 | Base64 → Vision API |
| PDF 업로드 | ⚠️ 부분 완료 | 크기 제한/403/타임아웃 미해결 |
| Google OCR | ❌ 보류 | Vision API로 임시 대체 |
| S3 영구 저장 | ❌ 미구현 | 이력서 재사용 기능 추후 |
| 면접 결과 저장 | ❌ 미구현 | `interview_resume` 테이블 준비됨 |

---

## 주요 파일 목록

| 파일 | 역할 |
|------|------|
| `AIInterviewComponent.tsx` | 면접 메인 컴포넌트 |
| `AIInterviewIntro.tsx` | 이력서 입력 탭 UI |
| `InterviewService.java` | 면접 로직 (3분기 처리) |
| `InterviewStartRequest.java` | 요청 DTO |
| `OpenAiClient.java` | Vision API 멀티모달 지원 |
| `V26__add_interview_resume.sql` | interview_resume 테이블 |
