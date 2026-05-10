# Phase 4 패치 가이드: 멤버십 게이팅

Phase 1~3이 모두 배포된 후 적용합니다.

---

## 1. 파일 교체 요약

Phase 4는 **기존 파일 교체**만 필요합니다. 신규 파일은 없습니다.

| 교체 대상 | 소스 | 변경 내용 |
|----------|------|---------|
| `domain/ai/controller/AiChatController.java` | `impl/phase4/controller/AiChatController.java` | `canConverse()` 권한 체크 추가 |
| `domain/ai/controller/AiInterviewController.java` | `impl/phase4/controller/AiInterviewController.java` | `canLearn()` 권한 체크 추가 |

---

## 2. 멤버십 권한 매핑

| AI 기능 | 필요 멤버십 권한 | 베이직 | 프리미엄 |
|---------|---------------|--------|---------|
| 면접 (`/api/ai/interview/**`) | `canLearn` | ✅ | ✅ |
| 자유 대화 (`/api/ai/chat/stream`) | `canConverse` | ❌ | ✅ |

---

## 3. 에러 SSE 포맷

권한 없을 때 SSE 스트림으로 에러 반환:
```
data: {"error":"면접 기능은 베이직 이상 멤버십이 필요합니다."}
```

프론트엔드에서 `chunk` 대신 `error` 키 존재 여부로 에러 처리 가능:
```typescript
// SSE 이벤트 파싱 예시
const data = JSON.parse(event.data);
if (data.error) {
  showMembershipUpgradeModal(data.error);
  return;
}
if (data.sessionId) {
  setSessionId(data.sessionId);  // 면접 시작 시 세션 저장
  return;
}
if (data.chunk) {
  appendText(data.chunk);
}
```

---

## 4. STT 게이팅 (선택)

STT(`/api/ai/stt`)는 현재 인증만 요구. 멤버십 게이팅 추가 원할 시:
```java
// AiSttController.java에 추가
if (!userMembershipService.canLearn(userId)) {
    return ResponseEntity.status(HttpStatus.FORBIDDEN)
        .body(ApiResponse.error("STT 기능은 베이직 이상 멤버십이 필요합니다."));
}
```

---

## 5. 전체 구현 완료 체크리스트

### Phase 2 ✅
- [x] `AsyncConfig.java`
- [x] `OpenAiClient.java`
- [x] DTO: `ChatMessage`, `ChatRequest`, `SttResponse`, `InterviewStartRequest`
- [x] `SttService`, `ChatService`, `InterviewService` (Phase 2 버전)
- [x] `AiSttController`, `AiChatController`, `AiInterviewController` (Phase 2 버전)
- [ ] SecurityConfig 패치 (`/api/ai/**` 추가) → phase2/PATCHES.md
- [ ] `application.yml` openai 설정 추가 → phase2/PATCHES.md

### Phase 1 ✅
- [x] `V22__create_memberships.sql`
- [x] `V23__create_user_memberships.sql`
- [x] `Membership`, `MembershipRepository`
- [x] `UserMembership`, `UserMembershipRepository`
- [x] `MembershipService`, `UserMembershipService`
- [x] `MembershipController`, `UserMembershipController`
- [x] DTOs
- [ ] SecurityConfig 패치 (`/api/v1/**` 추가) → phase1/PATCHES.md

### Phase 3 ✅
- [x] `InterviewSession.java`
- [x] `InterviewSessionService.java`
- [x] `InterviewAnswerRequest.java` (sessionId 버전)
- [x] `InterviewService.java` (Redis 세션 버전)
- [x] `AiInterviewController.java` (Phase 3 버전)

### Phase 4 ✅
- [x] `AiChatController.java` (canConverse 게이팅)
- [x] `AiInterviewController.java` (canLearn 게이팅)
