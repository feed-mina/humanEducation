# Architect — Plan

> 이 파일은 아키텍처 설계 계획을 기록한다.
> 사용자의 명시적 승인("YES") 후에만 담당 엔지니어에게 구현을 위임한다.
> 인라인 메모(수정 사항, 제약 조건)를 이 파일에 남기면 즉시 반영한다.

---

## Plan 작성 템플릿

새로운 아키텍처 변경 계획은 아래 형식으로 작성한다.

```markdown
## [계획 제목] — {날짜}

### 배경
- 왜 이 변경이 필요한가?
- 어떤 문제를 해결하는가?

### 접근 방식

#### Option A: [방식 이름]
- 설명
- 영향 파일:
  - `SDUI-server/src/.../TargetClass.java`
  - `metadata-project/components/.../TargetComponent.tsx`
- 핵심 스니펫:
  ```java / typescript
  // 변경 방향을 보여주는 코드 예시
  ```
- 트레이드오프:
  - 장점: ...
  - 단점: ...

#### Option B: [방식 이름]
- ...

### 권장안
Option A를 권장한다. 이유: ...

### 담당자 지정
- Backend: backend_engineer
- Frontend: frontend_engineer
- QA: qa_engineer

### 승인 상태
[ ] 사용자 승인 대기 중
[x] 사용자 승인 완료 (날짜: ...)
[ ] 구현 완료
```

---

## 현재 계획 없음

아직 작성된 계획이 없습니다. 요청이 오면 위 템플릿으로 작성합니다.