# Planner — Plan

> 이 파일은 신규 기능/화면 기획 계획을 기록한다.
> 사용자의 명시적 승인("YES") 후에만 담당자에게 배포한다.
> plan.md에 남긴 인라인 메모(수정, 제약)는 최우선 반영한다.

---

## Plan 작성 템플릿

새로운 기능/화면 기획은 아래 형식으로 작성한다.

```markdown
## [기능/화면 이름] 기획 — {날짜}

### 목적
- 이 기능이 해결하는 사용자 문제
- 비즈니스 가치

### 화면 정의서

| 항목 | 내용 |
|------|------|
| screen_id | NEW_SCREEN |
| URL 경로 | /NEW_SCREEN |
| 인증 필요 | Yes / No |
| 접근 역할 | ROLE_USER, ROLE_ADMIN |

### 컴포넌트 구성표

| 순서 | component_type | component_id | group_id | parent_group_id | ref_data_id | action_type | label_text | css_class |
|------|----------------|--------------|----------|-----------------|-------------|-------------|------------|-----------|
| 1    | GROUP          | root-group   | root-group | null           | -           | -           | -          | main-card |
| 2    | INPUT          | name-input   | -        | root-group      | userName    | -           | 이름       | input-base |
| 3    | BUTTON         | submit-btn   | -        | root-group      | -           | SUBMIT      | 저장       | btn-primary |

### 사용자 플로우

```
[이전 화면] → [NEW_SCREEN] → [다음 화면]
  action_type=LINK          action_type=SUBMIT

인터랙션 시나리오:
1. 사용자가 폼 작성
2. SUBMIT 버튼 클릭 → POST /api/{endpoint}
3. 성공 → ROUTE to [다음 화면]
4. 실패 → 에러 메시지 표시 (MODAL)
```

### 데이터 요구사항

| 항목 | 내용 |
|------|------|
| 데이터 소스 | query_master.sql_key = 'NEW_QUERY' |
| AUTO_FETCH | Yes / No |
| 예상 응답 구조 | `{ id, name, ... }[]` |
| 페이지네이션 | Yes (pageSize=5) / No |

### 신규 component_type 필요 여부
- [ ] 불필요 (기존 타입으로 해결 가능)
- [ ] 필요 → architect 에스컬레이션: [새 타입 설명]

### 신규 action_type 필요 여부
- [ ] 불필요
- [ ] 필요 → frontend_engineer 협의: [새 액션 설명]

### 담당자 배포

- Architect: 스키마/패턴 검토
- Designer: UI 레이아웃 설계
- Backend Engineer: query_master 쿼리, API 엔드포인트
- Frontend Engineer: 신규 component_type (해당 시), 액션 핸들러
- QA Engineer: 테스트 시나리오

### 승인 상태
[ ] 사용자 승인 대기 중
[x] 사용자 승인 완료 (날짜: ...)
[ ] 배포 완료
```

---

## 현재 계획 없음

아직 작성된 기획 계획이 없습니다. 요청이 오면 위 템플릿으로 작성합니다.