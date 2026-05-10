# QA Engineer — Research

> 이 파일은 QA 분석 결과를 기록한다. 테스트 전략 수립의 근거가 된다.

---

## 현재 테스트 환경 (2026-02-28 기준)

### 테스트 스택

| 도구 | 버전 | 용도 |
|------|------|------|
| Jest | 29.7.0 | 단위/통합 테스트 러너 |
| React Testing Library | 16.3.2 | React 컴포넌트 테스팅 |
| MSW (Mock Service Worker) | 2.7.0 | API 모킹 |
| @swc/jest | 0.2.37 | TypeScript 변환 (빠른 빌드) |
| ts-jest | 29.2.5 | TypeScript 타입 지원 |
| Playwright | 1.50.0 | E2E 브라우저 테스팅 |

### 설정 파일

| 파일 | 역할 |
|------|------|
| `jest.config.js` | Jest 설정 (모듈 별칭, transform, coverage) |
| `jest.setup.js` | @testing-library/jest-dom 설정 |
| `playwright.config.ts` | E2E 브라우저/기기 설정 |

---

## 현재 테스트 파일 목록

| 파일 경로 | 테스트 유형 | 검증 내용 |
|----------|-----------|---------|
| `tests/api_duplicated.test.tsx` | 통합 | 동일 화면에서 메타데이터 API 2회 이상 호출 방지 |
| `tests/rendering_optimization.test.tsx` | 통합/성능 | withRenderTrack 렌더 카운트 |
| `tests/components/TimeSelect.test.tsx` | 단위 | TimeSelect 컴포넌트 동작 |
| `tests/auth.setup.ts` | 설정 | 인증 MSW 핸들러 (로그인 모킹) |
| `tests/test-utils.tsx` | 유틸 | 커스텀 render (모든 Provider 포함) |
| `tests/TestLogger.ts` | 유틸 | HTML 리포트 생성기 |

### 테스트 명령어

```bash
npm run test                          # 전체 Jest 테스트
npx jest tests/path/to/file.test.tsx  # 단일 파일
npx playwright test                   # E2E 전체
```

### 리포트 위치 //[매모] 백앤드는 SUDI-server/tests/logs/backend-report.html , 프론트는 metadata-project/tests/logs/frontend-report.html
- `tests/logs/frontend-report.html`

---

## SDUI 특화 테스트 시나리오

### DynamicEngine 핵심 테스트 케이스

#### 1. componentMap 매핑 검증 
```typescript
// 모든 component_type이 componentMap에서 올바른 컴포넌트를 반환하는가?
// 알 수 없는 component_type은 에러 없이 스킵되는가?
```

#### 2. 데이터 바인딩 우선순위 검증  
```typescript
// 조건: formData[refId] 있음 → formData 값 사용
// 조건: formData[refId] 없음, pageData[refId] 있음 → pageData 값 사용
// 조건: 둘 다 없음 → undefined/empty string (에러 아님)
```

#### 3. Repeater(배열 렌더링) 검증
```typescript
// ref_data_id 있는 group + pageData[refId] = ['a','b','c']
// → 3개의 자식 컴포넌트 렌더링
// → pageData[refId] = [] → 자식 0개 (에러 아님)
// → pageData[refId] = null → 렌더링 스킵 (에러 아님)
```

#### 4. 그룹 레이아웃 검증
```typescript
// group_direction=ROW → className에 'flex-row-layout' 포함
// group_direction=COLUMN → className에 'flex-col-layout' 포함
```

#### 5. 엣지 케이스
```typescript
// 빈 metadata 배열 → 빈 DOM (에러 없음)
// 중복 componentId → 첫 번째 것 사용 (경고 로그)
// 고아 노드 (parent 없음) → 루트 노드로 렌더링
// MODAL 컴포넌트 → activeModal === componentId 일 때만 표시
```

---

## MSW 모킹 시나리오

### API 모킹 경로

```typescript
// 메타데이터 API
GET /api/ui/{screenId} → { data: mockMetadataTree }

// 인증 API
POST /api/auth/login → { data: { accessToken, user } }
POST /api/auth/login (실패) → { code: 'AUTH_001', message: '...' }
POST /api/auth/refresh → { data: { accessToken } }
POST /api/auth/refresh (실패) → 401

// 데이터 API
POST /api/execute/{sqlKey} → { data: mockPageData }
POST /api/execute/{sqlKey} (빈 결과) → { data: [] }

// 콘텐츠 API
POST /api/diary → { data: { diaryId: 1 } }
```

---

## 커버리지 갭 분석

### 현재 미테스트 영역 (확인 필요)

| 영역 | 우선순위 | 설명 |
|------|---------|------|
| useUserActions (LOGIN_SUBMIT) | High | 로그인 성공/실패 시나리오 |
| useUserActions (REGISTER_SUBMIT) | High | 회원가입 + 이메일 발송 |
| useBusinessActions (SUBMIT) | High | 콘텐츠 저장 플로우 |
| 401 자동 토큰 갱신 | High | Axios 인터셉터 |
| 보호 화면 리다이렉트 | Medium | 비로그인 접근 |
| 페이지네이션 | Medium | DIARY_LIST 페이지 전환 |
| AddressSearchGroup | Low | 다음 API 연동 |
| EmotionSelectField | Low | 감정 선택 UI |

---

## E2E 테스트 핵심 플로우

```
1. 회원가입 → 이메일 인증 → 로그인
2. 로그인 → 콘텐츠 작성 (CONTENT_WRITE) → 목록 확인 (CONTENT_LIST)
3. 콘텐츠 목록 → 상세 보기 (CONTENT_DETAIL) → 수정 (CONTENT_MODIFY)
4. 비로그인 → 보호 화면 접근 → 로그인 페이지 리다이렉트
5. 카카오 로그인 플로우
6. 토큰 만료 → 자동 갱신 → 화면 유지
7. MAIN_PAGE GUEST → 벤토 그리드(로그인유도 카드) 렌더링 확인
8. MAIN_PAGE USER → 벤토 그리드(약속위젯 + 콘텐츠 카드) 렌더링 확인
```

---

## [P3] 모달 시스템 버그 분석 결과 — 코드 재검증 (2026-02-28)

> **⚠️ 상태 업데이트:** 실제 코드를 재분석한 결과, 메인 데이터 파이프라인은 이미 수정되어 있음.
> 기존 분석은 이전 버전 코드 기준이었으며, 현재 코드 기준으로 갱신함.

---

### 현재 파이프라인 상태 (실제 코드 기준)

| 지점 | 파일 | 라인 | activeModal 포함 | 상태 |
|------|------|------|-----------------|------|
| activeModal 생성 | useUserActions.tsx | 17, 20 | YES | ✅ 정상 |
| usePageHook 반환 | usePageHook.tsx | 34-35 | YES (명시적 추가) | ✅ 수정됨 |
| CommonPage destructure | page.tsx | 42 | YES | ✅ 수정됨 |
| DynamicEngine props 전달 | page.tsx | 89-90 | YES | ✅ 수정됨 |
| DynamicEngine props 수신 | DynamicEngine.tsx | 16 | YES | ✅ 정상 |
| renderModals() 로직 | DynamicEngine.tsx | 138-157 | YES | ✅ 정상 |

**결론: 핵심 파이프라인 버그는 이미 해결됨. 아래 잔존 이슈만 남아 있음.**

---

### 잔존 이슈 (현재 코드 기준)

#### Issue-1 [Medium] useBusinessActions 반환에 modal 상태 누락

**파일:** `metadata-project/components/DynamicEngine/hook/useBusinessActions.tsx`
**라인:** 104

```typescript
// 현재:
return { ...base, handleAction };
// activeModal, closeModal 미포함 → base spread에도 없음
```

**영향:** 비즈니스 도메인 전용 액션에서 모달 트리거 시 일관성 문제.
현재는 `usePageHook`이 `userActions.activeModal`을 명시적으로 가져오므로 렌더링은 가능하나,
비즈니스 액션 핸들러 내부에서 `setActiveModal`을 직접 호출하는 로직이 있다면 참조 불일치 발생.

#### Issue-2 [Low] renderModals 최상위 레벨만 필터

**파일:** `metadata-project/components/DynamicEngine/DynamicEngine.tsx`
**라인:** 141

```typescript
// 현재:
nodes.filter(node => (node.componentType || node.component_type) === 'MODAL')
// 최상위 nodes 배열만 순회 → 중첩 그룹 내 MODAL 발견 불가
```

**영향:** group 하위에 MODAL이 배치된 경우 절대 렌더링 안 됨.
현재 DB 구조에서 MODAL이 최상위에만 위치한다면 실질 영향 없음.

#### Issue-3 [Low] Modal.tsx 컴포넌트 버튼/내용 미완성

**파일:** `metadata-project/components/fields/Modal.tsx`
**라인:** 18-20

**영향:** 모달은 표시되나 버튼 텍스트/내용 렌더링이 불완전할 수 있음.

---

### 수정이 필요한 잔존 이슈 Diff

#### 수정 1: `useBusinessActions.tsx` — modal 더미 함수 일관성 보장 (선택)
```diff
- return { ...base, handleAction };
+ return {
+     ...base,
+     handleAction,
+     // usePageHook이 userActions에서 직접 가져오므로 필수는 아니나 일관성 유지용
+ };
```
→ 현재 구조상 필수 수정은 아님. usePageHook 설계 의도 확인 후 결정.

#### 수정 2: `DynamicEngine.tsx` — 재귀적 MODAL 탐색 (필요 시)
```typescript
// 최상위 + 중첩 MODAL 모두 탐색이 필요할 경우:
const collectModals = (nodes: Metadata[]): Metadata[] =>
    nodes.flatMap(n => [
        ...(n.component_type === 'MODAL' ? [n] : []),
        ...(n.children ? collectModals(n.children) : [])
    ]);
```

---

### 이전 분석과의 차이점 요약

| 항목 | 이전 분석 (구버전) | 현재 상태 (재검증) |
|------|-----------------|-----------------|
| usePageHook activeModal 반환 | ❌ 누락 | ✅ lines 34-35에서 명시적 반환 |
| page.tsx destructure | ❌ 누락 | ✅ line 42에서 포함 |
| DynamicEngine props 전달 | ❌ 미전달 | ✅ lines 89-90에서 전달 |
| renderModals return null | 주석 처리 의심 | 실제 확인 결과 정상 구현됨 |

---

### 현재 권고사항

1. **즉시 조치 불필요** — 핵심 파이프라인 이미 수정됨
2. **Issue-1 추적** — 비즈니스 액션에서 모달 트리거 케이스 있으면 수정
3. **Issue-2 추적** — DB에서 중첩 MODAL 사용 케이스 생기면 수정
4. **모달 통합 테스트 작성** — 현재 파이프라인이 실제 E2E로 검증되지 않음

### 테스트 체크리스트 (모달 검증용)

- [ ] LOGIN 화면에서 모달 트리거 액션 → 모달 렌더링 확인
- [ ] DIARY 화면에서 모달 트리거 액션 → 모달 렌더링 확인
- [ ] 모달 닫기 (closeModal) → 화면 정상 복귀 확인
- [ ] 여러 모달 ID가 있을 때 올바른 모달만 표시 확인

---

## 분석 히스토리

| 날짜 | 분석 내용 | 결론 |
|------|-----------|------|
| 2026-02-28 | 전체 테스트 환경 초기 분석 | 위 내용 도출, 커버리지 갭 식별 |
| 2026-02-28 | [P3] 모달 시스템 버그 추적 | 아래 섹션 참고 |
| 2026-03-06 | E2E 플로우 DIARY→CONTENT 전환 반영, MAIN_PAGE 벤토 그리드 시나리오 추가 | E2E 플로우 7·8번 추가 |
