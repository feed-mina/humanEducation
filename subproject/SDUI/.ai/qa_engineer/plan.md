# QA Engineer — Plan

> 이 파일은 테스트 계획을 기록한다.
> 사용자의 명시적 승인("YES") 후에만 테스트 코드를 작성한다.
> 테스트 설계도 '기획'이다 — 구현보다 먼저 설계한다.

---

## Plan 작성 템플릿

```markdown
## [기능 이름] 테스트 계획 — {날짜}

### 배경
- 테스트 대상: 어떤 기능/컴포넌트/API
- 관련 구현 계획: frontend_engineer plan.md / backend_engineer plan.md

### 테스트 전략

| 레벨 | 도구 | 범위 |
|------|------|------|
| 단위 | Jest + RTL | 개별 컴포넌트, 훅 로직 |
| 통합 | Jest + MSW | DynamicEngine + 메타데이터 + API 모킹 |
| E2E | Playwright | 전체 사용자 플로우 |

### 테스트 케이스 목록

#### 단위 테스트

```typescript
// 파일: tests/components/{ComponentName}.test.tsx

describe('{ComponentName}', () => {
  // TC-001: 정상 렌더링
  // Given: 기본 props (value, meta)
  // When: 컴포넌트 렌더링
  // Then: label_text 표시됨, value 표시됨

  // TC-002: 읽기 전용 상태
  // Given: isReadonly=true
  // When: 입력 시도
  // Then: 입력 불가 (disabled 또는 readOnly)

  // TC-003: onChange 콜백
  // Given: onChange 모킹
  // When: 사용자 입력 (userEvent.type)
  // Then: onChange(componentId, 입력값) 호출됨

  // TC-004: onAction 콜백
  // Given: onAction 모킹, action_type 있는 버튼
  // When: 버튼 클릭
  // Then: onAction(meta, undefined) 호출됨

  // TC-005: 엣지 케이스 — null value
  // Given: value=null
  // When: 렌더링
  // Then: 에러 없이 빈 상태 표시
});
```

#### 통합 테스트 (DynamicEngine)

```typescript
// 파일: tests/integration/{feature}.test.tsx

describe('DynamicEngine — {feature}', () => {
  // TC-010: 메타데이터 → 올바른 컴포넌트 렌더링
  // Given: MSW로 /api/ui/{screenId} → mock metadata 반환
  // When: DynamicEngine 마운트
  // Then: componentMap 매핑대로 렌더링

  // TC-011: 데이터 바인딩 우선순위
  // Given: formData[refId]='user-input', pageData[refId]='server-data'
  // When: 컴포넌트 렌더링
  // Then: 'user-input' 표시 (formData 우선)

  // TC-012: Repeater 렌더링
  // Given: ref_data_id='items', pageData.items=['a','b','c']
  // When: DynamicEngine 렌더링
  // Then: 자식 컴포넌트 3개 렌더링
});
```

#### E2E 테스트 (Playwright)

```typescript
// 파일: e2e/{flow}.spec.ts

test('{사용자 플로우 이름}', async ({ page }) => {
  // Step 1: [페이지 이동]
  await page.goto('/view/LOGIN_PAGE');

  // Step 2: [사용자 인터랙션]
  await page.fill('[data-testid="userId-input"]', 'testuser');

  // Step 3: [결과 검증]
  await expect(page).toHaveURL('/view/DIARY_LIST');
});
```

### MSW 핸들러

```typescript
// 이 테스트에 필요한 API 모킹
http.get('/api/ui/{screenId}', () => {
  return HttpResponse.json({ data: mockMetadata });
}),
http.post('/api/auth/login', () => {
  return HttpResponse.json({ data: { accessToken: 'mock-token' } });
}),
```

### 테스트 파일 위치

| 파일 | 설명 |
|------|------|
| `tests/components/{ComponentName}.test.tsx` | 단위 테스트 |
| `tests/integration/{feature}.test.tsx` | 통합 테스트 |
| `e2e/{flow}.spec.ts` | E2E 테스트 |

### 성공 기준

| 기준 | 목표 |
|------|------|
| 단위 테스트 커버리지 | 80% 이상 |
| 렌더 카운트 (withRenderTrack) | 이전 대비 증가 없음 |
| E2E 통과율 | 100% |
| frontend-report.html | 모든 테스트 green |

### 담당자 연락 사항

- Frontend Engineer: 테스트 작성에 필요한 data-testid 속성 추가 요청
- Backend Engineer: API 응답 구조 확정 (MSW 모킹 전 필요)

### 승인 상태
[ ] 사용자 승인 대기 중
[x] 사용자 승인 완료 (날짜: ...)
[ ] 테스트 작성 완료
[ ] 전체 테스트 통과 (frontend-report.html 링크: ...)
```

---

## [P1] Modal 시스템 + 보안 회귀 테스트 계획 — 2026-02-28

### 배경
- 요청 출처: research.md `[P3] Modal 시스템 버그 분석`, `[P2] Security Audit` (2026-02-28)
- **Modal:** 핵심 파이프라인(usePageHook → page.tsx → DynamicEngine)은 이미 수정됨. 실제 동작을 테스트로 검증.
- **Security:** backend FIX 적용 후 회귀 테스트(인증 보호 엔드포인트 접근 차단), JWT cookie 전환 후 동작 확인.

---

### 테스트 전략

| 레벨 | 도구 | 테스트 대상 |
|------|------|-----------|
| 단위 | Jest + RTL | Modal 컴포넌트 자체 렌더링 |
| 통합 | Jest + MSW | usePageHook → DynamicEngine 모달 파이프라인 |
| 통합 | Jest + MSW | axios.tsx JWT 쿠키 전환 (토큰 없이 요청, 401 처리) |
| E2E | Playwright | 로그인 → 모달 트리거 → 닫기 |

---

### 테스트 케이스 목록

#### TC-M001 ~ M005: Modal 컴포넌트 단위 테스트

**파일:** `tests/components/Modal.test.tsx`

```typescript
describe('Modal 컴포넌트', () => {

  // TC-M001: 비활성 상태 렌더링 안 됨
  // Given: activeModal="OTHER_MODAL", componentId="MY_MODAL"
  // When: 렌더링
  // Then: 모달 DOM 없음 (null 반환)

  // TC-M002: 활성 상태 렌더링 됨
  // Given: activeModal="MY_MODAL", componentId="MY_MODAL"
  // When: 렌더링
  // Then: 모달 DOM 존재

  // TC-M003: onConfirm 콜백
  // Given: activeModal="MY_MODAL", onConfirm 모킹
  // When: 확인 버튼 클릭
  // Then: onConfirm() 1회 호출

  // TC-M004: onClose 콜백
  // Given: activeModal="MY_MODAL", onClose 모킹
  // When: 닫기 버튼 클릭
  // Then: onClose() 1회 호출, 이후 모달 사라짐

  // TC-M005: activeModal null 상태
  // Given: activeModal=null
  // When: 렌더링
  // Then: 에러 없이 렌더링 안 됨

});
```

---

#### TC-M010 ~ M015: Modal 파이프라인 통합 테스트

**파일:** `tests/integration/modal_pipeline.test.tsx`

```typescript
// 사용 메타데이터 구조 (mock):
const mockMetadata = [
  {
    component_id: 'confirm_modal',
    component_type: 'MODAL',
    label_text: '확인',
    parent_group_id: null,
    children: []
  },
  {
    component_id: 'open_btn',
    component_type: 'BUTTON',
    label_text: '열기',
    action_type: 'OPEN_MODAL',
    parent_group_id: null
  }
];

describe('Modal 파이프라인 — usePageHook → DynamicEngine', () => {

  // TC-M010: 버튼 클릭 → 모달 활성화
  // Given: MSW로 /api/ui/TEST_SCREEN → mockMetadata 반환
  // When: CommonPage 렌더링 후 '열기' 버튼 클릭
  // Then: 'confirm_modal' 모달 DOM 나타남

  // TC-M011: 모달 닫기
  // Given: 모달 활성화 상태
  // When: 닫기 버튼 클릭
  // Then: 모달 DOM 사라짐, activeModal = null

  // TC-M012: usePageHook activeModal 반환 확인
  // Given: usePageHook hook 렌더링
  // When: 초기 상태
  // Then: { activeModal: null, closeModal: function } 포함됨

  // TC-M013: 비즈니스 화면(isUserDomain=false)에서도 모달 동작
  // Given: screenId = "DIARY_WRITE" (비즈니스 도메인)
  // When: 모달 트리거 액션 실행
  // Then: activeModal 업데이트 → 모달 표시됨

  // TC-M014: 여러 모달 중 올바른 것만 표시
  // Given: 메타데이터에 MODAL 2개 (id: 'modal_a', 'modal_b')
  // When: activeModal = 'modal_a'
  // Then: 'modal_a'만 표시, 'modal_b'는 숨김

  // TC-M015: DynamicEngine renderModals null safe
  // Given: metadata.children = undefined
  // When: DynamicEngine 렌더링
  // Then: 에러 없이 렌더링 완료 (renderModals가 null 반환)

});
```

---

#### TC-S001 ~ S006: 인증 보안 회귀 테스트

**파일:** `tests/integration/auth_security.test.tsx`

```typescript
// backend FIX 적용 후 실행하는 회귀 테스트:
describe('인증 보안 — API 보호 엔드포인트', () => {

  // TC-S001: 토큰 없이 보호 화면 접근 → 리다이렉트
  // Given: AuthContext.isLoggedIn = false, screenId in PROTECTED_SCREENS
  // When: CommonPage 렌더링
  // Then: router.replace('/view/LOGIN_PAGE') 호출됨

  // TC-S002: 401 자동 갱신 → 성공 시 원래 요청 재시도
  // Given: MSW 첫 요청 401, refresh 성공, 두 번째 요청 200
  // When: API 요청
  // Then: 최종적으로 200 응답 처리됨

  // TC-S003: 401 자동 갱신 → 실패 시 로그인 리다이렉트
  // Given: MSW refresh 401 반환
  // When: API 요청 후 refresh 실패
  // Then: AuthContext.logout() 호출 → LOGIN_PAGE 이동

  // TC-S004: JWT 쿠키 전환 후 — localStorage 미사용 확인
  // Given: 로그인 성공 MSW 모킹
  // When: 로그인 처리
  // Then: localStorage.getItem('accessToken') 호출 안 됨 (spy 확인)

  // TC-S005: axios withCredentials — 쿠키 자동 전송
  // Given: axios 인스턴스
  // When: 설정 확인
  // Then: withCredentials = true 확인

  // TC-S006: PROTECTED_SCREENS에 없는 화면 — 로그아웃 상태로 접근 허용
  // Given: screenId not in PROTECTED_SCREENS, isLoggedIn = false
  // When: CommonPage 렌더링
  // Then: 리다이렉트 없음 (LOGIN_PAGE 자체, MAIN_PAGE 등)

});
```

---

### MSW 핸들러 추가 목록

```typescript
// tests/mocks/handlers.ts에 추가:

// 모달 트리거 액션 핸들러 (화면 메타데이터 반환)
http.get('/api/ui/MODAL_TEST_SCREEN', () =>
  HttpResponse.json({ data: mockModalMetadata })
),

// 401 → refresh → 재시도 시나리오
let callCount = 0;
http.post('/api/execute/:sqlKey', () => {
  if (callCount++ === 0) return new HttpResponse(null, { status: 401 });
  return HttpResponse.json({ data: [] });
}),
http.post('/api/auth/refresh', () =>
  HttpResponse.json({ data: { accessToken: 'new-token' } })
),

// refresh 실패 시나리오
http.post('/api/auth/refresh-fail', () =>
  new HttpResponse(null, { status: 401 })
),
```

---

### 테스트 파일 위치

| 파일 | 테스트 유형 |
|------|-----------|
| `tests/components/Modal.test.tsx` | 단위 — Modal 컴포넌트 |
| `tests/integration/modal_pipeline.test.tsx` | 통합 — 모달 파이프라인 전체 |
| `tests/integration/auth_security.test.tsx` | 통합 — 인증/보안 회귀 |

---

### 성공 기준

| 기준 | 목표 |
|------|------|
| Modal 단위 테스트 (TC-M001~M005) | 5/5 통과 |
| Modal 파이프라인 통합 (TC-M010~M015) | 6/6 통과 |
| 인증 보안 회귀 (TC-S001~S006) | 6/6 통과 |
| 기존 테스트 회귀 없음 | `npm run test` 전체 green |
| frontend-report.html | 모든 케이스 green |

---

### 담당자 협의 사항

- **Frontend Engineer:**
  - `Modal.tsx` 컴포넌트에 `data-testid` 속성 필요 (`data-testid="modal-{componentId}"`) //[매모]  아직 모달을 사용하는 페이지가 없음
  - axios.tsx 수정 후 기존 MSW 모킹 테스트 영향 여부 확인 요청 

- **Backend Engineer:**
  - `/api/execute/**` 권한 적용 후 인증 에러 응답 형식 공유 요청 (MSW 모킹용) //[매모]  우선 '*'로 테스트 

---

### TODO 리스트 (승인 후 순서대로 실행)

- [ ] 1. `tests/components/Modal.test.tsx` — TC-M001~M005 작성 //[매모]  테스트 파일 만들고 실행 결과 표시
- [ ] 2. `tests/integration/modal_pipeline.test.tsx` — TC-M010~M015 작성 //[매모]  테스트 파일 만들고 실행 결과 표시
- [ ] 3. `tests/mocks/handlers.ts` — 모달 관련 MSW 핸들러 추가 //[매모]  테스트 파일 만들고 실행 결과 표시
- [ ] 4. `tests/integration/auth_security.test.tsx` — TC-S001~S006 작성 //[매모]  테스트 파일 만들고 실행 결과 표시
- [ ] 5. `tests/mocks/handlers.ts` — 401/refresh 시나리오 핸들러 추가 //[매모]  테스트 파일 만들고 실행 결과 표시
- [ ] 6. `npm run test` 전체 통과 확인 //[매모]  테스트 파일 만들고 실행 결과 표시
- [ ] 7. `tests/logs/frontend-report.html` 결과 확인 //[매모]  테스트 파일 만들고 실행 결과 표시

---

### 승인 상태
- [x] 사용자 승인 대기 중
- 