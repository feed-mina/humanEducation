# 관리자 페이지 - 회원 권한 관리 기능 구현 계획 (v2)

## 1. 목표

기존 `CONTENT_LIST` 영역을 대체하여, 관리자가 회원을 검색하고 여러 회원을 선택(1명 이상 5명 이하)하여 일괄적으로 권한을 변경할 수 있는 UI와 기능을 구현합니다. 권한 변경 후에는 변경된 내용이 즉시 화면에 반영되어야 합니다.

---

## 2. Backend 개발 계획 (SDUI-server)

### 2.1. 신규 API 엔드포인트 정의

1.  **회원 목록 조회 API (`GET /api/admin/users`)**
    *   (변경 없음)

2.  **회원 권한 일괄 변경 API (`PUT /api/admin/users/roles`)**
    *   (변경 없음)

### 2.2. `query_master` 연동 및 서비스 로직 구현

1.  **`permission_queries.sql` 파일 내용 `query_master`에 등록**
    *   `find_users_for_admin` 쿼리를 등록합니다.
    *   `update_user_roles` 쿼리를 등록합니다. (주석에 명시된 비즈니스 로직 확인)

2.  **Repository/DAO 계층 구현**
    *   (변경 없음)

3.  **Service 계층 구현**
    *   `AdminUserService` (가칭)와 같은 서비스 클래스를 생성합니다.
    *   **`getUsers(searchTerm, pageable)`**: (변경 없음)
    *   **`updateUserRoles(userIds, newRole)`**:
        *   **[수정]** 요청된 `userIds`의 개수가 **1에서 5 사이인지 검증**합니다. 범위를 벗어날 경우, `BadRequestException`과 같은 예외를 발생시켜 클라이언트에게 오류를 알립니다.
        *   요청한 관리자의 권한을 확인하여 `ADMIN` 권한이 있는지 검증합니다.
        *   `userIds` 목록과 `newRole`을 받아 Repository를 통해 권한을 업데이트하는 로직을 구현합니다.

4.  **Controller 계층 구현**
    *   (변경 없음)

---

## 3. Frontend 개발 계획 (metadata-project)

### 3.1. 신규 페이지 및 컴포넌트 생성

1.  **`AdminPermissionPage.tsx` (가칭) 페이지 컴포넌트 생성**
    *   (변경 없음)

### 3.2. 상태 관리 (State Management)

*   (변경 없음)

### 3.3. 기능 구현

1.  **회원 목록 조회 및 렌더링**
    *   (변경 없음)

2.  **검색 기능**
    *   (변경 없음)

3.  **체크박스 기능**
    *   "전체 선택" 체크박스 클릭 시 모든 회원을 선택/해제하는 로직을 구현합니다.
    *   **[수정]** 개별 체크박스 선택 시, 현재 선택된 회원의 수가 **5명을 초과하는 경우 "최대 5명까지 선택할 수 있습니다." 경고창을 표시**하고 추가 선택을 막습니다.
    *   선택된 회원은 `selectedUserIds` 상태로 관리합니다.

4.  **권한 변경 기능**
    *   "선택된 회원 권한 변경" 버튼 클릭 시 `changePermissions` 함수를 실행합니다.
    *   **[수정]** `selectedUserIds`가 비어있는지(0명) 또는 5명을 초과하는지 확인하고, 해당될 경우 경고창을 띄웁니다.
    *   `window.confirm`을 사용하여 목업과 동일한 확인 메시지를 띄웁니다.
    *   사용자가 "확인"을 누르면 `PUT /api/admin/users/roles` API를 호출합니다.
    *   **[수정]** API 호출 성공 시, 성공했다는 `alert`을 띄우고 **회원 목록 API를 다시 호출하여 화면을 새로고침**합니다. 이를 통해 사용자는 변경된 권한을 즉시 확인할 수 있습니다.
    *   API 호출 실패 시, 백엔드에서 전달된 에러 메시지를 `alert`으로 표시합니다.

---

## 4. 작업 순서 제안

1.  **[Backend]** `query_master` 등록 및 DAO/Repository 구현
2.  **[Backend]** **Service 로직 수정 (1~5명 유효성 검사 추가)** 및 Controller 구현
3.  **[Frontend]** `AdminPermissionPage` 기본 UI 컴포넌트 생성
4.  **[Frontend]** 회원 목록 조회 API 연동 및 데이터 렌더링
5.  **[Frontend]** **체크박스 기능 수정 (최대 5개 선택 제한 로직 추가)**
6.  **[Frontend]** **권한 변경 기능 수정 (API 호출 후 목록 새로고침 로직 명시)**
7.  **[QA]** 통합 테스트 및 버그 수정

이 계획을 따라 순서대로 개발을 진행하면 체계적으로 기능을 완성할 수 있습니다.
