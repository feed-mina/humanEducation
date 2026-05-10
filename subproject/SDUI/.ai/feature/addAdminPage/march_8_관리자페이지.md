draft plan : 현재 관리자페이지를 만들 예정에 있습니다.  JwtAuthenticationFilter와 SecurityConfig 와 `GET/POST /api/execute/{sqlKey}` 부분이 어떻게 되어있는지 파악이 필요합니다.

`(/api/execute/**` → `hasRole('ADMIN')` + SecurityConfig permitAll 제거) 이 부분에 대해 수정이 먼저 필요합니다.

이후 아래 파일 3개 목록까지 확인해주세요.

기능은 페이지에서는 검색기능과 회원리스트에 체크박스 하여 권한을 바꿔줄수있는 기능과 . 권한종류는 select창에서 선택할 수 있고 넣기 전에 alert을 띄우고 회원 000의 권한을 ㅁㅁㅁ로 바꿉니다 라고 띄웁니다.

체크박스에 체크할때 5개 이하로 체크할 수 있도록 체크박스가 6개이면 alet을 울려야합니다. 회원권한을 1명 이상 ~5명 까지할수 있도록 query_master에 관한 쿼리를 작성이 필요합니다. 권한을 부여한 후 다시 리스트 페이지로 돌아올때 부여한 권한을 볼 수 있어야 합니다.

**`.ai/backend_engineer/permission_queries.sql`)**

**`.ai/architect/plan_admin_permission_page.md`**

**`.ai/frontend_engineer/Mock/mockup_admin_permissions.html`**

이 3개의 목록을 참고하여 현재 프로젝트에 맞게 적용합니다.

---

## Research 결과 (2026-03-08)

### 1. SecurityConfig.java 현황

- `/api/execute/**` → **`permitAll()`** : 인증 없이 누구나 모든 sqlKey 호출 가능
- `/api/admin/**` → **정의 없음** → `anyRequest().denyAll()` 에 걸려 403 반환
- **결론**: 드래프트 플랜의 "execute를 통째로 ADMIN으로 바꾼다"는 불가. 공개 화면(GUEST/USER)도 `/api/execute/**` 를 사용 중이기 때문에 전체를 ADMIN으로 잠그면 일반 화면이 모두 깨짐.
- **올바른 방향**: `/api/admin/**` 패턴을 새로 추가하고 `hasRole("ADMIN")` 부여 → 관리자 전용 기능은 이 경로로 분리

### 2. JwtAuthenticationFilter 현황

- JWT `role` 클레임 추출 → `SimpleGrantedAuthority(role)` 등록
- DB에 `ROLE_ADMIN` 형태로 저장 → JWT에 `ROLE_ADMIN` → Spring Security `hasRole("ADMIN")` 과 호환 (Spring이 내부적으로 `ROLE_` prefix 처리)
- `EXCLUDE_URLS`: `/api/auth/login`, `/api/auth/refresh`, `/api/kakao/**`, `/api/ui/LOGIN_PAGE` — 관리자 API는 제외 목록 없음 → JWT 필터 정상 통과

### 3. CommonQueryController (`GET/POST /api/execute/{sqlKey}`) 현황

- `Authentication` 객체를 받지만, 현재는 `userSqno`·`userId` 파라미터 주입에만 사용
- 역할 체크 로직 없음 — `sqlKey` 이름에 관계없이 모든 키가 실행됨
- `/api/execute/**` 를 건드리지 않고, 관리자 전용 컨트롤러를 신규 경로(`/api/admin/`)로 별도 구현하는 것이 안전

### 4. ⚠️ DynamicExecutor 호환성 문제 (핵심)

`permission_queries.sql`의 `update_user_roles` 쿼리가 **MyBatis `<foreach>` 문법** 사용:

```xml
WHERE user_id IN <foreach ...> #{userId} </foreach>
```

→ `DynamicExecutor`는 `NamedParameterJdbcTemplate` 기반 (`:paramName` 문법) 이므로 **호환 불가**.

추가로 `executeUpdate()`는 `List` 타입 파라미터를 **JSON 문자열로 직렬화**:

```java
if (value instanceof List ...) {
    objectMapper.writeValueAsString(value)  // "[1,2,3]" 문자열
}
```

→ `WHERE user_sqno IN (:userIds)` 에 JSON 문자열이 바인딩되어 실행 실패.

**결론**: 권한 일괄 변경(`update_user_roles`)은 query_master / DynamicExecutor 경로를 사용하지 않고, **Admin 전용 컨트롤러 + JPA `UserRepository`로 직접 처리** 필요.

`find_users_for_admin` SELECT 쿼리는 query_master로 등록 가능하지만 파라미터 문법 수정 필요:

- `${searchTerm}` → `:searchTerm`
- `LIKE '%${searchTerm}%'` → `LIKE '%' || :searchTerm || '%'`

### 5. User 엔티티 & UserRepository

- 테이블: `users`, `role` 컬럼 형식: `ROLE_USER` / `ROLE_ADMIN`
- `UserRepository`는 `JpaRepository<User, Long>` 상속 → `findAllById(List<Long>)` 기본 제공
- 벌크 role 업데이트: `@Modifying @Query` 어노테이션 메서드 추가 또는 `findAllById` 후 `setRole` + `saveAll()` 로 처리 가능

### 6. USER_LIST 스크린 현재 상태

- V12 마이그레이션(admin_users_btn)에서 `action_url = '/view/USER_LIST'` 참조
- 하지만 `USER_LIST` screen_id에 해당하는 `ui_metadata` 행이 **존재하지 않음**
- 신규 Flyway 마이그레이션으로 `USER_LIST` 스크린 메타데이터 INSERT 필요

### 7. 권한 값 일관성 확인

| 위치                                 | 형식                             |
| ------------------------------------ | -------------------------------- |
| DB `users.role`                    | `ROLE_USER`, `ROLE_ADMIN`    |
| JWT claim                            | `ROLE_ADMIN` (DB에서 가져옴)   |
| Spring Security `hasRole("ADMIN")` | 내부적으로 `ROLE_ADMIN` 체크   |
| 목업 select value                    | `USER`, `MANAGER`, `ADMIN` |

→ API PUT 요청 시 `ROLE_ADMIN` 형식으로 전달하거나, 서비스 레이어에서 `ROLE_` prefix를 붙이는 처리 필요

---

## 구현 Plan

### [P0 완료] Option B: query_master.required_role 컬럼 기반 권한 검증

`/api/execute/**` 를 `hasRole("ADMIN")` 으로 일괄 변경하면 모든 USER 화면이 깨지므로,
**각 쿼리에 required_role을 명시**하고 컨트롤러에서 검증하는 방식으로 결정.

**변경된 파일:**

- [QueryMaster.java](SDUI-server/src/main/java/com/domain/demo_backend/domain/query/domain/QueryMaster.java) — `requiredRole` 필드 추가 ✅
- [CommonQueryController.java](SDUI-server/src/main/java/com/domain/demo_backend/domain/query/controller/CommonQueryController.java) — role 검증 로직 추가 ✅
- [V15 초안](.ai/backend_engineer/V15__add_required_role_to_query_master.sql) → `SDUI-server/src/main/resources/db/migration/V15__add_required_role_to_query_master.sql` 로 이동 필요

**required_role 값 규칙:**

| 값             | 의미                              |
| -------------- | --------------------------------- |
| `NULL`       | 누구나 실행 가능 (기존 동작 유지) |
| `ROLE_USER`  | 로그인한 사용자만 실행 가능       |
| `ROLE_ADMIN` | ADMIN 권한 보유자만 실행 가능     |

**⚠️ V15 적용 전 확인:** 실제 등록된 sql_key 목록 조회 후 UPDATE 대상 확정 필요

```sql
SELECT sql_key, description FROM query_master ORDER BY sql_key;
```

---

### Phase 1: SecurityConfig.java 수정

```java
// 기존 유지 (공개 SDUI 쿼리 — required_role로 쿼리별 제어)
.requestMatchers("/api/execute/**").permitAll()

// 신규 추가
.requestMatchers("/api/admin/**").hasRole("ADMIN")
```

파일: [SecurityConfig.java](SDUI-server/src/main/java/com/domain/demo_backend/global/config/SecurityConfig.java)

### Phase 2: Admin 전용 컨트롤러 + 서비스 신규 생성

**신규 파일 위치** (`.ai` 폴더에 초안 작성 후 이동):

- `.ai/backend_engineer/AdminUserController.java` → `SDUI-server/src/main/java/com/domain/demo_backend/domain/admin/controller/AdminUserController.java`
- `.ai/backend_engineer/AdminUserService.java` → `SDUI-server/src/main/java/com/domain/demo_backend/domain/admin/service/AdminUserService.java`

**엔드포인트 정의**:

| Method | URL                              | 역할                                                        |
| ------ | -------------------------------- | ----------------------------------------------------------- |
| GET    | `/api/admin/users?searchTerm=` | 회원 목록 조회 (query_master `find_users_for_admin` 실행) |
| PUT    | `/api/admin/users/roles`       | 권한 일괄 변경 (JPA 직접 처리)                              |

**PUT 요청 body**:

```json
{ "userIds": [1, 2, 3], "newRole": "ROLE_ADMIN" }
```

**서비스 레이어 검증 규칙**:

1. `userIds.size() < 1` → `BadRequestException("최소 1명을 선택해야 합니다.")`
2. `userIds.size() > 5` → `BadRequestException("최대 5명까지 변경 가능합니다.")`
3. `newRole` 값이 `ROLE_USER`, `ROLE_ADMIN` 중 하나가 아니면 거부

### Phase 3: Flyway 마이그레이션 — query_master 등록

신규 마이그레이션 파일: `.ai/` 초안 → `SDUI-server/src/main/resources/db/migration/V16__admin_user_query.sql`

```sql
INSERT INTO query_master (sql_key, query_text, return_type, description, created_at, updated_at)
VALUES (
  'find_users_for_admin',
  'SELECT user_sqno, user_id, email, role
   FROM users
   WHERE del_yn = ''N''
     AND (:searchTerm = '''' OR LOWER(user_id) LIKE ''%'' || LOWER(:searchTerm) || ''%''
       OR LOWER(email) LIKE ''%'' || LOWER(:searchTerm) || ''%'')
   ORDER BY created_at DESC',
  'MULTI',
  '관리자 전용 회원 목록 조회 (searchTerm 빈 문자열이면 전체)',
  NOW(), NOW()
);
```

→ `update_user_roles`는 DynamicExecutor 미호환으로 **query_master 미등록**, JPA로 처리

### Phase 4: Flyway 마이그레이션 — USER_LIST 스크린 ui_metadata

신규 마이그레이션 파일: `.ai/` 초안 → `SDUI-server/src/main/resources/db/migration/V16__user_list_screen.sql`

- `screen_id = 'USER_LIST'`, `allowed_roles = 'ROLE_ADMIN'`
- 검색 INPUT + 검색 BUTTON 그룹
- 권한 SELECT + 변경 BUTTON 그룹
- 회원 테이블 (ADMIN_USER_TABLE 신규 컴포넌트 타입으로 등록 예정)

### Phase 5: UserRepository 메서드 추가 (필요 시)

```java
// 벌크 update 쿼리 (옵션 A — @Modifying 사용)
@Modifying
@Query("UPDATE User u SET u.role = :newRole WHERE u.userSqno IN :userIds")
int updateRoleByIds(@Param("userIds") List<Long> userIds, @Param("newRole") String newRole);
```

또는 `findAllById(userIds)` + 반복 `setRole` + `saveAll()` (더티 체킹, 건수 적으므로 성능 문제 없음)

### `query_master`에 `required_role` 컬럼 추가

```sql
ALTER TABLE query_master ADD COLUMN required_role VARCHAR(50) DEFAULT NULL;
-- NULL = 누구나, ROLE_USER = 로그인 필요, ROLE_ADMIN = 관리자만
```

`CommonQueryController`에서 체크:

```java
String requiredRole = queryMaster.getRequiredRole();
if (requiredRole != null) {
    if (authentication == null) return 401;
    boolean hasRole = authentication.getAuthorities().stream()
        .anyMatch(a -> a.getAuthority().equals(requiredRole));
    if (!hasRole) return 403;
}
```

* [X] V15 마이그레이션 초안 작성 (.ai 폴더) — query_master required_role 컬럼 추가 ✅
* [X] QueryMaster.java — requiredRole 필드 추가 ✅
* [X] CommonQueryController.java — role 검증 로직 추가 ✅
* [X] march_8_관리자페이지.md — Plan 업데이트 ✅

---

## 현재 구현 현황 (2026-03-08 기준, 로컬 환경 진행 중)

### ✅ 완료

| 항목                                                                  | 파일                                                                                                                                                                                                                                                 |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0 — QueryMaster.requiredRole 필드 추가                              | `QueryMaster.java`                                                                                                                                                                                                                                 |
| P0 — CommonQueryController required_role 검증 로직 추가              | `CommonQueryController.java:46-59`                                                                                                                                                                                                                 |
| V15 → migration 폴더 배치 + 로컬 DB 적용 완료                        | `V15__add_required_role_to_query_master.sql`                                                                                                                                                                                                       |
| V13, V14 `sql_text` → `query_text` 컬럼명 버그 수정              | `V13__admin_stats_query.sql`, `V14__admin_logs_query.sql`                                                                                                                                                                                        |
| application.yml → 로컬 PostgreSQL(5432/testdb) 전환                  | `application.yml`                                                                                                                                                                                                                                  |
| 어드민 URL 라우팅 `/view/admin/{screenId}` 패턴 도입                | `MetadataProvider.tsx`                                                                                                                                                                                                                             |
| USER_LIST PROTECTED_SCREENS 추가 (로그인 필수)                        | `page.tsx`                                                                                                                                                                                                                                         |
| 순서 1 — 서버 실행 + V12~V15 Flyway 마이그레이션 로컬 DB 적용        | `./gradlew bootRun` 로그 확인                                                                                                                                                                                                                      |
| 순서 2 — SecurityConfig `/api/admin/**` hasRole("ADMIN") 추가      | `SecurityConfig.java:90`                                                                                                                                                                                                                           |
| 순서 3 — AdminUserController + AdminUserService 신규 생성            | `domain/admin/controller/AdminUserController.java`, `domain/admin/service/AdminUserService.java`, `domain/admin/dto/AdminUserResponse.java`, `domain/admin/dto/UpdateUserRoleRequest.java`, `UserRepository.java` (findUsersForAdmin 추가) |
| 순서 4 — V16 마이그레이션 (query_master 등록 + GET_ADMIN_STATS 수정) | `V16__admin_user_query.sql`                                                                                                                                                                                                                        |
| 순서 5 — V17 마이그레이션 (USER_LIST 스크린 ui_metadata)             | `V17__user_list_screen.sql`                                                                                                                                                                                                                        |
| 순서 6 — Frontend ADMIN_USER_TABLE 컴포넌트 구현                     | `useAdminUsers.ts`, `AdminUserTable.tsx`, `componentMap.tsx`, `pages.css`                                                                                                                                                                    |

---

### 🚀 로컬 실행 체크리스트

#### 사전 조건

- [ ] Redis 실행 확인: `redis-cli ping` → `PONG` 응답
- [ ] 로컬 PostgreSQL(5432/testdb) 실행 확인

#### 서버 실행

```bash
cd SDUI-server
./gradlew bootRun
```

#### 실행 후 확인

- [ ] Flyway 마이그레이션 성공 로그 확인 (`Successfully applied N migrations to schema "public"`)
- [ ] V12~V15 순서대로 오류 없이 적용됐는지 확인 (오류 시 해당 버전 로그 상세 확인)
- [ ] sql_key 목록 확인 — V15 UPDATE 적용 여부 검증:
  ```sql
  SELECT sql_key, required_role FROM query_master ORDER BY sql_key;
  ```

  → `GET_MY_GOAL_TIME`, `GET_GOAL_LIST`, `GET_CONTENT_LIST`, `GET_MY_CONTENT` → `ROLE_USER`
  → `GET_SYSTEM_LOGS` → `ROLE_ADMIN`
  → 나머지 → `NULL`

---

* //[메모] 로컬에서 테스트 결과 :
1. {total_users}, {total_diaries}, {new_users} , {log_level}, {message}, {log_time} 등 텍스트로 하드코딩된 부분 확인
2. 시스템현광, 회원관리, 최근 시스템 로그 카드 누르면 링크 이동 가능하도록
3. sidebar 부분도 이에 맞춰서 변경할 수 있게 (다크모드) 로 같이 UI 업데이트 필요

---

## 테스트 결과 Research (2026-03-08)

### Issue 1 — 템플릿 변수 미치환 (`{total_users}`, `{log_level}` 등)

#### 근본 원인 분석

**`TextField.tsx` (components/fields/TextField.tsx:40)**

```typescript
let finalValue = value || meta?.labelText || ... || "";
// finalValue = '{total_users}\n총 사용자' 로 바로 설정
// → data 체크 블록에 진입하지 않음 (finalValue가 falsy가 아니므로)
// → {key} 치환 로직 자체가 없음
```

→ **TextField에 `{key}` → `data[key]` 치환 로직 추가 필요**

**통계 데이터 접근 경로 문제 (stats 전용 이슈)**

| 컴포넌트 | ref_data_id | data 값 (getComponentData 결과) |
|----------|-------------|----------------------------------|
| `admin_stat_users` TEXT | 없음 | `pageData` 전체 = `{admin_stats_source: [{...}]}` |
| | | `pageData.total_users` = **undefined** |

→ 치환 시 `{total_users}` → `undefined` → 치환 실패

**로그 데이터 접근 경로 (logs는 Repeater 구조 올바름)**

```
admin_logs_list (GROUP, ref_data_id='admin_logs_source') ← Repeater
  └── admin_log_item_template (GROUP)
        ├── admin_log_msg TEXT label='{[log_level]} {message}'
        └── admin_log_time TEXT label='{log_time}'
```
→ Repeater가 rowData를 올바르게 전달 → 치환 로직만 추가하면 해결

**통계 데이터 접근 해결 방안**

`admin_stats_row` GROUP에 `ref_data_id='admin_stats_source'` 추가 (Flyway UPDATE):
- admin_stats_row가 Repeater가 되어 `admin_stats_source = [{total_users, today_diaries, new_users}]` 1건 순회
- 3개 TEXT 자식이 `rowData = {total_users: '7', ...}` 수신
- 치환 후: `'7\n총 사용자'` 정상 표시

**리터럴 `\n` 문제**

PostgreSQL `standard_conforming_strings=on` 기본값에서:
- `'text\n더보기'` → DB에는 `\` + `n` 2글자 저장
- Jackson JSON 직렬화 → `"text\\nmore"` → JavaScript `finalValue = 'text\\n더보기'` (2글자)
- CSS `white-space: normal` (기본) → 개행 무시 → 화면에 `\n` 그대로 출력

→ JavaScript 단에서 `finalValue.replace(/\\n/g, '\n')` + CSS `white-space: pre-line` 처리 필요

---

### Issue 2 — 카드 클릭 불가

**DynamicEngine.tsx:72**

```typescript
const hasAction = !!(node.actionType || node.action_type);
// hasAction = false → cursor:default, onClick=undefined
```

V12 migration에서 GROUP 카드들에 `action_type` 미설정:
- `admin_stats_card` → action_type = null
- `admin_users_card` → action_type = null
- `admin_logs_card` → action_type = null

→ Flyway UPDATE로 `admin_users_card`에 `action_type='LINK'`, `action_url='/view/USER_LIST'` 추가

> ⚠️ `admin_stats_card`, `admin_logs_card`의 링크 목적지 페이지(`/view/ADMIN_STATS`, `/view/ADMIN_LOGS`)는 미구현 상태 → 이번 배치에서 USER_LIST 카드만 적용, 나머지는 추후 작업

---

### Issue 3 — 사이드바 어드민 전용 UI

**Sidebar.tsx:43**

```tsx
// 현재: isLoggedIn 여부만 체크 → 모든 로그인 사용자에게 동일 nav 표시
{isRealLoggedIn ? (
    <nav>
        <div>콘텐츠 리스트 보기</div>
        <div>약속 관리</div>
    </nav>
```

`user?.role === 'ROLE_ADMIN'` 분기 추가:
- admin: 회원 관리 (`/view/USER_LIST`) nav + `is-admin` CSS 클래스 적용
- user: 기존 nav 유지

---

## 구현 계획 (2026-03-08)

| 순서 | 파일 | 변경 내용 | 비고 |
|------|------|-----------|------|
| 1 | `components/fields/TextField.tsx` | `{key}` → `data[key]` 치환 + `\n` → 줄바꿈 | 프론트 코드 변경 |
| 2 | `db/migration/V16__admin_ui_fixes.sql` | admin_stats_row ref_data_id 추가 / admin_users_card LINK 추가 | 신규 Flyway 마이그레이션 |
| 3 | `components/layout/Sidebar.tsx` | 어드민 분기 nav + `is-admin` 클래스 | 프론트 코드 변경 |
| 4 | `app/styles/pages.css` | `.pc-sidebar.is-admin` 다크모드 CSS | 스타일 추가 |


### ⏳ 다음 구현 순서

| 순서 | 작업                                                                                                   | 파일                                         | 상태                     |
| ---- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------- | ------------------------ |
| 1    | 서버 실행 + V12~V15 Flyway 마이그레이션 로컬 DB 적용                                                   | `./gradlew bootRun`                        | ✅                       |
| 2    | SecurityConfig에 `/api/admin/**` hasRole("ADMIN") 추가                                               | `SecurityConfig.java`                      | ✅                       |
| 3    | AdminUserController + AdminUserService 신규 생성                                                       | `domain/admin/controller`, `service`     | ✅                       |
| 4    | V16 마이그레이션 —`find_users_for_admin` query_master 등록 + `GET_ADMIN_STATS` required_role 수정 | `V16__admin_user_query.sql`                | ✅                       |
| 5    | V17 마이그레이션 — USER_LIST 스크린 ui_metadata                                                       | `V17__user_list_screen.sql`                | ✅                       |
| 6    | Frontend — ADMIN_USER_TABLE 컴포넌트 구현 + componentMap 등록                                         | `AdminUserTable.tsx`, `componentMap.tsx` | ✅<br /><br /><br />   |

---

## 구현 실행 결과 (2026-03-08)

### 1. `TextField.tsx` — 템플릿 치환 + 줄바꿈 처리

`components/fields/TextField.tsx` — finalValue 계산 후 추가:
- `{key}` → `data[key]` 정규식 치환 (`/\{([^}]+)\}/g`)
- 리터럴 `\n` → 실제 개행 (`finalValue.replace(/\n/g, '\n')`)
- `customStyle`에 `whiteSpace: 'pre-line'` 추가

**효과**: `'{total_users}\n총 사용자'` + `rowData = {total_users: '7'}` → `'7\n총 사용자'` → 2줄 표시

### 2. `V16__admin_ui_fixes.sql` — 신규 생성

`SDUI-server/src/main/resources/db/migration/V16__admin_ui_fixes.sql`

```sql
-- ⚠️ 버그 수정 이력: 최초 생성 시 screen_id = 'ADMIN_PAGE' 로 잘못 작성됨
-- 어드민 대시보드 컴포넌트는 V12~V13 기준 screen_id = 'MAIN_PAGE' 사용 (V13 확인)
-- 수정: 'ADMIN_PAGE' → 'MAIN_PAGE'
UPDATE ui_metadata SET ref_data_id = 'admin_stats_source'
WHERE component_id = 'admin_stats_row' AND screen_id = 'MAIN_PAGE';

UPDATE ui_metadata SET action_type = 'LINK', action_url = '/view/USER_LIST'
WHERE component_id = 'admin_users_card' AND screen_id = 'MAIN_PAGE';
```

**효과**: stats Repeater 활성화 + 회원관리 카드 클릭 가능

> ⚠️ **screen_id 버그 원인**: 어드민 URL이 `/view/admin/ADMIN_PAGE` 이지만 실제 DB `ui_metadata.screen_id`는 `'MAIN_PAGE'`임. MetadataProvider가 `/view/admin/{screenId}` 경로에서 screenId를 추출해 `GET /api/ui/MAIN_PAGE` 로 호출하기 때문.

### 3. `Sidebar.tsx` — 어드민 분기 nav

- `isAdmin = user?.role === 'ROLE_ADMIN'` 추가
- `<aside>` className: `isAdmin ? ' is-admin' : ''` 조건부 추가
- Admin nav: 대시보드, 회원 관리 / User nav: 기존 유지

### 4. `pages.css` — 어드민 다크 테마

파일 끝에 `.pc-sidebar.is-admin` 다크 테마 CSS 추가 (배경 `#1e2433`, 글자 `#e2e8f0`)

---

## 트러블슈팅 기록 (2026-03-08)

### Issue 4 — `./gradlew bootRun` 후 Redis 캐시가 구버전 메타데이터 반환

#### 증상
서버 재시작 후 `GET /api/ui/MAIN_PAGE` 응답이 여전히 구버전 메타데이터(V16 이전) 반환.

#### 원인
Redis는 Spring Boot 재시작과 무관한 독립 컨테이너. Flyway가 DB를 업데이트해도 Redis 캐시(1시간 TTL) 는 자동 무효화되지 않음.

#### 해결 방법

```bash
docker exec sdui-redis redis-cli FLUSHALL
```

> ⚠️ **Flyway 마이그레이션 후 항상 Redis 플러시 필요**

#### 관련 캐시 키 구조

| 캐시 종류 | Redis 키 형식 | TTL |
|----------|-------------|-----|
| UI 메타데이터 | `ROLE_ADMIN_MAIN_PAGE` 등 | 1시간 |
| SQL 쿼리 결과 | `SQL:GET_ADMIN_STATS` 등 | 1시간 |

---

### Issue 5 — Redis 플러시 후에도 통계 미치환 + 카드 클릭 불가 (V16 미적용)

#### 증상
Redis 플러시 후 새로고침해도 `{total_users}`, `{today_diaries}`, `{new_users}` 리터럴 표시.
API 응답(`GET /api/ui/MAIN_PAGE`)에서 `admin_stats_row.refDataId = null` 확인.

#### 원인 분석

| 단계 | 내용 |
|------|------|
| V16 최초 실행 | `WHERE screen_id = 'ADMIN_PAGE'` (잘못됨) → 0행 업데이트 → flyway_schema_history에 "적용 완료" 기록 |
| V16 파일 수정 | `screen_id = 'MAIN_PAGE'`로 수정 → `validate-on-migrate: false`라 체크섬 에러 없음 |
| 다음 bootRun | Flyway는 "V16 이미 적용됨" 판단 → **재실행 안 함** → DB 변경 없음 |
| 최종 상태 | DB: `admin_stats_row.ref_data_id = null`, `admin_users_card.action_type = null` |

**핵심**: `validate-on-migrate: false`는 체크섬 에러는 막아주지만, 이미 기록된 마이그레이션을 재실행시키지는 않음.

#### 해결 방법: V17 마이그레이션 신규 생성

`SDUI-server/src/main/resources/db/migration/V17__fix_admin_stats_and_rename.sql`

```sql
-- 1. Repeater 활성화
UPDATE ui_metadata SET ref_data_id = 'admin_stats_source'
WHERE component_id = 'admin_stats_row' AND screen_id = 'MAIN_PAGE';

-- 2. 카드 클릭 이동
UPDATE ui_metadata SET action_type = 'LINK', action_url = '/view/USER_LIST'
WHERE component_id = 'admin_users_card' AND screen_id = 'MAIN_PAGE';

-- 3. today_diaries → today_contents (V3 마이그레이션과 일치)
UPDATE query_master SET query_text = '...' WHERE sql_key = 'GET_ADMIN_STATS';

-- 4. UI 템플릿 키 일치
UPDATE ui_metadata SET label_text = '{today_contents}\n오늘의 콘텐츠'
WHERE component_id = 'admin_stat_diaries' AND screen_id = 'MAIN_PAGE';
```

#### 적용 순서

```bash
./gradlew bootRun           # V17 자동 실행
docker exec sdui-redis redis-cli FLUSHALL  # Redis 캐시 무효화
# 브라우저 새로고침
```

---

### 테스트 체크리스트

- [x] `./gradlew bootRun` → V16 마이그레이션 (파일은 올바르나 DB 미적용)
- [x] V17 생성 → `admin_stats_row.ref_data_id`, `admin_users_card.action_type` 수정 + today_diaries→today_contents
- [x] `./gradlew bootRun` → V17 마이그레이션 정상 적용 확인
- [x] `docker exec sdui-redis redis-cli FLUSHALL` → Redis 캐시 플러시
- [x] 어드민 로그인 → 사이드바 다크 테마 + 대시보드/회원관리 nav
- [x] 어드민 대시보드 → 통계 숫자 정상 표시 (`{total_users}` 치환 확인)
- [x] 어드민 대시보드 → 시스템 로그 목록 정상 표시
- [x] 회원 관리 카드 클릭 → `/view/USER_LIST` 이동
- [x] 일반 유저 로그인 → 기존 nav 유지 (회귀 테스트)

---

## 트러블슈팅 기록 (2026-03-09)

### Issue 6 — USER_LIST 화면 빈 화면 (`data: []`)

#### 증상
`/view/admin/USER_LIST` 접근 시 `{"status":"success","data":[],"message":null}` 반환.
DB 쿼리 `SELECT * FROM ui_metadata WHERE screen_id = 'USER_LIST'` 결과 0행.

#### 원인
V15 파일에 `required_role` ALTER TABLE 부분만 먼저 DB에 적용 → Flyway가 V15를 "완료"로 기록.
이후 V15 파일 하단에 USER_LIST INSERT가 추가되었으나 Flyway는 재실행하지 않음.
V16/V17은 MAIN_PAGE 수정만 포함 → USER_LIST 메타데이터 미적용 상태 지속.

#### 해결
`V18__user_list_screen.sql` 신규 생성 — V15 하단 INSERT를 별도 마이그레이션으로 분리 적용.
돌아가기 버튼 URL: `/view/MAIN_PAGE` → `/view/admin/MAIN_PAGE` (어드민 경로 일치).

---

### Issue 7 — PC 헤더 레이아웃: 돌아가기 버튼 가운데 배치 문제

#### 증상
PC에서 "회원 권한 관리" 제목과 "← 돌아가기" 버튼이 `justify-content: space-between`임에도
버튼이 가운데(또는 아래)에 위치.

#### 원인
`flex-row-layout` 전역 클래스에 `flex-wrap: wrap` 선언 → 타이틀 TEXT 컴포넌트가 flex item으로
남은 공간을 점유 → 버튼이 다음 줄로 래핑됨.

#### 해결 (`pages.css`)
```css
.admin-page-header.flex-row-layout { flex-wrap: nowrap; align-items: center; }
.admin-page-header .admin-page-title { flex: 1; }
.admin-back-btn { white-space: nowrap; flex-shrink: 0; }
```
결과: PC에서 제목은 왼쪽 확장, 버튼은 오른쪽 고정.

---

### Issue 8 — 돌아가기 버튼 `is-readonly` 스타일 (회색 비활성화)

#### 증상
버튼에 `.is-readonly` 클래스 자동 적용 → `background-color: #F1F5F9`, `cursor: default` 스타일.

#### 원인
`ui_metadata.is_readonly` 칼럼 DB 기본값: `DEFAULT true`.
V18 INSERT에서 `is_readonly` 미지정 → 모든 컴포넌트에 `is-readonly` 클래스 부여.

#### 해결 (`pages.css`)
```css
.admin-back-btn.is-readonly {
    background: none !important;
    background-color: transparent !important;
    cursor: pointer !important;
    border-color: #cbd5e1 !important;
}
```
DB 마이그레이션 없이 CSS 오버라이드로 처리 (버튼의 readonly 의미가 없는 경우 CSS가 적절).

---

### 최종 완료 체크리스트 (2026-03-09)

- [x] V18 마이그레이션 생성 → USER_LIST ui_metadata 정상 등록
- [x] pages.css — 헤더 flex-wrap 수정 + admin-back-btn is-readonly 오버라이드
- [x] `/view/admin/USER_LIST` 정상 렌더링 확인
- [x] 검색, 체크박스 선택(최대 5명), 권한 변경 기능 정상 동작 확인
