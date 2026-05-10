# RBAC 통합 테스트 가이드 (Manual Test)

**작성일:** 2026-03-01
**작성자:** QA Engineer
**테스트 대상:** RBAC (Role-Based Access Control) 구현
**관련 커밋:** RBAC Phase 1-3 완료

---

## 📋 목차

1. [테스트 개요](#테스트-개요)
2. [사전 준비](#사전-준비)
3. [테스트 시나리오](#테스트-시나리오)
4. [트러블슈팅](#트러블슈팅)
5. [테스트 결과 기록](#테스트-결과-기록)

---

## 테스트 개요

### 목적
RBAC 구현이 요구사항대로 동작하는지 수동으로 검증

### 테스트 범위
- ✅ 카카오 로그인 후 ROLE_GUEST → ROLE_USER 승격 플로우
- ✅ 역할별 메타데이터 필터링 (백엔드)
- ✅ 역할별 label_text/css_class 오버라이드
- ✅ 추가 정보 입력 API 동작
- ✅ AuthContext 리다이렉트 로직

### 핵심 요구사항
1. 같은 페이지여도 권한에 따라 **label_text**를 다르게 표시
2. 같은 페이지여도 권한에 따라 **css_class**를 다르게 적용
3. 카카오 로그인 후 권한에 따라 **추가 개인정보 입력** 화면 표시
4. SDUI 원칙 준수: **재배포 없이 DB만으로 UI 변경** 가능

---

## 사전 준비

### 1. 데이터베이스 마이그레이션 실행

#### PostgreSQL 접속 및 스크립트 실행
```bash
# 방법 1: psql 커맨드라인
psql -U postgres -d testdb -f V2__add_rbac_support.sql

# 방법 2: pgAdmin Query Tool
# V2__add_rbac_support.sql 파일 내용을 복사하여 실행
```

#### 실행 결과 확인
```sql
-- 1. ui_metadata 테이블 구조 확인
\d ui_metadata

-- 예상 결과: allowed_roles, label_text_overrides, css_class_overrides 컬럼 존재

-- 2. ADDITIONAL_INFO_PAGE 메타데이터 확인
SELECT component_id, label_text, component_type, sort_order, is_required
FROM ui_metadata
WHERE screen_id = 'ADDITIONAL_INFO_PAGE'
ORDER BY sort_order;

-- 예상 결과:
-- component_id         | label_text              | component_type       | sort_order | is_required
-- HEADER_TEXT          | 추가 정보를 입력해주세요  | TEXT                | 10         | false
-- PHONE_INPUT          | 전화번호                | INPUT               | 20         | true
-- ADDRESS_GROUP        | 주소                    | ADDRESS_SEARCH_GROUP| 30         | true
-- SUBMIT_BTN           | 제출하기                | BUTTON              | 40         | false

-- 3. RBAC 샘플 데이터 확인
SELECT component_id, screen_id, allowed_roles, label_text_overrides, css_class_overrides
FROM ui_metadata
WHERE allowed_roles IS NOT NULL
   OR label_text_overrides IS NOT NULL
   OR css_class_overrides IS NOT NULL;

-- 예상 결과:
-- DIARY_LIST_TITLE: label_text_overrides = {"ROLE_ADMIN":"전체 사용자 콘텐츠","ROLE_USER":"내 콘텐츠"}
-- ADMIN_DELETE_ALL_BTN: allowed_roles = 'ROLE_ADMIN'

-- 4. 인덱스 생성 확인
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'ui_metadata' AND indexname = 'idx_ui_metadata_screen_role';

-- 예상 결과: idx_ui_metadata_screen_role 존재
```

### 2. 서버 실행

#### 백엔드 서버 시작
```bash
cd SDUI-server
./gradlew bootRun
```

**로그 확인 사항:**
- ✅ Hibernate가 새 컬럼(allowed_roles, label_text_overrides, css_class_overrides)을 인식했는지
- ✅ UiService, UiController 빈이 정상적으로 로드되었는지
- ✅ PostgreSQL 연결 성공 (`HikariPool` 로그)

#### 프론트엔드 서버 시작
```bash
cd metadata-project
npm run dev
```

**확인 사항:**
- ✅ http://localhost:3000 접속 가능
- ✅ 콘솔 에러 없음

### 3. 테스트 계정 준비

#### ROLE_GUEST 테스트 (카카오 로그인)
- 카카오 개발자 계정 준비
- 테스트 앱 설정 확인

#### ROLE_USER 테스트 (일반 로그인)
```sql
-- 일반 사용자 생성 (이미 존재하면 스킵)
INSERT INTO users (user_id, email, password, role, phone, zip_code, road_address, verify_yn)
VALUES ('testuser', 'testuser@example.com', 'hashed_password', 'ROLE_USER', '010-1234-5678', '12345', '서울시 강남구', 'Y');
```

#### ROLE_ADMIN 테스트
```sql
-- 관리자 계정 생성 또는 기존 계정의 role 변경
UPDATE users SET role = 'ROLE_ADMIN' WHERE email = 'admin@example.com';
```

---

## 테스트 시나리오

### 🧪 TC-RBAC-001: 카카오 로그인 → 추가 정보 입력 플로우

#### 사전 조건
- 카카오 로그인이 처음인 사용자 (DB에 없는 이메일)
- 백엔드가 신규 카카오 사용자를 `ROLE_GUEST`로 생성

#### 테스트 단계
1. **카카오 로그인 실행**
   - http://localhost:3000/view/LOGIN_PAGE 접속
   - "카카오 로그인" 버튼 클릭
   - 카카오 계정 인증 완료

2. **자동 리다이렉트 확인**
   - **예상 동작:** 로그인 성공 후 `/view/MAIN_PAGE` → 즉시 `/view/ADDITIONAL_INFO_PAGE`로 리다이렉트
   - **확인 방법:** 브라우저 주소창 확인
   - **로그 확인:** 브라우저 콘솔에 `"ROLE_GUEST 감지: 추가 정보 입력 페이지로 리다이렉트"` 출력

3. **ADDITIONAL_INFO_PAGE 렌더링 확인**
   - **예상 화면 구성:**
     - "추가 정보를 입력해주세요" 헤더 텍스트
     - 전화번호 입력 필드 (필수)
     - 주소 검색 그룹 (필수)
     - 제출하기 버튼

4. **필수 항목 유효성 검사**
   - 전화번호 미입력 상태로 "제출하기" 클릭
   - **예상 결과:** "필수 항목을 입력해주세요" 알림

5. **추가 정보 입력**
   - 전화번호: `010-9999-8888`
   - 우편번호: `06000`
   - 도로명 주소: `서울시 강남구 테헤란로 123`
   - 상세 주소: `4층`
   - "제출하기" 클릭

6. **API 호출 확인**
   - **확인 방법:** 브라우저 개발자 도구 (F12) → Network 탭
   - **요청 URL:** `POST /api/auth/update-profile`
   - **요청 Body:**
     ```json
     {
       "phone": "010-9999-8888",
       "zipCode": "06000",
       "roadAddress": "서울시 강남구 테헤란로 123",
       "detailAddress": "4층"
     }
     ```
   - **응답 확인:**
     ```json
     {
       "message": "추가 정보가 저장되었습니다",
       "role": "ROLE_USER"
     }
     ```

7. **권한 업그레이드 확인**
   - **예상 동작:** "정보가 저장되었습니다" 알림 → `/view/DIARY_LIST`로 이동
   - **DB 확인:**
     ```sql
     SELECT email, role, phone, road_address, detail_address, zip_code, updated_at
     FROM users
     WHERE email = '카카오_이메일@kakao.com';
     ```
   - **예상 결과:**
     - `role`: `ROLE_USER` (ROLE_GUEST → ROLE_USER 변경됨)
     - `phone`, `road_address`, `detail_address`, `zip_code` 업데이트됨
     - `updated_at` 최신 시간

8. **재로그인 시 리다이렉트 안 됨 확인**
   - 로그아웃 후 다시 카카오 로그인
   - **예상 동작:** `/view/MAIN_PAGE`로 이동 (ADDITIONAL_INFO_PAGE로 리다이렉트 안 됨)
   - **이유:** role이 `ROLE_USER`이므로 AuthContext의 리다이렉트 조건 불충족

#### 예상 결과
✅ ROLE_GUEST → ADDITIONAL_INFO_PAGE 자동 리다이렉트
✅ 필수 항목 유효성 검사 동작
✅ `/api/auth/update-profile` API 호출 성공
✅ DB에 추가 정보 저장 및 role 업그레이드
✅ DIARY_LIST로 정상 이동
✅ 재로그인 시 리다이렉트 안 됨

#### 실패 시 확인 사항
- [ ] AuthContext.tsx의 useEffect가 실행되는가?
- [ ] `/api/auth/me` 응답에 `role: "ROLE_GUEST"`가 포함되는가?
- [ ] ADDITIONAL_INFO_PAGE 메타데이터가 DB에 존재하는가?
- [ ] useUserActions.tsx의 SUBMIT_ADDITIONAL_INFO case가 실행되는가?
- [ ] 백엔드 AuthController의 `/update-profile` 엔드포인트가 호출되는가?

---

### 🧪 TC-RBAC-002: 역할별 메타데이터 필터링 (백엔드)

#### 사전 조건
- DB에 `ADMIN_DELETE_ALL_BTN` 컴포넌트 존재 (`allowed_roles = 'ROLE_ADMIN'`)
- ROLE_USER와 ROLE_ADMIN 계정 준비

#### 테스트 단계

##### 케이스 A: ROLE_USER는 관리자 전용 컴포넌트 못 봄
1. ROLE_USER 계정으로 로그인
2. `/view/DIARY_LIST` 접속
3. 브라우저 개발자 도구 (F12) → Network 탭 열기
4. `/api/ui/DIARY_LIST` 요청 확인
5. **응답 Body 확인:**
   ```json
   {
     "success": true,
     "data": [
       {
         "componentId": "DIARY_LIST_TITLE",
         "labelText": "내 콘텐츠",  // 오버라이드 적용됨
         ...
       }
       // ADMIN_DELETE_ALL_BTN 컴포넌트는 응답에 없어야 함
     ]
   }
   ```
6. **화면 확인:** "전체 삭제" 버튼이 렌더링되지 않음

##### 케이스 B: ROLE_ADMIN은 모든 컴포넌트 접근 가능
1. ROLE_ADMIN 계정으로 로그인
2. `/view/DIARY_LIST` 접속
3. 브라우저 개발자 도구 (F12) → Network 탭 열기
4. `/api/ui/DIARY_LIST` 요청 확인
5. **응답 Body 확인:**
   ```json
   {
     "success": true,
     "data": [
       {
         "componentId": "DIARY_LIST_TITLE",
         "labelText": "전체 사용자 콘텐츠",  // 오버라이드 적용됨
         ...
       },
       {
         "componentId": "ADMIN_DELETE_ALL_BTN",
         "labelText": "전체 삭제",
         "componentType": "BUTTON",
         "cssClass": "btn-danger",
         ...
       }
     ]
   }
   ```
6. **화면 확인:** "전체 삭제" 버튼이 렌더링됨

##### 케이스 C: allowed_roles=NULL이면 모두 접근 가능
1. DB에서 특정 컴포넌트의 `allowed_roles`를 NULL로 설정:
   ```sql
   UPDATE ui_metadata SET allowed_roles = NULL WHERE component_id = 'SOME_COMPONENT';
   ```
2. ROLE_USER와 ROLE_ADMIN 모두 로그인하여 해당 컴포넌트가 보이는지 확인

#### 예상 결과
✅ ROLE_USER는 `allowed_roles='ROLE_ADMIN'` 컴포넌트가 응답에 없음
✅ ROLE_ADMIN은 모든 컴포넌트가 응답에 포함됨
✅ allowed_roles=NULL인 컴포넌트는 모두 접근 가능
✅ 백엔드 로그에 `userRole: ROLE_USER` 또는 `userRole: ROLE_ADMIN` 출력

#### 실패 시 확인 사항
- [ ] UiController에서 `extractRole()` 메서드가 올바른 역할을 반환하는가?
- [ ] UiService의 `isAccessible()` 메서드가 호출되는가?
- [ ] DB의 `allowed_roles` 컬럼 값이 올바른가?
- [ ] JWT 토큰에 role 클레임이 포함되어 있는가?

---

### 🧪 TC-RBAC-003: 역할별 label_text 오버라이드

#### 사전 조건
- DB에 label_text_overrides 설정된 컴포넌트 존재:
  ```sql
  UPDATE ui_metadata
  SET label_text = '콘텐츠 목록',
      label_text_overrides = '{"ROLE_ADMIN":"전체 사용자 콘텐츠","ROLE_USER":"내 콘텐츠"}'::jsonb
  WHERE component_id = 'DIARY_LIST_TITLE' AND screen_id = 'DIARY_LIST';
  ```

#### 테스트 단계

##### 케이스 A: ROLE_USER는 "내 콘텐츠" 표시
1. ROLE_USER 계정으로 로그인
2. `/view/DIARY_LIST` 접속
3. **화면 확인:** 페이지에 "내 콘텐츠" 텍스트가 표시됨
4. **API 응답 확인:**
   ```json
   {
     "componentId": "DIARY_LIST_TITLE",
     "labelText": "내 콘텐츠"  // 오버라이드 적용
   }
   ```

##### 케이스 B: ROLE_ADMIN은 "전체 사용자 콘텐츠" 표시
1. ROLE_ADMIN 계정으로 로그인
2. `/view/DIARY_LIST` 접속
3. **화면 확인:** 페이지에 "전체 사용자 콘텐츠" 텍스트가 표시됨
4. **API 응답 확인:**
   ```json
   {
     "componentId": "DIARY_LIST_TITLE",
     "labelText": "전체 사용자 콘텐츠"  // 오버라이드 적용
   }
   ```

##### 케이스 C: ROLE_GUEST는 기본값 표시
1. ROLE_GUEST 계정으로 로그인 (또는 비로그인)
2. 해당 컴포넌트가 접근 가능한 페이지 접속
3. **예상 결과:** "콘텐츠 목록" (기본 label_text) 표시

#### 예상 결과
✅ ROLE_USER는 "내 콘텐츠" 표시
✅ ROLE_ADMIN은 "전체 사용자 콘텐츠" 표시
✅ 오버라이드가 없는 역할은 기본값 표시
✅ 재배포 없이 DB만 수정하여 텍스트 변경 가능

#### 실패 시 확인 사항
- [ ] UiResponseDto의 `resolveOverriddenValue()` 메서드가 호출되는가?
- [ ] DB의 `label_text_overrides` 컬럼이 올바른 JSONB 형식인가?
- [ ] ObjectMapper가 JSONB를 올바르게 파싱하는가?

---

### 🧪 TC-RBAC-004: 역할별 css_class 오버라이드

#### 사전 조건
- DB에 css_class_overrides 설정:
  ```sql
  UPDATE ui_metadata
  SET css_class = 'btn-primary',
      css_class_overrides = '{"ROLE_ADMIN":"btn-danger","ROLE_USER":"btn-success"}'::jsonb
  WHERE component_id = 'SOME_BUTTON';
  ```

#### 테스트 단계

##### 케이스 A: ROLE_USER는 btn-success 클래스 적용
1. ROLE_USER 계정으로 로그인
2. 해당 버튼이 있는 페이지 접속
3. **API 응답 확인:**
   ```json
   {
     "componentId": "SOME_BUTTON",
     "cssClass": "btn-success"  // 오버라이드 적용
   }
   ```
4. **화면 확인:** 브라우저 개발자 도구 → Elements 탭에서 버튼의 class 확인
   ```html
   <button class="btn-success">...</button>
   ```

##### 케이스 B: ROLE_ADMIN은 btn-danger 클래스 적용
1. ROLE_ADMIN 계정으로 로그인
2. 해당 버튼이 있는 페이지 접속
3. **화면 확인:**
   ```html
   <button class="btn-danger">...</button>
   ```

#### 예상 결과
✅ 역할별로 다른 CSS 클래스 적용
✅ 버튼 색상/스타일이 역할에 따라 다르게 렌더링
✅ 재배포 없이 DB만 수정하여 스타일 변경 가능

---

### 🧪 TC-RBAC-005: React Query 캐시 키 분리

#### 목적
역할이 바뀌면 메타데이터 캐시가 자동으로 갱신되는지 확인

#### 테스트 단계
1. ROLE_USER로 로그인 → DIARY_LIST 접속
2. 브라우저 개발자 도구 → React Query Devtools 열기 (있는 경우)
3. **캐시 키 확인:** `USER_DIARY_LIST` (rolePrefix + screenId)
4. 로그아웃
5. ROLE_ADMIN으로 로그인 → DIARY_LIST 접속
6. **캐시 키 확인:** `ADMIN_DIARY_LIST`
7. **예상 동작:** 별도의 캐시 엔트리로 저장되어 역할별 다른 메타데이터 사용

#### 예상 결과
✅ rolePrefix 기반 캐시 키 분리
✅ 역할 변경 시 자동으로 다른 메타데이터 조회
✅ 캐시 충돌 없음

---

## 트러블슈팅

### ❌ 문제 1: ADDITIONAL_INFO_PAGE로 리다이렉트되지 않음

#### 증상
- 카카오 로그인 후 MAIN_PAGE에 머무름
- ADDITIONAL_INFO_PAGE로 이동하지 않음

#### 원인 및 해결

##### 원인 1: AuthContext의 useEffect가 실행되지 않음
```javascript
// 확인 방법: 브라우저 콘솔에 로그가 출력되는지 확인
useEffect(() => {
    console.log('AuthContext useEffect - user:', user, 'isLoading:', isLoading);
    if (!isLoading && user?.role === 'ROLE_GUEST') {
        console.log('ROLE_GUEST 감지: 추가 정보 입력 페이지로 리다이렉트');
        router.push('/view/ADDITIONAL_INFO_PAGE');
    }
}, [isLoading, user, router]);
```
**해결:** AuthProvider가 앱 최상단에서 감싸고 있는지 확인 (layout.tsx)

##### 원인 2: `/api/auth/me` 응답에 role이 없음
```bash
# 확인 방법: Network 탭에서 /api/auth/me 응답 확인
{
  "isLoggedIn": true,
  "email": "user@example.com",
  "role": "ROLE_USER"  // ← 이 필드가 있어야 함
}
```
**해결:** AuthController의 `/me` 엔드포인트에서 role을 응답에 포함하는지 확인

##### 원인 3: 백엔드가 ROLE_GUEST를 설정하지 않음
```sql
-- 확인 방법: DB에서 카카오 로그인한 사용자의 role 확인
SELECT email, role FROM users WHERE email = '카카오이메일@kakao.com';
```
**해결:** KakaoService의 `registerKakaoUser()` 메서드에서 신규 사용자 생성 시 `role = "ROLE_GUEST"` 설정 확인

---

### ❌ 문제 2: 제출 버튼 클릭 시 에러 발생

#### 증상
- "제출하기" 클릭 시 에러 알림 또는 아무 동작 없음

#### 원인 및 해결

##### 원인 1: formData의 키 이름이 맞지 않음
```javascript
// useUserActions.tsx에서 확인
const phone = currentFormData.PHONE_INPUT;  // 대문자 확인
const roadAddress = currentFormData.road_address;  // snake_case 확인
```
**해결:** ADDITIONAL_INFO_PAGE 메타데이터의 component_id와 일치하는지 확인

##### 원인 2: API 호출 실패 (401 Unauthorized)
```
POST /api/auth/update-profile 401
```
**해결:**
- JWT 쿠키가 제대로 전송되는지 확인 (withCredentials: true)
- 백엔드 SecurityConfig에서 `/api/auth/update-profile` 엔드포인트가 인증 필요한지 확인

##### 원인 3: 백엔드 엔드포인트 없음 (404 Not Found)
```
POST /api/auth/update-profile 404
```
**해결:** AuthController에 `/update-profile` 메서드가 추가되었는지 확인

---

### ❌ 문제 3: 관리자 전용 버튼이 보이지 않음 (ROLE_ADMIN)

#### 증상
- ROLE_ADMIN으로 로그인했는데 관리자 버튼이 안 보임

#### 원인 및 해결

##### 원인 1: DB에 데이터가 없음
```sql
SELECT * FROM ui_metadata WHERE component_id = 'ADMIN_DELETE_ALL_BTN';
-- 결과가 없으면 데이터 삽입 필요
```
**해결:** V2__add_rbac_support.sql 스크립트 실행 확인

##### 원인 2: 사용자 role이 실제로 ROLE_ADMIN이 아님
```sql
SELECT email, role FROM users WHERE email = '현재_로그인_이메일';
```
**해결:** role을 'ROLE_ADMIN'으로 업데이트

##### 원인 3: JWT에 role 클레임이 없음
```java
// JwtUtil.java의 createAccessToken() 확인
Claims claims = Jwts.claims().setSubject(user.getEmail());
claims.put("role", user.getRole());  // ← 이 부분이 있는지 확인
```
**해결:** JwtUtil에서 role 클레임을 추가하는 코드 확인

---

### ❌ 문제 4: label_text가 오버라이드되지 않음

#### 증상
- ROLE_USER로 로그인했는데 "내 콘텐츠" 대신 기본값 "콘텐츠 목록"이 표시됨

#### 원인 및 해결

##### 원인 1: DB의 label_text_overrides 형식 오류
```sql
-- 확인
SELECT component_id, label_text_overrides FROM ui_metadata WHERE component_id = 'DIARY_LIST_TITLE';

-- 올바른 형식
'{"ROLE_ADMIN":"전체 사용자 콘텐츠","ROLE_USER":"내 콘텐츠"}'

-- 잘못된 형식 (작은따옴표 사용)
"{'ROLE_ADMIN':'전체 사용자 콘텐츠','ROLE_USER':'내 콘텐츠'}"  -- X
```
**해결:** 올바른 JSON 형식으로 수정

##### 원인 2: UiResponseDto에서 역할이 전달되지 않음
```java
// UiService.java 확인
.map(entity -> new UiResponseDto(entity, userRole))  // userRole 전달 확인
```
**해결:** UiService에서 DTO 생성 시 userRole 파라미터 전달 확인

##### 원인 3: ObjectMapper 파싱 실패
```java
// UiResponseDto.java의 resolveOverriddenValue() 메서드에 로그 추가
try {
    ObjectMapper mapper = new ObjectMapper();
    Map<String, String> overrides = mapper.readValue(overridesJson, Map.class);
    System.out.println("Parsed overrides: " + overrides);  // 디버깅 로그
    return overrides.getOrDefault(userRole, defaultValue);
} catch (Exception e) {
    System.err.println("JSON parsing error: " + e.getMessage());  // 에러 로그
    return defaultValue;
}
```
**해결:** 백엔드 로그에서 파싱 에러 확인

---

### ❌ 문제 5: 컴파일 에러 발생

#### 증상
- Gradle 빌드 실패 또는 서버 시작 실패

#### 원인 및 해결

##### 원인 1: import 누락
```java
// UiService.java
import java.util.Arrays;  // 추가 필요

// UiResponseDto.java
import com.fasterxml.jackson.databind.ObjectMapper;  // 추가 필요
import java.util.Map;  // 추가 필요
```
**해결:** 누락된 import 추가

##### 원인 2: UserRepository 빈 주입 실패
```java
// AuthController.java
@Autowired
private UserRepository userRepository;  // 필드 주입 확인

// 생성자 주입으로 변경 권장
public AuthController(
    AuthService authService,
    JwtUtil jwtUtil,
    RefreshTokenRepository refreshTokenRepository,
    UserRepository userRepository  // 생성자에 추가
) {
    this.authService = authService;
    this.jwtUtil = jwtUtil;
    this.refreshTokenRepository = refreshTokenRepository;
    this.userRepository = userRepository;
}
```
**해결:** 생성자 주입으로 변경 또는 @Autowired 확인

---

## 테스트 결과 기록

### 테스트 정보
- **테스터:** _____________
- **테스트 일시:** 2026년 ___월 ___일 ___시___분
- **환경:**
  - OS: Windows 11 / macOS / Linux
  - 브라우저: Chrome / Firefox / Safari (버전: _______)
  - DB: PostgreSQL 15 (port 5433)
  - Backend: Spring Boot 3.1.4 (Java 17)
  - Frontend: Next.js 16.1.3 (React 19)

---

### TC-RBAC-001: 카카오 로그인 → 추가 정보 입력 플로우

| 단계 | 예상 결과 | 실제 결과 | 통과 여부 | 비고 |
|------|----------|----------|----------|------|
| 1. 카카오 로그인 | 로그인 성공 | | ☐ Pass ☐ Fail | |
| 2. ADDITIONAL_INFO_PAGE 리다이렉트 | 자동 이동 | | ☐ Pass ☐ Fail | |
| 3. 화면 렌더링 | 4개 컴포넌트 표시 | | ☐ Pass ☐ Fail | |
| 4. 필수 항목 검증 | 알림 표시 | | ☐ Pass ☐ Fail | |
| 5. 정보 제출 | API 성공 | | ☐ Pass ☐ Fail | |
| 6. DB 업데이트 | role=ROLE_USER | | ☐ Pass ☐ Fail | |
| 7. DIARY_LIST 이동 | 페이지 변경 | | ☐ Pass ☐ Fail | |
| 8. 재로그인 테스트 | 리다이렉트 안 됨 | | ☐ Pass ☐ Fail | |

**종합 결과:** ☐ 전체 통과 ☐ 일부 실패

**실패 상세:**
```
(실패한 단계와 에러 메시지 기록)
```

---

### TC-RBAC-002: 역할별 메타데이터 필터링

| 케이스 | 예상 결과 | 실제 결과 | 통과 여부 | 비고 |
|--------|----------|----------|----------|------|
| A. ROLE_USER 필터링 | ADMIN_BTN 응답 없음 | | ☐ Pass ☐ Fail | |
| B. ROLE_ADMIN 전체 접근 | ADMIN_BTN 응답 있음 | | ☐ Pass ☐ Fail | |
| C. allowed_roles=NULL | 모두 접근 가능 | | ☐ Pass ☐ Fail | |

**종합 결과:** ☐ 전체 통과 ☐ 일부 실패

---

### TC-RBAC-003: 역할별 label_text 오버라이드

| 케이스 | 예상 텍스트 | 실제 텍스트 | 통과 여부 | 비고 |
|--------|-------------|------------|----------|------|
| A. ROLE_USER | "내 콘텐츠" | | ☐ Pass ☐ Fail | |
| B. ROLE_ADMIN | "전체 사용자 콘텐츠" | | ☐ Pass ☐ Fail | |
| C. ROLE_GUEST | "콘텐츠 목록" (기본값) | | ☐ Pass ☐ Fail | |

**종합 결과:** ☐ 전체 통과 ☐ 일부 실패

---

### TC-RBAC-004: 역할별 css_class 오버라이드

| 케이스 | 예상 클래스 | 실제 클래스 | 통과 여부 | 비고 |
|--------|------------|------------|----------|------|
| A. ROLE_USER | btn-success | | ☐ Pass ☐ Fail | |
| B. ROLE_ADMIN | btn-danger | | ☐ Pass ☐ Fail | |

**종합 결과:** ☐ 전체 통과 ☐ 일부 실패

---

### TC-RBAC-005: React Query 캐시 키 분리

| 항목 | 예상 결과 | 실제 결과 | 통과 여부 | 비고 |
|------|----------|----------|----------|------|
| ROLE_USER 캐시 키 | USER_DIARY_LIST | | ☐ Pass ☐ Fail | |
| ROLE_ADMIN 캐시 키 | ADMIN_DIARY_LIST | | ☐ Pass ☐ Fail | |
| 역할 변경 시 재조회 | 자동 갱신 | | ☐ Pass ☐ Fail | |

**종합 결과:** ☐ 전체 통과 ☐ 일부 실패

---

### 전체 요약

| 테스트 케이스 | 통과 | 실패 | 비고 |
|-------------|-----|------|------|
| TC-RBAC-001 | ☐ | ☐ | |
| TC-RBAC-002 | ☐ | ☐ | |
| TC-RBAC-003 | ☐ | ☐ | |
| TC-RBAC-004 | ☐ | ☐ | |
| TC-RBAC-005 | ☐ | ☐ | |

**총 통과율:** ___% (통과 ___개 / 전체 ___개)

---

### 발견된 이슈

| 이슈 ID | 심각도 | 설명 | 재현 단계 | 상태 |
|---------|--------|------|----------|------|
| RBAC-BUG-001 | High/Medium/Low | | | Open/Fixed |
| RBAC-BUG-002 | High/Medium/Low | | | Open/Fixed |

**이슈 상세 설명:**
```
(스크린샷, 로그, 에러 메시지 첨부)
```

---

### 개선 제안

1. **성능 최적화:**
   -

2. **UX 개선:**
   -

3. **보안 강화:**
   -

---

### 다음 단계

- [ ] 발견된 이슈 수정
- [ ] Phase 4 (자동화 테스트 코드 작성) 진행
- [ ] 프로덕션 배포 준비

---

## 참고 자료

- **RBAC 구현 Plan:** `.claude/plans/glittery-nibbling-whistle.md`
- **SQL 마이그레이션:** `V2__add_rbac_support.sql`
- **백엔드 코드:**
  - `UiService.java` - 역할 기반 필터링 로직
  - `UiResponseDto.java` - 오버라이드 로직
  - `AuthController.java` - 추가 정보 입력 API
- **프론트엔드 코드:**
  - `AuthContext.tsx` - ROLE_GUEST 리다이렉트
  - `useUserActions.tsx` - SUBMIT_ADDITIONAL_INFO 액션

---

**문서 버전:** v1.0
**최종 수정일:** 2026-03-01
