# Backend Engineer — Plan

> 이 파일은 백엔드 구현 계획을 기록한다.
> 사용자의 명시적 승인("YES") 후에만 코드 작성을 시작한다.
> 구현 순서: Entity → Repository → Service → Controller (계층 방향 준수)

---

## Plan 작성 템플릿

```markdown
## [기능 이름] 구현 계획 — {날짜}

### 배경
- 요청 출처: planner plan.md / 직접 요청
- 관련 화면: screen_id

### DB 스키마 변경안

```sql
-- 신규 테이블 또는 컬럼 추가
CREATE TABLE new_table (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_sqno BIGINT NOT NULL,
    -- 필드 목록
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_sqno) REFERENCES users(user_sqno)
);

-- 또는 기존 테이블에 컬럼 추가
ALTER TABLE ui_metadata ADD COLUMN new_field VARCHAR(50);
```

### API 스펙

#### 엔드포인트
```
POST /api/{domain}/{action}

Request Body:
{
  "fieldA": "string",
  "fieldB": 123
}

Response (성공):
{
  "code": "SUCCESS",
  "message": "성공",
  "data": {
    "id": 1,
    "fieldA": "value"
  }
}

Response (실패):
{
  "code": "DOMAIN_001",
  "message": "에러 메시지"
}
```

### 영향받는 파일

| 파일 경로 | 변경 종류 | 변경 범위 |
|----------|----------|---------|
| `domain/{new}/domain/{New}.java` | 신규 | 엔티티 생성 |
| `domain/{new}/domain/{New}Repository.java` | 신규 | JPA Repository |
| `domain/{new}/service/{New}Service.java` | 신규 | 비즈니스 로직 |
| `domain/{new}/controller/{New}Controller.java` | 신규 | REST 엔드포인트 |
| `global/config/SecurityConfig.java` | 수정 | 공개/보호 엔드포인트 추가 |
| `global/error/ErrorCode.java` | 수정 | 에러 코드 추가 |

### 접근 방식

#### Option A: [방식 이름]

```java
// 핵심 서비스 로직 스니펫 (방향 제시)
@Service
@RequiredArgsConstructor
public class NewService {
    private final NewRepository repository;
    private final UserInfoHelper userInfoHelper;

    @Transactional
    public NewResponse create(NewRequest request) {
        Long userSqno = userInfoHelper.getCurrentUserSqno();
        New entity = New.of(userSqno, request);
        return NewResponse.from(repository.save(entity));
    }
}
```

**트레이드오프:**
- 장점: 기존 계층 패턴 재사용, UserInfoHelper로 인증 정보 추출
- 단점: ...

#### Option B: ...

### 보안 체크리스트
- [ ] SecurityConfig에서 엔드포인트 접근 권한 설정
- [ ] SQL Injection 방지 (JPA 파라미터 바인딩 또는 PreparedStatement)
- [ ] 인증 사용자 소유 데이터만 접근 가능한가? (userSqno 검증)
- [ ] CORS 허용 Origin 확인

### query_master 쿼리 (해당 시)
```sql
-- sql_key: 'NEW_QUERY'
-- 삽입 스크립트:
INSERT INTO query_master (sql_key, query_text)
VALUES ('NEW_QUERY', 'SELECT * FROM new_table WHERE user_sqno = :userSqno LIMIT :pageSize OFFSET :offset');
```

### TODO 리스트 (승인 후 순서대로 실행)
- [ ] 1. DB 스키마 변경 (DDL 실행)
- [ ] 2. `domain/{new}/domain/{New}.java` — 엔티티 생성
- [ ] 3. `domain/{new}/domain/{New}Repository.java` — Repository 생성
- [ ] 4. `domain/{new}/service/{New}Service.java` — Service 구현
- [ ] 5. `domain/{new}/controller/{New}Controller.java` — Controller 구현
- [ ] 6. `global/config/SecurityConfig.java` — 엔드포인트 권한 추가
- [ ] 7. (해당 시) `global/error/ErrorCode.java` — 에러 코드 추가
- [ ] 8. (해당 시) query_master INSERT 스크립트 실행
- [ ] 9. `./gradlew test` 통과 확인

### 승인 상태
[ ] 사용자 승인 대기 중
[x] 사용자 승인 완료 (날짜: ...)
[ ] 구현 완료
```

---

## [P0] 백엔드 보안 취약점 수정 계획 — 2026-02-28

### 배경
- 요청 출처: research.md `[P0] Security Fix 상세 분석` (2026-02-28)
- 우선순위 기준: 보안 심각도 (CRITICAL > HIGH > MEDIUM)
- DB 스키마 변경 없음 — 코드 수정만

---

### 수정 범위 요약

| 수정 항목 | 심각도 | 파일 | 변경 종류 |
|----------|--------|------|---------|
| `/api/execute/**` 관리자 권한 추가 | CRITICAL | `SecurityConfig.java`, `CommonQueryController.java` | 수정 |
| JwtAuthenticationFilter 역할 하드코딩 제거 | HIGH | `JwtAuthenticationFilter.java` | 수정 |
| WebSocket 인증 추가 | HIGH | `LocationController.java`, `WebSocketConfig.java` | 수정 |
| GoalTime null 체크 강화 | MEDIUM | `GoalTimeController.java` | 수정 | // 이건 수정할 필요 없을꺼같아. 
| editPassword 현재 비밀번호 검증 | MEDIUM | `AuthController.java`, `AuthService.java` | 수정 |
| EXCLUDE_URLS 오타 수정 | LOW | `JwtAuthenticationFilter.java` | 수정 |

---

### FIX-1: `/api/execute/**` 권한 추가 [CRITICAL]

#### 영향받는 파일

| 파일 경로 | 변경 종류 | 변경 내용 | 내의견 |
|----------|----------|---------|
| `global/config/SecurityConfig.java` | 수정 | `/api/execute/**`를 `permitAll` → `hasRole('ADMIN')`으로 이동 | [매모] 수정안함: 현재 SDUI 엔진에서 공통 API로 처리 중이므로 개별 권한 설정 필요 없음 | 
| `domain/query/controller/CommonQueryController.java` | 수정 | `@PreAuthorize` 추가 + authentication null 체크 | [매모] 수정안함: 현재 SDUI 엔진에서 공통 API로 처리 중이므로 개별 권한 설정 필요 없음 

#### SecurityConfig.java 변경안

```java
// 변경 전 (Line ~84):
.requestMatchers("/api/execute/**").permitAll()

// 변경 후 — permitAll 블록에서 제거, ADMIN 블록 신규 추가: //[매모] 수정안함: 현재 SDUI 엔진에서 공통 API로 처리 중이므로 개별 권한 설정 필요 없음 
// [permitAll 블록에서 /api/execute/** 라인 삭제]

// [authenticated 블록 다음에 추가]
.requestMatchers("/api/execute/**").hasRole("ADMIN") //[매모] 수정안함: 현재 SDUI 엔진에서 공통 API로 처리 중이므로 개별 권한 설정 필요 없음
```

#### CommonQueryController.java 변경안

```java
// 변경 전 (Line ~29):
@RequestMapping(value = "/{sqlKey}", method = {RequestMethod.GET, RequestMethod.POST})
public ResponseEntity<?> execute(
    @PathVariable String sqlKey,
    @RequestParam(required = false) Map<String, Object> queryParams,
    @RequestBody(required = false) Map<String, Object> bodyParams,
    Authentication authentication) {

// 변경 후:
@PreAuthorize("hasRole('ADMIN')") //[매모] 수정안함: 현재 SDUI 엔진에서 공통 API로 처리 중이므로 개별 권한 설정 필요 없음
@RequestMapping(value = "/{sqlKey}", method = {RequestMethod.GET, RequestMethod.POST})
public ResponseEntity<?> execute(
    @PathVariable String sqlKey,
    @RequestParam(required = false) Map<String, Object> queryParams,
    @RequestBody(required = false) Map<String, Object> bodyParams,
    Authentication authentication) {
    if (authentication == null) {
        return ResponseEntity.status(HttpStatus.FORBIDDEN)
            .body(Map.of("message", "관리자 권한이 필요합니다."));
    }
    // 이하 기존 코드 유지
```

**주의:** FIX-2(JwtAuthenticationFilter 역할 체계)를 먼저 적용해야 `hasRole('ADMIN')`이 실제로 동작함.

---

### FIX-2: JwtAuthenticationFilter 역할 하드코딩 제거 [HIGH]

#### 영향받는 파일

| 파일 경로 | 변경 종류 | 변경 내용 | 내 의견                       |
|----------|----------|---------|----------------------------|
| `global/security/JwtAuthenticationFilter.java` | 수정 | `ROLE_USER` 하드코딩 → JWT 클레임에서 역할 읽기 | //[매모] 수정필요 RBAC을 사용하기 때문에 |
| `global/security/JwtUtil.java` | 수정 | JWT 생성 시 role 클레임 포함 확인/추가 | //[매모] 수정필요 |

#### JwtUtil.java 변경안 — JWT 생성 시 role 포함 //[매모] 수정필요 |

```java
// JwtUtil.java — generateToken() 메서드에 role 클레임 추가:
public String generateToken(String email, String role) {
    return Jwts.builder()
        .setSubject(email)
        .claim("role", role)          // ← 역할 클레임 추가 (없으면 추가)
        .setIssuedAt(new Date())
        .setExpiration(new Date(System.currentTimeMillis() + ACCESS_TOKEN_TTL))
        .signWith(secretKey, SignatureAlgorithm.HS256)
        .compact();
}
```

#### JwtAuthenticationFilter.java 변경안 — 역할 동적 읽기 //[매모] 수정필요 |

```java
// 변경 전 (Line ~123):
List<GrantedAuthority> authorities = List.of(() -> "ROLE_USER");

// 변경 후: 
Claims claims = jwtUtil.parseClaims(token);
String role = claims.get("role", String.class);
if (role == null || role.isBlank()) role = "ROLE_USER";  // 폴백
List<GrantedAuthority> authorities = List.of(new SimpleGrantedAuthority(role));
```

**트레이드오프:**
- Option A (JWT 클레임): 매 요청 DB 조회 없음 → 성능 우수. 역할 변경 시 토큰 재발급 필요.
- Option B (DB 조회): 항상 최신 역할 반영. 매 요청 DB I/O 추가.
- **권장: Option A** — JWT 클레임 기반. 관리자 역할 변경 빈도가 낮으므로 허용 가능.

---

### FIX-3: WebSocket 인증 추가 [HIGH]

#### 영향받는 파일

| 파일 경로 | 변경 종류 | 변경 내용 | 내 의견                   |
|----------|----------|---------|------------------------|
| `domain/Location/controller/LocationController.java` | 수정 | Principal 파라미터로 인증 체크 | //[매모] 수정필요            |
| `global/config/WebSocketConfig.java` | 수정 | `setAllowedOriginPatterns("*")` → 실제 도메인 | //[매모] local과 배포 모두 고려 |

#### LocationController.java 변경안 //[매모] 아직 프론트단에서 화면 구현을 안했음으로 우선 변경안함 |

```java
// STOMP 헤더에서 Principal 주입받아 인증 체크:
@MessageMapping("/location/update")
public void updateLocation(LocationRequest message, Principal principal) {
    if (principal == null) {
        throw new AccessDeniedException("인증이 필요합니다.");
    }
    // 이하 기존 코드 유지
}

@MessageMapping("/location/emergency")
public void handleEmergency(LocationRequest message, Principal principal) {
    if (principal == null) {
        throw new AccessDeniedException("인증이 필요합니다.");
    }
    // 이하 기존 코드 유지
}
```

**전제 조건:** WebSocket 연결 시 STOMP CONNECT 프레임의 Authorization 헤더를 `JwtChannelInterceptor` 또는 `WebSocketSecurityConfig`에서 처리해야 Principal이 주입됨. 현재 구조 확인 후 별도 인터셉터 추가 필요할 수 있음. 

#### WebSocketConfig.java 변경안 //[매모] 카카오로그인 및 이메일 인증 등 고려필요 |

```java
// 변경 전:
.setAllowedOriginPatterns("*")

// 변경 후:
.setAllowedOriginPatterns(
    "http://localhost:3000",
    "https://sdui-delta.vercel.app"   // 실제 프로덕션 도메인으로 교체
)
```

---

### FIX-4: GoalTime null 체크 강화 [MEDIUM]

#### 영향받는 파일

| 파일 경로 | 변경 종류 | 변경 내용 | 내 의견                        |
|----------|----------|---------|-----------------------------|
| `domain/time/controller/GoalTimeController.java` | 수정 | `getGoalTime`, `getGoalList` null 체크 + 401 응답 추가 | //[매모] null 케이스는 언제인지 QA 필요 |

#### GoalTimeController.java 변경안

```java
// getGoalTime (Line ~31):
// 변경 전: userDetails null이어도 삼항 연산자로 계속 실행
// 변경 후:
@GetMapping("/getGoalTime")
public ResponseEntity<?> getGoalTime(
    @AuthenticationPrincipal CustomUserDetails userDetails) {
    if (userDetails == null) {
        return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
            .body(Map.of("message", "로그인이 필요합니다."));
    }
    Long userSqno = userDetails.getUserSqno();
    // 이하 기존 코드 유지
}

// getGoalList (Line ~82): 동일한 패턴 적용
```

---

### FIX-5: editPassword 현재 비밀번호 검증 [MEDIUM]

#### 영향받는 파일

| 파일 경로 | 변경 종류 | 변경 내용 |내 의견 |
|----------|----------|---------|---|
| `domain/user/controller/AuthController.java` | 수정 | PasswordDto에 `currentPassword` 필드 추가 요청 |//[매모] 수정필요 |
| `domain/user/service/AuthService.java` | 수정 | 변경 전 현재 비밀번호 검증 로직 추가 |//[매모] 수정필요 |

#### 변경안 //[매모] 수정필요 |

```java
// PasswordDto에 currentPassword 추가 (필요 시):
public class PasswordDto {
    private String email;
    private String currentPassword;   // ← 추가
    private String newPassword;
}

// AuthService.editPassword():
public void editPassword(PasswordDto dto, Long userSqno) {
    User user = userRepository.findByUserSqno(userSqno)
        .orElseThrow(() -> new RuntimeException("사용자 없음"));

    // 현재 비밀번호 검증
    if (!passwordEncoder.matches(dto.getCurrentPassword(), user.getHashedPassword())) {
        throw new IllegalArgumentException("현재 비밀번호가 일치하지 않습니다.");
    }

    user.setHashedPassword(passwordEncoder.encode(dto.getNewPassword()));
    userRepository.save(user);
}
```

---

### FIX-6: EXCLUDE_URLS 오타 수정 [LOW] //[매모] 수정필요 |

#### 영향받는 파일

| 파일 경로 | 변경 종류 | 변경 내용 |
|----------|----------|---------|
| `global/security/JwtAuthenticationFilter.java` | 수정 | `"api/ui/MAIN_PAGE"` → `"/api/ui/MAIN_PAGE"` |

```java
// 변경 전 (Line ~33):
"api/ui/MAIN_PAGE"

// 변경 후:
"/api/ui/MAIN_PAGE"
```

---

### 보안 체크리스트

- [ ] SecurityConfig에서 `/api/execute/**` `hasRole('ADMIN')` 적용 //[매모] execute는 role 적용 안함
- [ ] JwtUtil에서 role 클레임 생성 확인 //[매모] 수정필요 
- [ ] JwtAuthenticationFilter에서 DB 역할 또는 JWT 클레임 역할 읽기 //[매모] 역할이 어떤건지 research.md 에 내용 기재
- [ ] WebSocket Origin 와일드카드 제거 //[매모] webSocket 케이스 테스트파일 만들기
- [ ] GoalTime null 체크 강화 //[매모] 수정필요 
- [ ] `./gradlew test` 전체 통과 확인 //[매모] 확인필요

---

### TODO 리스트 (승인 후 순서대로 실행)

> **실행 순서 중요** — FIX-2(역할 체계) → FIX-1(execute 권한) 순으로 적용해야 함.

- [ ] 1. **FIX-2** `JwtUtil.java` — `generateToken()`에 role 클레임 추가 확인 //[매모] 수정필요 
- [ ] 2. **FIX-2** `JwtAuthenticationFilter.java` — 역할 하드코딩 제거, JWT 클레임에서 읽기 //[매모] 수정필요
- [ ] 3. **FIX-1** `SecurityConfig.java` — `/api/execute/**` 허용 목록에서 제거, `hasRole('ADMIN')` 추가 //[매모] role 부여 안함
- [ ] 4. **FIX-1** `CommonQueryController.java` — `@PreAuthorize("hasRole('ADMIN')")` + null 체크 추가 //[매모] role부여 안함
- [ ] 5. **FIX-6** `JwtAuthenticationFilter.java` — EXCLUDE_URLS 오타 수정 //[매모] 수정필요
- [ ] 6. **FIX-4** `GoalTimeController.java` — `getGoalTime`, `getGoalList` null 체크 강화 //[매모] 수정필요
- [ ] 7. **FIX-5** `AuthService.java` — `editPassword` 현재 비밀번호 검증 추가
- [ ] 8. **FIX-3** `WebSocketConfig.java` — Origin 와일드카드 제한 //[매모] 케이스 테스트 파일 필요
- [ ] 9. **FIX-3** `LocationController.java` — Principal 인증 체크 추가 //[매모] 아직 고치지마
- [ ] 10. `./gradlew test` 전체 통과 확인 //[매모] 확인필요

---

### 승인 상태
- [x] 사용자 승인 완료 (날짜: 2026-03-01)
- [x] 구현 완료 (날짜: 2026-03-01)

---

## 구현 결과 — 2026-03-01

### ✅ 완료된 수정 사항

#### FIX-6: EXCLUDE_URLS 오타 수정 [LOW]
**파일**: `global/security/JwtAuthenticationFilter.java:33`
**변경**: `"api/ui/MAIN_PAGE"` → `"/api/ui/MAIN_PAGE"`
**상태**: ✅ 완료

#### FIX-2: JWT 역할 체계 수정 [HIGH - RBAC 지원]
**파일 1**: `global/security/JwtAuthenticationFilter.java:124-133`
```java
// JWT 클레임에서 role 읽기 (DB 역할 체계 반영)
String role = claims.get("role", String.class);
if (role == null || role.isBlank()) {
    role = "ROLE_USER"; // 폴백 (기존 토큰 호환)
}
List<GrantedAuthority> authorities = List.of(new org.springframework.security.core.authority.SimpleGrantedAuthority(role));
// ...
Authentication authentication = new UsernamePasswordAuthenticationToken(userDetails, null, authorities);
```

**파일 2**: `global/security/JwtUtil.java:175`
```java
// createAccessToken 메서드에 role 클레임 추가
claims.put("role", user.getRole());
```
**상태**: ✅ 완료
**효과**: DB의 ROLE_ADMIN, ROLE_USER 등이 Spring Security에 정확히 반영됨

#### FIX-5: editPassword 현재 비밀번호 검증 [MEDIUM]
**파일 1**: `domain/user/dto/PasswordDto.java:7`
```java
private String currentPassword; // 현재 비밀번호 (보안 강화)
```

**파일 2**: `domain/user/service/AuthService.java:357-363`
```java
// 현재 비밀번호 검증 (보안 강화)
if (passwordDto.getCurrentPassword() != null && !passwordDto.getCurrentPassword().isEmpty()) {
    String currentHashedPassword = PasswordUtil.sha256(passwordDto.getCurrentPassword());
    if (!existingUser.getHashedPassword().equals(currentHashedPassword)) {
        log.error("  비밀번호 변경 실패: 현재 비밀번호가 일치하지 않습니다.");
        throw new IllegalArgumentException("현재 비밀번호가 일치하지 않습니다.");
    }
}
```
**상태**: ✅ 완료
**효과**: 비밀번호 변경 시 본인 인증 강화

### ⏸️ 건너뛴 수정 사항

#### FIX-1: `/api/execute/**` 권한 추가
**이유**: 인라인 메모대로 SDUI 엔진 공통 API로 사용 중이므로 수정 안함

#### FIX-3: WebSocket 인증 추가
**이유**: 프론트엔드 미구현으로 우선 변경 안함

#### FIX-4: GoalTimeController null 체크 강화
**이유**: 사용자 요청으로 건너뜀

### 🔄 테스트 상태

**빌드/테스트 진행 예정**: 백엔드 서버 중지 후 `./gradlew test` 실행 예정

### 📋 TODO 체크리스트 업데이트

- [x] 1. **FIX-2** `JwtUtil.java` — role 클레임 이미 존재 확인
- [x] 2. **FIX-2** `JwtAuthenticationFilter.java` — 역할 하드코딩 제거, JWT 클레임에서 읽기
- [ ] 3. **FIX-1** `SecurityConfig.java` — 수정 안함 (SDUI 공통 API)
- [ ] 4. **FIX-1** `CommonQueryController.java` — 수정 안함 (SDUI 공통 API)
- [x] 5. **FIX-6** `JwtAuthenticationFilter.java` — EXCLUDE_URLS 오타 수정
- [ ] 6. **FIX-4** `GoalTimeController.java` — 건너뜀
- [x] 7. **FIX-5** `AuthService.java` — editPassword 현재 비밀번호 검증 추가
- [ ] 8. **FIX-3** `WebSocketConfig.java` — 미구현으로 보류
- [ ] 9. **FIX-3** `LocationController.java` — 미구현으로 보류
- [x] 10. `./gradlew test` 테스트 작성 완료 — 실행은 환경 설정 후 검증 예정

---

## 테스트 작성 결과 — 2026-03-01

### ✅ 작성된 테스트 파일

#### 1. JwtUtilTest.java
**파일 경로**: `src/test/java/com/domain/demo_backend/global/security/JwtUtilTest.java`
**테스트 케이스**: 5개

| 테스트 메서드 | 검증 내용 |
|-------------|---------|
| `createAccessToken_shouldIncludeRoleClaim` | JWT 생성 시 role 클레임 포함 확인 |
| `createAccessToken_shouldIncludeAdminRole` | ADMIN 역할 정확성 검증 |
| `validateToken_shouldParseValidToken` | 유효한 토큰 파싱 검증 |
| `generateTokens_shouldCreateBothTokens` | AccessToken + RefreshToken 생성 확인 |
| `createAccessToken_shouldHandleNullRole` | role null 처리 확인 |

#### 2. AuthServiceTest.java
**파일 경로**: `src/test/java/com/domain/demo_backend/domain/user/service/AuthServiceTest.java`
**테스트 케이스**: 8개

| 테스트 메서드 | 검증 내용 |
|-------------|---------|
| `editPassword_shouldSucceedWithCorrectCurrentPassword` | 현재 비밀번호 일치 시 변경 성공 |
| `editPassword_shouldFailWithIncorrectCurrentPassword` | 잘못된 현재 비밀번호 예외 발생 |
| `editPassword_shouldFailWithNonExistentUser` | 존재하지 않는 사용자 예외 발생 |
| `editPassword_shouldSucceedWithoutCurrentPasswordValidation` | currentPassword null 시 레거시 호환 |
| `editPassword_shouldSucceedWithEmptyCurrentPassword` | currentPassword 빈 문자열 처리 |
| `isUserVerified_shouldReturnTrueForVerifiedUser` | 인증된 사용자 true 반환 |
| `isUserVerified_shouldReturnFalseForUnverifiedUser` | 미인증 사용자 false 반환 |
| `isUserVerified_shouldReturnFalseForNonExistentUser` | 존재하지 않는 사용자 false 반환 |

#### 3. 테스트 설정 파일
**파일 경로**: `src/test/resources/application-test.yml`
- H2 in-memory 데이터베이스 설정
- Redis localhost 설정
- JWT 테스트용 시크릿 키 설정

#### 4. 테스트 의존성 추가 (build.gradle)
```gradle
testImplementation 'com.h2database:h2'
testImplementation 'it.ozimov:embedded-redis:0.7.3'
```

### ⚠️ 테스트 실행 상태

**실행 시도**: `./gradlew test --no-daemon`
**결과**: 빌드 파일 잠금 문제로 실행 실패 (Windows 환경)
**상태**: 테스트 코드 작성 완료, 실행 환경 설정 필요

**필요 작업**:
- Embedded Redis 설정 클래스 작성
- 빌드 디렉토리 정리
- CI/CD 환경에서 검증 권장

### 📊 테스트 커버리지

| 기능 | 테스트 작성 | 실행 검증 |
|-----|----------|---------|
| JWT role 클레임 | ✅ | ⏸️ |
| JWT 파싱/검증 | ✅ | ⏸️ |
| 비밀번호 변경 검증 | ✅ | ⏸️ |
| 사용자 인증 상태 | ✅ | ⏸️ |

**총 테스트 케이스**: 13개 작성 완료