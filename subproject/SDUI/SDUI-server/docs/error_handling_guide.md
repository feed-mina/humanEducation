# 백엔드 에러 처리 가이드

## 📋 개요

프론트엔드 에러 처리 유틸리티와 호환되는 일관된 API 응답 형식을 제공합니다.

---

## 🏗️ 구조

```
global/
├── common/response/
│   └── ApiResponse.java        # 공통 응답 형식
├── exception/
│   ├── BusinessException.java  # 비즈니스 예외 베이스
│   ├── GlobalExceptionHandler.java  # 전역 예외 처리
│   └── custom/
│       ├── DuplicateEmailException.java
│       ├── DuplicatePhoneException.java
│       ├── ResourceNotFoundException.java
│       └── UnauthorizedException.java
```

---

## 🛠️ 사용 방법

### 1. 성공 응답

```java
@PostMapping("/content/create")
public ResponseEntity<ApiResponse<Content>> createContent(@RequestBody ContentRequest request) {
    Content content = contentService.create(request);

    // 방법 1: 데이터만
    return ResponseEntity.ok(ApiResponse.success(content));

    // 방법 2: 메시지 + 데이터
    return ResponseEntity.ok(ApiResponse.success("콘텐츠가 생성되었습니다", content));
}
```

**응답 예제:**
```json
{
  "status": "success",
  "message": "콘텐츠가 생성되었습니다",
  "data": {
    "contentId": 123,
    "title": "제목"
  },
  "timestamp": "2026-03-02T19:00:00"
}
```

---

### 2. 커스텀 예외 사용 (권장)

```java
@PostMapping("/register")
public ResponseEntity<ApiResponse<User>> register(@RequestBody RegisterRequest request) {
    // 이메일 중복 체크
    if (userRepository.existsByEmail(request.getEmail())) {
        throw new DuplicateEmailException(request.getEmail());
    }

    // 전화번호 중복 체크
    if (userRepository.existsByPhone(request.getPhone())) {
        throw new DuplicatePhoneException(request.getPhone());
    }

    User user = authService.register(request);
    return ResponseEntity.status(201).body(ApiResponse.success("회원가입이 완료되었습니다", user));
}
```

**에러 응답 (자동 처리됨):**
```json
{
  "status": "error",
  "message": "이미 사용 중인 이메일입니다: test@example.com",
  "error": "DuplicateEmailException",
  "path": "/api/auth/register",
  "timestamp": "2026-03-02T19:00:00"
}
```

---

### 3. 직접 에러 응답 반환

```java
@PostMapping("/login")
public ResponseEntity<ApiResponse<TokenResponse>> login(@RequestBody LoginRequest request) {
    try {
        TokenResponse token = authService.login(request);
        return ResponseEntity.ok(ApiResponse.success("로그인 성공", token));
    } catch (BadCredentialsException e) {
        // 직접 에러 응답 생성
        return ResponseEntity
                .status(401)
                .body(ApiResponse.error("이메일 또는 비밀번호가 올바르지 않습니다"));
    }
}
```

---

### 4. 간단한 Map 응답 (하위 호환성)

```java
@PostMapping("/logout")
public ResponseEntity<?> logout() {
    authService.logout();

    // 간단한 성공 응답
    return ResponseEntity.ok(ApiResponse.simpleSuccess("로그아웃되었습니다"));

    // 간단한 에러 응답
    // return ResponseEntity.badRequest().body(ApiResponse.simpleError("로그아웃 실패"));
}
```

---

## 🎯 AuthController 적용 예제

### Before (기존 코드)

```java
@PostMapping("/register")
public ResponseEntity<?> register(@RequestBody RegisterRequest request) {
    try {
        User user = authService.register(request);
        return ResponseEntity.status(201).body(user);
    } catch (Exception e) {
        return ResponseEntity.badRequest().body(e.getMessage());  // ❌ 일관성 없음
    }
}
```

### After (개선 코드)

```java
@PostMapping("/register")
public ResponseEntity<ApiResponse<User>> register(@RequestBody RegisterRequest request) {
    // 중복 체크 (예외 자동 처리)
    if (userRepository.existsByEmail(request.getEmail())) {
        throw new DuplicateEmailException(request.getEmail());
    }
    if (userRepository.existsByPhone(request.getPhone())) {
        throw new DuplicatePhoneException(request.getPhone());
    }

    User user = authService.register(request);
    return ResponseEntity.status(201).body(
        ApiResponse.success("회원가입이 완료되었습니다", user)
    );
}
```

---

## 📦 커스텀 예외 만들기

### 예제: InvalidPasswordException

```java
package com.domain.demo_backend.global.exception.custom;

import com.domain.demo_backend.global.exception.BusinessException;
import org.springframework.http.HttpStatus;

public class InvalidPasswordException extends BusinessException {

    public InvalidPasswordException() {
        super("비밀번호가 올바르지 않습니다", HttpStatus.UNAUTHORIZED);
    }

    public InvalidPasswordException(String message) {
        super(message, HttpStatus.UNAUTHORIZED);
    }
}
```

### 사용

```java
@PostMapping("/change-password")
public ResponseEntity<ApiResponse<Void>> changePassword(@RequestBody PasswordRequest request) {
    if (!authService.verifyPassword(request.getOldPassword())) {
        throw new InvalidPasswordException("현재 비밀번호가 일치하지 않습니다");
    }

    authService.changePassword(request.getNewPassword());
    return ResponseEntity.ok(ApiResponse.success("비밀번호가 변경되었습니다", null));
}
```

---

## 🔄 자동 처리되는 예외들

GlobalExceptionHandler가 자동으로 처리하는 예외:

| 예외 타입 | HTTP 상태 | 기본 메시지 |
|-----------|-----------|-------------|
| `BusinessException` | 예외에 설정된 상태 | 예외 메시지 |
| `MethodArgumentNotValidException` | 400 | "입력값을 확인해주세요" |
| `DataIntegrityViolationException` | 400 | "데이터 저장에 실패했습니다" |
| `IllegalArgumentException` | 400 | 예외 메시지 |
| `NullPointerException` | 500 | "서버 내부 오류가 발생했습니다" |
| `Exception` | 500 | "서버 오류가 발생했습니다" |

---

## 🚀 마이그레이션 가이드

### 단계 1: 기존 컨트롤러 확인

```bash
# 기존 에러 응답 패턴 검색
grep -r "ResponseEntity.badRequest" src/
grep -r "ResponseEntity.status(400)" src/
grep -r "catch (Exception" src/
```

### 단계 2: 점진적 적용

1. **새 엔드포인트**: `ApiResponse` 사용
2. **중요 엔드포인트**: 인증, 회원가입, 결제 등 우선 적용
3. **나머지 엔드포인트**: 시간 날 때 리팩토링

### 단계 3: 테스트

```java
@Test
void registerWithDuplicateEmail_ShouldReturnConflict() {
    // Given
    RegisterRequest request = new RegisterRequest("test@test.com", "password");

    // When
    ResponseEntity<ApiResponse> response = authController.register(request);

    // Then
    assertEquals(409, response.getStatusCodeValue());
    assertEquals("error", response.getBody().getStatus());
    assertEquals("이미 사용 중인 이메일입니다", response.getBody().getMessage());
}
```

---

## 💡 모범 사례

### ✅ DO

```java
// 명확한 커스텀 예외 사용
throw new DuplicateEmailException(email);

// 성공 시 의미 있는 메시지
return ResponseEntity.ok(ApiResponse.success("저장되었습니다", data));

// 중요한 에러는 로깅
log.error("결제 실패", e);
throw new PaymentException("결제에 실패했습니다");
```

### ❌ DON'T

```java
// 일반 Exception 던지기
throw new Exception("에러 발생");  // ❌

// 에러를 문자열로 반환
return ResponseEntity.badRequest().body("실패");  // ❌

// 중요한 에러 정보 숨기기
catch (Exception e) {
    return ResponseEntity.status(500).body("에러");  // ❌ 로깅 없음
}
```

---

## 🎉 완료 체크리스트

- [ ] `ApiResponse` 클래스 생성
- [ ] `GlobalExceptionHandler` 생성
- [ ] `BusinessException` 베이스 클래스 생성
- [ ] 커스텀 예외 클래스 생성 (DuplicateEmailException 등)
- [ ] AuthController에 적용
- [ ] 나머지 컨트롤러에 점진적 적용
- [ ] 프론트엔드와 통합 테스트
