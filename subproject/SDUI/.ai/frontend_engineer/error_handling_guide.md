# 에러 처리 가이드 (2026-03-02)

## 📋 개요

`[object Object]` alert 문제를 해결하기 위한 공통 에러 처리 유틸리티입니다.

---

## 🛠️ 사용 방법

### 1. handleError 함수 (권장)

에러를 콘솔에 로깅하고 사용자에게 alert을 표시합니다.

```typescript
import { handleError } from '@/utils/errorHandler';

try {
    const res = await axios.post('/api/auth/register', data);
    // 성공 처리
} catch (error: any) {
    handleError(error, 'REGISTER_SUBMIT', '회원가입에 실패했습니다');
}
```

**파라미터:**
- `error`: catch된 에러 객체
- `context`: 에러 발생 위치 (디버깅용, 예: 'REGISTER_SUBMIT')
- `defaultMessage`: 기본 메시지 (백엔드에서 메시지가 없을 때 사용)

---

### 2. extractErrorMessage 함수

에러 객체에서 메시지만 추출합니다 (alert을 직접 제어하고 싶을 때).

```typescript
import { extractErrorMessage } from '@/utils/errorHandler';

try {
    const res = await axios.post('/api/auth/login', data);
    // 성공 처리
} catch (error: any) {
    const message = extractErrorMessage(error, '로그인에 실패했습니다');

    // 커스텀 처리
    if (error?.response?.status === 401) {
        alert('인증 정보가 올바르지 않습니다');
    } else {
        alert(message);
    }
}
```

---

### 3. analyzeError 함수 (고급)

에러를 상세 분석하여 조건부 처리가 필요할 때 사용합니다.

```typescript
import { analyzeError } from '@/utils/errorHandler';

try {
    const res = await axios.post('/api/content/create', data);
    // 성공 처리
} catch (error: any) {
    const analysis = analyzeError(error);

    console.log('에러 분석:', analysis);
    // {
    //   message: "이미 존재하는 콘텐츠입니다",
    //   statusCode: 409,
    //   statusText: "Conflict",
    //   defaultMessage: "중복된 데이터입니다",
    //   data: { message: "이미 존재하는 콘텐츠입니다" },
    //   isNetworkError: false,
    //   isTimeout: false
    // }

    if (analysis.statusCode === 409) {
        alert('이미 존재하는 콘텐츠입니다. 다른 제목을 사용해주세요.');
    } else if (analysis.isNetworkError) {
        alert('네트워크 연결을 확인해주세요.');
    } else {
        alert(analysis.message);
    }
}
```

---

## 🔧 백엔드 응답 형식 (권장)

백엔드에서는 **일관된 에러 응답 형식**을 사용하세요.

### 표준 형식

```json
{
  "message": "사용자에게 표시할 메시지",
  "error": "상세 기술 정보 (선택 사항)",
  "timestamp": "2026-03-02T19:00:00Z",
  "path": "/api/auth/register"
}
```

### Spring Boot 예제

```java
@PostMapping("/register")
public ResponseEntity<?> register(@RequestBody RegisterRequest request) {
    try {
        // 비즈니스 로직
        return ResponseEntity.ok(result);
    } catch (DuplicateEmailException e) {
        // 400 Bad Request
        return ResponseEntity.badRequest().body(Map.of(
            "message", "이미 사용 중인 이메일입니다",
            "error", e.getMessage()
        ));
    } catch (InvalidInputException e) {
        // 422 Unprocessable Entity
        return ResponseEntity.status(422).body(Map.of(
            "message", "입력값을 확인해주세요",
            "error", e.getMessage()
        ));
    } catch (Exception e) {
        // 500 Internal Server Error
        log.error("Register error", e);
        return ResponseEntity.status(500).body(Map.of(
            "message", "서버 오류가 발생했습니다",
            "error", "Internal server error"
        ));
    }
}
```

---

## 📊 HTTP 상태 코드별 기본 메시지

`getStatusMessage()` 함수가 제공하는 기본 메시지:

| 상태 코드 | 기본 메시지 |
|-----------|-------------|
| 400 | 잘못된 요청입니다 |
| 401 | 인증이 필요합니다 |
| 403 | 접근 권한이 없습니다 |
| 404 | 요청한 리소스를 찾을 수 없습니다 |
| 409 | 중복된 데이터입니다 |
| 422 | 입력값을 확인해주세요 |
| 500 | 서버 오류가 발생했습니다 |
| 502 | 게이트웨이 오류가 발생했습니다 |
| 503 | 서비스를 일시적으로 사용할 수 없습니다 |

---

## ✅ 적용된 곳

- `useUserActions.tsx`: LOGIN_SUBMIT, REGISTER_SUBMIT, SUBMIT_ADDITIONAL_INFO
- (추가 적용 필요): `useBusinessActions.tsx`의 모든 에러 처리

---

## 🔍 디버깅

개발 환경에서는 자동으로 콘솔에 상세 에러가 출력됩니다:

```
[REGISTER_SUBMIT] Error: AxiosError {...}
[REGISTER_SUBMIT] Error Response: {status: 400, data: {...}}
```

---

## 🚀 다음 단계

### 전체 프로젝트에 적용

1. **useBusinessActions.tsx** 수정
2. **axios interceptor**에 공통 에러 처리 추가 (선택)
3. 백엔드 모든 컨트롤러에서 일관된 에러 응답 형식 사용

### Axios Interceptor 예제 (선택 사항)

```typescript
// services/axios.tsx
import { extractErrorMessage } from '@/utils/errorHandler';

axios.interceptors.response.use(
    (response) => response,
    (error) => {
        // 401 Unauthorized: 자동 로그아웃
        if (error.response?.status === 401) {
            // 로그아웃 처리
            window.location.href = '/view/LOGIN_PAGE';
        }

        // 에러를 그대로 throw (각 컴포넌트에서 처리)
        return Promise.reject(error);
    }
);
```

---

## 💡 핵심 교훈

1. **절대 `alert(error)`를 직접 호출하지 마세요** → `[object Object]` 표시됨
2. **항상 `extractErrorMessage()` 또는 `handleError()` 사용**
3. **백엔드는 일관된 응답 형식 사용** → `{ message: "..." }`
4. **상태 코드별로 적절한 메시지 제공**
