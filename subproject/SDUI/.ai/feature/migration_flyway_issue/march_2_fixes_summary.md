# 백엔드 수정 완료 요약 (2026-03-02)

## 수정 완료 파일 (5개)

### ✅ 1. AuthController.java
- **Line 73**: `"ROLE_USER"` 하드코딩 → `userDetails.getRole()` 사용
- **Line 114-122**: role 쿠키 추가
- **Line 128**: role 쿠키를 응답 헤더에 추가

### ✅ 2. KakaoService.java
- **Line 128**: 신규 카카오 사용자 role을 `"ROLE_GUEST"`로 변경

### ✅ 3. KakaoController.java (/login 메서드)
- **Line 93-101**: role 쿠키 생성 추가
- **Line 107**: role 쿠키를 응답 헤더에 추가

### ✅ 4. KakaoController.java (/callback 메서드)
- **Line 165-173**: role 쿠키 생성 추가
- **Line 183**: role 쿠키를 응답 헤더에 추가
- **Line 187-189**: ROLE_GUEST 감지 시 ADDITIONAL_INFO_PAGE로 리다이렉트

### ✅ 5. TokenResponse.java (이미 완료됨)
- role 필드가 이미 존재함

### ✅ 6. JwtUtil.java (이미 완료됨)
- generateTokens()가 이미 role을 설정함 (line 159)

## 다음 단계

### 1. SQL 스크립트 실행
프로젝트 루트에 생성된 `fix_sdui_issues.sql` 파일을 PostgreSQL에서 실행:

```bash
# pgAdmin 사용 또는 psql 커맨드라인 사용
# Windows PowerShell:
Get-Content fix_sdui_issues.sql | docker exec -i <postgres-container> psql -U postgres -d testdb
```

### 2. Redis 캐시 클리어
```bash
docker exec -it sdui-redis-1 redis-cli FLUSHDB
```

### 3. 백엔드 재시작
```bash
cd SDUI-server
./gradlew clean build
./gradlew bootRun
```

### 4. 테스트
- 신규 카카오 사용자 로그인 → ADDITIONAL_INFO_PAGE 확인
- 일반 로그인 → role 쿠키 확인
- 콘텐츠 리스트 화면 → 3개 아이템 표시 확인
- 회원가입 → 주소 버튼 활성화 확인

## 수정 내용 상세

### 문제 1: Role 쿠키 및 RBAC 리다이렉트
- `/api/auth/me`가 실제 role 반환
- 로그인 시 role 쿠키 설정
- 신규 카카오 사용자 ROLE_GUEST로 생성
- ROLE_GUEST 자동 리다이렉트

### 문제 2: 콘텐츠 리스트
- SQL 스크립트로 해결 (ui_metadata + query_master)

### 문제 3: 주소 버튼
- SQL 스크립트로 해결 (is_readonly = false)
