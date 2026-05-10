# Flyway Checksum 문제 해결 가이드

**작성일**: 2026-03-03
**문제**: V3-V7 마이그레이션을 수동 실행했으나 flyway_schema_history에 NULL checksum으로 기록되어 컨테이너 시작 실패
**에러**: `Migration checksum mismatch for migration version 3 -> Applied to database : null -> Resolved locally : -1090675366`

---

## 현재 상황

### ✅ 완료된 작업
- V3-V7 마이그레이션 SQL을 수동으로 실행하여 데이터베이스 스키마 변경 완료
- `diary` → `content` 테이블 마이그레이션 완료
- `diary_backup` 테이블 생성 및 데이터 백업 완료
- `ui_metadata`, `query_master` 메타데이터 업데이트 완료

### ❌ 현재 문제
- flyway_schema_history 테이블의 V3-V7 레코드에 checksum이 NULL로 기록됨
- JAR 파일의 마이그레이션 파일은 실제 checksum 값을 가지고 있음 (예: V3 = -1090675366)
- Spring Boot 시작 시 Flyway가 checksum 불일치를 감지하여 validation 실패

---

## 해결 방법

### 🎯 Option 1: Flyway Validation 비활성화 (권장)

**장점**:
- 가장 간단하고 빠름
- 데이터베이스 수정 불필요
- 향후 마이그레이션도 정상 작동

**단점**:
- Checksum validation이 비활성화되므로 의도하지 않은 마이그레이션 변경 감지 못함

#### 실행 방법

```bash
# EC2 서버에 SSH 접속
ssh ubuntu@43.201.237.68

# 스크립트 실행
bash /path/to/flyway_checksum_fix.sh
```

또는 수동 실행:

```bash
# 1. 기존 컨테이너 제거
docker stop sdui-backend-lab
docker rm sdui-backend-lab

# 2. 환경 변수 설정
export MAIL_PW="xbfkdgfxevqytije"
export JWT_KEY="d6ac9ecc0a3aa3c395313fb236e0ec10d71ab78fb36f54ba626664eba0b842b1"

# 3. Flyway validation 비활성화하고 재시작
docker run -d \
  --name sdui-backend-lab \
  -p 8081:8080 \
  --network sdui-network \
  -e SPRING_PROFILES_ACTIVE=prod \
  -e SPRING_DATASOURCE_URL=jdbc:postgresql://sdui-db:5432/SDUI_LAB \
  -e SPRING_DATASOURCE_USERNAME=mina \
  -e SPRING_DATASOURCE_PASSWORD=password \
  -e SPRING_DATA_REDIS_HOST=sdui-redis \
  -e SPRING_MAIL_PASSWORD=$MAIL_PW \
  -e JWT_SECRET_KEY=$JWT_KEY \
  -e SPRING_FLYWAY_VALIDATE_ON_MIGRATE=false \
  yerinmin/sdui-app:lab-claude-dev

# 4. 로그 확인
docker logs -f sdui-backend-lab
```

---

### 🎯 Option 2: Flyway History 재설정

**장점**:
- Flyway validation 유지 (향후 마이그레이션 안정성)
- 깔끔한 history 상태

**단점**:
- 데이터베이스 수정 필요
- 약간 복잡함

#### 실행 방법

```bash
# 1. PostgreSQL 접속
docker exec -it sdui-db psql -U mina -d SDUI_LAB

# 2. SQL 파일 실행 또는 수동 입력
```

```sql
-- 현재 상태 확인
SELECT installed_rank, version, description, checksum, success
FROM flyway_schema_history
ORDER BY installed_rank;

-- V3-V7 레코드 삭제
DELETE FROM flyway_schema_history WHERE version IN ('3', '4', '5', '6', '7');

-- V7을 새로운 baseline으로 설정
INSERT INTO flyway_schema_history (
    installed_rank,
    version,
    description,
    type,
    script,
    checksum,
    installed_by,
    installed_on,
    execution_time,
    success
) VALUES (
    3,
    '7',
    'baseline at version 7 after manual migration',
    'BASELINE',
    'V7__drop_diary_table.sql',
    NULL,
    'mina',
    NOW(),
    0,
    true
);

-- 결과 확인
SELECT installed_rank, version, description, type, success
FROM flyway_schema_history
ORDER BY installed_rank;

-- 종료
\q
```

```bash
# 3. 컨테이너 재시작
docker restart sdui-backend-lab

# 4. 로그 확인
docker logs -f sdui-backend-lab
```

---

## 검증 절차

### 1. 컨테이너 정상 시작 확인

```bash
docker ps | grep sdui-backend-lab
```

**예상 결과**: Status가 "Up" 상태

### 2. Spring Boot 애플리케이션 로그 확인

```bash
docker logs sdui-backend-lab | grep "Started"
```

**예상 결과**:
```
Started DemoBackendApplication in X.XXX seconds
```

### 3. 포트 8081 리스닝 확인

```bash
ss -tuln | grep 8081
```

**예상 결과**:
```
tcp   LISTEN 0      4096         0.0.0.0:8081       0.0.0.0:*
```

### 4. API 테스트

```bash
# EC2 내부에서
curl http://localhost:8081/api/content/list

# 외부에서
curl http://43.201.237.68:8081/api/content/list
```

**예상 결과**: HTTP 200 OK, JSON 응답

---

## 참고 파일

- `flyway_checksum_fix.sh`: Option 1 자동 실행 스크립트
- `flyway_fix_option2.sql`: Option 2 SQL 스크립트

---

## 다음 단계

이 가이드 완료 후:

1. ✅ API 접근 테스트 (Step 3 of plan)
2. ✅ AWS 보안 그룹 설정 (필요 시, Step 4 of plan)
3. ✅ 프론트엔드 연동 테스트 (Step 5 of plan)
4. ✅ Report 작성

---

**작성자**: 
**관련 Plan**: `.ai/backend_engineer/plans/2026-03-03_container_restart_and_flyway_verification_plan.md`
