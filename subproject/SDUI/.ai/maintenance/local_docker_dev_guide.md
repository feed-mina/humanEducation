# 로컬 Docker 개발 환경 가이드

**작성일**: 2026-03-07
**대상**: 개발자
**목적**: IntelliJ + Docker DB 조합으로 로컬 개발 환경을 구성하고 Flyway 마이그레이션을 실행하는 방법 안내

---

## 📚 목차

1. [포트 구성](#1-포트-구성)
2. [로컬 테스트 표준 순서](#2-로컬-테스트-표준-순서)
3. [Docker Compose 서비스 이름 vs 컨테이너 이름](#3-docker-compose-서비스-이름-vs-컨테이너-이름)
4. [Flyway 마이그레이션 트러블슈팅](#4-flyway-마이그레이션-트러블슈팅)
5. [application.yml Flyway 설정 기준값](#5-applicationyml-flyway-설정-기준값)
6. [자주 겪는 실수](#6-자주-겪는-실수)

---

## 1. 포트 구성

| 서비스 | 포트 | 비고 |
|--------|------|------|
| 로컬 PostgreSQL 18 (기본) | 5432 | OS 직접 설치 |
| 로컬 PostgreSQL 13 (구버전) | 5433 | OS 직접 설치 |
| Docker sdui-db | **5434** | 5432/5433 충돌 방지 |
| Docker sdui-redis | 6379 | |
| Docker sdui-backend | 8080 | IntelliJ 사용 시 중지 필요 |
| IntelliJ Spring Boot | 8080 | Docker sdui-backend와 충돌 |
| Next.js dev server | 3000 | |

> docker-compose.yml 기준: `sdui-db` 컨테이너는 `5434:5432` 포트 매핑

---

## 2. 로컬 테스트 표준 순서

IntelliJ에서 Spring Boot를 직접 실행하는 경우:

```bash
# Step 1: Docker DB 초기화 (프로젝트 루트에서)
docker-compose stop db
docker-compose rm -f db
docker-compose up -d db

# Step 2: DB 기동 확인 (5~10초 대기)
docker ps   # sdui-db 컨테이너 Up 상태 확인

# Step 3: sdui-backend Docker 컨테이너 중지 (8080 포트 해제)
docker stop sdui-backend

# Step 4: IntelliJ에서 DemoBackendApplication 실행
# → 콘솔에서 Flyway V1~V8 순차 실행 로그 확인
# → "Started DemoBackendApplication" 확인

# Step 5: Next.js 프론트엔드 실행
cd metadata-project
npm run dev

# Step 6: 브라우저에서 확인
# http://localhost:3000/view/MAIN_PAGE
```

### 예상 Flyway 실행 로그

```
Migrating schema "public" to version 1
Migrating schema "public" to version 2
...
Migrating schema "public" to version 8
Successfully applied 8 migrations to schema "public"
```

---

## 3. Docker Compose 서비스 이름 vs 컨테이너 이름

`docker-compose` 명령어는 **서비스 이름**을 사용한다. 컨테이너 이름과 다르다.

| docker-compose.yml 서비스 이름 | 컨테이너 이름 | 올바른 명령어 |
|-------------------------------|-------------|-------------|
| `db` | `sdui-db` | `docker-compose stop db` ✅ |
| `redis` | `sdui-redis` | `docker-compose stop redis` ✅ |
| `app` | `sdui-backend` | `docker-compose stop app` ✅ |

```bash
# ❌ 잘못된 예 (서비스 이름이 아닌 컨테이너 이름 사용)
docker-compose stop sdui-db   # Error: no such service: sdui-db

# ✅ 올바른 예
docker-compose stop db        # 정상 동작

# docker 명령어는 컨테이너 이름 사용 (docker-compose와 다름)
docker stop sdui-backend      # ✅ docker 명령어는 컨테이너 이름 사용
```

---

## 4. Flyway 마이그레이션 트러블슈팅

### 문제 1: V2 실패 — "relation 'users' does not exist"

**원인**: `application.yml`의 `baseline-version: 1` 설정 시 V1은 baseline 마커로만 처리되어 실제 SQL이 실행되지 않음.
빈 Docker DB에 users 테이블가 없어서 V2의 FK 참조가 실패.

**해결**:
1. V1 SQL을 실제 DDL(CREATE TABLE)로 재작성
2. `baseline-version: 0` 으로 변경 → V1부터 실제 실행됨

```yaml
# application.yml 기준값
flyway:
  enabled: true
  baseline-on-migrate: true
  baseline-version: 0   # ← 0으로 설정해야 V1 실행됨
  locations: classpath:db/migration
  validate-on-migrate: false
```

### 문제 2: "there is already a transaction in progress (SQL State: 25001)"

**원인**: 마이그레이션 SQL 파일 내에 명시적 `BEGIN;` / `COMMIT;`이 있으면 Flyway 자체 트랜잭션과 중첩됨.

**해결**: V2~V8 SQL에서 `BEGIN;` 과 `COMMIT;` 제거. Flyway가 트랜잭션을 자동 관리함.

```sql
-- ❌ 제거 대상
BEGIN;
-- ... SQL ...
COMMIT;

-- ✅ 올바른 형태 (BEGIN/COMMIT 없이 SQL만)
-- ... SQL ...
```

### 문제 3: V8 실패 후 재실행

```sql
-- Docker DB psql 접속 후 실행
DELETE FROM flyway_schema_history WHERE version = '8';
-- 이후 Spring Boot 재시작 → V8 자동 재실행
```

### 문제 4: Port 8080 already in use

**원인**: `sdui-backend` Docker 컨테이너가 8080 포트를 점유 중.

```bash
# 확인
docker ps | grep 8080

# 해결: sdui-backend 컨테이너만 중지 (db, redis는 유지)
docker stop sdui-backend
```

---

## 5. application.yml Flyway 설정 기준값

```yaml
spring:
  flyway:
    enabled: true
    baseline-on-migrate: true
    baseline-version: 0      # 0으로 설정 → V1부터 실행
    locations: classpath:db/migration
    validate-on-migrate: false  # 로컬 개발 시 false 권장
```

> AWS 환경의 `application-prod.yml`은 별도 설정 유지 (`validate-on-migrate: true`)

---

## 6. 자주 겪는 실수

| 실수 | 증상 | 올바른 방법 |
|------|------|------------|
| `docker-compose stop sdui-db` | `no such service: sdui-db` | `docker-compose stop db` |
| `docker stop backend` | `No such container` | `docker stop sdui-backend` |
| IntelliJ 실행 전 sdui-backend 미중지 | Port 8080 already in use | `docker stop sdui-backend` 후 실행 |
| V8 실패 후 flyway_schema_history 미정리 | V8 재실행 안 됨 | `DELETE FROM flyway_schema_history WHERE version = '8'` |
| `baseline-version: 1` 설정 | V1 실행 안 됨, users 테이블 없음 | `baseline-version: 0` 으로 변경 |
| BEGIN/COMMIT 포함된 마이그레이션 | SQL State 25001 경고 | 마이그레이션 SQL에서 BEGIN/COMMIT 제거 |

---

## 관련 문서

- [`troubleshooting_guide.md`](troubleshooting_guide.md) — 전반적인 문제 해결 가이드
- [`database_change_guide.md`](database_change_guide.md) — Flyway 마이그레이션 작성 가이드
- [`aws_environment_guide.md`](aws_environment_guide.md) — AWS 환경 설정
- `.ai/frontend_engineer/reports/2026-03-07_main_page_bento_grid_implementation.md` — 벤토 그리드 구현 상세 보고서
