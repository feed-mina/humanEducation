# 컨테이너 재시작 및 Flyway 마이그레이션 검증 Plan

**작성일**: 2026-03-03 01:00
**이전 작업**: SDUI_LAB 스키마 복사 완료 (pg_dump)
**목표**: sdui-backend-lab 재시작 → Flyway V1~V7 실행 → API 접근 확인

---

## 현재 상태

### ✅ 완료된 작업
1. Flyway 설정 파일 수정 (application.yml, application-prod.yml)
2. 마이그레이션 스크립트 7개 작성 (V1~V7)
3. GitHub Actions 배포 성공
4. SDUI_LAB 데이터베이스 생성
5. **pg_dump로 SDUI_TD 스키마 → SDUI_LAB 복사 완료**

### ❌ 남은 문제
1. sdui-backend-lab 컨테이너가 V2 마이그레이션 실패로 중지됨
2. 포트 8081 API 접근 불가 (ERR_CONNECTION_TIMED_OUT)

---

## Step 1: 컨테이너 재시작 (5분)

### 1.1 현재 컨테이너 상태 확인

```bash
ssh ubuntu@43.201.237.68
docker ps -a | grep sdui-backend-lab
```

**예상 결과**:
- 컨테이너가 Exited 상태일 가능성 높음 (V2 마이그레이션 실패로 인한 중지)

### 1.2 컨테이너 재시작

```bash
docker restart sdui-backend-lab
```

### 1.3 로그 실시간 모니터링

```bash
docker logs -f sdui-backend-lab
```

**확인할 내용**:
- ✅ Flyway baseline 설정 확인
- ✅ V1~V7 마이그레이션 순차 실행
- ✅ Spring Boot 애플리케이션 정상 기동
- ✅ "Started DemoBackendApplication" 메시지 확인

**예상 로그**:
```
Flyway Community Edition 9.x.x by Redgate
Database: jdbc:postgresql://sdui-db:5432/SDUI_LAB (PostgreSQL 15.x)
Successfully validated 7 migrations (execution time 00:00.052s)
Creating Schema History table "public"."flyway_schema_history" with baseline ...
Successfully baselined schema with version: 0
Current version of schema "public": 0
Migrating schema "public" to version "1 - baseline schema"
Migrating schema "public" to version "2 - create content table"
Migrating schema "public" to version "3 - migrate diary to content"
Migrating schema "public" to version "4 - update ui metadata rbac"
Migrating schema "public" to version "5 - update query master redis"
Migrating schema "public" to version "6 - update metadata content refs"
Migrating schema "public" to version "7 - drop diary table"
Successfully applied 7 migrations to schema "public" (execution time 00:00.234s)
```

### 1.4 컨테이너 재시작 실패 시 대응

**시나리오 A**: 컨테이너가 계속 중지되는 경우

```bash
# 컨테이너 제거
docker stop sdui-backend-lab
docker rm sdui-backend-lab

# 새로 실행
docker run -d --name sdui-backend-lab \
  -p 8081:8080 \
  --network sdui-network \
  -e SPRING_PROFILES_ACTIVE=prod \
  -e SPRING_DATASOURCE_URL=jdbc:postgresql://sdui-db:5432/SDUI_LAB \
  -e SPRING_DATASOURCE_USERNAME=mina \
  -e SPRING_DATASOURCE_PASSWORD=password \
  -e SPRING_DATA_REDIS_HOST=sdui-redis \
  -e SPRING_MAIL_PASSWORD=$MAIL_PASSWORD \
  -e JWT_SECRET_KEY=$JWT_SECRET \
  <DOCKER_USERNAME>/sdui-app:lab-claude-dev
```

**시나리오 B**: Flyway 마이그레이션 일부 실패

```bash
# Flyway repair 실행
docker exec sdui-backend-lab sh -c "./gradlew flywayRepair"
docker restart sdui-backend-lab
```

---

## Step 2: Flyway 마이그레이션 검증 (5분)

### 2.1 Flyway 히스토리 확인

```bash
docker exec sdui-db psql -U mina -d SDUI_LAB -c "
SELECT
  installed_rank,
  version,
  description,
  type,
  script,
  installed_on,
  execution_time,
  success
FROM flyway_schema_history
ORDER BY installed_rank;
"
```

**예상 결과**:
| installed_rank | version | description | success |
|----------------|---------|-------------|---------|
| 1 | 0 | << Flyway Baseline >> | true |
| 2 | 1 | baseline schema | true |
| 3 | 2 | create content table | true |
| 4 | 3 | migrate diary to content | true |
| 5 | 4 | update ui metadata rbac | true |
| 6 | 5 | update query master redis | true |
| 7 | 6 | update metadata content refs | true |
| 8 | 7 | drop diary table | true |

### 2.2 content 테이블 확인

```bash
docker exec sdui-db psql -U mina -d SDUI_LAB -c "
SELECT
  (SELECT COUNT(*) FROM content) AS content_count,
  (SELECT COUNT(*) FROM diary_backup) AS backup_count,
  (SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'diary')) AS diary_exists;
"
```

**예상 결과**:
- `content_count`: diary에서 이관된 데이터 수 (SDUI_TD 기준)
- `backup_count`: diary_backup에 백업된 데이터 수 (동일)
- `diary_exists`: false (V7에서 삭제됨)

### 2.3 ui_metadata 변경 확인

```bash
docker exec sdui-db psql -U mina -d SDUI_LAB -c "
SELECT screen_id, COUNT(*) AS component_count
FROM ui_metadata
WHERE screen_id LIKE 'CONTENT%'
GROUP BY screen_id
ORDER BY screen_id;
"
```

**예상 결과**:
- `CONTENT_LIST`: N개
- `CONTENT_WRITE`: M개
- `CONTENT_DETAIL`: K개

---

## Step 3: API 접근 테스트 (5분)

### 3.1 포트 8081 리스닝 확인

```bash
ssh ubuntu@43.201.237.68
ss -tuln | grep 8081
```

**예상 결과**:
```
tcp   LISTEN 0      4096         0.0.0.0:8081       0.0.0.0:*
```

### 3.2 로컬호스트 테스트 (EC2 내부)

```bash
curl http://localhost:8081/api/content/list
```

**예상 결과**:
- HTTP 200 OK
- JSON 응답 (content 목록 또는 빈 배열)

### 3.3 외부에서 API 테스트

```bash
# 로컬 머신에서 실행
curl http://43.201.237.68:8081/api/content/list
```

**예상 결과 A** (성공):
- HTTP 200 OK
- JSON 응답

**예상 결과 B** (실패 - 보안 그룹 미설정):
- `ERR_CONNECTION_TIMED_OUT`
- → Step 4로 이동

---

## Step 4: AWS 보안 그룹 설정 (필요 시, 10분)

### 4.1 현재 보안 그룹 확인

```bash
aws ec2 describe-security-groups \
  --group-ids <SECURITY_GROUP_ID> \
  --query 'SecurityGroups[*].IpPermissions[?ToPort==`8081`]'
```

### 4.2 포트 8081 인바운드 규칙 추가

**AWS Console 방법**:
1. EC2 Dashboard → Security Groups
2. SDUI EC2 인스턴스의 보안 그룹 선택
3. Inbound rules → Edit inbound rules
4. Add rule:
   - **Type**: Custom TCP
   - **Port Range**: 8081
   - **Source**: 0.0.0.0/0 (또는 Vercel IP)
   - **Description**: SDUI lab/claude-dev API
5. Save rules

**AWS CLI 방법**:
```bash
aws ec2 authorize-security-group-ingress \
  --group-id <SECURITY_GROUP_ID> \
  --protocol tcp \
  --port 8081 \
  --cidr 0.0.0.0/0
```

### 4.3 검증

```bash
curl http://43.201.237.68:8081/api/content/list
```

---

## Step 5: 프론트엔드 연동 테스트 (5분)

### 5.1 Vercel 환경 변수 확인

**변수**: `NEXT_PUBLIC_API_BASE_URL`
**값**: `http://43.201.237.68:8081` (lab/claude-dev 환경)

### 5.2 프론트엔드 화면 테스트

- [ ] `/view/CONTENT_LIST` 접근
- [ ] 콘텐츠 목록 렌더링 확인
- [ ] `/view/CONTENT_WRITE` 접근
- [ ] 콘텐츠 작성 폼 렌더링 확인

### 5.3 API 호출 확인 (브라우저 DevTools)

- [ ] Network 탭에서 `/api/content/list` 호출 확인
- [ ] Response 200 확인
- [ ] CORS 에러 없는지 확인

---

## 성공 기준

### 필수 (Must Have)
- ✅ sdui-backend-lab 컨테이너 정상 실행 중
- ✅ Flyway 마이그레이션 V1~V7 모두 success
- ✅ content 테이블 존재 및 데이터 이관 완료
- ✅ diary 테이블 삭제됨
- ✅ `curl http://43.201.237.68:8081/api/content/list` 응답 성공

### 권장 (Should Have)
- ✅ 프론트엔드에서 CONTENT_LIST 정상 렌더링
- ✅ 콘텐츠 작성/수정 기능 정상 작동

---

## 예상 소요 시간

| Step | 작업 | 예상 시간 |
|------|------|----------|
| 1 | 컨테이너 재시작 | 5분 |
| 2 | Flyway 마이그레이션 검증 | 5분 |
| 3 | API 접근 테스트 | 5분 |
| 4 | AWS 보안 그룹 설정 (필요 시) | 10분 |
| 5 | 프론트엔드 연동 테스트 | 5분 |
| **총** |  | **30분** |

---

## 리스크 및 대응

### 리스크 1: V3 마이그레이션 데이터 이관 실패
- **원인**: diary 테이블이 비어있거나 손상됨
- **대응**: SDUI_TD의 diary 데이터를 먼저 SDUI_LAB로 복사 후 재시도

### 리스크 2: V7 실행 시 diary_backup 손실
- **확률**: 낮음 (V2에서 백업 생성)
- **대응**: V2 재실행하여 백업 재생성

### 리스크 3: 포트 8081 외부 접근 불가
- **확률**: 높음 (AWS 보안 그룹 미설정)
- **대응**: Step 4에서 인바운드 규칙 추가

---

## 다음 단계

이 plan 완료 후:
1. ✅ Report 작성 (`.ai/backend_engineer/reports/2026-03-03_container_restart_report.md`)
2. ⏳ main 브랜치 병합 준비
3. ⏳ SDUI_TD 배포 (프로덕션)

---

**자동 승인**: YES (사용자 요청에 따라 묻지 않고 진행)
**실행 시작**: 즉시
