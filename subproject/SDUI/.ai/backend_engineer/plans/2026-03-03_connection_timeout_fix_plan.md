# ERR_CONNECTION_TIMED_OUT 문제 해결 Plan

**작성일**: 2026-03-03 00:35
**문제**: lab/claude-dev 배포 후 API 연결 불가
**증상**: `ERR_CONNECTION_TIMED_OUT`
**목표**: 포트 8081로 API 접근 가능하도록 수정

---

## 문제 분석

### 현재 상태
- ✅ GitHub Actions 배포 성공
- ✅ 프론트엔드 Vercel 배포 완료
- ❌ API 연결 불가 (http://43.201.237.68:8081/api/content/list)

### 가설
1. **AWS 보안 그룹에 포트 8081 미개방** (가능성: 90%)
   - main 브랜치는 8080만 사용
   - lab/claude-dev는 8081 사용 (처음)

2. **Docker 컨테이너 미실행** (가능성: 5%)
   - Flyway 마이그레이션 실패
   - 애플리케이션 기동 오류

3. **네트워크 설정 문제** (가능성: 5%)
   - Docker 네트워크
   - 포트 매핑 오류

---

## 해결 Plan

### Step 1: 진단 (Diagnosis) - 10분

#### 1.1 Docker 컨테이너 상태 확인
```bash
ssh ubuntu@43.201.237.68
docker ps | grep sdui-backend-lab
docker logs --tail 100 sdui-backend-lab
```

**예상 결과**:
- 컨테이너가 실행 중이면 → Step 1.2로
- 컨테이너가 없거나 중지됨 → Step 2로

#### 1.2 포트 리스닝 확인
```bash
netstat -tuln | grep 8081
# 또는
ss -tuln | grep 8081
```

**예상 결과**:
- 8081 포트가 LISTEN 중이면 → Step 1.3으로
- 8081 포트가 없으면 → Step 2로

#### 1.3 AWS 보안 그룹 확인
```bash
# AWS CLI 또는 AWS Console에서 확인
aws ec2 describe-security-groups \
  --group-ids sg-xxxxxx \
  --query 'SecurityGroups[*].IpPermissions'
```

**예상 결과**:
- 포트 8081이 없으면 → **Step 3로 (주요 해결책)**
- 포트 8081이 있으면 → Step 4로

---

### Step 2: Docker 컨테이너 문제 해결 - 15분

#### 2.1 로그 분석
```bash
docker logs sdui-backend-lab | grep -i "error\|exception\|flyway\|failed"
```

#### 2.2 Flyway 마이그레이션 상태 확인
```bash
docker exec sdui-db psql -U postgres -d SDUI_LAB \
  -c "SELECT version, description, installed_on, success FROM flyway_schema_history ORDER BY installed_rank;"
```

#### 2.3 컨테이너 재시작
```bash
docker stop sdui-backend-lab
docker rm sdui-backend-lab

# 재배포
docker run -d --name sdui-backend-lab \
  -p 8081:8080 \
  --network sdui-network \
  -e SPRING_PROFILES_ACTIVE=prod \
  -e SPRING_DATASOURCE_URL=jdbc:postgresql://sdui-db:5432/SDUI_LAB \
  -e SPRING_DATASOURCE_USERNAME=$DB_USERNAME \
  -e SPRING_DATASOURCE_PASSWORD=$DB_PASSWORD \
  -e SPRING_DATA_REDIS_HOST=sdui-redis \
  -e SPRING_MAIL_PASSWORD=$MAIL_PASSWORD \
  -e JWT_SECRET_KEY=$JWT_SECRET \
  $DOCKER_USERNAME/sdui-app:lab-claude-dev
```

---

### Step 3: AWS 보안 그룹 수정 (주요 해결책) - 10분

#### 3.1 AWS Console 접속
1. AWS Management Console 로그인
2. EC2 Dashboard → Security Groups
3. SDUI EC2 인스턴스의 보안 그룹 선택

#### 3.2 인바운드 규칙 추가
**추가할 규칙**:
- **Type**: Custom TCP
- **Protocol**: TCP
- **Port Range**: 8081
- **Source**: 0.0.0.0/0 (또는 특정 IP)
- **Description**: SDUI lab/claude-dev backend API

#### 3.3 규칙 적용
- "Save rules" 클릭
- 즉시 적용됨 (재부팅 불필요)

#### 3.4 검증
```bash
# 로컬에서 테스트
curl http://43.201.237.68:8081/api/content/list

# 또는
telnet 43.201.237.68 8081
```

**예상 결과**:
- 연결 성공 → API 응답 확인
- 연결 실패 → Step 4로

---

### Step 4: 네트워크 디버깅 - 15분

#### 4.1 EC2 인스턴스 내부에서 테스트
```bash
ssh ubuntu@43.201.237.68

# 로컬호스트 테스트
curl http://localhost:8081/api/content/list

# Docker 네트워크 테스트
curl http://172.17.0.1:8081/api/content/list
```

#### 4.2 방화벽 확인
```bash
# UFW 상태 확인
sudo ufw status

# iptables 확인
sudo iptables -L -n | grep 8081
```

#### 4.3 Docker 포트 매핑 확인
```bash
docker port sdui-backend-lab
```

**예상 출력**:
```
8080/tcp -> 0.0.0.0:8081
```

---

### Step 5: 검증 및 테스트 - 10분

#### 5.1 API 테스트
```bash
# Content List
curl http://43.201.237.68:8081/api/content/list

# UI Metadata
curl http://43.201.237.68:8081/api/ui/CONTENT_LIST

# Health Check
curl http://43.201.237.68:8081/actuator/health
```

#### 5.2 Flyway 마이그레이션 검증
```bash
ssh ubuntu@43.201.237.68

# 마이그레이션 히스토리
docker exec sdui-db psql -U postgres -d SDUI_LAB \
  -c "SELECT version, description, installed_on, success FROM flyway_schema_history ORDER BY installed_rank;"

# content 테이블 확인
docker exec sdui-db psql -U postgres -d SDUI_LAB \
  -c "\dt content"

# 데이터 개수
docker exec sdui-db psql -U postgres -d SDUI_LAB \
  -c "SELECT COUNT(*) FROM content;"
```

#### 5.3 프론트엔드 연동 테스트
- Vercel 프론트엔드에서 API 호출
- CONTENT_LIST, CONTENT_WRITE, CONTENT_DETAIL 화면 테스트
- 콘텐츠 작성/수정/삭제 기능 테스트

---

## 예상 결과

### 성공 시나리오 (90% 확률)
1. ✅ AWS 보안 그룹에 포트 8081 추가
2. ✅ API 연결 성공
3. ✅ Flyway 마이그레이션 V1~V7 모두 성공
4. ✅ content 테이블 생성 및 데이터 이관 완료
5. ✅ 프론트엔드 연동 정상 작동

### 대체 시나리오 (10% 확률)
1. Docker 컨테이너 재시작 필요
2. Flyway 마이그레이션 일부 실패 → flywayRepair 실행
3. 네트워크 설정 수정 필요

---

## 리스크 관리

### 주요 리스크
1. **포트 개방으로 인한 보안 위협**
   - 대응: 특정 IP만 허용하도록 제한 (Vercel IP 또는 사무실 IP)
   - 또는 ALB/CloudFront 사용

2. **Flyway 마이그레이션 실패**
   - 대응: diary_backup으로 복원
   - flywayRepair 실행 후 재시도

3. **기존 main 브랜치 영향**
   - 대응: 포트 8081은 lab 전용이므로 main(8080)에 영향 없음

---

## 타임라인

| Step | 작업 | 예상 시간 | 누적 시간 |
|------|------|-----------|----------|
| 1 | 진단 (컨테이너/포트/보안그룹) | 10분 | 10분 |
| 2 | Docker 문제 해결 (필요시) | 15분 | 25분 |
| 3 | AWS 보안 그룹 수정 | 10분 | 35분 |
| 4 | 네트워크 디버깅 (필요시) | 15분 | 50분 |
| 5 | 검증 및 테스트 | 10분 | 60분 |

**총 예상 시간**: 1시간 (최악의 경우)
**실제 예상 시간**: 20분 (Step 1 + Step 3만 실행)

---

## 성공 기준

### 필수 (Must Have)
- ✅ `curl http://43.201.237.68:8081/api/content/list` 응답 성공
- ✅ Flyway 마이그레이션 V1~V7 모두 success
- ✅ content 테이블 생성 완료

### 권장 (Should Have)
- ✅ 프론트엔드에서 API 호출 성공
- ✅ CONTENT_LIST 화면 정상 렌더링
- ✅ 콘텐츠 작성 기능 정상 작동

### 선택 (Nice to Have)
- ✅ Redis 캐시 정상 작동
- ✅ JWT 인증 정상 작동
- ✅ WebSocket 연결 정상

---

## 후속 조치

### 즉시
1. 문제 해결 후 report 작성
2. main 브랜치 병합 준비
3. 보안 설정 재검토

### 단기 (1주일 내)
1. 포트 8081 보안 강화 (IP 제한)
2. 모니터링 설정 (CloudWatch, Sentry)
3. 알림 설정 (Slack, Email)

### 중기 (1개월 내)
1. ALB/CloudFront 도입 검토
2. 성능 테스트 및 최적화
3. 백업 자동화

---

**검토 필요**: AWS 보안 그룹 접근 권한 확인
**승인 대기**: ⏳ 사용자 승인 대기

---

## 다음 단계

이 plan에 동의하시면 "YES"라고 답변해주세요.
즉시 Step 1 (진단)부터 시작하겠습니다.

또는 특정 Step부터 시작하고 싶으시면 말씀해주세요.
