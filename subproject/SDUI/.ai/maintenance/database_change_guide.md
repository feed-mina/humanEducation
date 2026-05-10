# 데이터베이스 변경 가이드 (Flyway 기반)

**작성일**: 2026-03-03
**대상**: 백엔드 개발자
**목적**: Flyway를 사용한 DB 스키마 변경 및 ui_metadata 페이지 추가 방법 제공

---

## 📚 목차

1. [Flyway를 사용한 DB 변경 워크플로우](#1-flyway를-사용한-db-변경-워크플로우)
2. [마이그레이션 스크립트 작성 규칙](#2-마이그레이션-스크립트-작성-규칙)
3. [테이블 추가 가이드](#3-테이블-추가-가이드)
4. [ui_metadata에 새 페이지 추가](#4-ui_metadata에-새-페이지-추가)
5. [로컬과 AWS 환경 차이](#5-로컬과-aws-환경-차이)
6. [일반적인 DB 변경 작업](#6-일반적인-db-변경-작업)
7. [트러블슈팅](#7-트러블슈팅)

---

## 1. Flyway를 사용한 DB 변경 워크플로우

### 전체 프로세스

```
로컬 (testdb) → lab/claude-dev (SDUI_LAB) → main (SDUI_TD)
     ↓                  ↓                         ↓
   개발/테스트        테스트 검증              프로덕션 배포
```

### Step 1: 로컬에서 마이그레이션 스크립트 작성

**1.1 브랜치 생성**
```bash
# main 또는 lab/claude-dev에서 새 브랜치 생성
git checkout lab/claude-dev
git pull origin lab/claude-dev
git checkout -b feature/add-notifications-table
```

**1.2 마이그레이션 스크립트 생성**
```bash
# 위치: SDUI-server/src/main/resources/db/migration/
cd SDUI-server/src/main/resources/db/migration

# 파일명 규칙: V{순차번호}__{설명}.sql
# 예: V8__add_notifications_table.sql
```

**1.3 스크립트 작성**
```sql
-- V8__add_notifications_table.sql
BEGIN;

-- 알림 테이블 생성
CREATE TABLE IF NOT EXISTS notifications (
    notification_id BIGSERIAL PRIMARY KEY,
    user_sqno BIGINT NOT NULL,
    message TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_notification_user
        FOREIGN KEY (user_sqno)
        REFERENCES users (user_sqno)
        ON DELETE CASCADE
);

-- 인덱스 생성 (성능 최적화)
CREATE INDEX IF NOT EXISTS idx_notification_user_sqno
    ON notifications(user_sqno);

CREATE INDEX IF NOT EXISTS idx_notification_created_at
    ON notifications(created_at DESC);

COMMIT;
```

**중요**:
- 반드시 `BEGIN;` / `COMMIT;` 사용 (트랜잭션)
- `IF NOT EXISTS` 사용하여 멱등성 보장
- Foreign Key는 참조 테이블 존재 확인 필요

### Step 2: 로컬에서 테스트

**2.1 Docker Compose 실행**
```bash
# 프로젝트 루트에서
docker-compose up -d

# PostgreSQL 상태 확인
docker ps | grep postgres
```

**2.2 Flyway 마이그레이션 실행**
```bash
cd SDUI-server

# 마이그레이션 정보 확인 (실행 전)
./gradlew flywayInfo

# 예상 출력:
# +-----------+---------+---------------------+------+---------------------+---------+
# | Category  | Version | Description         | Type | Installed On        | State   |
# +-----------+---------+---------------------+------+---------------------+---------+
# | Versioned | 1       | baseline schema     | SQL  | 2026-03-01 10:00:00 | Success |
# | Versioned | 2       | create content      | SQL  | 2026-03-01 10:00:05 | Success |
# ...
# | Versioned | 8       | add notifications   | SQL  |                     | Pending |
# +-----------+---------+---------------------+------+---------------------+---------+

# 마이그레이션 실행
./gradlew flywayMigrate

# 결과 확인
./gradlew flywayInfo
```

**2.3 DB에서 직접 확인**
```bash
# PostgreSQL 접속 (로컬 docker-compose, port 5433)
psql -U postgres -d testdb -h localhost -p 5433

# Flyway 히스토리 확인
SELECT version, description, type, installed_on, success
FROM flyway_schema_history
ORDER BY installed_rank;

# 테이블 생성 확인
\d notifications

# 인덱스 확인
\di notifications*

# 테이블 목록
\dt

# 종료
\q
```

**2.4 애플리케이션 실행 테스트**
```bash
cd SDUI-server
./gradlew bootRun

# 로그에서 Flyway 실행 확인
# Flyway Community Edition 9.22.3 by Redgate
# Successfully validated 8 migrations
# Current version of schema "public": 8
```

### Step 3: lab 환경 배포

**3.1 Git 커밋 및 푸시**
```bash
# 변경사항 확인
git status

# 마이그레이션 스크립트만 추가
git add SDUI-server/src/main/resources/db/migration/V8__add_notifications_table.sql

# (선택) 관련 엔티티/서비스 코드도 함께 추가
git add SDUI-server/src/main/java/com/domain/demo_backend/domain/notification/

# 커밋 (명확한 메시지)
git commit -m "feat: Add notifications table (V8 migration)

- 알림 기능을 위한 notifications 테이블 생성
- user_sqno 외래 키 설정
- 성능 최적화를 위한 인덱스 추가

Co-Authored-By:  <noreply@anthropic.com>"

# lab/claude-dev 브랜치로 푸시 (또는 PR 생성)
git push origin feature/add-notifications-table
```

**3.2 Pull Request 생성**
- GitHub에서 PR 생성: `feature/add-notifications-table` → `lab/claude-dev`
- PR 설명에 변경사항 상세 기재
- 리뷰어 지정 (필요 시)

**3.3 PR 병합 및 배포**
```bash
# PR 병합 후 자동으로 GitHub Actions 실행
# lab/claude-dev → SDUI_LAB 배포
```

**3.4 배포 확인**
```bash
# SSH 접속
ssh ubuntu@43.201.237.68

# 컨테이너 로그 확인
docker logs sdui-backend-lab | grep -i "flyway\|migration"

# Flyway 히스토리 확인
docker exec sdui-db psql -U mina -d SDUI_LAB -c "
SELECT version, description, installed_on, success
FROM flyway_schema_history
ORDER BY installed_rank;
"

# 테이블 생성 확인
docker exec sdui-db psql -U mina -d SDUI_LAB -c "\d notifications"
```

### Step 4: 검증 및 프로덕션 배포

**4.1 SDUI_LAB에서 테스트**
- API 테스트 (Postman, curl 등)
- 프론트엔드 통합 테스트
- 데이터 CRUD 작업 확인

**4.2 main 브랜치 병합**
```bash
# 로컬에서 main 브랜치로 전환
git checkout main
git pull origin main

# lab/claude-dev 병합
git merge lab/claude-dev

# 충돌 해결 (필요 시)
# git status
# git mergetool

# 최종 확인
git log --oneline -10

# 푸시
git push origin main
```

**4.3 프로덕션 배포 확인**
```bash
# GitHub Actions에서 main 브랜치 워크플로우 확인
# main → SDUI_TD 배포

# SSH 접속 후 확인
docker logs sdui-backend | grep -i "flyway"

docker exec sdui-db psql -U mina -d SDUI_TD -c "
SELECT version, description, success FROM flyway_schema_history ORDER BY installed_rank;
"
```

---

## 2. 마이그레이션 스크립트 작성 규칙

### 파일명 규칙

**형식**: `V{순차번호}__{설명}.sql`

**올바른 예시**:
```
V8__add_notifications_table.sql
V9__update_users_add_profile_image.sql
V10__migrate_old_diary_data.sql
```

**잘못된 예시**:
```
v8_add_notifications.sql          # V 대문자 필요
V8_add_notifications_table.sql    # __ 두 개 필요
V08__add_notifications_table.sql  # 0 패딩 불필요 (선택사항)
add_notifications_table.sql       # V 버전 누락
```

### 스크립트 작성 원칙

#### 1. 트랜잭션 사용 (필수)

```sql
-- ✅ 올바른 방법
BEGIN;

CREATE TABLE notifications (...);
CREATE INDEX idx_notification_user ON notifications(user_sqno);

COMMIT;

-- ❌ 잘못된 방법 (트랜잭션 없음)
CREATE TABLE notifications (...);
CREATE INDEX idx_notification_user ON notifications(user_sqno);
```

#### 2. 멱등성 보장 (IF NOT EXISTS)

```sql
-- ✅ 올바른 방법
CREATE TABLE IF NOT EXISTS notifications (...);
CREATE INDEX IF NOT EXISTS idx_notification_user ON notifications(user_sqno);

-- ❌ 잘못된 방법 (재실행 시 에러)
CREATE TABLE notifications (...);
CREATE INDEX idx_notification_user ON notifications(user_sqno);
```

#### 3. Foreign Key 참조 순서 확인

```sql
-- ✅ 올바른 방법 (users 테이블이 이미 존재)
CREATE TABLE IF NOT EXISTS notifications (
    notification_id BIGSERIAL PRIMARY KEY,
    user_sqno BIGINT NOT NULL,

    CONSTRAINT fk_notification_user
        FOREIGN KEY (user_sqno)
        REFERENCES users (user_sqno)  -- users 테이블 존재 확인 필수
);

-- ❌ 잘못된 방법 (참조 테이블 미존재)
-- users 테이블이 없으면 마이그레이션 실패
```

#### 4. 인덱스 추가 (성능 최적화)

```sql
-- 자주 조회되는 컬럼에 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_notification_user_sqno
    ON notifications(user_sqno);

CREATE INDEX IF NOT EXISTS idx_notification_created_at
    ON notifications(created_at DESC);

-- 복합 인덱스 (필요 시)
CREATE INDEX IF NOT EXISTS idx_notification_user_read
    ON notifications(user_sqno, is_read);
```

#### 5. 기본값 및 제약조건 설정

```sql
CREATE TABLE IF NOT EXISTS notifications (
    notification_id BIGSERIAL PRIMARY KEY,
    message TEXT NOT NULL,                     -- NOT NULL 제약
    is_read BOOLEAN DEFAULT FALSE,             -- 기본값
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CHECK (LENGTH(message) > 0)                -- CHECK 제약 (선택)
);
```

### 롤백 스크립트 (선택사항)

Flyway는 기본적으로 롤백을 지원하지 않지만, 별도 파일로 작성 가능:

```sql
-- U8__rollback_notifications_table.sql (수동 실행용)
BEGIN;

DROP INDEX IF EXISTS idx_notification_created_at;
DROP INDEX IF EXISTS idx_notification_user_sqno;
DROP TABLE IF EXISTS notifications;

COMMIT;
```

**중요**: Flyway는 `U` 파일을 자동 실행하지 않습니다. 수동으로 psql에서 실행 필요.

---

## 3. 테이블 추가 가이드

### 3.1 새 테이블 추가 체크리스트

- [ ] 테이블명이 복수형인가? (notifications, users, contents 등)
- [ ] Primary Key가 정의되었는가? (BIGSERIAL 사용 권장)
- [ ] Foreign Key 참조가 올바른가?
- [ ] NOT NULL 제약이 필요한 컬럼에 설정되었는가?
- [ ] 기본값(DEFAULT)이 설정되었는가?
- [ ] 인덱스가 필요한 컬럼에 추가되었는가?
- [ ] 타임스탬프 컬럼(created_at, updated_at)이 있는가?
- [ ] 소프트 삭제 컬럼(del_yn)이 필요한가?

### 3.2 테이블 추가 템플릿

```sql
-- V{N}__add_{table_name}_table.sql
BEGIN;

CREATE TABLE IF NOT EXISTS {table_name} (
    {table_name}_id BIGSERIAL PRIMARY KEY,

    -- 비즈니스 컬럼
    title VARCHAR(255) NOT NULL,
    content TEXT,
    status VARCHAR(50) DEFAULT 'ACTIVE',

    -- 외래 키
    user_sqno BIGINT NOT NULL,

    -- 타임스탬프
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 소프트 삭제
    del_yn VARCHAR(1) DEFAULT 'N',
    del_dt TIMESTAMP,

    -- 제약조건
    CONSTRAINT fk_{table_name}_user
        FOREIGN KEY (user_sqno)
        REFERENCES users (user_sqno)
        ON DELETE CASCADE,

    CHECK (del_yn IN ('Y', 'N'))
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_{table_name}_user_sqno
    ON {table_name}(user_sqno);

CREATE INDEX IF NOT EXISTS idx_{table_name}_created_at
    ON {table_name}(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_{table_name}_del_yn
    ON {table_name}(del_yn) WHERE del_yn = 'N';

COMMIT;
```

### 3.3 JPA Entity 클래스 생성

테이블 추가 후 JPA Entity 클래스도 생성 필요:

```java
// SDUI-server/src/main/java/com/domain/demo_backend/domain/notification/domain/Notification.java
package com.domain.demo_backend.domain.notification.domain;

import com.domain.demo_backend.domain.user.domain.Users;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "notifications")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Notification {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "notification_id")
    private Long notificationId;

    @Column(name = "message", nullable = false, columnDefinition = "TEXT")
    private String message;

    @Column(name = "is_read")
    private Boolean isRead = false;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_sqno", nullable = false)
    private Users user;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "del_yn", length = 1)
    private String delYn = "N";

    @Column(name = "del_dt")
    private LocalDateTime delDt;
}
```

### 3.4 Repository, Service, Controller 구현

```java
// Repository
public interface NotificationRepository extends JpaRepository<Notification, Long> {
    List<Notification> findByUser_UserSqnoAndDelYn(Long userSqno, String delYn);
}

// Service
@Service
@RequiredArgsConstructor
public class NotificationService {
    private final NotificationRepository notificationRepository;

    public List<Notification> getUserNotifications(Long userSqno) {
        return notificationRepository.findByUser_UserSqnoAndDelYn(userSqno, "N");
    }
}

// Controller
@RestController
@RequestMapping("/api/notifications")
@RequiredArgsConstructor
public class NotificationController {
    private final NotificationService notificationService;

    @GetMapping("/list")
    public ResponseEntity<List<Notification>> getNotifications(@AuthenticationPrincipal CustomUserDetails user) {
        List<Notification> notifications = notificationService.getUserNotifications(user.getUserSqno());
        return ResponseEntity.ok(notifications);
    }
}
```

---

## 4. ui_metadata에 새 페이지 추가

SDUI는 UI 구조를 `ui_metadata` 테이블에 메타데이터로 저장합니다. 새 페이지 추가 시 이 테이블에 데이터를 INSERT합니다.

### Option 1: SQL INSERT 문으로 추가 (권장)

**4.1 Flyway 마이그레이션 스크립트 작성**

```sql
-- V9__add_settings_page_metadata.sql
BEGIN;

-- 1. 페이지 최상위 그룹 (컨테이너)
INSERT INTO ui_metadata (
    screen_id, component_id, component_type,
    label_text, sort_order, css_class, group_direction
) VALUES (
    'SETTINGS_PAGE',           -- 화면 ID
    'SETTINGS_CONTAINER',      -- 컴포넌트 ID
    'GROUP',                   -- 그룹 타입
    '설정',                    -- 라벨
    1,                         -- 정렬 순서
    'settings-page-container', -- CSS 클래스
    'COLUMN'                   -- 세로 방향
) ON CONFLICT (screen_id, component_id) DO NOTHING;

-- 2. 프로필 섹션 그룹
INSERT INTO ui_metadata (
    screen_id, component_id, parent_group_id, component_type,
    label_text, sort_order, css_class, group_direction
) VALUES (
    'SETTINGS_PAGE', 'PROFILE_SECTION', 'SETTINGS_CONTAINER', 'GROUP',
    '프로필 설정', 10, 'profile-section', 'COLUMN'
) ON CONFLICT (screen_id, component_id) DO NOTHING;

-- 3. 프로필 이미지 입력
INSERT INTO ui_metadata (
    screen_id, component_id, parent_group_id, component_type,
    label_text, ref_data_id, sort_order, css_class
) VALUES (
    'SETTINGS_PAGE', 'PROFILE_IMAGE', 'PROFILE_SECTION', 'IMAGE',
    '프로필 사진', 'profileImageUrl', 10, 'profile-image'
) ON CONFLICT (screen_id, component_id) DO NOTHING;

-- 4. 닉네임 입력 필드
INSERT INTO ui_metadata (
    screen_id, component_id, parent_group_id, component_type,
    label_text, ref_data_id, placeholder_text, sort_order
) VALUES (
    'SETTINGS_PAGE', 'NICKNAME_INPUT', 'PROFILE_SECTION', 'INPUT',
    '닉네임', 'nickname', '닉네임을 입력하세요', 20
) ON CONFLICT (screen_id, component_id) DO NOTHING;

-- 5. 알림 섹션
INSERT INTO ui_metadata (
    screen_id, component_id, parent_group_id, component_type,
    label_text, sort_order, css_class, group_direction
) VALUES (
    'SETTINGS_PAGE', 'NOTIFICATION_SECTION', 'SETTINGS_CONTAINER', 'GROUP',
    '알림 설정', 20, 'notification-section', 'COLUMN'
) ON CONFLICT (screen_id, component_id) DO NOTHING;

-- 6. 알림 토글
INSERT INTO ui_metadata (
    screen_id, component_id, parent_group_id, component_type,
    label_text, ref_data_id, sort_order
) VALUES (
    'SETTINGS_PAGE', 'NOTIFICATION_TOGGLE', 'NOTIFICATION_SECTION', 'TOGGLE',
    '알림 받기', 'notificationEnabled', 10
) ON CONFLICT (screen_id, component_id) DO NOTHING;

-- 7. 저장 버튼
INSERT INTO ui_metadata (
    screen_id, component_id, parent_group_id, component_type,
    label_text, action_type, action_url, sort_order, css_class
) VALUES (
    'SETTINGS_PAGE', 'SAVE_BTN', 'SETTINGS_CONTAINER', 'BUTTON',
    '저장', 'SAVE_SETTINGS', '/api/settings/save', 30, 'primary-button'
) ON CONFLICT (screen_id, component_id) DO NOTHING;

COMMIT;
```

**중요 컬럼 설명**:
- `screen_id`: 페이지 식별자 (예: SETTINGS_PAGE, CONTENT_LIST)
- `component_id`: 컴포넌트 고유 ID (화면 내에서 유일해야 함)
- `parent_group_id`: 부모 그룹 ID (계층 구조)
- `component_type`: 컴포넌트 타입 (GROUP, INPUT, BUTTON, TEXT, IMAGE 등)
- `ref_data_id`: 데이터 바인딩 키 (pageData, formData에서 사용)
- `action_type`: 액션 핸들러 (SAVE_SETTINGS, LOGIN_SUBMIT 등)
- `action_url`: API 엔드포인트
- `group_direction`: 그룹 레이아웃 (ROW, COLUMN)
- `sort_order`: 정렬 순서 (작을수록 먼저 렌더링)

**4.2 로컬 테스트**

```bash
# Flyway 마이그레이션 실행
./gradlew flywayMigrate

# DB 확인
psql -U postgres -d testdb -h localhost -p 5433

# SETTINGS_PAGE 메타데이터 조회
SELECT component_id, parent_group_id, component_type, label_text, sort_order
FROM ui_metadata
WHERE screen_id = 'SETTINGS_PAGE'
ORDER BY sort_order;

# 계층 구조 확인
SELECT
    COALESCE(parent_group_id, 'ROOT') AS parent,
    component_id,
    component_type,
    label_text
FROM ui_metadata
WHERE screen_id = 'SETTINGS_PAGE'
ORDER BY parent_group_id NULLS FIRST, sort_order;
```

**4.3 프론트엔드에서 접근**

```typescript
// screenMap.ts에 추가
export const SCREEN_MAP: Record<string, string> = {
    // ... 기존 매핑
    '/view/settings': 'SETTINGS_PAGE',
};

// 브라우저에서 접근
// http://localhost:3000/view/settings
// → MetadataProvider가 'SETTINGS_PAGE' 메타데이터 로드
// → DynamicEngine이 렌더링
```

### Option 2: pgAdmin을 통한 수동 추가 (로컬만)

**주의**: 로컬 testdb에서만 사용하고, 나중에 SQL 스크립트로 변환 필요

```sql
-- pgAdmin 또는 psql에서 직접 실행
INSERT INTO ui_metadata (
    screen_id, component_id, component_type,
    label_text, sort_order
) VALUES (
    'TEST_PAGE', 'TEST_CONTAINER', 'GROUP', '테스트 페이지', 1
);
```

**프로덕션 배포 전**:
- 위 INSERT문을 Flyway 스크립트로 변환
- `ON CONFLICT DO NOTHING` 추가하여 멱등성 보장

---

## 5. 로컬과 AWS 환경 차이

### 5.1 로컬 환경 (testdb)

| 항목 | 설정 |
|------|------|
| **DB명** | testdb |
| **포트** | 5433 (docker-compose.yml) |
| **접속** | `psql -U postgres -d testdb -h localhost -p 5433` |
| **Flyway 실행** | 수동: `./gradlew flywayMigrate` |
| **데이터** | 개발/테스트 데이터 자유롭게 추가/삭제 가능 |
| **롤백** | `./gradlew flywayClean` 가능 (주의!) |

### 5.2 AWS lab 환경 (SDUI_LAB)

| 항목 | 설정 |
|------|------|
| **DB명** | SDUI_LAB |
| **포트** | 5432 (Docker 네트워크 내부) |
| **접속** | `docker exec -it sdui-db psql -U mina -d SDUI_LAB` |
| **Flyway 실행** | 자동: GitHub Actions 배포 시 (Spring Boot 시작 시) |
| **데이터** | 테스트 데이터 (프로덕션과 분리) |
| **롤백** | flywayRepair 또는 수동 복구 |

### 5.3 AWS 프로덕션 환경 (SDUI_TD)

| 항목 | 설정 |
|------|------|
| **DB명** | SDUI_TD |
| **포트** | 5432 (Docker 네트워크 내부) |
| **접속** | `docker exec -it sdui-db psql -U mina -d SDUI_TD` |
| **Flyway 실행** | 자동: GitHub Actions 배포 시 |
| **데이터** | **프로덕션 데이터 (변경 전 반드시 백업!)** |
| **롤백** | 백업에서 복원 (시간 소요) |

### 5.4 환경별 작업 방법

**로컬 (testdb)**:
```bash
# Flyway 마이그레이션 수동 실행
./gradlew flywayMigrate

# 마이그레이션 취소 (개발 중에만 사용)
./gradlew flywayClean
./gradlew flywayMigrate

# 직접 SQL 실행 가능
psql -U postgres -d testdb -h localhost -p 5433 -c "INSERT INTO ..."
```

**AWS (SDUI_LAB, SDUI_TD)**:
```bash
# GitHub Actions를 통한 자동 배포만 사용
git push origin lab/claude-dev  # → SDUI_LAB 자동 배포
git push origin main            # → SDUI_TD 자동 배포

# 직접 SQL 실행은 최소화 (긴급 시에만)
# 프로덕션(SDUI_TD)에는 절대 직접 실행 금지!
```

---

## 6. 일반적인 DB 변경 작업

### 6.1 컬럼 추가

```sql
-- V10__add_user_profile_image.sql
BEGIN;

ALTER TABLE users
ADD COLUMN IF NOT EXISTS profile_image_url VARCHAR(500);

ALTER TABLE users
ADD COLUMN IF NOT EXISTS bio TEXT;

COMMIT;
```

### 6.2 컬럼 이름 변경

```sql
-- V11__rename_diary_to_content_id.sql
BEGIN;

-- 컬럼명 변경
ALTER TABLE content
RENAME COLUMN diary_id TO content_id;

-- 시퀀스명 변경
ALTER SEQUENCE diary_diary_id_seq
RENAME TO content_content_id_seq;

COMMIT;
```

### 6.3 데이터 마이그레이션

```sql
-- V12__migrate_legacy_users.sql
BEGIN;

-- 백업 테이블 생성
CREATE TABLE IF NOT EXISTS users_backup AS
SELECT * FROM users_legacy;

-- 데이터 이관
INSERT INTO users (email, password, role_cd, role_nm, created_at)
SELECT email, password, 'USER', 'USER', reg_dt
FROM users_legacy
WHERE NOT EXISTS (
    SELECT 1 FROM users WHERE users.email = users_legacy.email
);

-- 검증
DO $$
DECLARE
    legacy_count INTEGER;
    new_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO legacy_count FROM users_legacy;
    SELECT COUNT(*) INTO new_count FROM users WHERE created_at >= '2026-03-01';

    IF new_count < legacy_count THEN
        RAISE EXCEPTION 'Data migration failed: expected %, got %', legacy_count, new_count;
    END IF;
END $$;

COMMIT;
```

### 6.4 인덱스 추가/삭제

```sql
-- V13__optimize_content_queries.sql
BEGIN;

-- 복합 인덱스 추가 (자주 함께 조회되는 컬럼)
CREATE INDEX IF NOT EXISTS idx_content_user_date
ON content(user_sqno, date DESC);

-- 사용하지 않는 인덱스 삭제
DROP INDEX IF EXISTS idx_content_old_column;

COMMIT;
```

### 6.5 Foreign Key 추가

```sql
-- V14__add_foreign_keys.sql
BEGIN;

-- Foreign Key 제약조건 추가
ALTER TABLE content
ADD CONSTRAINT fk_content_user
    FOREIGN KEY (user_sqno)
    REFERENCES users (user_sqno)
    ON DELETE CASCADE;

-- 제약조건 이름 확인
SELECT conname, contype
FROM pg_constraint
WHERE conrelid = 'content'::regclass;

COMMIT;
```

### 6.6 테이블 삭제

```sql
-- V15__drop_legacy_diary_table.sql
BEGIN;

-- 백업 확인 (이미 백업되어 있어야 함)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'diary_backup') THEN
        RAISE EXCEPTION 'Backup table diary_backup does not exist!';
    END IF;
END $$;

-- Foreign Key 제거 (있는 경우)
ALTER TABLE IF EXISTS some_table
DROP CONSTRAINT IF EXISTS fk_some_table_diary;

-- 테이블 삭제
DROP TABLE IF EXISTS diary CASCADE;

COMMIT;
```

---

## 7. 트러블슈팅

### 7.1 Flyway checksum mismatch

**증상**:
```
Migration checksum mismatch for migration version 5
Detected difference: ...
```

**원인**:
- 기존 마이그레이션 스크립트를 수정함

**해결책**:

**Option 1: flywayRepair (개발 환경만)**
```bash
./gradlew flywayRepair
./gradlew flywayMigrate
```

**Option 2: 원본 스크립트 복원**
```bash
# Git에서 원본 파일 복원
git checkout HEAD -- SDUI-server/src/main/resources/db/migration/V5__*.sql

# 변경사항은 새 버전으로 작성
# V11__fix_v5_migration.sql
```

### 7.2 Foreign Key 제약 위반

**증상**:
```
ERROR: insert or update on table "notifications" violates foreign key constraint "fk_notification_user"
Detail: Key (user_sqno)=(999) is not present in table "users".
```

**원인**:
- 참조하는 테이블(users)에 해당 레코드가 없음

**해결책**:
```sql
-- 마이그레이션 스크립트에서 데이터 존재 확인
BEGIN;

INSERT INTO notifications (user_sqno, message)
SELECT 123, '알림 메시지'
WHERE EXISTS (SELECT 1 FROM users WHERE user_sqno = 123);

COMMIT;
```

### 7.3 빈 데이터베이스에서 Flyway 실패

**증상**:
```
Migration V2__create_content_table.sql failed
ERROR: relation "users" does not exist
```

**원인**:
- SDUI_LAB가 빈 상태로 생성됨
- V2 스크립트가 users 테이블 존재를 가정

**해결책**:

**Option 1: 스키마 복사 (권장)**
```bash
# SDUI_TD의 스키마를 SDUI_LAB로 복사
docker exec sdui-db pg_dump -U mina -d SDUI_TD --schema-only | \
docker exec -i sdui-db psql -U mina -d SDUI_LAB
```

**Option 2: V1 스크립트 수정**
```sql
-- V1__baseline_schema.sql을 실제 테이블 생성 스크립트로 변경
BEGIN;

CREATE TABLE IF NOT EXISTS users (...);
CREATE TABLE IF NOT EXISTS diary (...);
-- ... 모든 기본 테이블 생성

COMMIT;
```

### 7.4 마이그레이션 버전 충돌

**증상**:
```
Detected resolved migration not applied to database: V8
But database contains migration V8 with different checksum
```

**원인**:
- 두 명의 개발자가 동시에 V8 스크립트를 작성함

**해결책**:
```bash
# 한 명의 V8을 V9로 변경
mv V8__add_feature_a.sql V9__add_feature_a.sql

# Git에서 충돌 해결 후 병합
```

### 7.5 Flyway 히스토리 테이블 손상

**증상**:
```
Unable to check whether table "flyway_schema_history" exists
```

**원인**:
- flyway_schema_history 테이블이 삭제되거나 손상됨

**해결책**:

**개발 환경**:
```bash
# Flyway 초기화
./gradlew flywayClean
./gradlew flywayMigrate
```

**프로덕션 환경**:
```sql
-- flyway_schema_history 재생성 (주의!)
CREATE TABLE IF NOT EXISTS flyway_schema_history (
    installed_rank INT NOT NULL,
    version VARCHAR(50),
    description VARCHAR(200) NOT NULL,
    type VARCHAR(20) NOT NULL,
    script VARCHAR(1000) NOT NULL,
    checksum INT,
    installed_by VARCHAR(100) NOT NULL,
    installed_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    execution_time INT NOT NULL,
    success BOOLEAN NOT NULL,
    PRIMARY KEY (installed_rank)
);

-- 기존 마이그레이션 기록 복원
INSERT INTO flyway_schema_history (installed_rank, version, description, type, script, checksum, installed_by, installed_on, execution_time, success)
VALUES
(1, '1', 'baseline schema', 'SQL', 'V1__baseline_schema.sql', 123456, 'mina', '2026-03-01 10:00:00', 50, true),
(2, '2', 'create content table', 'SQL', 'V2__create_content_table.sql', 234567, 'mina', '2026-03-01 10:00:05', 100, true);
-- ... 모든 성공한 마이그레이션 기록
```

---

## 참고 자료

- **AWS 환경 가이드**: `.ai/maintenance/aws_environment_guide.md`
- **배포 체크리스트**: `.ai/maintenance/deployment_checklist.md`
- **트러블슈팅**: `.ai/maintenance/troubleshooting_guide.md`
- **Flyway 공식 문서**: https://flywaydb.org/documentation
- **PostgreSQL 데이터 타입**: https://www.postgresql.org/docs/current/datatype.html

---

**문서 관리**:
 
- 최종 업데이트: 2026-03-03
- 다음 리뷰 예정일: 2026-04-03
