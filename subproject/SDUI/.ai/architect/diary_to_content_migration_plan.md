# diary → content 테이블 마이그레이션 계획

**작성자**: Architect
**작성일**: 2026-03-01
**목적**: diary 테이블을 content 테이블로 변경하는 마이그레이션의 영향도 분석 및 전략 수립

---

## 📋 변경 개요

### 변경 사항
- **기존**: `diary` 테이블
- **변경 후**: `content` 테이블 (같은 컬럼 구조 유지)

### 변경 이유
- 도메인 용어 일반화 (일기 → 콘텐츠)
- 향후 확장성 고려 (일기 외 다양한 콘텐츠 타입 수용)

---

## 🔍 영향도 분석

### 1. 데이터베이스 레벨

#### 1.1 테이블 구조 // [메모] pgAdmin에서 완료

```sql
-- 기존 diary 테이블 (generate_pg.sql 참조)
CREATE TABLE diary (
    diary_id bigserial NOT NULL,
    content varchar(255),
    date varchar(255),
    title varchar(255),
    user_sqno bigint,
    -- ... 기타 27개 컬럼
    CONSTRAINT diary_pkey PRIMARY KEY (diary_id),
    CONSTRAINT fkq7liav7xmcxfdvhy1iten6bxy FOREIGN KEY (user_sqno)
        REFERENCES users (user_sqno)
);

-- 변경 후 content 테이블
CREATE TABLE content (
    content_id bigserial NOT NULL,          -- diary_id → content_id
    content_text varchar(255),              -- content → content_text (컬럼명 중복 방지)
    date varchar(255),
    title varchar(255),
    user_sqno bigint,
    -- ... 기타 컬럼 동일
    CONSTRAINT content_pkey PRIMARY KEY (content_id),
    CONSTRAINT fk_content_user FOREIGN KEY (user_sqno)
        REFERENCES users (user_sqno) // [메모] pgAdmin에서 완료
);
```

#### 1.2 마이그레이션 전략

**Option A: DROP + CREATE (권장 - 개발 환경)**
```sql
-- 기존 데이터 백업 (필요 시)
CREATE TABLE diary_backup AS SELECT * FROM diary; // [메모] pgAdmin에서 완료

-- 테이블 삭제
DROP TABLE diary CASCADE; // [메모] diary와 user가 연관되어있어서 우선 DRP 안함 

-- 새 테이블 생성
CREATE TABLE content ( ... ); // [메모] pgAdmin에서 완료
```

**추천**: 개발 환경이므로 **Option A (DROP + CREATE)** 사용

#### 1.3 ui_metadata 영향
- `screen_id`: `DIARY_LIST` → `CONTENT_LIST`
- `screen_id`: `DIARY_WRITE` → `CONTENT_WRITE`
- `screen_id`: `DIARY_DETAIL` → `CONTENT_DETAIL`
- `component_id`: `MY_DIARY_*` → `MY_CONTENT_*`
- `data_sql_key`: `GET_DIARY_LIST` → `GET_CONTENT_LIST`
- `action_url`: `/api/diary/*` → `/api/content/*`

---

### 2. 백엔드 레벨

#### 2.1 패키지 구조 변경 // [메모] bash로 작업할 명령어 를 밑에 적어줘
```
domain/
  diary/              → content/
    controller/
      DiaryController → ContentController
    service/
      DiaryService    → ContentService
    domain/
      Diary.java      → Content.java
      DiaryRepository → ContentRepository
    dto/
      DiaryRequest    → ContentRequest
      DiaryResponse   → ContentResponse
```

**📌 실행 방법: 자동화 스크립트 사용**

**파일**: `rename_diary_to_content.sh` (프로젝트 루트에 생성 완료)

```bash
# 실행 권한 부여
chmod +x rename_diary_to_content.sh

# 스크립트 실행 (1분 소요)
./rename_diary_to_content.sh
```

**스크립트 자동 처리 항목**:
- ✅ `domain/content` 패키지 디렉토리 생성
- ✅ 6개 Java 파일 복사 및 리네임
- ✅ 패키지명 변경 (`domain.diary` → `domain.content`)
- ✅ 클래스명 변경 (`Diary` → `Content`)
- ✅ 테이블명 변경 (`@Table(name = "diary")` → `@Table(name = "content")`)
- ✅ 컬럼명 변경 (`diary_id` → `content_id`, `content` → `content_text`)
- ✅ API 경로 변경 (`/api/diary` → `/api/content`)

**스크립트 실행 후 수동 작업** (IntelliJ):
1. 프로젝트 다시 로드: `Ctrl + Alt + Y`
2. Find & Replace (`Ctrl + Shift + R`):
   - `getDiaryId()` → `getContentId()`
   - `setDiaryId()` → `setContentId()`
   - `getContent()` → `getContentText()` (엔티티 메서드)
   - `setContent()` → `setContentText()`
   - `"diary"` → `"content"` (로그 메시지, 주석)
3. Import 정리: `Ctrl + Alt + O`
4. 컴파일 확인: `./gradlew clean build`
5. diary 패키지 삭제 (컴파일 성공 확인 후):
   ```bash
   rm -rf SDUI-server/src/main/java/com/domain/demo_backend/domain/diary
   ```

#### 2.2 API 엔드포인트 변경 // [메모] 2.1 이 끝나면 인텔리제이에서 수동으로 한번에 할 예정 
| 기존 | 변경 후 |
|------|---------|
| `POST /api/diary/write` | `POST /api/content/write` |
| `GET /api/diary/list` | `GET /api/content/list` |
| `GET /api/diary/{id}` | `GET /api/content/{id}` |
| `PUT /api/diary/{id}` | `PUT /api/content/{id}` |
| `DELETE /api/diary/{id}` | `DELETE /api/content/{id}` |
| `DELETE /api/diary/delete-all` | `DELETE /api/content/delete-all` |

---

### 3. 프론트엔드 레벨

#### 3.1 화면 ID 변경 // [메모] 2.1 이 끝나면 인텔리제이에서 수동으로 한번에 할 예정
| 기존 | 변경 후 |
|------|---------|
| `DIARY_LIST` | `CONTENT_LIST` |
| `DIARY_WRITE` | `CONTENT_WRITE` |
| `DIARY_DETAIL` | `CONTENT_DETAIL` |

#### 3.2 액션 타입 변경 // [메모] 2.1 이 끝나면 인텔리제이에서 수동으로 한번에 할 예정
| 기존 | 변경 후 |
|------|---------|
| `DIARY_WRITE_SUBMIT` | `CONTENT_WRITE_SUBMIT` |
| `DIARY_UPDATE` | `CONTENT_UPDATE` |
| `DIARY_DELETE` | `CONTENT_DELETE` |
| `DELETE_ALL_DIARIES` | `DELETE_ALL_CONTENTS` |

#### 3.3 URL 라우팅 변경  // [메모] ui_metadata 테이블에서 DML 필요 , 쿼리문 필요
| 기존 | 변경 후 |
|------|---------|
| `/view/DIARY_LIST` | `/view/CONTENT_LIST` |
| `/view/DIARY_WRITE` | `/view/CONTENT_WRITE` |
| `/view/DIARY_DETAIL` | `/view/CONTENT_DETAIL` |

**📌 실행 방법: SQL 쿼리 실행**

**파일**: `V3__diary_to_content_ui_metadata.sql` (프로젝트 루트에 생성 완료)

```bash
# psql로 실행
psql -U postgres -d testdb -p 5433 -f V3__diary_to_content_ui_metadata.sql

# 또는 pgAdmin Query Tool에서 파일 내용 복사하여 실행
```

**SQL 자동 처리 항목**:
- ✅ `screen_id`: `DIARY_LIST` → `CONTENT_LIST`, `DIARY_WRITE` → `CONTENT_WRITE`, `DIARY_DETAIL` → `CONTENT_DETAIL`
- ✅ `component_id`: `MY_DIARY_*` → `MY_CONTENT_*`, `DIARY_*` → `CONTENT_*`
- ✅ `action_type`: `DIARY_WRITE_SUBMIT` → `CONTENT_WRITE_SUBMIT`, `DIARY_UPDATE` → `CONTENT_UPDATE`, etc.
- ✅ `action_url`: `/api/diary/*` → `/api/content/*`
- ✅ `data_api_url`: `/api/diary/*` → `/api/content/*`
- ✅ `data_sql_key`: `GET_DIARY_*` → `GET_CONTENT_*`
- ✅ `label_text`: `일기` → `콘텐츠`
- ✅ `label_text_overrides` (JSONB): `일기` → `콘텐츠`
- ✅ `query_master` 테이블: `sql_key`, `query_text`, `description` 함께 변경

**검증 쿼리** (SQL 스크립트에 포함):
```sql
-- CONTENT 관련 데이터 확인
SELECT screen_id, component_id, label_text, action_type, action_url
FROM ui_metadata
WHERE screen_id LIKE 'CONTENT%'
ORDER BY screen_id, sort_order;

-- DIARY 잔존 데이터 확인 (결과가 없어야 함)
SELECT screen_id, component_id, label_text
FROM ui_metadata
WHERE screen_id LIKE 'DIARY%';
```

---

## 🚀 마이그레이션 실행 가이드

### ✅ 사전 완료 작업 (2026-03-01)
- [x] content 테이블 생성 (pgAdmin)
- [x] diary_backup 백업 생성 (pgAdmin)
- [x] RBAC 컬럼 추가 완료 (V2 SQL 실행)
- [x] rename_diary_to_content.sh 스크립트 생성
- [x] V3__diary_to_content_ui_metadata.sql 스크립트 생성

---

### Step 1: 백엔드 패키지 리네임 (bash 자동화)

**소요 시간**: 1분

```bash
cd C:\Users\Samsung\Documents\Development\Personal_Projects\2026\SDUI

# 실행 권한 부여
chmod +x rename_diary_to_content.sh

# 스크립트 실행
./rename_diary_to_content.sh
```

**처리 내용**:
- `domain/content` 패키지 생성
- 6개 Java 파일 복사 및 리네임
- 패키지명, 클래스명, 테이블명 자동 변경

---

### Step 2: IntelliJ 백엔드 수동 작업

**소요 시간**: 30분

1. **프로젝트 다시 로드**: `Ctrl + Alt + Y`

2. **Find & Replace** (`Ctrl + Shift + R`) - 프로젝트 전체 범위:

   | 검색어 | 치환어 | 설명 |
   |-------|--------|------|
   | `getDiaryId()` | `getContentId()` | Getter 메서드 |
   | `setDiaryId(` | `setContentId(` | Setter 메서드 |
   | `getContent()` | `getContentText()` | Content 필드 getter |
   | `setContent(` | `setContentText(` | Content 필드 setter |
   | `"diary"` | `"content"` | 로그 메시지 |
   | `// diary` | `// content` | 주석 |

3. **Import 정리**: `Ctrl + Alt + O` (전체 프로젝트)

4. **컴파일 확인**:
   ```bash
   cd SDUI-server
   ./gradlew clean build
   ```

5. **diary 패키지 삭제** (컴파일 성공 확인 후):
   ```bash
   rm -rf src/main/java/com/domain/demo_backend/domain/diary
   ```

---

### Step 3: ui_metadata 업데이트 (SQL)

**소요 시간**: 1분

**pgAdmin에서 실행**:
1. Query Tool 열기
2. `V3__diary_to_content_ui_metadata.sql` 내용 복사
3. 실행 후 검증 쿼리 결과 확인

**또는 psql**:
```bash
psql -U postgres -d testdb -p 5433 -f V3__diary_to_content_ui_metadata.sql
```

**검증**:
- CONTENT_LIST, CONTENT_WRITE, CONTENT_DETAIL screen_id 확인
- DIARY 관련 데이터가 없는지 확인

---

### Step 4: 프론트엔드 수동 작업 (IntelliJ)

**소요 시간**: 20분

**Find & Replace** (`Ctrl + Shift + R`) - `metadata-project/` 범위:

#### 4.1 화면 ID 변경
| 검색어 | 치환어 |
|-------|--------|
| `'DIARY_LIST'` | `'CONTENT_LIST'` |
| `'DIARY_WRITE'` | `'CONTENT_WRITE'` |
| `'DIARY_DETAIL'` | `'CONTENT_DETAIL'` |

#### 4.2 액션 타입 변경
| 검색어 | 치환어 |
|-------|--------|
| `DIARY_WRITE_SUBMIT` | `CONTENT_WRITE_SUBMIT` |
| `DIARY_UPDATE` | `CONTENT_UPDATE` |
| `DIARY_DELETE` | `CONTENT_DELETE` |
| `DELETE_ALL_DIARIES` | `DELETE_ALL_CONTENTS` |

#### 4.3 API URL 변경
| 검색어 | 치환어 |
|-------|--------|
| `/api/diary/` | `/api/content/` |

#### 4.4 필드명 변경
| 검색어 | 치환어 |
|-------|--------|
| `formData.content` | `formData.contentText` |
| `formData.diaryId` | `formData.contentId` |

#### 4.5 라벨 변경
| 검색어 | 치환어 |
|-------|--------|
| `일기 목록` | `콘텐츠 목록` |
| `일기 작성` | `콘텐츠 작성` |
| `내 일기` | `내 콘텐츠` |

**수정 대상 파일 (8개)**:
- `components/constants/screenMap.ts`
- `components/DynamicEngine/hook/useBusinessActions.tsx`
- `components/DynamicEngine/hook/useUserActions.tsx`
- `components/layout/Header.tsx`
- `app/view/[...slug]/page.tsx` (PROTECTED_SCREENS)
- `components/DynamicEngine/hook/usePageMetadata.tsx`
- `components/fields/ButtonField.tsx`
- `tests/rendering_optimization.test.tsx`

---

### Step 5: 검증 및 테스트

**소요 시간**: 10분

#### 5.1 백엔드 검증
```bash
cd SDUI-server

# 컴파일 성공 확인
./gradlew clean build

# 테스트 실행
./gradlew test

# 서버 시작
./gradlew bootRun
```

#### 5.2 프론트엔드 검증
```bash
cd metadata-project

# 린트 확인
npm run lint

# 컴파일 확인
npm run build

# 테스트 실행
npm run test

# 개발 서버 시작
npm run dev
```

#### 5.3 API 테스트 (Postman/Thunder Client)
- ✅ `POST /api/content/write` - 콘텐츠 작성
- ✅ `GET /api/content/list` - 콘텐츠 목록 조회
- ✅ `GET /api/content/{id}` - 콘텐츠 상세 조회
- ✅ `PUT /api/content/{id}` - 콘텐츠 수정
- ✅ `DELETE /api/content/{id}` - 콘텐츠 삭제
- ✅ `DELETE /api/content/delete-all` - 전체 삭제 (ROLE_ADMIN만)

---

### Step 6: diary 테이블 삭제 (최종)

**소요 시간**: 1분

**모든 검증 완료 후 실행**:

```sql
-- content 테이블 사용 확인 후
DROP TABLE diary CASCADE;

-- 백업 테이블도 삭제 (필요 시)
DROP TABLE diary_backup;
```

**주의**: user 테이블과 FK 관계 있으므로 CASCADE 옵션 필수

---

## 📊 전체 소요 시간 (실측)

| 단계 | 작업 | 예상 시간 | 실제 시간 |
|-----|------|----------|----------|
| **Step 1** | bash 스크립트 실행 | 1분 | - |
| **Step 2** | IntelliJ 백엔드 수정 | 30분 | - |
| **Step 3** | SQL 실행 | 1분 | - |
| **Step 4** | 프론트엔드 수정 | 20분 | - |
| **Step 5** | 검증 및 테스트 | 10분 | - |
| **Step 6** | diary 테이블 삭제 | 1분 | - |
| **총 예상 시간** |  | **약 1시간** |  |

---

## ⚠️ 주의사항

### 1. 하위 호환성
- 기존 diary 관련 코드는 **모두 제거**됨 (하위 호환 없음)
- 사용자가 북마크한 `/view/DIARY_LIST` URL은 404 발생

### 2. 데이터 손실 위험
- **개발 환경**: 기존 diary 데이터 삭제됨 (문제 없음)

### 3. 외부 의존성
- **Redis 캐시**: `SQL:GET_DIARY_LIST` → `SQL:GET_CONTENT_LIST` 키 변경
- **캐시 무효화 필요**: 배포 후 Redis FLUSHDB 실행

---

## 🔗 생성된 파일

### 1. 자동화 스크립트
- ✅ `rename_diary_to_content.sh` - 백엔드 패키지 리네임 bash 스크립트

### 2. SQL 마이그레이션
- ✅ `V3__diary_to_content_ui_metadata.sql` - ui_metadata 업데이트 쿼리

### 3. 관련 문서
- `.ai/backend_engineer/diary_to_content_migration_plan.md` - 백엔드 상세 수정 가이드
- `.ai/frontend_engineer/diary_to_content_migration_plan.md` - 프론트엔드 상세 수정 가이드

---

## ✅ 체크리스트

### 데이터베이스
- [x] content 테이블 생성 (pgAdmin)
- [x] diary_backup 백업
- [ ] V3 SQL 실행 (ui_metadata 업데이트)
- [ ] diary 테이블 DROP (최종 단계)

### 백엔드
- [ ] rename_diary_to_content.sh 실행
- [ ] IntelliJ Find & Replace (getter/setter)
- [ ] Import 정리
- [ ] 컴파일 확인
- [ ] diary 패키지 삭제
- [ ] 테스트 실행

### 프론트엔드
- [ ] 화면 ID 변경 (screenMap.ts 등)
- [ ] 액션 타입 변경 (useBusinessActions, useUserActions)
- [ ] API URL 변경
- [ ] 필드명 변경 (formData)
- [ ] 라벨 변경 (일기 → 콘텐츠)
- [ ] 린트 및 컴파일 확인

### 검증
- [ ] 백엔드 컴파일 성공
- [ ] 프론트엔드 컴파일 성공
- [ ] API 동작 테스트 (CRUD)
- [ ] RBAC 동작 확인
