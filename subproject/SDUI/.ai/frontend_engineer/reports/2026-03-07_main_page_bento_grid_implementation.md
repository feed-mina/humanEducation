# MAIN_PAGE 벤토 그리드 구현 보고서

**작성일**: 2026-03-07
**브랜치**: feature/main_page_product
**상태**: ✅ CSS 추가 완료 / 🔲 브라우저 QA 대기 중
**관련 파일**: `product_main_plan.md`, `mockup2.html`

---

## Executive Summary

MAIN_PAGE를 기존 리스트형 레이아웃에서 3열 벤토 그리드로 전환하는 작업을 완료했습니다.
백엔드/Java 코드 수정 없이 **DB 메타데이터(V8 마이그레이션) + 프론트엔드 CSS 추가**만으로 구현했습니다.
Docker 로컬 환경에서 Flyway V1-V8 마이그레이션 정상 실행 확인 완료. CSS는 `mockup2.html` 디자인 기준으로 작성됐습니다.

---

## 1. 구현 범위

### 1.1 화면별 카드 구성 (mockup2.html 기준)

| 상태 | Card 1 (col 1-2) | Card 2 (col 3) | Card 3 (full width) |
|------|-----------------|----------------|---------------------|
| **PC/모바일 로그인** | 약속 위젯 (TIME_RECORD_WIDGET) | 콘텐츠 쓰러가기 | 콘텐츠 보기 |
| **PC/모바일 비로그인** | 시간 설정하기 (TIME_RECORD_WIDGET) | 로그인 하러가기 | 튜토리얼 보기 |

### 1.2 RBAC 분기 전략

```
allowed_roles = NULL       → ROLE_GUEST + ROLE_USER 모두 표시
allowed_roles = 'ROLE_USER'  → 로그인 유저만 표시
allowed_roles = 'ROLE_GUEST' → 비로그인(GUEST)만 표시
```

> UiController: 미인증 요청 → `userRole = "ROLE_GUEST"` 자동 적용

---

## 2. 변경 파일 목록

| 파일 | 변경 종류 | 내용 |
|------|----------|------|
| `SDUI-server/src/main/resources/db/migration/V8__main_page_bento_grid.sql` | 신규 (기존 세션) | MAIN_PAGE 벤토 카드 메타데이터 INSERT/UPDATE |
| `metadata-project/app/styles/pages.css` | 수정 (line 1717~) | 벤토 그리드 CSS 추가 |
| `SDUI-server/src/main/resources/db/migration/V1__baseline_schema.sql` | 전면 재작성 (기존 세션) | 실제 DDL로 교체 — Docker 빈 DB 대응 |
| `SDUI-server/src/main/resources/application.yml` (+ out/, bin/) | 수정 (기존 세션) | `baseline-version: 1 → 0` |
| `V2~V7__*.sql` | 수정 (기존 세션) | BEGIN/COMMIT 제거 (Flyway 트랜잭션 중첩 방지) |

---

## 3. CSS 설계 — `pages.css` (line 1717~)

### 3.1 핵심 설계 결정 사항

#### TIME_RECORD_WIDGET 처리
`TIME_RECORD_WIDGET`은 DynamicEngine에서 GROUP이 아닌 컴포넌트로 렌더링되므로 `css_class` 속성이 자동 적용되지 않는다.
→ `.main-bento > :first-child { grid-column: 1 / 3; }` 로 CSS에서 직접 2칸 점유 지정.

#### overflow: visible 유지
`goal-list-popup`이 `position: absolute; top: 100%`로 렌더링된다.
`.bento-card`에 `overflow: hidden`을 적용하면 팝업이 잘린다.
→ `.bento-card { overflow: visible; position: relative; }` 유지 필수.

#### flex-col-layout vs main-bento 충돌 방지
DynamicEngine은 모든 GROUP에 `flex-col-layout`을 자동 추가한다.
MAIN_SECTION에는 `main-bento`와 `flex-col-layout`이 동시에 적용된다.
→ CSS에서 `flex-col-layout`을 `main-bento`보다 **먼저** 선언하여 `display: grid`가 우선 적용되도록 함.

#### RecordTimeComponent 기존 CSS 충돌
`pages.css` 기존 코드에 충돌하는 속성이 2개 존재:

| 클래스 | 충돌 속성 | 영향 | override |
|--------|-----------|------|---------|
| `.time-record-container` | `position: sticky; top: 0; z-index: 100` | 벤토 카드 내 스크롤 시 카드 상단에 고정됨 | `position: relative !important` |
| `.no-goal-container` | `flex: 8; justify-content: flex-end` | 헤더 비율 전용 값, 벤토 카드 내 콘텐츠 하단 쏠림 | `flex: 1 !important; justify-content: center !important` |

### 3.2 CSS 클래스 목록

```
.flex-row-layout       — DynamicEngine GROUP 방향 헬퍼 (ROW)
.flex-col-layout       — DynamicEngine GROUP 방향 헬퍼 (COLUMN/기본값)

.main-bento            — 3열 CSS Grid 컨테이너
.main-bento > :first-child  — TIME_RECORD_WIDGET col-span-2 강제
.col-span-2            — GROUP 자식 열 확장 (1 / 3)
.col-span-3            — GROUP 자식 열 확장 (1 / -1)

.bento-card            — 카드 공통 (radius 16px, padding 24px, overflow: visible)
.bento-card-appointment  — 흰 배경 + #e5e7eb 테두리
.bento-card-no-goal    — 연한 그래디언트 배경, 중앙 정렬
.bento-card-diary      — 녹색 그래디언트 (#166534 → #15803d)
.bento-card-login      — Teal (#0f766e)
.bento-card-dark       — 다크 슬레이트 (#1e293b)

.bento-card-body       — 내부 flex column, gap 4px
.bento-card-icon       — 이모지 아이콘 (1.6rem)
.bento-card-title      — 제목 텍스트 (1.1rem, 700)
.bento-card-desc       — 설명 텍스트 (0.82rem, opacity .75)
.bento-card-arrow      — 화살표 버튼 (ButtonField → "content-btn bento-card-arrow")
.bento-card-tag        — 뱃지 버튼 (ButtonField → "content-btn bento-card-tag")

.main-bento .time-record-container  — sticky 해제 override
.main-bento .no-goal-container      — flex 비율 + 정렬 override
```

### 3.3 반응형 처리

```css
@media (max-width: 768px) {
    .main-bento { grid-template-columns: 1fr; }
    .main-bento > :first-child, .col-span-2, .col-span-3 { grid-column: 1 !important; }
}
```

---

## 4. DynamicEngine 렌더링 구조 분석

V8 메타데이터 기준 실제 DOM 구조 (GUEST 예시):

```html
<div class="engine-container is-pc">
  <div class="content-area">
    <!-- MAIN_SECTION: GROUP, css_class="main-bento" -->
    <div class="group-MAIN_SECTION main-bento flex-col-layout">

      <!-- TIME_RECORD_WIDGET: 컴포넌트 직접 렌더, css_class 미적용 -->
      <!-- → .main-bento > :first-child 로 col 1-2 점유 -->
      <div class="no-goal-container">  ← RecordTimeComponent (goalTime=null)
        <p>오늘의 약속 시간은 언제인가요?</p>
        <button class="setup-button">시간 설정하기</button>
      </div>

      <!-- GROUP: css_class 정상 적용 -->
      <div class="group-main_bento_login_grp bento-card bento-card-login flex-col-layout">
        <div class="group-main_bento_login_body bento-card-body flex-col-layout">
          <div class="ui-text-field bento-card-title">로그인 하러가기</div>
          <div class="ui-text-field bento-card-desc">계정이 있으신가요?...</div>
        </div>
        <button class="content-btn bento-card-arrow">→</button>
      </div>

      <!-- col-span-3: GROUP 자식 → css_class에 직접 포함되어 적용됨 -->
      <div class="group-main_bento_tutorial_grp bento-card bento-card-dark col-span-3 flex-col-layout">
        ...
        <button class="content-btn bento-card-tag">📖 튜토리얼</button>
      </div>

    </div>
  </div>
</div>
```

### 렌더러 타입별 css_class 적용 방식

| 컴포넌트 타입 | 렌더러 | css_class 적용 방식 |
|-------------|--------|-------------------|
| GROUP | DynamicEngine (div 직접 생성) | ✅ className에 직접 추가 |
| TEXT | TextField → `<div class="ui-text-field {css_class}">` | ✅ meta.cssClass로 적용 |
| BUTTON | ButtonField → `<button class="content-btn {css_class}">` | ✅ cn()으로 병합 |
| TIME_RECORD_WIDGET | RecordTimeComponent → 하드코딩된 className | ❌ css_class 미적용 → CSS `:first-child` 우회 |

---

## 5. 로컬 테스트 환경 구성 이력

### 5.1 문제 원인

| 문제 | 원인 | 해결책 |
|------|------|--------|
| V2 마이그레이션 실패 "relation 'users' does not exist" | `baseline-version: 1` → V1이 baseline 마커로만 처리되어 실행 안 됨 (users 테이블 없음) | V1을 실제 DDL로 재작성 + `baseline-version: 0`으로 변경 |
| Flyway "already in transaction" 경고 | V2~V8에 명시적 BEGIN/COMMIT이 있어 Flyway 자체 트랜잭션과 중첩 | V2~V8 SQL에서 BEGIN/COMMIT 제거 |
| Spring Boot Port 8080 이미 사용 중 | sdui-backend Docker 컨테이너가 8080을 점유 | `docker stop sdui-backend` 후 IntelliJ 재기동 |
| docker-compose 서비스 이름 오류 | `docker-compose stop sdui-db` → 서비스명은 `db` (컨테이너명 `sdui-db`와 다름) | `docker-compose stop db` 사용 |

### 5.2 로컬 테스트 순서 (확정)

```bash
# 1. Docker DB 초기화 (프로젝트 루트에서)
docker-compose stop db
docker-compose rm -f db
docker-compose up -d db

# 2. sdui-backend Docker 컨테이너가 실행 중이면 중지
docker stop sdui-backend

# 3. IntelliJ에서 DemoBackendApplication 실행
#    → Flyway V1~V8 순차 실행됨

# 4. 프론트엔드 실행
cd metadata-project && npm run dev

# 5. 브라우저 확인
# http://localhost:3000/view/MAIN_PAGE
```

### 5.3 Flyway 재실행 방법 (V8 실패 시)

```sql
-- Docker DB psql에서 실행
DELETE FROM flyway_schema_history WHERE version = '8';
-- 이후 Spring Boot 재시작 → V8 재실행
```

---

## 6. QA 체크리스트 (TODO 6~11)

| # | 확인 항목 | 상태 |
|---|----------|------|
| 6 | 로컬 DB Flyway V1-V8 정상 실행 | ✅ 완료 (이전 세션) |
| 7 | PC 로그인: 약속위젯(col1-2) + 콘텐츠쓰러가기(col3) + 콘텐츠보기(full) | 🔲 |
| 8 | PC 비로그인: 시간설정하기(col1-2) + 로그인(col3) + 튜토리얼(full) | 🔲 |
| 9 | 모바일 단일 컬럼 렌더링 (두 상태 모두) | 🔲 |
| 10 | RecordTimeComponent sticky 해제 확인, goal-list-popup 카드 밖으로 표시 확인 | 🔲 |
| 11 | qa_engineer와 테스트 케이스 협의 | 🔲 |

---

## 7. AWS 배포 시 주의사항

- Docker DB 포트는 **5434** (로컬 5432/5433과 충돌 방지용으로 변경됨)
  → AWS EC2 환경은 기존 설정 유지, 변경 불필요
- V8 SQL은 `label_text NOT NULL` 제약 대응 완료 (GROUP/WIDGET 행에 `''` 명시)
- `TUTORIAL_PAGE` screen_id DB 존재 여부는 AWS 배포 전 확인 필요 (TODO 3)

---

## 8. 참고 파일

| 파일 | 설명 |
|------|------|
| `.ai/frontend_engineer/Mock/mockup2.html` | 벤토 그리드 디자인 기준 목업 (v2) |
| `.ai/frontend_engineer/Mock/mockup_main_bento.html` | 초안 목업 (v1, 참고용) |
| `.ai/frontend_engineer/product_main_plan.md` | 구현 계획 및 TODO 목록 |
| `SDUI-server/src/main/resources/db/migration/V8__main_page_bento_grid.sql` | DB 메타데이터 마이그레이션 |
| `metadata-project/app/styles/pages.css` (line 1717~) | 벤토 CSS |
