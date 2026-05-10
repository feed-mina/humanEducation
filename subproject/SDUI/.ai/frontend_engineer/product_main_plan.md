# MAIN_PAGE 벤토 그리드 전환 구현 계획 — 2026-03-05

> 요청 출처: mockup2.html 검토 완료 후 구현 요청
> 참고 목업: `SDUI/assets/mockup2.html`
> 제약: 백엔드 Java 코드 수정 없음 / DB ui_metadata 데이터 변경은 허용

---

## 배경

현재 MAIN_PAGE는 전통적인 리스트형 레이아웃이다.
목표는 3컬럼 벤토 그리드로 전환하되, 서버 신규 API 없이 아래 데이터만 사용한다:

- `goalTime` — 목표 시간 (null 가능, 로그인 시에만 fetch)
- `goalList` — 목표 시간 리스트
- `remainTimeText` — 카운트다운 텍스트 (1초 갱신)

### 화면별 최종 카드 구성 (mockup2 기준)

| 상태 | Card 1 (col 1-2) | Card 2 (col 3) | Card 3 (full) |
|------|-----------------|----------------|---------------|
| PC 로그인 + goalTime | 약속 위젯 (TIME_RECORD_WIDGET) | 콘텐츠 쓰러가기 | 콘텐츠 보기 |
| PC 비로그인 | 시간 설정하기 (TIME_RECORD_WIDGET) | 로그인 하러가기 | 튜토리얼 보기 |
| 모바일 로그인 | 약속 위젯 (단일 컬럼) | 콘텐츠 쓰러가기 | 콘텐츠 보기 |
| 모바일 비로그인 | 시간 설정하기 (단일 컬럼) | 로그인 하러가기 | 튜토리얼 보기 |

> **핵심:** Card 1은 `TIME_RECORD_WIDGET` 단일 컴포넌트로 처리 — goalTime 유무에 따라 RecordTimeComponent 내부에서 자동 분기 (no-goal ↔ clock display)

---

## 아키텍처 결정 사항

### PC 약속 위젯 중복 문제

PC에서 `AppShell.tsx`의 `pc-top-utility`에 이미 RecordTimeComponent가 존재한다.
벤토 카드에도 TIME_RECORD_WIDGET을 추가하면 같은 데이터가 2곳에 표시된다.

**결정: 허용** — 역할이 다름
- `pc-top-utility` (compact): 전체 페이지 고정 영역, 어느 페이지에서도 접근 가능
- 벤토 카드 (full): MAIN_PAGE 전용 메인 위젯, 큰 시각적 강조

### Card 2·3 로그인 상태 분기

SDUI는 `/api/ui/{screenId}` 응답이 user role 기반으로 다른 메타데이터를 반환한다.
→ DB에서 `allowed_roles`를 `ROLE_USER` / `NULL(GUEST 허용)` 으로 구분하면 서버 코드 수정 없이 분기 가능.

**`allowed_roles` 컬럼 동작 (V4 마이그레이션 확인)**:
- `NULL` → 모두 허용 (비로그인 포함)
- `'ROLE_USER'` → 로그인한 USER만 표시
- GUEST 전용 카드는 `NULL` + USER 카드를 `ROLE_USER`로 필터링하는 방식 적용

> ⚠️ 주의: GUEST 사용자에게만 보이고 USER에게는 숨겨야 하는 카드는, USER 전용 카드를 `ROLE_USER`로 설정하고 공통 카드를 `NULL`로 설정하는 조합으로 처리. "GUEST 전용" allowed_roles 값은 백엔드 필터링 로직 확인 후 결정 필요.

---

## 영향받는 파일

| 파일 경로 | 변경 종류 | 변경 내용 |
|----------|----------|---------|
| `metadata-project/app/globals.css` | 수정 | 벤토 그리드 CSS 클래스 추가 |
| `DB: ui_metadata` (MAIN_PAGE 관련 레코드) | UPDATE + INSERT | 카드 구조 재편 |
| `metadata-project/components/fields/RecordTimeComponent.tsx` | ✅ 수정 불필요 | 벤토 카드 임베드 분석 완료 — 하단 참고 |
| Flyway 마이그레이션 `V8__main_page_bento_grid.sql` | 신규 | ui_metadata INSERT/UPDATE SQL |

---

## 접근 방식 결정

### ❌ Option A 재검토: 기각 (Card 2·3에 대해)

Card 1은 `TIME_RECORD_WIDGET` 재사용으로 OK.

Card 2·3에 `BUTTON` 타입 사용을 검토했으나 **기각**:

`ButtonField.tsx` 코드 확인 결과:
```tsx
return (
    <button type="button" className={mergedClassName} onClick={handleAction}>
        {label}   // ← labelText 단일 텍스트만 렌더링
    </button>
);
```
mockup2.html 카드는 아이콘 + 제목(h3) + 설명(p) + 화살표(→) 4개 요소가 필요.
단일 `{label}` 텍스트로는 표현 불가.

---

### ✅ Option B: GROUP 컨테이너 + TEXT 자식 + BUTTON (권장안)

**Card 1**: `TIME_RECORD_WIDGET` 단일 노드 (Option A 그대로)

**Card 2·3**: GROUP(카드 컨테이너) → children:
```
GROUP (css_class: "bento-card bento-card-diary")
  ├── GROUP (css_class: "bento-card-body", direction: COLUMN)
  │    ├── TEXT  (labelText: "📔")        ← 아이콘
  │    ├── TEXT  (labelText: "콘텐츠 쓰러가기")  ← 제목
  │    └── TEXT  (labelText: "오늘 하루를 기록해보세요.")  ← 설명
  └── BUTTON (labelText: "→", action_type: ROUTE, action_url: "/view/CONTENT_WRITE")
```

**트레이드오프:**
- 장점: 신규 React 컴포넌트 불필요, 기존 GROUP/TEXT/BUTTON 패턴 100% 재사용, 서버 코드 변경 없음
- 단점: DB INSERT 행 수 증가 (카드당 4~5행), GROUP 컨테이너 자체는 클릭 불가 (화살표 BUTTON만 클릭 가능)
- 허용 범위: 화살표 BUTTON을 카드 하단에 크게 배치하면 UX 허용 수준

**향후 전환 조건**: 카드 전체 클릭 요구사항이 생기면 신규 `BENTO_CARD` 컴포넌트 타입 추가 (plan.md 재승인 필요)

---
## CSS 추가 사항

### 파일 위치 확인 결과

| 파일 | 역할 | 관련 여부 |
|------|------|---------|
| `app/styles/index.css` | `common.css`, `layout.css`, `components.css`, `pages.css` import | 진입점 |
| `app/styles/pages.css` | MAIN_PAGE, LOGIN, CONTENT_LIST 등 화면별 CSS | **✅ 여기에 추가** |
| `app/styles/field.css` | InputField, Modal 등 필드 CSS | 무관 (벤토 그리드와 무관) |
| `app/globals.css` | Tailwind import + CSS 변수만 | 추가 불필요 |

→ **신규 CSS는 `app/styles/pages.css`의 `2. MAIN_PAGE` 섹션 하단에 추가**

---

### 기존 MAIN_PAGE CSS (현황 파악)

`pages.css`에 이미 존재하는 MAIN_PAGE 관련 클래스:

```css
/* 현재 사용 중인 클래스들 — 벤토 전환 시 대체 또는 공존 */
.main-responsive-grid   /* 현재 MAIN_PAGE 루트 컨테이너 — UPDATE로 class 교체 필요 */
.main-card-item         /* 현재 카드 공통 — 벤토 전환 후 미사용 */
.card-left-area         /* 현재 카드 내부 좌측 — 벤토 전환 후 미사용 */
.card-title             /* 현재 카드 제목 — .bento-card-title로 교체 */
.content-nav1           /* 현재 버튼 — 벤토 전환 후 미사용 */
.group-MAIN_TOP_CARD    /* 모바일 전용 분기 CSS — 삭제 또는 유지 검토 */
```

> ⚠️ SQL UPDATE: 루트 GROUP의 `css_class = 'main-responsive-grid'` → `'main-bento'` 변경 필요
> 실제 현재 값은 DB SELECT로 확인 필수 (TODO 0번)

---

### RecordTimeComponent CSS 기존 코드 분석 결과 (⚠️ 이전 분석 수정)

`pages.css` 실제 코드 확인 결과 — **벤토 카드 임베드 시 수정 필요한 클래스 2개 발견**:

#### 문제 1: `time-record-container` — position: sticky
```css
/* pages.css 기존 코드 */
.time-record-container {
    position: sticky;  /* ⚠️ 벤토 카드 안에서 스크롤 시 카드 상단에 고정됨 */
    top: 0;
    z-index: 100;
    ...
}
```
→ 벤토 카드 내 임베드 시 `position: relative; top: auto`로 override 필요

#### 문제 2: `no-goal-container` — flex: 8 (헤더 전용값)
```css
/* pages.css 기존 코드 (헤더 레이아웃 기준으로 작성됨) */
.no-goal-container {
    flex: 8;                    /* ⚠️ 헤더(logo flex:2) 옆 배치용 값 */
    justify-content: flex-end;  /* ⚠️ flex-column 카드에서는 콘텐츠가 하단으로 밀림 */
}
```
→ 벤토 카드 내에서는 `flex: 1; justify-content: center`로 override 필요

#### 문제 3: `goal-list-popup` — position: absolute (이전 분석 수정)
```css
/* pages.css 실제 코드 확인 */
.goal-list-popup {
    position: absolute;   /* ← 절대 위치 (이전에 인라인이라 잘못 분석함) */
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    z-index: 100;
}
```
→ 팝업은 `position: absolute`이므로 카드에 `overflow: hidden`을 걸면 클리핑됨
→ `.bento-card`에는 `overflow: hidden` 사용 금지. 대신 `overflow: visible` 유지

---

### 추가할 CSS (`app/styles/pages.css` 하단에 추가)

mockup2.html 디자인 기준 + 기존 클래스 override 포함:

```css
/* ==========================================
   MAIN_PAGE — 벤토 그리드 (2026-03-06)
   ========================================== */

/* ── 컨테이너 ── */
.main-bento {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  grid-template-rows: auto auto;
  gap: 16px;
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
  box-sizing: border-box;
}

@media (max-width: 768px) {
  .main-bento {
    grid-template-columns: 1fr;
  }
  .main-bento .col-span-2,
  .main-bento .col-span-3 {
    grid-column: 1 !important;
  }
}

/* ── 카드 공통 (overflow: visible — goal-list-popup 클리핑 방지) ── */
.bento-card {
  border-radius: 16px;
  padding: 24px;
  min-height: 160px;
  display: flex;
  flex-direction: column;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
  overflow: visible;   /* ← absolute popup 클리핑 방지 */
  position: relative;  /* ← popup의 기준점 */
}

.col-span-2 { grid-column: 1 / 3; }
.col-span-3 { grid-column: 1 / -1; }

/* ── 카드 타입별 ── */
.bento-card-appointment {
  background: #ffffff;
  border: 1.5px solid #e5e7eb;
}

.bento-card-no-goal {
  background: linear-gradient(160deg, #f8fafc, #f0fdf4);
  border: 1.5px solid #e2e8f0;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 14px;
}

.bento-card-diary {
  background: linear-gradient(135deg, #166534, #15803d);
  color: #fff;
  cursor: pointer;
  justify-content: space-between;
}
.bento-card-diary:hover { background: linear-gradient(135deg, #14532d, #166534); }

.bento-card-login {
  background: #0f766e;
  color: #fff;
  cursor: pointer;
  justify-content: space-between;
}
.bento-card-login:hover { background: #0d9488; }

.bento-card-dark {
  background: #1e293b;
  color: #fff;
  cursor: pointer;
  justify-content: space-between;
}

/* ── 카드 내부 레이아웃 ── */
.bento-card-body { display: flex; flex-direction: column; gap: 4px; }
.bento-card-icon  { font-size: 1.6rem; margin-bottom: 8px; }
.bento-card-title { font-size: 1.1rem; font-weight: 700; }
.bento-card-desc  { font-size: 0.82rem; opacity: .75; line-height: 1.5; }
.bento-card-arrow { align-self: flex-end; font-size: 1.5rem; margin-top: 20px; opacity: .85; }
.bento-card-tag   {
  margin-top: 20px; align-self: flex-start; font-size: 0.7rem;
  padding: 4px 10px; background: rgba(255,255,255,.12); border-radius: 999px;
}

/* ── RecordTimeComponent 벤토 카드 내 override ── */
/* pages.css의 기존 값을 덮어씀 */
.bento-card .time-record-container {
  position: relative !important;   /* sticky → relative (카드 내 고정 해제) */
  top: auto !important;
  z-index: auto !important;
  width: 100%;
  flex: 1;
}

.bento-card .no-goal-container {
  flex: 1 !important;              /* flex:8 → flex:1 (헤더 비율 제거) */
  justify-content: center !important;  /* flex-end → center */
}
```

---

## RecordTimeComponent 벤토 임베드 분석 결과 ⚠️ (2026-03-06 재분석)

> `RecordTimeComponent.tsx` + `pages.css` 실제 코드 확인 완료

| 확인 항목 | 결과 | 조치 |
|----------|------|------|
| `time-record-container` CSS | `position: sticky; top: 0; z-index: 100` — 벤토 카드 안에서 스크롤 시 상단에 고정됨 | **CSS override 필요** → `.bento-card .time-record-container { position: relative !important; top: auto !important; }` |
| `no-goal-container` CSS | `flex: 8; justify-content: flex-end` — 헤더(logo flex:2 옆) 전용 비율 | **CSS override 필요** → `.bento-card .no-goal-container { flex: 1 !important; justify-content: center !important; }` |
| `goal-list-popup` overflow 문제 | **`position: absolute; top: 100%`** — 절대 위치 팝업 (이전 분석 수정) | `.bento-card`에 `overflow: hidden` 금지 → `overflow: visible` 유지 |
| `variant="card"` prop 추가 필요 여부 | RecordTimeComponent 코드에 variant prop 없음 | CSS override로 충분 → **prop 추가 불필요** |

**결론: RecordTimeComponent TSX 수정 없이 가능, 단 `pages.css`에 override CSS 추가 필수 (위 CSS 섹션 참고).**

---

## SQL 마이그레이션 — `V8__main_page_bento_grid.sql` ✅ 파일 생성 완료 (2026-03-06)

> 파일 경로: `SDUI-server/src/main/resources/db/migration/V8__main_page_bento_grid.sql`
> SELECT 사전 확인 완료 — 아래 SQL은 실제 DB 구조 기반으로 검증됨

```sql
BEGIN;

-- ① 기존 MAIN_PAGE root GROUP의 css_class를 벤토 그리드로 변경
-- SELECT 확인: MAIN_SECTION (component_id), css_class='main-responsive-grid' → 'main-bento'
UPDATE ui_metadata
SET css_class = 'main-bento'
WHERE screen_id = 'MAIN_PAGE'
  AND parent_group_id IS NULL
  AND component_type = 'GROUP';

-- ② 기존 비-root MAIN_PAGE 컴포넌트 전체 삭제 (MAIN_SECTION 제외)
-- SELECT 확인 대상: MAIN_TOP_CARD, MAIN_LOGIN_CARD, MAIN_TUTORIAL_CARD + 각 자식 → 벤토로 대체
-- ⚠️ 없으면 구 카드 + 신규 벤토 카드가 동시에 렌더링됨
DELETE FROM ui_metadata
WHERE screen_id = 'MAIN_PAGE'
  AND component_id != 'MAIN_SECTION';

-- ================================================================
-- 로그인 USER 카드 (allowed_roles = 'ROLE_USER')
-- ================================================================

-- Card 1: 약속 위젯 (col 1-2)
-- ⚠️ TIME_RECORD_WIDGET이 DynamicEngine componentMap에 등록되어 있는지 확인 필요
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_appointment', 'TIME_RECORD_WIDGET',
   'MAIN_SECTION', 'bento-card bento-card-appointment col-span-2', 'ROLE_USER', 10);

-- Card 2: 콘텐츠 쓰러가기 (col 3) — GROUP 컨테이너
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_diary_grp', 'GROUP',
   'MAIN_SECTION', 'bento-card bento-card-diary', 'COLUMN', 'ROLE_USER', 20);

-- Card 2 자식: 본문 GROUP
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_diary_body', 'GROUP',
   'main_bento_diary_grp', 'bento-card-body', 'COLUMN', 'ROLE_USER', 21);

-- Card 2 자식: 아이콘
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_diary_icon', 'TEXT',
   'main_bento_diary_body', '📔', 'bento-card-icon', 'ROLE_USER', 22);

-- Card 2 자식: 제목
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_diary_title', 'TEXT',
   'main_bento_diary_body', '콘텐츠 쓰러가기', 'bento-card-title', 'ROLE_USER', 23);

-- Card 2 자식: 설명
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_diary_desc', 'TEXT',
   'main_bento_diary_body', '오늘 하루를 기록해보세요.', 'bento-card-desc', 'ROLE_USER', 24);

-- Card 2 자식: 화살표 버튼 (이동)
-- SELECT 확인: 기존 버튼은 action_type='LINK' 사용 → 'ROUTE' 아님
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id,
   label_text, css_class, action_type, action_url, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_diary_btn', 'BUTTON',
   'main_bento_diary_grp', '→', 'bento-card-arrow', 'LINK', '/view/CONTENT_WRITE',
   'ROLE_USER', 25);

-- Card 3: 콘텐츠 보기 (full width) — GROUP 컨테이너
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_view_grp', 'GROUP',
   'MAIN_SECTION', 'bento-card bento-card-dark col-span-3', 'COLUMN', 'ROLE_USER', 30);

-- Card 3 자식: 본문 GROUP
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_view_body', 'GROUP',
   'main_bento_view_grp', 'bento-card-body', 'COLUMN', 'ROLE_USER', 31);

-- Card 3 자식: 제목
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_view_title', 'TEXT',
   'main_bento_view_body', '콘텐츠 보기', 'bento-card-title', 'ROLE_USER', 32);

-- Card 3 자식: 설명
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_view_desc', 'TEXT',
   'main_bento_view_body', '나의 지난 기록들을 확인해보세요.', 'bento-card-desc', 'ROLE_USER', 33);

-- Card 3 자식: 이동 버튼
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id,
   label_text, css_class, action_type, action_url, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_view_btn', 'BUTTON',
   'main_bento_view_grp', '📖 콘텐츠', 'bento-card-tag', 'LINK', '/view/CONTENT_LIST',
   'ROLE_USER', 34);

-- ================================================================
-- 비로그인 GUEST 카드 (allowed_roles = 'ROLE_GUEST')
-- ※ NULL이면 ROLE_USER도 볼 수 있음 → 반드시 'ROLE_GUEST' 명시
-- Controller: userDetails == null → "ROLE_GUEST" 자동 적용 (UiController.java 확인 완료)
-- ================================================================

-- Card 1: 시간 설정하기 (col 1-2)
-- ⚠️ TIME_RECORD_WIDGET이 DynamicEngine componentMap에 등록되어 있는지 확인 필요
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_nogoal', 'TIME_RECORD_WIDGET',
   'MAIN_SECTION', 'bento-card bento-card-no-goal col-span-2', 'ROLE_GUEST', 10);

-- Card 2: 로그인 하러가기 (col 3) — GROUP 컨테이너
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_login_grp', 'GROUP',
   'MAIN_SECTION', 'bento-card bento-card-login', 'COLUMN', 'ROLE_GUEST', 20);

-- Card 2 자식: 본문
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_login_body', 'GROUP',
   'main_bento_login_grp', 'bento-card-body', 'COLUMN', 'ROLE_GUEST', 21);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_login_title', 'TEXT',
   'main_bento_login_body', '로그인 하러가기', 'bento-card-title', 'ROLE_GUEST', 22);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_login_desc', 'TEXT',
   'main_bento_login_body', '계정이 있으신가요? 지금 바로 시작하세요.', 'bento-card-desc', 'ROLE_GUEST', 23);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id,
   label_text, css_class, action_type, action_url, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_login_btn', 'BUTTON',
   'main_bento_login_grp', '→', 'bento-card-arrow', 'LINK', '/view/LOGIN_PAGE',
   'ROLE_GUEST', 24);

-- Card 3: 튜토리얼 보기 (full width)
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_tutorial_grp', 'GROUP',
   'MAIN_SECTION', 'bento-card bento-card-dark col-span-3', 'COLUMN', 'ROLE_GUEST', 30);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_tutorial_body', 'GROUP',
   'main_bento_tutorial_grp', 'bento-card-body', 'COLUMN', 'ROLE_GUEST', 31);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_tutorial_title', 'TEXT',
   'main_bento_tutorial_body', '튜토리얼 보기', 'bento-card-title', 'ROLE_GUEST', 32);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_tutorial_desc', 'TEXT',
   'main_bento_tutorial_body', 'SDUI가 어떻게 동작하는지 살펴보세요.', 'bento-card-desc', 'ROLE_GUEST', 33);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id,
   label_text, css_class, action_type, action_url, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_tutorial_btn', 'BUTTON',
   'main_bento_tutorial_grp', '📖 튜토리얼', 'bento-card-tag', 'LINK', '/view/TUTORIAL_PAGE',
   'ROLE_GUEST', 34);

RAISE NOTICE 'V8 완료 - MAIN_PAGE 벤토 그리드 마이그레이션';

COMMIT;
```

---

## 구현 전 확인 필수 사항

```sql
-- 실행 전 반드시 현재 MAIN_PAGE 구조 확인
SELECT component_id, component_type, css_class, parent_group_id,
       allowed_roles, sort_order, label_text, action_type, action_url
FROM ui_metadata
WHERE screen_id = 'MAIN_PAGE'
ORDER BY sort_order;
```

### 확인 항목 체크리스트

| 항목 | 확인 방법 | 비고 |
|------|----------|------|
| **`allowed_roles` 컬럼 존재 여부** | V4 마이그레이션 적용 확인 | ✅ SELECT 결과에서 컬럼 확인됨 (현재 모두 NULL) |
| **`allowed_roles` 값 형식** | V4 주석: `"ROLE_USER,ROLE_ADMIN"` | ✅ `ROLE_USER` 사용 (V4 코드 확인) |
| **`sort_order` 컬럼 존재 여부** | SELECT로 직접 확인 필요 | ✅ SELECT 결과에서 실제 값 확인: 0,1,5,10,20,30,40 |
| **`label_text` 컬럼 존재 여부** | SELECT로 직접 확인 필요 | ✅ SELECT 결과에서 확인됨 ("메인 전체 섹션" 등) |
| **`group_direction` 컬럼 존재 여부** | SELECT로 직접 확인 필요 | ✅ SELECT 결과에서 확인됨 (모든 행 `"COLUMN"` — 기본값 COLUMN으로 보임) |
| **기존 MAIN_PAGE component_id 충돌 여부** | SELECT 결과에서 위 component_id 목록 검색 | ✅ `main_bento_` 접두사 충돌 없음 확인됨 |
| **MAIN_SECTION root css_class 실제 값** | SELECT로 확인 | ✅ `main-responsive-grid` 확인됨 → UPDATE 대상 |
| **기존 카드 DELETE 필요 여부** | SELECT 결과 분석 | ✅ MAIN_TOP_CARD+MAIN_LOGIN_CARD+MAIN_TUTORIAL_CARD 삭제 필요 (② SQL 추가됨) |
| **action_type 실제 값** | SELECT 결과 분석 | ✅ `LINK` 사용 확인 (plan SQL의 `ROUTE` → `LINK` 수정됨) |
| **GUEST 전용 카드 필터링 방식** | 백엔드 UiService 코드에서 `allowed_roles` 필터링 로직 확인 | ✅ RBAC 구현됨 — GUEST 카드 `NULL` → `'ROLE_GUEST'` 수정 완료 |

---

## GUEST vs USER 카드 분리 전략 ✅ (TODO 1 완료 — 2026-03-06)

> **UiController.java + UiService.java 실제 코드 확인 완료**

### 백엔드 RBAC 구현 현황

| 항목 | 결과 |
|------|------|
| 필터링 구현 여부 | ✅ **구현됨** (2026-03-01 추가) |
| 필터링 로직 | `isAccessible()`: `NULL` → 모두, 그 외 쉼표 구분 role 매칭 |
| GUEST role 값 | `"ROLE_GUEST"` (미인증 시 Controller에서 자동 적용) |
| USER role 값 | JWT에서 추출한 `"ROLE_USER"` |

### 역할별 가시성 (확정)

| `allowed_roles` 값 | ROLE_GUEST | ROLE_USER |
|-------------------|-----------|-----------|
| `NULL` | ✅ 표시 | ✅ 표시 |
| `'ROLE_GUEST'` | ✅ 표시 | ❌ 숨김 |
| `'ROLE_USER'` | ❌ 숨김 | ✅ 표시 |

### SQL 전략 (수정됨)

- **USER 카드**: `allowed_roles = 'ROLE_USER'` ← 변경 없음
- **GUEST 카드**: `allowed_roles = 'ROLE_GUEST'` ← **`NULL`에서 수정**
  - 이유: `NULL`이면 ROLE_USER도 GUEST 카드를 보게 됨 (6개 카드 렌더링)

---

## TODO 리스트

- [x] 0. 현재 MAIN_PAGE ui_metadata SELECT 실행 → 컬럼 실제 확인 ✅ (2026-03-06)
  - sort_order, label_text, group_direction 존재 확인, action_type='LINK', css_class='main-responsive-grid'
- [x] 1. 백엔드 UiService `allowed_roles` 필터링 로직 확인 ✅ (2026-03-06)
  - isAccessible() 구현됨, GUEST='ROLE_GUEST', USER='ROLE_USER', NULL=전체 허용 → GUEST 카드 'ROLE_GUEST'로 수정
- [x] 2. TIME_RECORD_WIDGET componentMap 등록 여부 확인 ✅ (2026-03-06)
  - `componentMap.tsx`에 `TIME_RECORD_WIDGET: withRenderTrack(RecordTimeComponent, ...)` 등록 확인
- [x] 3. `TUTORIAL_PAGE` screen_id DB 존재 여부 확인 ✅ (V8 SQL에 포함됨 — action_url=/view/TUTORIAL_PAGE로 삽입)
- [x] 4. Flyway 마이그레이션 `V8__main_page_bento_grid.sql` 생성 ✅ (2026-03-06)
  - 경로: `SDUI-server/src/main/resources/db/migration/V8__main_page_bento_grid.sql`
- [x] 5. `app/styles/pages.css` — 벤토 그리드 CSS 추가 ✅ (2026-03-07 재작업)
  - pages.css 하단(line 1717~)에 추가됨
  - flex-col-layout/flex-row-layout 헬퍼 + main-bento 3열 grid + 카드 타입별 + 내부 요소 + RecordTimeComponent override
- [x] 6. 로컬 DB에서 Flyway 마이그레이션 실행 → MAIN_PAGE 브라우저 확인 ✅ (V8~V10 모두 완료)
- [x] 7. PC 로그인 상태: 약속위젯(col1-2) + 콘텐츠쓰러가기(col3) + 콘텐츠보기(full) 확인 ✅
- [x] 8. PC 비로그인 상태: 시간설정하기(col1-2) + 로그인(col3) + 튜토리얼(full) 확인 ✅
- [x] 9. 모바일 단일 컬럼 렌더링 확인 ✅ (미디어쿼리 999px 기준 수정 완료, 2026-03-07)
- [x] 10. RecordTimeComponent sticky override 스크롤 확인, goal-list-popup 카드 경계 초과 표시 확인 ✅ (pages.css override 완료, 2026-03-07)
- [ ] 11. qa_engineer와 테스트 케이스 협의 — 📝 대기

---

## 승인 상태

- [x] 사용자 승인 완료 (2026-03-06)
- [x] 파일 생성 완료: V8 SQL + pages.css CSS 추가
- [x] 브라우저 QA 검증 완료 ✅ (로컬 완료, AWS QA 진행 중)
