-- V8: MAIN_PAGE 벤토 그리드 전환 (2026-03-06)
-- 기존 responsive grid → 3열 bento grid
-- RBAC: USER 카드 'ROLE_USER', GUEST 카드 'ROLE_GUEST' (NULL 사용 금지)
-- 수정 이력: label_text NOT NULL 제약 대응 — GROUP/WIDGET 행에 '' 추가 (2026-03-06)

-- ① root GROUP css_class 변경
UPDATE ui_metadata
SET css_class = 'main-bento'
WHERE screen_id = 'MAIN_PAGE'
  AND parent_group_id IS NULL
  AND component_type = 'GROUP';

-- ② 기존 비-root MAIN_PAGE 컴포넌트 전체 삭제 (MAIN_SECTION 제외)
DELETE FROM ui_metadata
WHERE screen_id = 'MAIN_PAGE'
  AND component_id != 'MAIN_SECTION';

-- ================================================================
-- USER 카드 (allowed_roles = 'ROLE_USER')
-- ================================================================

-- Card 1: 약속 위젯 (col 1-2)
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_appointment', 'TIME_RECORD_WIDGET',
   'MAIN_SECTION', '', 'bento-card bento-card-appointment col-span-2', 'ROLE_USER', 10);

-- Card 2: 콘텐츠 쓰러가기 (col 3) — GROUP 컨테이너
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_diary_grp', 'GROUP',
   'MAIN_SECTION', '', 'bento-card bento-card-diary', 'COLUMN', 'ROLE_USER', 20);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_diary_body', 'GROUP',
   'main_bento_diary_grp', '', 'bento-card-body', 'COLUMN', 'ROLE_USER', 21);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_diary_icon', 'TEXT',
   'main_bento_diary_body', '📔', 'bento-card-icon', 'ROLE_USER', 22);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_diary_title', 'TEXT',
   'main_bento_diary_body', '콘텐츠 쓰러가기', 'bento-card-title', 'ROLE_USER', 23);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_diary_desc', 'TEXT',
   'main_bento_diary_body', '오늘 하루를 기록해보세요.', 'bento-card-desc', 'ROLE_USER', 24);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id,
   label_text, css_class, action_type, action_url, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_diary_btn', 'BUTTON',
   'main_bento_diary_grp', '→', 'bento-card-arrow', 'LINK', '/view/CONTENT_WRITE',
   'ROLE_USER', 25);

-- Card 3: 콘텐츠 보기 (full width)
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_view_grp', 'GROUP',
   'MAIN_SECTION', '', 'bento-card bento-card-dark col-span-3', 'COLUMN', 'ROLE_USER', 30);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_view_body', 'GROUP',
   'main_bento_view_grp', '', 'bento-card-body', 'COLUMN', 'ROLE_USER', 31);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_view_title', 'TEXT',
   'main_bento_view_body', '콘텐츠 보기', 'bento-card-title', 'ROLE_USER', 32);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_view_desc', 'TEXT',
   'main_bento_view_body', '나의 지난 기록들을 확인해보세요.', 'bento-card-desc', 'ROLE_USER', 33);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id,
   label_text, css_class, action_type, action_url, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_view_btn', 'BUTTON',
   'main_bento_view_grp', '📖 콘텐츠', 'bento-card-tag', 'LINK', '/view/CONTENT_LIST',
   'ROLE_USER', 34);

-- ================================================================
-- GUEST 카드 (allowed_roles = 'ROLE_GUEST')
-- ※ NULL 사용 금지 — NULL이면 ROLE_USER도 GUEST 카드를 봄
-- UiController: userDetails == null → "ROLE_GUEST" 자동 적용
-- ================================================================

-- Card 1: 시간 설정하기 (col 1-2)
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_nogoal', 'TIME_RECORD_WIDGET',
   'MAIN_SECTION', '', 'bento-card bento-card-no-goal col-span-2', 'ROLE_GUEST', 10);

-- Card 2: 로그인 하러가기 (col 3)
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_login_grp', 'GROUP',
   'MAIN_SECTION', '', 'bento-card bento-card-login', 'COLUMN', 'ROLE_GUEST', 20);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_login_body', 'GROUP',
   'main_bento_login_grp', '', 'bento-card-body', 'COLUMN', 'ROLE_GUEST', 21);

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
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_tutorial_grp', 'GROUP',
   'MAIN_SECTION', '', 'bento-card bento-card-dark col-span-3', 'COLUMN', 'ROLE_GUEST', 30);

INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, group_direction, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'main_bento_tutorial_body', 'GROUP',
   'main_bento_tutorial_grp', '', 'bento-card-body', 'COLUMN', 'ROLE_GUEST', 31);

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

DO $$ BEGIN RAISE NOTICE 'V8 완료 - MAIN_PAGE 벤토 그리드 마이그레이션'; END $$;
