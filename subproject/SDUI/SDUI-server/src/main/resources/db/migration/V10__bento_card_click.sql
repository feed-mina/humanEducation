-- V10: MAIN_PAGE 벤토 카드 전체 클릭 가능하도록 전환 (2026-03-07)
-- 각 카드 GROUP에 action_type/action_url 추가 → 카드 전체가 클릭 영역이 됨
-- 카드 내부 BUTTON(→, 📖) → TEXT로 변경 (GROUP이 이동 담당하므로 중복 방지)

-- ================================================================
-- USER 카드 GROUP에 LINK 액션 추가
-- ================================================================
UPDATE ui_metadata
SET action_type = 'LINK',
    action_url  = '/view/CONTENT_WRITE'
WHERE component_id = 'main_bento_diary_grp';

UPDATE ui_metadata
SET action_type = 'LINK',
    action_url  = '/view/CONTENT_LIST'
WHERE component_id = 'main_bento_view_grp';

-- ================================================================
-- GUEST 카드 GROUP에 LINK 액션 추가
-- ================================================================
UPDATE ui_metadata
SET action_type = 'LINK',
    action_url  = '/view/LOGIN_PAGE'
WHERE component_id = 'main_bento_login_grp';

UPDATE ui_metadata
SET action_type = 'LINK',
    action_url  = '/view/TUTORIAL_PAGE'
WHERE component_id = 'main_bento_tutorial_grp';

-- ================================================================
-- 카드 내부 화살표/태그 BUTTON → TEXT 변경 (GROUP이 이동 담당)
-- ================================================================
UPDATE ui_metadata
SET component_type = 'TEXT'
WHERE component_id IN (
    'main_bento_diary_btn',
    'main_bento_view_btn',
    'main_bento_login_btn',
    'main_bento_tutorial_btn'
);

DO $$ BEGIN RAISE NOTICE 'V10 완료 - 벤토 카드 전체 클릭 가능 전환'; END $$;
