-- V16: Admin UI 수정
-- 1. admin_stats_row에 ref_data_id 추가 → Repeater 활성화 (stats 데이터 바인딩)
UPDATE ui_metadata
SET ref_data_id = 'admin_stats_source'
WHERE component_id = 'admin_stats_row'
  AND screen_id = 'MAIN_PAGE';

-- 2. admin_users_card에 LINK 액션 추가 → 카드 클릭 시 회원 관리 페이지 이동
UPDATE ui_metadata
SET action_type = 'LINK',
    action_url  = '/view/USER_LIST'
WHERE component_id = 'admin_users_card'
  AND screen_id = 'MAIN_PAGE';
