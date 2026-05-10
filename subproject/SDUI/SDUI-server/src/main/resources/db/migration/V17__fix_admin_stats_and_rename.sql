-- V17: 어드민 대시보드 수정
-- 배경: V16은 최초 실행 시 screen_id='MAIN_PAGE'(잘못됨)로 0행 업데이트 후 flyway_schema_history에 기록됨.
-- validate-on-migrate=false 설정으로 파일 수정 후에도 에러 없이 서버 기동되었으나 재실행되지 않음.
-- 이 마이그레이션이 V16의 미적용 변경사항을 처리함.
-- 1. admin_stats_row Repeater 활성화 (ref_data_id 설정)
--    → DynamicEngine이 admin_stats_source 데이터를 rowData로 전달, {key} 템플릿 치환 활성화
UPDATE ui_metadata
SET ref_data_id = 'admin_stats_source'
WHERE component_id = 'admin_stats_row'
  AND screen_id = 'MAIN_PAGE';
-- 2. 회원 관리 카드 클릭 이동 활성화
UPDATE ui_metadata
SET action_type = 'LINK',
  action_url = '/view/USER_LIST'
WHERE component_id = 'admin_users_card'
  AND screen_id = 'MAIN_PAGE';
-- 3. today_diaries → today_contents 컬럼 별칭 변경
--    (V3에서 diary→content 마이그레이션 완료, GET_ADMIN_STATS 쿼리만 미적용 상태)
UPDATE query_master
SET query_text = 'SELECT
       TO_CHAR((SELECT COUNT(*) FROM users), ''FM9,999'') as total_users,
       TO_CHAR((SELECT COUNT(*) FROM content WHERE reg_dt >= CURRENT_DATE), ''FM9,999'') as today_contents,
       TO_CHAR((SELECT COUNT(*) FROM users WHERE reg_dt >= CURRENT_DATE), ''FM9,999'') as new_users'
WHERE sql_key = 'GET_ADMIN_STATS';
-- 4. UI 템플릿 키도 일치시켜 변경
UPDATE ui_metadata
SET label_text = '{today_contents}\n오늘의 콘텐츠'
WHERE component_id = 'admin_stat_diaries'
  AND screen_id = 'MAIN_PAGE';