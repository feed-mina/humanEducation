-- V13__admin_stats_query.sql
-- 관리자 대시보드 통계 데이터 연동
-- 1. users 테이블에 가입일 컬럼 추가 (통계용)
-- 기존 데이터는 현재 시간으로 설정됨
ALTER TABLE users
ADD COLUMN IF NOT EXISTS reg_dt TIMESTAMP DEFAULT NOW();
-- 2. 관리자 통계 쿼리 등록 (천단위 콤마 포맷팅 포함)
INSERT INTO query_master (sql_key, query_text)
VALUES (
        'GET_ADMIN_STATS',
        'SELECT
       TO_CHAR((SELECT COUNT(*) FROM users), ''FM9,999'') as total_users,
       TO_CHAR((SELECT COUNT(*) FROM content WHERE reg_dt >= CURRENT_DATE), ''FM9,999'') as today_diaries,
       TO_CHAR((SELECT COUNT(*) FROM users WHERE reg_dt >= CURRENT_DATE), ''FM9,999'') as new_users'
    );
-- 3. 데이터 소스 컴포넌트 추가 (AUTO_FETCH)
-- 페이지 로드 시 GET_ADMIN_STATS 쿼리를 실행하여 데이터를 가져옴
INSERT INTO ui_metadata (
        screen_id,
        component_id,
        component_type,
        action_type,
        data_sql_key,
        allowed_roles,
        sort_order,
        label_text
    )
VALUES (
        'MAIN_PAGE',
        'admin_stats_source',
        'DATA_SOURCE',
        'AUTO_FETCH',
        'GET_ADMIN_STATS',
        'ROLE_ADMIN',
        5,
        'Admin Stats Data Source'
    );
-- 4. 기존 텍스트 컴포넌트를 데이터 바인딩 형식({key})으로 수정
UPDATE ui_metadata
SET label_text = '{total_users}\n총 사용자'
WHERE component_id = 'admin_stat_users';
UPDATE ui_metadata
SET label_text = '{today_diaries}\n오늘의 콘텐츠'
WHERE component_id = 'admin_stat_diaries';
-- 5. 신규 가입 통계 아이템 추가 (V12에서 누락된 항목)
INSERT INTO ui_metadata (
        screen_id,
        component_id,
        component_type,
        parent_group_id,
        label_text,
        css_class,
        allowed_roles,
        sort_order
    )
VALUES (
        'MAIN_PAGE',
        'admin_stat_new_users',
        'TEXT',
        'admin_stats_row',
        '{new_users}\n신규 가입',
        'stat-item stat-value-text',
        'ROLE_ADMIN',
        3
    );