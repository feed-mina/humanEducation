-- V15__add_required_role_to_query_master.sql
-- [P0 Security Fix] query_master 테이블에 required_role 컬럼 추가
-- + USER_LIST 스크린 ui_metadata 등록
--
-- 목적: /api/execute/{sqlKey} 엔드포인트의 무인증 SQL 실행 취약점 해소
-- 방식: 각 쿼리에 실행 권한 요구사항을 명시 → CommonQueryController에서 검증
--
-- required_role 값 규칙:
--   NULL        → 누구나 실행 가능 (기존 동작 유지)
--   'ROLE_USER' → 로그인한 사용자만 실행 가능
--   'ROLE_ADMIN'→ ADMIN 권한 보유자만 실행 가능

ALTER TABLE query_master
    ADD COLUMN IF NOT EXISTS required_role VARCHAR(50) DEFAULT NULL;

COMMENT ON COLUMN query_master.required_role
    IS '실행 권한 요구사항: NULL=공개, ROLE_USER=로그인 필요, ROLE_ADMIN=관리자 전용';

-- 사용자별 데이터를 조회하는 쿼리는 ROLE_USER로 보호
UPDATE query_master SET required_role = 'ROLE_USER'
WHERE sql_key IN (
    'GET_MY_GOAL_TIME',
    'GET_GOAL_LIST',
    'GET_CONTENT_LIST',
    'GET_MY_CONTENT'
);

-- 관리자 전용 쿼리 보호 (V12~V14에서 등록된 쿼리들)
UPDATE query_master SET required_role = 'ROLE_ADMIN'
WHERE sql_key IN (
    'GET_SYSTEM_LOGS',
    'GET_ADMIN_STATS'
);

-- ===================================================
-- USER_LIST 스크린 ui_metadata 등록
-- 회원 권한 관리 페이지 (ROLE_ADMIN 전용)
-- ADMIN_USER_TABLE 컴포넌트: 검색/체크박스/권한변경/페이징을 내부에서 처리
-- API: GET /api/admin/users, PUT /api/admin/users/role
-- ===================================================

-- 1. 페이지 루트 컨테이너 (parent_group_id = NULL → 최상위 노드)
INSERT INTO ui_metadata (
    screen_id, component_id, component_type,
    parent_group_id, label_text, css_class,
    group_direction, allowed_roles, sort_order
)
VALUES (
    'USER_LIST', 'user_list_page', 'GROUP',
    NULL, 'User List Page', 'admin-page-container',
    'COLUMN', 'ROLE_ADMIN', 1
);

-- 2. 헤더 행 (제목 + 돌아가기 버튼)
INSERT INTO ui_metadata (
    screen_id, component_id, component_type,
    parent_group_id, label_text, css_class,
    group_direction, allowed_roles, sort_order
)
VALUES (
    'USER_LIST', 'user_list_header', 'GROUP',
    'user_list_page', 'User List Header', 'admin-page-header',
    'ROW', 'ROLE_ADMIN', 1
);

-- 2-1. 페이지 제목
INSERT INTO ui_metadata (
    screen_id, component_id, component_type,
    parent_group_id, label_text, css_class,
    allowed_roles, sort_order
)
VALUES (
    'USER_LIST', 'user_list_title', 'TEXT',
    'user_list_header', '회원 권한 관리', 'admin-page-title',
    'ROLE_ADMIN', 1
);

-- 2-2. 돌아가기 버튼 (MAIN_PAGE 이동)
INSERT INTO ui_metadata (
    screen_id, component_id, component_type,
    parent_group_id, label_text, css_class,
    action_type, action_url, allowed_roles, sort_order
)
VALUES (
    'USER_LIST', 'user_list_back_btn', 'BUTTON',
    'user_list_header', '← 돌아가기', 'admin-back-btn',
    'LINK', '/view/admin/MAIN_PAGE', 'ROLE_ADMIN', 2
);

-- 3. 회원 관리 테이블 (ADMIN_USER_TABLE 자체 완결형 컴포넌트)
INSERT INTO ui_metadata (
    screen_id, component_id, component_type,
    parent_group_id, label_text, css_class,
    allowed_roles, sort_order
)
VALUES (
    'USER_LIST', 'user_list_table', 'ADMIN_USER_TABLE',
    'user_list_page', 'Admin User Table', 'admin-user-table-wrapper',
    'ROLE_ADMIN', 2
);
