-- V18__user_list_screen.sql
-- 배경: V15 파일 하단에 USER_LIST INSERT가 추가되었으나,
--       Flyway가 V15를 이미 "완료"로 기록한 후 파일이 수정되어 재실행되지 않음.
--       V16/V17은 MAIN_PAGE 수정만 포함 → USER_LIST 메타데이터 미적용 상태.
-- 목적: USER_LIST 스크린 ui_metadata 신규 등록 (회원 권한 관리 페이지, ROLE_ADMIN 전용)

-- 1. 페이지 루트 컨테이너
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
