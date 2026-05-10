-- V12__add_admin_dashboard.sql
-- 관리자(ROLE_ADMIN) 전용 대시보드 컴포넌트 추가
-- ================================================================
-- 관리자 ADMIN 카드 (allowed_roles = 'ROLE_ADMIN')
-- ================================================================
-- Card 1: 시스템 현황 (Col 1-2)
-- 배경색: 어두운 네이비 톤 (.bento-card-admin-stats)
INSERT INTO ui_metadata (
    screen_id,
    component_id,
    component_type,
    parent_group_id,
    label_text,
    css_class,
    group_direction,
    allowed_roles,
    sort_order
  )
VALUES (
    'MAIN_PAGE',
    'admin_stats_card',
    'GROUP',
    'MAIN_SECTION',
    '시스템 현황',
    'bento-card bento-card-admin-stats col-span-2',
    'COLUMN',
    'ROLE_ADMIN',
    10
  );
-- Card 1 자식: 제목
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
    'admin_stats_title',
    'TEXT',
    'admin_stats_card',
    '📊 시스템 현황',
    'bento-card-title',
    'ROLE_ADMIN',
    1
  );
-- Card 1 자식: 통계 컨테이너 (ROW)
INSERT INTO ui_metadata (
    screen_id,
    component_id,
    component_type,
    parent_group_id,
    label_text,
    css_class,
    group_direction,
    allowed_roles,
    sort_order
  )
VALUES (
    'MAIN_PAGE',
    'admin_stats_row',
    'GROUP',
    'admin_stats_card',
    'Admin Stats Row',
    'stat-group',
    'ROW',
    'ROLE_ADMIN',
    2
  );
-- Card 1 자식: 통계 아이템 1 (총 사용자) - 데모용 정적 텍스트
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
    'admin_stat_users',
    'TEXT',
    'admin_stats_row',
    '1,240\n총 사용자',
    'stat-item stat-value-text',
    'ROLE_ADMIN',
    1
  );
-- Card 1 자식: 통계 아이템 2 (오늘의 콘텐츠)
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
    'admin_stat_diaries',
    'TEXT',
    'admin_stats_row',
    '58\n오늘의 콘텐츠',
    'stat-item stat-value-text',
    'ROLE_ADMIN',
    2
  );
-- Card 2: 회원 관리 (Col 3)
-- 배경색: 흰색 (.bento-card-admin-users)
INSERT INTO ui_metadata (
    screen_id,
    component_id,
    component_type,
    parent_group_id,
    label_text,
    css_class,
    group_direction,
    allowed_roles,
    sort_order
  )
VALUES (
    'MAIN_PAGE',
    'admin_users_card',
    'GROUP',
    'MAIN_SECTION',
    'Admin Users Card',
    'bento-card bento-card-admin-users',
    'COLUMN',
    'ROLE_ADMIN',
    20
  );
-- Card 2 자식: 헤더 그룹 (제목 + 화살표)
INSERT INTO ui_metadata (
    screen_id,
    component_id,
    component_type,
    parent_group_id,
    label_text,
    css_class,
    group_direction,
    allowed_roles,
    sort_order
  )
VALUES (
    'MAIN_PAGE',
    'admin_users_header',
    'GROUP',
    'admin_users_card',
    'Admin Users Header',
    'card-header',
    'ROW',
    'ROLE_ADMIN',
    1
  );
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
    'admin_users_title',
    'TEXT',
    'admin_users_header',
    '👥 회원 관리',
    'bento-card-title',
    'ROLE_ADMIN',
    1
  );
-- Card 2 자식: 이동 버튼 (화살표)
INSERT INTO ui_metadata (
    screen_id,
    component_id,
    component_type,
    parent_group_id,
    label_text,
    css_class,
    action_type,
    action_url,
    allowed_roles,
    sort_order
  )
VALUES (
    'MAIN_PAGE',
    'admin_users_btn',
    'BUTTON',
    'admin_users_header',
    '→',
    'arrow-btn',
    'LINK',
    '/view/USER_LIST',
    'ROLE_ADMIN',
    2
  );
-- Card 2 자식: 설명
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
    'admin_users_desc',
    'TEXT',
    'admin_users_card',
    '사용자 목록을 조회하고 권한을 관리합니다.',
    'card-desc',
    'ROLE_ADMIN',
    2
  );
-- Card 3: 시스템 로그 (Full Width)
-- 배경색: 흰색 (.bento-card-admin-logs)
INSERT INTO ui_metadata (
    screen_id,
    component_id,
    component_type,
    parent_group_id,
    label_text,
    css_class,
    group_direction,
    allowed_roles,
    sort_order
  )
VALUES (
    'MAIN_PAGE',
    'admin_logs_card',
    'GROUP',
    'MAIN_SECTION',
    'Admin Logs Card',
    'bento-card bento-card-admin-logs col-span-3',
    'COLUMN',
    'ROLE_ADMIN',
    30
  );
-- Card 3 자식: 제목
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
    'admin_logs_title',
    'TEXT',
    'admin_logs_card',
    '🚨 최근 시스템 로그',
    'bento-card-title',
    'ROLE_ADMIN',
    1
  );
-- Card 3 자식: 로그 리스트 (임시 정적 데이터 - 추후 DATA_SOURCE 연동)
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
    'admin_log_item_1',
    'TEXT',
    'admin_logs_card',
    '[ERROR] DB Connection Timeout (10:42 AM)',
    'log-item status-error',
    'ROLE_ADMIN',
    2
  );
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
    'admin_log_item_2',
    'TEXT',
    'admin_logs_card',
    '[INFO] New user registered (09:15 AM)',
    'log-item status-info',
    'ROLE_ADMIN',
    3
  );
