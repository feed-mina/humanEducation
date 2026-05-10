-- V14__admin_logs_query.sql
-- 시스템 로그 테이블 생성 및 관리자 대시보드 연동

-- 1. 시스템 로그 테이블 생성
CREATE TABLE IF NOT EXISTS system_logs (
    log_id BIGSERIAL PRIMARY KEY,
    log_level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 2. 샘플 데이터 삽입
INSERT INTO system_logs (log_level, message, created_at) VALUES
('ERROR', 'DB Connection Timeout (Retry 2)', NOW() - INTERVAL '10 minutes'),
('INFO', 'New user registered (user_id: test99)', NOW() - INTERVAL '1 hour'),
('INFO', 'Daily backup completed successfully', NOW() - INTERVAL '6 hours'),
('WARN', 'High memory usage detected (85%)', NOW() - INTERVAL '1 day'),
('INFO', 'System update completed', NOW() - INTERVAL '2 days');

-- 3. 로그 조회 쿼리 등록
INSERT INTO query_master (sql_key, query_text)
VALUES (
    'GET_SYSTEM_LOGS',
    'SELECT log_level, message, TO_CHAR(created_at, ''HH:MI AM'') as log_time FROM system_logs ORDER BY created_at DESC LIMIT 5'
);

-- 4. 데이터 소스 컴포넌트 추가 (AUTO_FETCH)
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, action_type, data_sql_key, allowed_roles, sort_order, label_text)
VALUES
  ('MAIN_PAGE', 'admin_logs_source', 'DATA_SOURCE', 'AUTO_FETCH', 'GET_SYSTEM_LOGS', 'ROLE_ADMIN', 6, 'Admin Logs Data Source');

-- 5. 기존 정적 로그 아이템 삭제 (V12에서 생성된 더미 데이터)
DELETE FROM ui_metadata
WHERE component_id IN ('admin_log_item_1', 'admin_log_item_2');

-- 6. 로그 리스트 리피터 컨테이너 추가
-- parent_group_id: admin_logs_card (V12에서 생성됨)
-- ref_data_id: admin_logs_source (위에서 생성한 데이터 소스 ID)
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, css_class, group_direction, ref_data_id, allowed_roles, sort_order, label_text)
VALUES
  ('MAIN_PAGE', 'admin_logs_list', 'GROUP',
   'admin_logs_card', 'log-list', 'COLUMN', 'admin_logs_source', 'ROLE_ADMIN', 10, 'Admin Logs List');

-- 7. 리피터 아이템 템플릿 (Row)
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, css_class, group_direction, allowed_roles, sort_order, label_text)
VALUES
  ('MAIN_PAGE', 'admin_log_item_template', 'GROUP',
   'admin_logs_list', 'log-item', 'ROW', 'ROLE_ADMIN', 1, 'Admin Log Item Template');

-- 8. 리피터 아이템 내용 (메시지)
-- [{log_level}] {message} 형식으로 표시
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'admin_log_msg', 'TEXT',
   'admin_log_item_template', '[{log_level}] {message}', 'log-message', 'ROLE_ADMIN', 1);

-- 9. 리피터 아이템 시간
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, allowed_roles, sort_order)
VALUES
  ('MAIN_PAGE', 'admin_log_time', 'TEXT',
   'admin_log_item_template', '{log_time}', 'log-time', 'ROLE_ADMIN', 2);