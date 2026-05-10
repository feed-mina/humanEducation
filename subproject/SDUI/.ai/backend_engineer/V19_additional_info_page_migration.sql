-- V19: ADDITIONAL_INFO_PAGE 메타데이터 추가
-- 카카오 OAuth 후 추가 정보 입력 화면 (ROLE_GUEST 신규 사용자용)
-- backup_aws_data.sql 기반 (ui_id 1036-1039)
-- ON CONFLICT DO NOTHING: 이미 존재하는 경우 무시

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, placeholder, css_class,
   ref_data_id, group_direction, is_visible, component_props)
VALUES
  ('ADDITIONAL_INFO_PAGE', 'HEADER_TEXT', '추가 정보를 입력해주세요', 'TEXT', 10,
   false, true, NULL, 'text-xl font-bold mb-6 text-center',
   NULL, 'COLUMN', 'true', '{}')
ON CONFLICT (component_id, screen_id) DO NOTHING;

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, placeholder, css_class,
   ref_data_id, group_direction, is_visible, component_props)
VALUES
  ('ADDITIONAL_INFO_PAGE', 'PHONE_INPUT', '전화번호', 'INPUT', 20,
   true, false, '010-1234-5678', 'mb-4',
   'phone', 'COLUMN', 'true', '{"type": "tel"}')
ON CONFLICT (component_id, screen_id) DO NOTHING;

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, placeholder, css_class,
   ref_data_id, group_direction, is_visible, component_props)
VALUES
  ('ADDITIONAL_INFO_PAGE', 'ADDRESS_GROUP', '주소', 'ADDRESS_SEARCH_GROUP', 30,
   true, false, NULL, 'mb-4',
   'ADDRESS_GROUP', 'COLUMN', 'true', '{}')
ON CONFLICT (component_id, screen_id) DO NOTHING;

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, action_type, action_url, css_class,
   group_direction, is_visible, component_props)
VALUES
  ('ADDITIONAL_INFO_PAGE', 'SUBMIT_BTN', '제출하기', 'BUTTON', 40,
   false, false, 'SUBMIT_ADDITIONAL_INFO', '/api/auth/update-profile', 'btn-primary w-full mt-6',
   'COLUMN', 'true', '{"variant": "primary"}')
ON CONFLICT (component_id, screen_id) DO NOTHING;

DO $$ BEGIN RAISE NOTICE 'V19: ADDITIONAL_INFO_PAGE 메타데이터 추가 완료'; END $$;
