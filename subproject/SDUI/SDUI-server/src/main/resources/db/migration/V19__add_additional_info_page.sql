-- V19: ADDITIONAL_INFO_PAGE 메타데이터 추가
-- 카카오 OAuth 후 추가 정보 입력 화면 (ROLE_GUEST 신규 사용자용)
-- WHERE NOT EXISTS로 중복 방지 (이미 존재하면 skip)

-- component_props 컬럼이 없는 경우 추가 (기존 운영 DB에서는 수동 추가됐음)
ALTER TABLE ui_metadata ADD COLUMN IF NOT EXISTS component_props JSONB;

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, css_class,
   group_direction, is_visible, component_props)
SELECT
  'ADDITIONAL_INFO_PAGE', 'HEADER_TEXT', '추가 정보를 입력해주세요', 'TEXT', 10,
  false, true, 'text-xl font-bold mb-6 text-center',
  'COLUMN', 'true', '{}'
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'ADDITIONAL_INFO_PAGE' AND component_id = 'HEADER_TEXT'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, placeholder, css_class,
   ref_data_id, group_direction, is_visible, component_props)
SELECT
  'ADDITIONAL_INFO_PAGE', 'PHONE_INPUT', '전화번호', 'INPUT', 20,
  true, false, '010-1234-5678', 'mb-4',
  'phone', 'COLUMN', 'true', '{"type": "tel"}'
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'ADDITIONAL_INFO_PAGE' AND component_id = 'PHONE_INPUT'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, css_class,
   ref_data_id, group_direction, is_visible, component_props)
SELECT
  'ADDITIONAL_INFO_PAGE', 'ADDRESS_GROUP', '주소', 'ADDRESS_SEARCH_GROUP', 30,
  true, false, 'mb-4',
  'ADDRESS_GROUP', 'COLUMN', 'true', '{}'
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'ADDITIONAL_INFO_PAGE' AND component_id = 'ADDRESS_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, action_type, action_url, css_class,
   group_direction, is_visible, component_props)
SELECT
  'ADDITIONAL_INFO_PAGE', 'SUBMIT_BTN', '제출하기', 'BUTTON', 40,
  false, false, 'SUBMIT_ADDITIONAL_INFO', '/api/auth/update-profile',
  'btn-primary w-full mt-6',
  'COLUMN', 'true', '{"variant": "primary"}'
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'ADDITIONAL_INFO_PAGE' AND component_id = 'SUBMIT_BTN'
);

DO $$ BEGIN RAISE NOTICE 'V19: ADDITIONAL_INFO_PAGE 메타데이터 추가 완료'; END $$;
