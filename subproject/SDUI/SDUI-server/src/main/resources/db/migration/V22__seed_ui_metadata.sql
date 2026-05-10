-- V22: ui_metadata 전체 시드 데이터 (프로덕션 Docker DB 기준)
-- 신규 로컬 DB에서 마이그레이션 실행 시 누락된 화면 데이터를 채웁니다.
-- WHERE NOT EXISTS로 중복 방지 (이미 존재하면 skip)
--
-- 포함 화면: LOGIN_PAGE, REGISTER_PAGE, VERIFY_CODE_PAGE,
--            GLOBAL_HEADER, SIDE_MENU, MAIN_PAGE,
--            CONTENT_LIST, CONTENT_DETAIL, CONTENT_WRITE, CONTENT_MODIFY,
--            SET_TIME_PAGE, RECORD_TIME_COMPONENT, ADDITIONAL_INFO_PAGE

-- 신규 컬럼 추가 (기존 DB에는 수동으로 추가됐음)
ALTER TABLE ui_metadata ADD COLUMN IF NOT EXISTS allowed_roles VARCHAR(255);
ALTER TABLE ui_metadata ADD COLUMN IF NOT EXISTS label_text_overrides JSONB;
ALTER TABLE ui_metadata ADD COLUMN IF NOT EXISTS css_class_overrides JSONB;

-- 시퀀스 동기화 (기존 DB에서 명시적 ui_id INSERT로 시퀀스가 뒤처진 경우 보정)
SELECT setval(
  pg_get_serial_sequence('ui_metadata', 'ui_id'),
  COALESCE((SELECT MAX(ui_id) FROM ui_metadata), 0)
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'ADDITIONAL_INFO_PAGE', 'HEADER_TEXT', '추가 정보를 입력해주세요', 'TEXT', 10,
  false, true, NULL, NULL,
  'text-xl font-bold mb-6 text-center', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'ADDITIONAL_INFO_PAGE' AND component_id = 'HEADER_TEXT'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'ADDITIONAL_INFO_PAGE', 'PHONE_INPUT', '전화번호', 'INPUT', 20,
  true, false, NULL, '010-1234-5678',
  'mb-4', NULL, NULL, NULL,
  NULL, NULL, NULL, 'phone',
  NULL, 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{"type": "tel"}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'ADDITIONAL_INFO_PAGE' AND component_id = 'PHONE_INPUT'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'ADDITIONAL_INFO_PAGE', 'ADDRESS_GROUP', '주소', 'ADDRESS_SEARCH_GROUP', 30,
  true, false, NULL, NULL,
  'mb-4', NULL, NULL, NULL,
  NULL, NULL, NULL, 'ADDRESS_GROUP',
  NULL, 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'ADDITIONAL_INFO_PAGE' AND component_id = 'ADDRESS_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'ADDITIONAL_INFO_PAGE', 'SUBMIT_BTN', '제출하기', 'BUTTON', 40,
  false, false, NULL, NULL,
  'btn-primary w-full mt-6', NULL, 'SUBMIT_ADDITIONAL_INFO', '/api/auth/update-profile',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{"variant": "primary"}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'ADDITIONAL_INFO_PAGE' AND component_id = 'SUBMIT_BTN'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'content_detail_source', '상세 데이터 소스', 'DATA_SOURCE', 0,
  false, true, NULL, NULL,
  'content_detail_source', NULL, 'AUTO_FETCH', NULL,
  'GET_CONTENT_DETAIL', '/api/execute/GET_DIARY_DETAIL', NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, NULL, 'false',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'content_detail_source'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'DETAIL_SECTION', '', 'GROUP', 1,
  false, false, NULL, NULL,
  'write_section1', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'DETAIL_SECTION', 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'DETAIL_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'sleep_time_select', '수면 시간', 'TIME_SELECT', 10,
  false, true, NULL, NULL,
  'time-select-wrapper', NULL, NULL, NULL,
  NULL, NULL, NULL, 'selected_times',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'sleep_time_select'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'label_contentTitle', '제목', 'TEXT', 20,
  false, false, NULL, NULL,
  'content-label', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'label_contentTitle'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'contentTitle', '제목', 'INPUT', 21,
  false, true, NULL, NULL,
  'contentInput', NULL, NULL, NULL,
  NULL, NULL, NULL, 'title',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'contentTitle'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'contentContent', '', 'TEXTAREA', 30,
  false, true, NULL, NULL,
  'contentTextarea', NULL, NULL, NULL,
  NULL, NULL, NULL, 'content',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'contentContent'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'DAYTAG_SUB_GROUP', '', 'GROUP', 40,
  false, false, NULL, NULL,
  'write_sub_group', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'DAYTAG_SUB_GROUP', 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'DAYTAG_SUB_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'daily_routine_section', '일과 기록', 'TIME_SLOT_RECORD', 50,
  false, true, NULL, NULL,
  'time-slot-container', NULL, NULL, NULL,
  NULL, NULL, NULL, 'daily_slots',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'daily_routine_section'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'contentEmotion', '오늘의 감정', 'EMOTION_SELECT', 52,
  false, true, NULL, NULL,
  'emotionSelect', NULL, NULL, NULL,
  NULL, NULL, NULL, 'emotion',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'EMOTION_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'contentEmotion'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'EMOTION_SUB_GROUP', '', 'GROUP', 60,
  false, false, NULL, NULL,
  'write_sub_group', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'EMOTION_SUB_GROUP', 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'EMOTION_SUB_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'CONTENTWRITE_BTN_GROUP', 'CONTENTWRITE_BTN_GROUP', 'GROUP', 70,
  false, false, NULL, NULL,
  'content_btn_group', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'DIARYWRITE_BTN_GROUP', 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'CONTENTWRITE_BTN_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'go_modify_btn', '수정하기', 'BUTTON', 71,
  false, false, NULL, NULL,
  'go_modify_btn', NULL, 'ROUTE_MODIFY', '/view/CONTENT_MODIFY',
  NULL, NULL, NULL, NULL,
  ' ', 'COLUMN', NULL, NULL,
  NULL, 'DIARYWRITE_BTN_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'go_modify_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'dayTag1', '태그 1', 'INPUT', 72,
  false, true, NULL, NULL,
  'contentInput', NULL, NULL, NULL,
  NULL, NULL, NULL, 'day_tag1',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DAYTAG_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'dayTag1'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'dayTag2', '태그 2', 'INPUT', 82,
  false, true, NULL, NULL,
  'contentInput', NULL, NULL, NULL,
  NULL, NULL, NULL, 'day_tag2',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DAYTAG_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'dayTag2'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'dayTag3', '태그 3', 'INPUT', 92,
  false, true, NULL, NULL,
  'contentInput', NULL, NULL, NULL,
  NULL, NULL, NULL, 'day_tag3',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DAYTAG_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'dayTag3'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_DETAIL', 'go_list_btn', '목록으로 돌아가기', 'BUTTON', 93,
  false, false, NULL, NULL,
  'go_list_btn', NULL, 'ROUTE', '/view/CONTENT_LIST',
  NULL, NULL, NULL, NULL,
  'DIARYWRITE_BTN_GROUP', 'COLUMN', NULL, NULL,
  NULL, 'DIARYWRITE_BTN_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_DETAIL' AND component_id = 'go_list_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'go_write_btn', '새 콘텐츠 쓰기', 'BUTTON', 0,
  false, false, NULL, NULL,
  'write-btn', NULL, 'LINK', '/view/CONTENT_WRITE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'LIST_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'go_write_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'content_list_source', '콘텐츠 목록 데이터', 'DATA_SOURCE', 0,
  false, true, NULL, NULL,
  'content_list_source', NULL, 'AUTO_FETCH', NULL,
  'GET_CONTENT_LIST_PAGE', '/api/execute/GET_CONTENT_LIST_PAGE', '{"pageSize": 5, "offset": 0, "filterId": ""}', NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'LIST_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'content_list_source'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'LIST_SECTION', '리스트 전체 섹션', 'GROUP', 1,
  false, true, NULL, NULL,
  'LIST_SECTION', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'LIST_SECTION', 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'LIST_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'CONTENT_CARD_HEADER', '카드 상단 영역', 'GROUP', 1,
  false, true, NULL, NULL,
  'content-card-header', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'CONTENT_CARD_HEADER', 'ROW', NULL, NULL,
  NULL, 'CONTENT_CARD', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'CONTENT_CARD_HEADER'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'TITLE_AUTHOR_GROUP', '제목작성자묶음', 'GROUP', 1,
  false, true, NULL, NULL,
  'title-author-wrapper', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'TITLE_AUTHOR_GROUP', 'ROW', NULL, NULL,
  NULL, 'CONTENT_CARD_HEADER', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'TITLE_AUTHOR_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'list_item_title', '제목', 'TEXT', 1,
  false, true, NULL, NULL,
  'contentTitle', NULL, NULL, NULL,
  NULL, NULL, NULL, 'title',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'TITLE_AUTHOR_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'list_item_title'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'list_item_date', '날짜', 'TEXT', 2,
  false, true, NULL, NULL,
  'contentDate', NULL, NULL, NULL,
  NULL, NULL, NULL, 'reg_dt',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'CONTENT_CARD_HEADER', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'list_item_date'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'CONTENT_CARD', '콘텐츠 카드', 'GROUP', 2,
  false, true, NULL, NULL,
  'content-post', NULL, 'ROUTE_DETAIL', '/view/CONTENT_DETAIL',
  NULL, NULL, NULL, 'content_list_source',
  'CONTENT_CARD', 'COLUMN', NULL, NULL,
  NULL, 'LIST_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'CONTENT_CARD'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'list_item_author', '작성자', 'TEXT', 2,
  false, true, NULL, NULL,
  'contentContent', NULL, NULL, NULL,
  NULL, NULL, NULL, 'user_id',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'TITLE_AUTHOR_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'list_item_author'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'content_list_container', '콘텐츠리스트', 'GROUP', 10,
  false, true, NULL, NULL,
  'content-list-container', NULL, NULL, NULL,
  NULL, NULL, NULL, 'content_list_source',
  'content_list_group', 'COLUMN', NULL, NULL,
  NULL, 'content_list_group', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'content_list_container'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'content_total_count', '전체 개수 조회', 'DATA_SOURCE', 99,
  false, true, NULL, NULL,
  'content_total_count', NULL, 'AUTO_FETCH', NULL,
  'COUNT_CONTENT_LIST', '/api/execute/COUNT_CONTENT_LIST', '{}', NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'LIST_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'content_total_count'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'content_pagination', '페이지네이션', 'PAGINATION', 100,
  false, true, NULL, NULL,
  'content-pagination', NULL, NULL, NULL,
  NULL, NULL, NULL, 'content_total_count',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'LIST_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'content_pagination'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_LIST', 'ADMIN_DELETE_ALL_BTN', '전체 삭제', 'BUTTON', 999,
  false, true, NULL, NULL,
  'btn-danger', NULL, 'DELETE_ALL_CONTENTS', '/api/content/delete-all',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  'ROLE_ADMIN', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_LIST' AND component_id = 'ADMIN_DELETE_ALL_BTN'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'content_detail_source', '상세 데이터 소스', 'DATA_SOURCE', 0,
  false, true, NULL, NULL,
  'content_detail_source', NULL, 'AUTO_FETCH', NULL,
  'GET_CONTENT_DETAIL', NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, NULL, 'false',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'content_detail_source'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'DETAIL_SECTION', '', 'GROUP', 1,
  false, false, NULL, NULL,
  'write_section1', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'DETAIL_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'selected_times', '수면 시간', 'TIME_SELECT', 10,
  false, false, NULL, NULL,
  'time-select-wrapper', NULL, NULL, NULL,
  NULL, NULL, NULL, 'selected_times',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'selected_times'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'label_contentTitle', '제목', 'TEXT', 20,
  false, false, NULL, NULL,
  'content-label', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'label_contentTitle'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'contentTitle', '제목', 'INPUT', 21,
  false, false, NULL, NULL,
  'contentInput', NULL, NULL, NULL,
  NULL, NULL, NULL, 'title',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'contentTitle'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'content', '', 'TEXTAREA', 30,
  false, false, NULL, NULL,
  'contentTextarea', NULL, NULL, NULL,
  NULL, NULL, NULL, 'content',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'content'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'DAYTAG_SUB_GROUP', '', 'GROUP', 40,
  false, false, NULL, NULL,
  'write_sub_group', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'DAYTAG_SUB_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'daily_slots', '일과 기록', 'TIME_SLOT_RECORD', 50,
  false, false, NULL, NULL,
  'time-slot-container', NULL, NULL, NULL,
  NULL, NULL, NULL, 'daily_slots',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'daily_slots'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'emotion', '오늘의 감정', 'EMOTION_SELECT', 52,
  false, false, NULL, NULL,
  'emotionSelect', NULL, NULL, NULL,
  NULL, NULL, NULL, 'emotion',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'EMOTION_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'emotion'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'EMOTION_SUB_GROUP', '', 'GROUP', 60,
  false, false, NULL, NULL,
  'write_sub_group', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'EMOTION_SUB_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'go_modify_btn', '수정하기', 'BUTTON', 70,
  false, false, NULL, NULL,
  'content-btn-primary', NULL, 'LINK', '/view/MAIN_PAGE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DIARYWRITE_BTN_GROUP', 'false',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'go_modify_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'go_list_btn', '수정 완료', 'BUTTON', 71,
  false, false, NULL, NULL,
  'save-button', NULL, 'SUBMIT', '/api/execute/UPDATE_CONTENT_DETAIL',
  'UPDATE_CONTENT_DETAIL', NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DETAIL_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'go_list_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'day_tag1', '태그 1', 'INPUT', 72,
  false, false, NULL, NULL,
  'contentInput', NULL, NULL, NULL,
  NULL, NULL, NULL, 'day_tag1',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DAYTAG_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'day_tag1'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'day_tag2', '태그 2', 'INPUT', 82,
  false, false, NULL, NULL,
  'contentInput', NULL, NULL, NULL,
  NULL, NULL, NULL, 'day_tag2',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DAYTAG_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'day_tag2'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_MODIFY', 'day_tag3', '태그 3', 'INPUT', 92,
  false, false, NULL, NULL,
  'contentInput', NULL, NULL, NULL,
  NULL, NULL, NULL, 'day_tag3',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DAYTAG_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_MODIFY' AND component_id = 'day_tag3'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'CONTENTWRITE_SECTION', '콘텐츠쓰기 섹션', 'GROUP', 1,
  false, false, NULL, NULL,
  'write_section1', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'DIARYWRITE_SECTION', 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'CONTENTWRITE_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'selected_times', '수면 시간 기록', 'TIME_SELECT', 10,
  false, false, NULL, NULL,
  'sleep_time_select', NULL, NULL, NULL,
  NULL, NULL, NULL, 'selected_times',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DIARYWRITE_SECTION', 'true',
  '{"endHour": 24, "startHour": 0, "slidesPerView": 6}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'selected_times'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'label_contentTitle', '제목', 'TEXT', 20,
  false, false, NULL, NULL,
  'content-label', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DIARYWRITE_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'label_contentTitle'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'contentTitle', '제목', 'INPUT', 21,
  false, false, NULL, '제목을 입력하세요',
  'contentInput', NULL, NULL, NULL,
  NULL, NULL, NULL, 'title',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DIARYWRITE_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'contentTitle'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'content', '내용', 'TEXTAREA', 30,
  false, false, NULL, '내용을 입력하세요',
  'contentTextarea', NULL, NULL, NULL,
  NULL, NULL, NULL, 'content',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DIARYWRITE_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'content'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'DAYTAG_SUB_GROUP', '하루해쉬태그그룹', 'GROUP', 40,
  false, false, NULL, NULL,
  'write_sub_group', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'DAYTAG_SUB_GROUP', 'ROW', NULL, NULL,
  NULL, 'DIARYWRITE_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'DAYTAG_SUB_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'daily_slots', '오늘의 흐름', 'TIME_SLOT_RECORD', 50,
  false, false, NULL, NULL,
  'daily_routine_section', NULL, NULL, NULL,
  NULL, NULL, NULL, 'daily_slots',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'DIARYWRITE_SECTION', 'true',
  '{"title": "나의 하루 기록", "description": "아침, 점심, 저녁 기록", "placeholders": {"lunch": "점심 식사 이후", "evening": "마무리", "morning": "기상 직후 "}}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'daily_slots'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'label_contentEmotion', '오늘의 감정은?', 'TEXT', 51,
  false, false, NULL, NULL,
  'content-label', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'EMOTION_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'label_contentEmotion'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'emotion', '감정', 'EMOTION_SELECT', 52,
  false, false, NULL, '감정값이 필요합니다.',
  'emotionSelect', NULL, NULL, NULL,
  NULL, NULL, NULL, 'emotion',
  NULL, 'ROW', NULL, NULL,
  NULL, 'EMOTION_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'emotion'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'EMOTION_SUB_GROUP', '감정 입력 그룹', 'GROUP', 60,
  false, false, NULL, NULL,
  'write_sub_group', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'EMOTION_SUB_GROUP', 'COLUMN', NULL, NULL,
  NULL, 'DIARYWRITE_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'EMOTION_SUB_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'GROUP_TAGS_ONELINE', '태그한줄그룹', 'GROUP', 61,
  false, false, NULL, NULL,
  'GROUP_TAGS_ONELINE', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'ROW', NULL, NULL,
  NULL, 'DAYTAG_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'GROUP_TAGS_ONELINE'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'GROUP_TAG_ROW1', '태그1행', 'GROUP', 70,
  false, false, NULL, NULL,
  'GROUP_TAG_ROW1', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'ROW', NULL, NULL,
  NULL, 'DAYTAG_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'GROUP_TAG_ROW1'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'save_btn', '저장하기', 'BUTTON', 70,
  false, false, NULL, NULL,
  'save-button', NULL, 'SUBMIT', '/api/execute/INSERT_CONTENT',
  'INSERT_CONTENT', NULL, NULL, NULL,
  'DIARYWRITE_BTN_GROUP', 'COLUMN', NULL, NULL,
  NULL, 'DIARYWRITE_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'save_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'title_dayTag1', '하루태그1', 'TEXT', 71,
  false, false, NULL, NULL,
  'content-label', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'GROUP_TAG_ROW1', 'COLUMN', NULL, NULL,
  NULL, 'GROUP_TAG_ROW1', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'title_dayTag1'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'day_tag1', '태그 1', 'INPUT', 72,
  false, false, NULL, '#tag1',
  'contentInput', NULL, NULL, NULL,
  NULL, NULL, NULL, 'day_tag1',
  'GROUP_TAGS_ONELINE', 'ROW', NULL, NULL,
  NULL, 'GROUP_TAGS_ONELINE', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'day_tag1'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'GROUP_TAG_ROW2', '태그2행', 'GROUP', 80,
  false, true, NULL, NULL,
  'GROUP_TAG_ROW2', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'ROW', NULL, NULL,
  NULL, 'DAYTAG_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'GROUP_TAG_ROW2'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'title_dayTag2', '하루태그2', 'TEXT', 81,
  false, false, NULL, NULL,
  'content-label', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'GROUP_TAG_ROW2', 'COLUMN', NULL, NULL,
  NULL, 'GROUP_TAG_ROW2', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'title_dayTag2'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'day_tag2', '태그 2', 'INPUT', 82,
  false, false, NULL, '#tag2',
  'contentInput', NULL, NULL, NULL,
  NULL, NULL, NULL, 'day_tag2',
  'GROUP_TAGS_ONELINE', 'ROW', NULL, NULL,
  NULL, 'GROUP_TAGS_ONELINE', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'day_tag2'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'GROUP_TAG_ROW3', '태그3행', 'GROUP', 90,
  false, true, NULL, NULL,
  'GROUP_TAG_ROW3', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'ROW', NULL, NULL,
  NULL, 'DAYTAG_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'GROUP_TAG_ROW3'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'title_dayTag3', '하루태그3', 'TEXT', 91,
  false, false, NULL, NULL,
  'content-label', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'GROUP_TAG_ROW3', 'COLUMN', NULL, NULL,
  NULL, 'GROUP_TAG_ROW3', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'title_dayTag3'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'CONTENT_WRITE', 'day_tag3', '태그 3', 'INPUT', 92,
  false, false, NULL, '#tag3',
  'contentInput', NULL, NULL, NULL,
  NULL, NULL, NULL, 'day_tag3',
  'GROUP_TAGS_ONELINE', 'ROW', NULL, NULL,
  NULL, 'GROUP_TAGS_ONELINE', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'CONTENT_WRITE' AND component_id = 'day_tag3'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'GLOBAL_HEADER', 'HEADER_SECTION', '메뉴 수정', 'GROUP', 0,
  false, true, NULL, NULL,
  'HEADER_SECTION', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'GLOBAL_HEADER' AND component_id = 'HEADER_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'GLOBAL_HEADER', 'header_logo', 'JustSaying', 'LINK', 1,
  false, true, NULL, NULL,
  'header_logo', NULL, 'ROUTE', '/view/MAIN_PAGE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'HEADER_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'GLOBAL_HEADER' AND component_id = 'header_logo'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'GLOBAL_HEADER', 'header_login_btn', '로그인', 'LINK', 12,
  false, true, NULL, NULL,
  'header_login_btn', NULL, 'ROUTE', '/view/LOGIN_PAGE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'HEADER_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'GLOBAL_HEADER' AND component_id = 'header_login_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'GLOBAL_HEADER', 'header_general_logout', '로그아웃', 'BUTTON', 13,
  false, true, NULL, NULL,
  'logout_button', NULL, 'LOGOUT', '/api/auth/logout',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'HEADER_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'GLOBAL_HEADER' AND component_id = 'header_general_logout'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'GLOBAL_HEADER', 'header_kakao_logout', '카카오 로그아웃', 'SNS_BUTTON', 14,
  false, false, NULL, NULL,
  'kakao_logout_button', NULL, 'KAKAO_LOGOUT', NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'HEADER_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'GLOBAL_HEADER' AND component_id = 'header_kakao_logout'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'LOGIN_SECTION', '로그인 섹션', 'GROUP', 0,
  false, true, NULL, NULL,
  'LOGIN_SECTION', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'LOGIN_SECTION', 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'LOGIN_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'EMAIL_SUB_GROUP', '이메일 입력 그룹', 'GROUP', 1,
  false, true, NULL, NULL,
  'EMAIL_SUB_GROUP', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'EMAIL_SUB_GROUP', 'ROW', NULL, NULL,
  NULL, 'LOGIN_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'EMAIL_SUB_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'label_email', 'Email', 'TEXT', 2,
  false, true, NULL, NULL,
  'label_email', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'EMAIL_SUB_GROUP', 'false',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'label_email'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'user_email', 'ID', 'INPUT', 3,
  true, false, NULL, '아이디 입력',
  'login_form-input', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'ROW', NULL, NULL,
  NULL, 'EMAIL_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'user_email'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'user_email_domain', '@', 'EMAIL_SELECT', 4,
  true, true, NULL, NULL,
  'user_email_domain', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'ROW', NULL, NULL,
  NULL, 'EMAIL_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'user_email_domain'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'PW_SUB_GROUP', '비밀번호 섹션', 'GROUP', 5,
  false, true, NULL, NULL,
  'PW_SUB_GROUP', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'PW_SUB_GROUP', 'COLUMN', NULL, NULL,
  NULL, 'LOGIN_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'PW_SUB_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'label_pw', 'Password', 'TEXT', 6,
  false, true, NULL, NULL,
  'label_pw', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'ROW', NULL, NULL,
  NULL, 'PW_SUB_GROUP', 'false',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'label_pw'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'user_pw', 'Password', 'PASSWORD', 7,
  true, false, NULL, '비밀번호를 입력하세요',
  'login_form-input', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'ROW', NULL, NULL,
  NULL, 'PW_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'user_pw'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'pw_toggle_btn', '보이기', 'BUTTON', 8,
  false, true, NULL, NULL,
  'pw_toggle_btn', NULL, 'TOGGLE_PW', NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'PW_SUB_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'pw_toggle_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'login_btn', '로그인', 'BUTTON', 9,
  false, false, NULL, NULL,
  'login_form_button', NULL, 'LOGIN_SUBMIT', '/api/auth/login',
  NULL, '/api/auth/login', NULL, NULL,
  'LOGIN_BTN_GROUP', 'COLUMN', NULL, NULL,
  NULL, 'LOGIN_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'login_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'SNS_SECTION', 'SNS 로그인', 'GROUP', 10,
  false, true, NULL, NULL,
  'SNS_SECTION', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'SNS_SECTION', 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'SNS_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'kakao_login_btn', 'Login with Kakao', 'SNS_BUTTON', 11,
  false, false, NULL, NULL,
  'kakao-button', NULL, 'LINK', 'https://kauth.kakao.com/oauth/authorize?client_id=2d22c7fa1d59eb77a5162a3948a0b6fe&redirect_uri=https://yerin.duckdns.org/api/kakao/callback&response_type=code',
  NULL, NULL, NULL, NULL,
  'SNS_GROUP', 'COLUMN', NULL, NULL,
  NULL, 'SNS_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'kakao_login_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'LOGIN_PAGE', 'join_btn', '회원가입', 'BUTTON', 12,
  false, false, NULL, NULL,
  'signup-nav', NULL, 'LINK', '/view/REGISTER_PAGE',
  NULL, NULL, NULL, NULL,
  'JOIN_GROUP', 'COLUMN', NULL, NULL,
  NULL, 'LOGIN_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'LOGIN_PAGE' AND component_id = 'join_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'MAIN_SECTION', '메인 전체 섹션', 'GROUP', 0,
  false, true, NULL, NULL,
  'main-bento', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'MAIN_SECTION', 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'MAIN_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_appointment', '', 'TIME_RECORD_WIDGET', 10,
  false, true, NULL, NULL,
  'bento-card bento-card-appointment col-span-2', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'MAIN_SECTION', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_appointment'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_nogoal', '', 'TIME_RECORD_WIDGET', 10,
  false, true, NULL, NULL,
  'bento-card bento-card-no-goal col-span-2', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'MAIN_SECTION', 'true',
  '{}'::jsonb,
  'ROLE_GUEST', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_nogoal'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_login_grp', '', 'GROUP', 20,
  false, true, NULL, NULL,
  'bento-card bento-card-login', NULL, 'LINK', '/view/LOGIN_PAGE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'MAIN_SECTION', 'true',
  '{}'::jsonb,
  'ROLE_GUEST', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_login_grp'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_diary_grp', '', 'GROUP', 20,
  false, true, NULL, NULL,
  'bento-card bento-card-diary', NULL, 'LINK', '/view/CONTENT_WRITE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'MAIN_SECTION', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_diary_grp'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_login_body', '', 'GROUP', 21,
  false, true, NULL, NULL,
  'bento-card-body', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_login_grp', 'true',
  '{}'::jsonb,
  'ROLE_GUEST', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_login_body'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_diary_body', '', 'GROUP', 21,
  false, true, NULL, NULL,
  'bento-card-body', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_diary_grp', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_diary_body'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_diary_icon', '📔', 'TEXT', 22,
  false, true, NULL, NULL,
  'bento-card-icon', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_diary_body', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_diary_icon'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_login_title', '로그인 하러가기', 'TEXT', 22,
  false, true, NULL, NULL,
  'bento-card-title', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_login_body', 'true',
  '{}'::jsonb,
  'ROLE_GUEST', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_login_title'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_diary_title', '콘텐츠 작성하기', 'TEXT', 23,
  false, true, NULL, NULL,
  'bento-card-title', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_diary_body', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_diary_title'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_login_desc', '계정이 있으신가요? 지금 바로 시작하세요.', 'TEXT', 23,
  false, true, NULL, NULL,
  'bento-card-desc', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_login_body', 'true',
  '{}'::jsonb,
  'ROLE_GUEST', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_login_desc'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_login_btn', '→', 'TEXT', 24,
  false, true, NULL, NULL,
  'bento-card-arrow', NULL, 'LINK', '/view/LOGIN_PAGE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_login_grp', 'true',
  '{}'::jsonb,
  'ROLE_GUEST', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_login_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_diary_desc', '오늘 하루를 기록해보세요.', 'TEXT', 24,
  false, true, NULL, NULL,
  'bento-card-desc', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_diary_body', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_diary_desc'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_diary_btn', '→', 'TEXT', 25,
  false, true, NULL, NULL,
  'bento-card-arrow', NULL, 'LINK', '/view/CONTENT_WRITE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_diary_grp', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_diary_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_tutorial_grp', '', 'GROUP', 30,
  false, true, NULL, NULL,
  'bento-card bento-card-dark col-span-3', NULL, 'LINK', '/view/TUTORIAL_PAGE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'MAIN_SECTION', 'true',
  '{}'::jsonb,
  'ROLE_GUEST', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_tutorial_grp'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_view_grp', '', 'GROUP', 30,
  false, true, NULL, NULL,
  'bento-card bento-card-dark col-span-3', NULL, 'LINK', '/view/CONTENT_LIST',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'MAIN_SECTION', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_view_grp'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_view_body', '', 'GROUP', 31,
  false, true, NULL, NULL,
  'bento-card-body', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_view_grp', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_view_body'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_tutorial_body', '', 'GROUP', 31,
  false, true, NULL, NULL,
  'bento-card-body', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_tutorial_grp', 'true',
  '{}'::jsonb,
  'ROLE_GUEST', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_tutorial_body'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_view_title', '콘텐츠 리스트 확인하기', 'TEXT', 32,
  false, true, NULL, NULL,
  'bento-card-title', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_view_body', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_view_title'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_tutorial_title', '튜토리얼 보기', 'TEXT', 32,
  false, true, NULL, NULL,
  'bento-card-title', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_tutorial_body', 'true',
  '{}'::jsonb,
  'ROLE_GUEST', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_tutorial_title'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_tutorial_desc', 'SDUI가 어떻게 동작하는지 살펴보세요.', 'TEXT', 33,
  false, true, NULL, NULL,
  'bento-card-desc', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_tutorial_body', 'true',
  '{}'::jsonb,
  'ROLE_GUEST', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_tutorial_desc'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_view_desc', '나의 지난 기록들을 확인해보세요.', 'TEXT', 33,
  false, true, NULL, NULL,
  'bento-card-desc', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_view_body', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_view_desc'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_view_btn', '📖 콘텐츠 목록', 'TEXT', 34,
  false, true, NULL, NULL,
  'bento-card-tag', NULL, 'LINK', '/view/CONTENT_LIST',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_view_grp', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_view_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'MAIN_PAGE', 'main_bento_tutorial_btn', '📖 튜토리얼', 'TEXT', 34,
  false, true, NULL, NULL,
  'bento-card-tag', NULL, 'LINK', '/view/TUTORIAL_PAGE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'main_bento_tutorial_grp', 'true',
  '{}'::jsonb,
  'ROLE_GUEST', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'MAIN_PAGE' AND component_id = 'main_bento_tutorial_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'RECORD_TIME_COMPONENT', 'arrival_btn', '도착 완료', 'BUTTON', 1,
  false, true, NULL, NULL,
  'arrival-button', NULL, 'SUBMIT', '/api/arrival/check',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'ACTION_BTN_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'RECORD_TIME_COMPONENT' AND component_id = 'arrival_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'RECORD_TIME_COMPONENT', 'no_goal_msg', '오늘의 약속 시간은 언제인가요?', 'TEXT', 1,
  false, true, NULL, NULL,
  'no-goal-container', NULL, 'NAVIGATE', '/view/SET_TIME_PAGE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'MAIN_CLOCK_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'RECORD_TIME_COMPONENT' AND component_id = 'no_goal_msg'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'RECORD_TIME_COMPONENT', 'remainTimeCountdown', '남은 시간', 'COUNTDOWN', 1,
  false, true, NULL, NULL,
  'countdown-timer', NULL, NULL, NULL,
  NULL, '/api/time/remain', NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'ACTIVE_INFO_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'RECORD_TIME_COMPONENT' AND component_id = 'remainTimeCountdown'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'RECORD_TIME_COMPONENT', 'MAIN_CLOCK_SECTION', '지각방지 섹션', 'GROUP', 1,
  false, true, NULL, NULL,
  'time-record-container', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'MAIN_CLOCK_SECTION', 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'RECORD_TIME_COMPONENT' AND component_id = 'MAIN_CLOCK_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'RECORD_TIME_COMPONENT', 'ACTION_BTN_GROUP', '버튼 그룹', 'GROUP', 2,
  false, true, NULL, NULL,
  'action-button-box', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'ACTION_BTN_GROUP', 'COLUMN', NULL, NULL,
  NULL, 'ACTIVE_INFO_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'RECORD_TIME_COMPONENT' AND component_id = 'ACTION_BTN_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'RECORD_TIME_COMPONENT', 'list_more_btn', '...', 'BUTTON', 2,
  false, true, NULL, NULL,
  'more-button', NULL, 'TOGGLE_LIST', NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'ACTION_BTN_GROUP', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'RECORD_TIME_COMPONENT' AND component_id = 'list_more_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'RECORD_TIME_COMPONENT', 'ACTIVE_INFO_GROUP', '실시간 정보 그룹', 'GROUP', 2,
  false, true, NULL, NULL,
  'active-display-row', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'ACTIVE_INFO_GROUP', 'ROW', NULL, NULL,
  NULL, 'MAIN_CLOCK_SECTION', 'false',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'RECORD_TIME_COMPONENT' AND component_id = 'ACTIVE_INFO_GROUP'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'reg_email', '이메일', 'INPUT', 1,
  true, false, NULL, 'example@email.com',
  'reg_email', NULL, NULL, NULL,
  NULL, NULL, NULL, 'email',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'INFO_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'reg_email'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'INFO_SECTION', '기본 정보 영역', 'SECTION', 1,
  false, true, NULL, NULL,
  'INFO_SECTION', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'INFO_SECTION', 'vertical', NULL, NULL,
  NULL, 'REG_CONTAINER', 'true',
  '{"gap": "15px"}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'INFO_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'reg_submit', '회원 가입 완료', 'BUTTON', 1,
  false, false, NULL, NULL,
  'reg_submit', NULL, 'REGISTER_SUBMIT', '/api/auth/register',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'ACTION_SECTION', 'true',
  '{"variant": "primary", "fullWidth": true}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'reg_submit'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'reg_zipcode', '우편번호', 'INPUT', 1,
  true, true, NULL, NULL,
  'reg_zipcode', NULL, NULL, NULL,
  NULL, NULL, NULL, 'zipCode',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'ADDR_SEARCH_ROW', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'reg_zipcode'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'ADDR_SEARCH_ROW', '우편번호 검색줄', 'ROW', 1,
  false, false, NULL, NULL,
  'ADDR_SEARCH_ROW', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'ADDR_SEARCH_ROW', 'horizontal', NULL, NULL,
  NULL, 'ADDR_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'ADDR_SEARCH_ROW'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'REG_CONTAINER', '회원가입 메인 컨테이너', 'CONTAINER', 1,
  false, true, NULL, NULL,
  'REG_CONTAINER', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'REG_CONTAINER', 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'REG_CONTAINER'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'reg_addr_btn', '주소 찾기', 'BUTTON', 2,
  false, false, NULL, NULL,
  'reg_addr_btn', NULL, 'OPEN_POSTCODE', NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'ADDR_SEARCH_ROW', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'reg_addr_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'reg_pw', '비밀번호', 'INPUT', 2,
  true, false, NULL, NULL,
  'reg_pw', NULL, NULL, NULL,
  NULL, NULL, NULL, 'password',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'INFO_SECTION', 'true',
  '{"type": "password"}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'reg_pw'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'reg_road_addr', '도로명 주소', 'INPUT', 2,
  false, true, NULL, NULL,
  'reg_road_addr', NULL, NULL, NULL,
  NULL, NULL, NULL, 'roadAddress',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'ADDR_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'reg_road_addr'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'ADDR_SECTION', '주소 정보 영역', 'SECTION', 2,
  false, false, NULL, NULL,
  'mt-30', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'ADDR_SECTION', 'COLUMN', NULL, NULL,
  NULL, 'REG_CONTAINER', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'ADDR_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'ACTION_SECTION', '버튼 영역', 'SECTION', 3,
  false, true, NULL, NULL,
  'mt-40', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'ACTION_SECTION', 'COLUMN', NULL, NULL,
  NULL, 'REG_CONTAINER', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'ACTION_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'reg_phone', '핸드폰 번호', 'INPUT', 3,
  false, false, NULL, '010-0000-0000',
  'reg_phone', NULL, NULL, NULL,
  NULL, NULL, NULL, 'phone',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'INFO_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'reg_phone'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'reg_detail_addr', '상세 주소', 'INPUT', 3,
  false, false, NULL, '나머지 주소를 입력하세요',
  'reg_detail_addr', NULL, NULL, NULL,
  NULL, NULL, NULL, 'detailAddress',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'ADDR_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'reg_detail_addr'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'REGISTER_PAGE', 'email_verify_modal', '회원 정보 입력', 'MODAL', 99,
  false, false, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'ROOT', 'true',
  '{"content": "인증 메일을 전송했습니다. 인증을 완료하신 후 이 페이지로 돌아와 확인 버튼을 눌러주세요.", "button_text": "확인"}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'REGISTER_PAGE' AND component_id = 'email_verify_modal'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'SET_TIME_PAGE', 'SET_TIME_SECTION', '시간 설정 전체 섹션', 'GROUP', 1,
  false, false, NULL, NULL,
  'SET_TIME_SECTION', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'SET_TIME_SECTION', 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'SET_TIME_PAGE' AND component_id = 'SET_TIME_SECTION'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'SET_TIME_PAGE', 'set_time_title', '퇴근/약속 목표 시간 설정', 'TEXT', 2,
  false, true, NULL, NULL,
  'page-title', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'SET_TIME_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'SET_TIME_PAGE' AND component_id = 'set_time_title'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'SET_TIME_PAGE', 'set_time_desc', '오늘의 목표 시간을 선택해 주세요.', 'TEXT', 3,
  false, true, NULL, NULL,
  'page-desc', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'SET_TIME_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'SET_TIME_PAGE' AND component_id = 'set_time_desc'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'SET_TIME_PAGE', 'targetTime', '목표 시간', 'DATETIME_PICKER', 4,
  true, false, NULL, '목표시간 입력 yyyy-MM-dd HH:mm:ss',
  'targetTime', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'SET_TIME_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'SET_TIME_PAGE' AND component_id = 'targetTime'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'SET_TIME_PAGE', 'messageInput', '오늘의 메모', 'INPUT', 5,
  false, false, NULL, '오늘의 각오 한마디',
  'time-input', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'SET_TIME_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'SET_TIME_PAGE' AND component_id = 'messageInput'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'SET_TIME_PAGE', 'save_time_btn', '시간정하기', 'BUTTON', 6,
  false, false, NULL, NULL,
  'save-button', NULL, 'SUBMIT', '/api/goalTime/save',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'SET_TIME_SECTION', 'true',
  '{"next_url": "/view/MAIN_PAGE"}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'SET_TIME_PAGE' AND component_id = 'save_time_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'SET_TIME_PAGE', 'back_btn', '취소', 'BUTTON', 7,
  false, false, NULL, NULL,
  'cancel-button', NULL, 'NAVIGATE', '/view/MAIN_PAGE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'SET_TIME_SECTION', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'SET_TIME_PAGE' AND component_id = 'back_btn'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'SIDE_MENU', 'menu_main', '메인페이지', 'MENU_ITEM', 1,
  false, false, NULL, NULL,
  'menu_main', NULL, 'LINK', '/view/MAIN_PAGE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'SIDE_MENU' AND component_id = 'menu_main'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'SIDE_MENU', 'menu_tutorial', '튜토리얼', 'MENU_ITEM', 2,
  false, false, NULL, NULL,
  'menu_tutorial', NULL, 'LINK', '/view/TUTORIAL_PAGE',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'SIDE_MENU' AND component_id = 'menu_tutorial'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'VERIFY_CODE_PAGE', 'VERIFY_CONTAINER', '이메일 인증', 'CONTAINER', 1,
  false, true, NULL, NULL,
  'verify-container', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'VERIFY_CONTAINER', 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'VERIFY_CODE_PAGE' AND component_id = 'VERIFY_CONTAINER'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'VERIFY_CODE_PAGE', 'reg_email', '가입 이메일', 'INPUT', 1,
  false, true, NULL, '가입하신 이메일입니다',
  'verify-input-readonly', NULL, NULL, NULL,
  NULL, NULL, NULL, 'email',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'VERIFY_CONTAINER', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'VERIFY_CODE_PAGE' AND component_id = 'reg_email'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'VERIFY_CODE_PAGE', 'reg_code', '인증 번호', 'INPUT', 2,
  false, false, NULL, '메일로 발송된 6자리 번호를 입력하세요',
  'verify-input-active', NULL, NULL, NULL,
  NULL, NULL, NULL, 'code',
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'VERIFY_CONTAINER', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'VERIFY_CODE_PAGE' AND component_id = 'reg_code'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'VERIFY_CODE_PAGE', 'verify_submit', '인증 완료', 'BUTTON', 3,
  false, false, NULL, NULL,
  'verify-submit-btn', NULL, 'VERIFY_CODE', '/api/auth/verify-code',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'VERIFY_CONTAINER', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'VERIFY_CODE_PAGE' AND component_id = 'verify_submit'
);

INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'VERIFY_CODE_PAGE', 'resend_btn', '인증 번호 재발송', 'LINK', 4,
  false, false, NULL, NULL,
  'resend-link', NULL, 'RESEND_CODE', '/api/auth/resend-code',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'VERIFY_CONTAINER', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'VERIFY_CODE_PAGE' AND component_id = 'resend_btn'
);


DO $$ BEGIN RAISE NOTICE 'V22: ui_metadata 시드 데이터 삽입 완료'; END $$;
