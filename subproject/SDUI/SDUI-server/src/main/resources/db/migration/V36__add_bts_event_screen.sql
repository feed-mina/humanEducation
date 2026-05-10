-- V36: BTS_EVENT_MAIN 화면 메타데이터 등록
-- bts-event 프로젝트의 data/screens/BTS_EVENT_MAIN.json을 ui_metadata 테이블에 INSERT합니다.
-- component_props의 showWhen 필드로 조건부 렌더링을 지원합니다.

-- ==========================================
-- BTS_EVENT_MAIN 화면
-- ==========================================

-- 1. 치어 모드 오버레이
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'cheer-overlay', '치어 모드', 'CHEER_MODE', 10,
  false, false, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, 'true',
  '{"showWhen": {"field": "showCheer", "equals": true}}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'cheer-overlay'
);

-- 2. 헤더 그룹
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'header', '헤더', 'GROUP', 20,
  false, false, NULL, NULL,
  'app-header', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'header', 'ROW', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'header'
);

-- 2-1. 헤더 타이틀 영역
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'title', '타이틀', 'GROUP', 21,
  false, false, NULL, NULL,
  'flex items-center gap-2', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'title', 'ROW', NULL, NULL,
  NULL, 'header', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'title'
);

-- 2-2. 헤더 컨트롤 영역
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'header-controls', '헤더 컨트롤', 'GROUP', 22,
  false, false, NULL, NULL,
  'flex items-center gap-2', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'header-controls', 'ROW', NULL, NULL,
  NULL, 'header', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'header-controls'
);

-- 2-2-1. 언어 토글
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'lang-toggle', '언어 토글', 'LANG_TOGGLE', 23,
  false, false, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'header-controls', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'lang-toggle'
);

-- 3. 레이어 필터 (지도 탭에서만 표시)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'map-layer-filter', '레이어 필터', 'LAYER_FILTER', 30,
  false, false, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, 'true',
  '{"showWhen": {"field": "tab", "equals": "map"}}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'map-layer-filter'
);

-- 4. 메인 콘텐츠 그룹
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'main-content', '메인 콘텐츠', 'GROUP', 40,
  false, false, NULL, NULL,
  'flex-1 overflow-hidden relative', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'main-content', 'COLUMN', NULL, NULL,
  NULL, NULL, 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'main-content'
);

-- 4-1. 지도 섹션 (tab=map일 때만 표시)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'map-section', '지도 섹션', 'GROUP', 41,
  false, false, NULL, NULL,
  'flex flex-col h-full', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'map-section', 'COLUMN', NULL, NULL,
  NULL, 'main-content', 'true',
  '{"showWhen": {"field": "tab", "equals": "map"}}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'map-section'
);

-- 4-1-1. 지도 영역 그룹
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'map-area', '지도 영역', 'GROUP', 42,
  false, false, NULL, NULL,
  'flex-1 overflow-hidden relative', NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  'map-area', 'COLUMN', NULL, NULL,
  NULL, 'map-section', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'map-area'
);

-- 4-1-1-1. Leaflet 지도
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'leaflet-map', '지도', 'EVENT_MAP', 43,
  false, false, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'map-area', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'leaflet-map'
);

-- 4-1-1-2. 상태 카드 (날씨/교통)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'status-card', '상태 카드', 'STATUS_CARD', 44,
  false, false, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'map-area', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'status-card'
);

-- 4-1-1-3. 공지 버튼 (플로팅)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'notice-btn', '교통 공지 버튼', 'GROUP', 45,
  false, false, NULL, NULL,
  'floating-notice-btn', NULL, 'SHOW_NOTICE', NULL,
  NULL, NULL, NULL, NULL,
  'notice-btn', NULL, NULL, NULL,
  NULL, 'map-area', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'notice-btn'
);

-- 4-1-1-4. 라이브 PIP
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'live-pip', '라이브 PIP', 'LIVE_PIP', 46,
  false, false, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'map-area', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'live-pip'
);

-- 4-1-2. 정보 패널
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'info-panel', '정보 패널', 'INFO_PANEL', 47,
  false, false, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'map-section', 'true',
  '{}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'info-panel'
);

-- 4-2. 채팅 섹션 (tab=chat일 때만 표시)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'chat-section', '채팅', 'GUEST_CHAT', 50,
  false, false, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'main-content', 'true',
  '{"showWhen": {"field": "tab", "equals": "chat"}}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'chat-section'
);

-- 4-3. 게시판 섹션 (tab=board일 때만 표시)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'board-section', '게시판', 'FAN_BOARD', 60,
  false, false, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, 'main-content', 'true',
  '{"showWhen": {"field": "tab", "equals": "board"}}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'board-section'
);

-- 5. 공지 모달 (showNotice=true일 때만 표시)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'notice-modal', '교통 공지 모달', 'NOTICE_MODAL', 70,
  false, false, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, 'true',
  '{"showWhen": {"field": "showNotice", "equals": true}}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'notice-modal'
);

-- 6. 후원 모달 (showSupport=true일 때만 표시)
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'BTS_EVENT_MAIN', 'support-modal', '후원 모달', 'SUPPORT_MODAL', 80,
  false, false, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL,
  NULL, NULL, 'true',
  '{"showWhen": {"field": "showSupport", "equals": true}}'::jsonb,
  NULL, NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'BTS_EVENT_MAIN' AND component_id = 'support-modal'
);

DO $$ BEGIN RAISE NOTICE 'V36: BTS_EVENT_MAIN 화면 메타데이터 등록 완료 (17개 컴포넌트)'; END $$;
