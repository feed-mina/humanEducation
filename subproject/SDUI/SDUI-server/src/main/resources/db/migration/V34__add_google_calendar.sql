-- V34: Google Calendar 연동
-- google_oauth_tokens 테이블 및 goal_settings 이벤트 ID 컬럼 추가

CREATE TABLE google_oauth_tokens (
    id            BIGSERIAL    PRIMARY KEY,
    user_sqno     BIGINT       NOT NULL UNIQUE REFERENCES users(user_sqno),
    access_token  TEXT         NOT NULL,
    refresh_token TEXT         NOT NULL,
    token_expiry  TIMESTAMPTZ  NOT NULL,
    created_at    TIMESTAMPTZ  DEFAULT NOW(),
    updated_at    TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX idx_google_tokens_user ON google_oauth_tokens (user_sqno);

ALTER TABLE goal_settings
    ADD COLUMN google_calendar_event_id VARCHAR(200);

-- SET_TIME_PAGE에 구글 캘린더 연결 버튼 추가
INSERT INTO ui_metadata
  (screen_id, component_id, label_text, component_type, sort_order,
   is_required, is_readonly, default_value, placeholder,
   css_class, inline_style, action_type, action_url,
   data_sql_key, data_api_url, data_params, ref_data_id,
   group_id, group_direction, submit_group_id, submit_group_order,
   submit_group_separator, parent_group_id, is_visible, component_props,
   allowed_roles, label_text_overrides, css_class_overrides)
SELECT
  'SET_TIME_PAGE', 'google_calendar_btn', '구글 캘린더 연결', 'BUTTON', 8,
  false, false, NULL, NULL,
  'google-calendar-btn', NULL, 'GOOGLE_CALENDAR_CONNECT', '/api/google/auth-url',
  NULL, NULL, NULL, NULL,
  NULL, 'COLUMN', NULL, NULL,
  NULL, 'SET_TIME_SECTION', 'true',
  '{}'::jsonb,
  'ROLE_USER', NULL::jsonb, NULL::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM ui_metadata
  WHERE screen_id = 'SET_TIME_PAGE' AND component_id = 'google_calendar_btn'
);
