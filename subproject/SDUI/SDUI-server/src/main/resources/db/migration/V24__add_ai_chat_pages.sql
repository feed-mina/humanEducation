-- V24: AI 채팅 화면 등록 (최종 상태)
-- 통합 범위: 구 V26(AI_ENGLISH/KOREAN_CHAT_PAGE) + V27(is_readonly) + V30(V2 테스트) + V31 + V32(V2 정식 승격)
-- AI_ENGLISH_CHAT_PAGE는 직접 V2 최종 상태로 등록 (중간 V1→V2 전환 과정 생략)
-- 2026-03-13 ~ 2026-03-15

-- ───────────────────────────────────────────────
-- query_master: AI 채팅 설정값
-- ───────────────────────────────────────────────

-- AI 영어 대화 V2 설정값 (AI_ENGLISH_CHAT_PAGE 최종 버전에서 사용)
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'AI_ENGLISH_CHAT_CONFIG_V2',
  'SELECT
          ''🎤 Start Recording''                                                       AS mic_btn_label,
          ''Submit''                                                                   AS submit_btn_label,
          ''End Chat''                                                                 AS end_btn_label,
          ''Hello! I''''m your English conversation partner. What would you like to practice today?'' AS welcome_message,
          ''en''                                                                       AS language,
          ''PREMIUM''                                                                  AS required_tier,
          ''Voice conversation requires a PREMIUM membership.''                        AS upgrade_message',
  'SINGLE', 'AI 영어 대화 화면 설정값 (V2)', 'Y', 3600, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'AI_ENGLISH_CHAT_CONFIG_V2');

-- AI 한국어 대화 설정값
INSERT INTO query_master (sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role)
SELECT 'AI_KOREAN_CHAT_CONFIG',
  'SELECT
          ''🎤 녹음 시작''                                                            AS mic_btn_label,
          ''답변완료''                                                                 AS submit_btn_label,
          ''대화 종료''                                                                AS end_btn_label,
          ''안녕하세요! 한국어 대화 연습을 도와드리겠습니다. 무엇을 연습하고 싶으신가요?'' AS welcome_message,
          ''ko''                                                                       AS language,
          ''PREMIUM''                                                                  AS required_tier,
          ''음성 대화 기능은 프리미엄 멤버십이 필요합니다.''                            AS upgrade_message',
  'SINGLE', 'AI 한국어 대화 화면 설정값', 'Y', 3600, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'AI_KOREAN_CHAT_CONFIG');

-- ───────────────────────────────────────────────
-- AI_ENGLISH_CHAT_PAGE (V2 최종 상태로 직접 등록)
-- component_id: ai_en2_* (V32에서 AI_ENGLISH_CHAT_PAGE2 → AI_ENGLISH_CHAT_PAGE로 승격된 구조)
-- ───────────────────────────────────────────────
INSERT INTO ui_metadata (
    screen_id, component_id, component_type, parent_group_id,
    label_text, css_class, group_direction, allowed_roles, sort_order
  )
VALUES (
    'AI_ENGLISH_CHAT_PAGE', 'ai_en2_root', 'GROUP', NULL,
    '', 'ai-page-root', 'COLUMN', 'ROLE_USER', 1
  );

INSERT INTO ui_metadata (
    screen_id, component_id, component_type, parent_group_id,
    label_text, action_type, data_sql_key, allowed_roles, sort_order
  )
VALUES (
    'AI_ENGLISH_CHAT_PAGE', 'ai_en2_config', 'DATA_SOURCE', 'ai_en2_root',
    '', 'AUTO_FETCH', 'AI_ENGLISH_CHAT_CONFIG_V2', 'ROLE_USER', 2
  );

INSERT INTO ui_metadata (
    screen_id, component_id, component_type, parent_group_id,
    label_text, action_type, ref_data_id, css_class, is_readonly, allowed_roles, sort_order
  )
VALUES (
    'AI_ENGLISH_CHAT_PAGE', 'ai_en2_chat', 'AI_CHAT_V2', 'ai_en2_root',
    'AI 영어 대화 V2', 'AI_CHAT_EN', 'ai_en2_config', 'ai-chat-en',
    false, 'ROLE_USER', 3
  );

-- ───────────────────────────────────────────────
-- AI_KOREAN_CHAT_PAGE (is_readonly=false 포함)
-- ───────────────────────────────────────────────
INSERT INTO ui_metadata (
    screen_id, component_id, component_type, parent_group_id,
    label_text, css_class, group_direction, allowed_roles, sort_order
  )
VALUES (
    'AI_KOREAN_CHAT_PAGE', 'ai_ko_root', 'GROUP', NULL,
    '', 'ai-page-root', 'COLUMN', 'ROLE_USER', 1
  );

INSERT INTO ui_metadata (
    screen_id, component_id, component_type, parent_group_id,
    label_text, action_type, data_sql_key, allowed_roles, sort_order
  )
VALUES (
    'AI_KOREAN_CHAT_PAGE', 'ai_ko_config', 'DATA_SOURCE', 'ai_ko_root',
    '', 'AUTO_FETCH', 'AI_KOREAN_CHAT_CONFIG', 'ROLE_USER', 2
  );

INSERT INTO ui_metadata (
    screen_id, component_id, component_type, parent_group_id,
    label_text, action_type, ref_data_id, css_class, is_readonly, allowed_roles, sort_order
  )
VALUES (
    'AI_KOREAN_CHAT_PAGE', 'ai_ko_chat', 'AI_CHAT', 'ai_ko_root',
    'AI 한국어 대화', 'AI_CHAT_KO', 'ai_ko_config', 'ai-chat-ko',
    false, 'ROLE_USER', 3
  );

DO $$ BEGIN RAISE NOTICE 'V24 완료 - AI_ENGLISH_CHAT_PAGE(V2), AI_KOREAN_CHAT_PAGE 등록'; END $$;
