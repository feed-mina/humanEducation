-- V25: AI 면접 + AI 일본어 채팅 기능 추가 (최종 상태)
-- 통합 범위: 구 V33(면접 페이지) + V34(system_prompt_template 컬럼) + V35(일본어 페이지)
--            + V36(is_readonly 활성화) + V37(면접 한국어 전환) + V38(label_text 한국어) + V39(프롬프트 강화)
-- 2026-03-15 ~ 2026-03-17

-- ───────────────────────────────────────────────
-- 1. system_prompt_template 컬럼 추가 (AI 페르소나 프롬프트용)
-- ───────────────────────────────────────────────
ALTER TABLE ui_metadata ADD COLUMN IF NOT EXISTS system_prompt_template TEXT;

COMMENT ON COLUMN ui_metadata.system_prompt_template IS 'AI 채팅/면접 컴포넌트용 시스템 프롬프트 템플릿';

-- ───────────────────────────────────────────────
-- 2. AI_INTERVIEW_PAGE (한국어 최종 설정)
-- ───────────────────────────────────────────────

-- query_master: 한국어 면접 설정값
INSERT INTO query_master (
    sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role
  )
SELECT
  'AI_INTERVIEW_CONFIG',
  'SELECT
  ''🎤 답변 녹음''                                                              AS mic_btn_label,
  ''답변 제출''                                                                  AS submit_btn_label,
  ''면접 종료''                                                                  AS end_btn_label,
  ''안녕하세요! AI 면접관입니다. 이력서를 입력하고 면접을 시작해주세요.''            AS welcome_message,
  ''이력서 내용을 여기에 붙여넣으세요...'' || chr(10) || chr(10) || ''예) 이름, 경력, 프로젝트, 기술 스택 등'' AS resume_placeholder,
  ''면접 시작하기''                                                               AS start_btn_label,
  ''ko''                                                                        AS language,
  ''PREMIUM''                                                                   AS required_tier,
  ''면접 기능은 프리미엄 멤버십이 필요합니다.''                                    AS upgrade_message',
  'SINGLE', 'AI 면접 인터뷰 설정값', 'Y', 3600, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'AI_INTERVIEW_CONFIG');

-- ui_metadata: AI_INTERVIEW_PAGE 루트 그룹
INSERT INTO ui_metadata (
    screen_id, component_id, component_type, parent_group_id,
    label_text, css_class, group_direction, allowed_roles, sort_order
  )
VALUES (
    'AI_INTERVIEW_PAGE', 'ai_interview_root', 'GROUP', NULL,
    '', 'ai-page-root', 'COLUMN', 'ROLE_USER', 1
  );

-- ui_metadata: 데이터 소스
INSERT INTO ui_metadata (
    screen_id, component_id, component_type, parent_group_id,
    label_text, action_type, data_sql_key, allowed_roles, sort_order
  )
VALUES (
    'AI_INTERVIEW_PAGE', 'ai_interview_config', 'DATA_SOURCE', 'ai_interview_root',
    '', 'AUTO_FETCH', 'AI_INTERVIEW_CONFIG', 'ROLE_USER', 2
  );

-- ui_metadata: AI 면접 컴포넌트 (label_text 한국어, is_readonly=false)
INSERT INTO ui_metadata (
    screen_id, component_id, component_type, parent_group_id,
    label_text, action_type, ref_data_id, css_class, is_readonly, allowed_roles, sort_order
  )
VALUES (
    'AI_INTERVIEW_PAGE', 'ai_interview_field', 'AI_INTERVIEW', 'ai_interview_root',
    'AI 면접관', 'AI_INTERVIEW_EN', 'ai_interview_config', 'ai-interview-en',
    false, 'ROLE_USER', 3
  );

-- ───────────────────────────────────────────────
-- 3. AI_JAPANESE_CHAT_PAGE (강화 프롬프트 + 활성화 최종 설정)
-- ───────────────────────────────────────────────

-- query_master: 일본어 채팅 설정값
INSERT INTO query_master (
    sql_key, query_text, return_type, description, use_redis_yn, redis_ttl_sec, required_role
  )
SELECT
  'AI_JAPANESE_CHAT_CONFIG',
  'SELECT
  ''🎤 録음 시작''                                                              AS mic_btn_label,
  ''Submit''                                                                    AS submit_btn_label,
  ''대화 종료''                                                                 AS end_btn_label,
  ''こんにちは！日本語の練習をお手伝いします。今日はどんなことを話したいですか？'' AS welcome_message,
  ''ja''                                                                        AS language,
  ''PREMIUM''                                                                   AS required_tier,
  ''음성 대화 기능은 프리미엄 멤버십이 필요합니다.''                               AS upgrade_message',
  'SINGLE', 'AI 일본어 대화 화면 설정값', 'Y', 3600, 'ROLE_USER'
WHERE NOT EXISTS (SELECT 1 FROM query_master WHERE sql_key = 'AI_JAPANESE_CHAT_CONFIG');

-- ui_metadata: AI_JAPANESE_CHAT_PAGE 루트 그룹
INSERT INTO ui_metadata (
    screen_id, component_id, component_type, parent_group_id,
    label_text, css_class, group_direction, allowed_roles, sort_order
  )
VALUES (
    'AI_JAPANESE_CHAT_PAGE', 'ai_ja_root', 'GROUP', NULL,
    '', 'ai-page-root', 'COLUMN', 'ROLE_USER', 1
  );

-- ui_metadata: 데이터 소스
INSERT INTO ui_metadata (
    screen_id, component_id, component_type, parent_group_id,
    label_text, action_type, data_sql_key, allowed_roles, sort_order
  )
VALUES (
    'AI_JAPANESE_CHAT_PAGE', 'ai_ja_config', 'DATA_SOURCE', 'ai_ja_root',
    '', 'AUTO_FETCH', 'AI_JAPANESE_CHAT_CONFIG', 'ROLE_USER', 2
  );

-- ui_metadata: AI 일본어 채팅 컴포넌트 (강화 프롬프트, is_readonly=false)
INSERT INTO ui_metadata (
    screen_id, component_id, component_type, parent_group_id,
    label_text, action_type, ref_data_id, css_class,
    system_prompt_template, is_readonly, allowed_roles, sort_order
  )
VALUES (
    'AI_JAPANESE_CHAT_PAGE', 'ai_ja_chat', 'AI_CHAT_V2', 'ai_ja_root',
    'AI Japanese Tutor', 'AI_CHAT_JA', 'ai_ja_config', 'ai-japanese-theme',
    'You are a friendly and professional Japanese tutor. Help the user improve their Japanese through natural conversation.

CRITICAL RULE: You MUST respond with ONLY a single JSON object. No text before or after the JSON.
Format: {"en": "<your Japanese response in kanji/kana>", "ko": "<Korean translation>"}

Examples:
{"en": "そうですね！とても面白いですね。次は何を話しましょうか？", "ko": "그렇군요! 매우 흥미롭네요. 다음에는 무엇을 이야기할까요?"}
{"en": "日本語がお上手ですね！どのくらい勉強していますか？", "ko": "일본어를 잘 하시네요! 얼마나 공부하셨나요?"}

NEVER include any explanation, markdown, or extra text outside the JSON object.',
    false, 'ROLE_USER', 3
  );

DO $$ BEGIN RAISE NOTICE 'V25 완료 - AI_INTERVIEW_PAGE + AI_JAPANESE_CHAT_PAGE 등록 (최종 설정값)'; END $$;
