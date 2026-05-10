-- V24__ai_chat_pages.sql
-- 목적: AI 영어/한국어 대화 화면 등록 (ui_metadata + query_master)
-- DESTINATION: SDUI-server/src/main/resources/db/migration/V24__ai_chat_pages.sql

-- ============================================================
-- 1. query_master: 화면 설정값 (버튼 라벨, 환영 메시지 등)
-- ============================================================

INSERT INTO query_master (sql_key, sql_text, description, required_role)
VALUES (
    'ai_english_chat_config',
    'SELECT
        ''🎤 녹음 시작''   AS mic_btn_label,
        ''답변완료''        AS submit_btn_label,
        ''대화 종료''       AS end_btn_label,
        ''Hello! I''''m your English conversation partner. What would you like to practice today?'' AS welcome_message,
        ''en''              AS language,
        ''BASIC''           AS required_tier,
        ''음성 대화 기능은 베이직 이상 멤버십이 필요합니다.'' AS upgrade_message',
    'AI 영어 대화 화면 설정값',
    'ROLE_USER'
);

INSERT INTO query_master (sql_key, sql_text, description, required_role)
VALUES (
    'ai_korean_chat_config',
    'SELECT
        ''🎤 녹음 시작''   AS mic_btn_label,
        ''답변완료''        AS submit_btn_label,
        ''대화 종료''       AS end_btn_label,
        ''안녕하세요! 한국어 대화 연습을 도와드리겠습니다. 무엇을 연습하고 싶으신가요?'' AS welcome_message,
        ''ko''              AS language,
        ''BASIC''           AS required_tier,
        ''음성 대화 기능은 베이직 이상 멤버십이 필요합니다.'' AS upgrade_message',
    'AI 한국어 대화 화면 설정값',
    'ROLE_USER'
);

-- ============================================================
-- 2. ui_metadata: AI_ENGLISH_CHAT_PAGE
-- ============================================================

-- 2-1. 데이터 소스 (query_master 자동 호출)
INSERT INTO ui_metadata (
    screen_id, component_id, label_text, component_type,
    action_type, data_sql_key, allowed_roles, sort_order
) VALUES (
    'AI_ENGLISH_CHAT_PAGE', 'ai_en_config', '', 'DATA_SOURCE',
    'AUTO_FETCH', 'ai_english_chat_config', 'ROLE_USER', 0
);

-- 2-2. AI 채팅 컴포넌트
INSERT INTO ui_metadata (
    screen_id, component_id, label_text, component_type,
    action_type, ref_data_id, css_class, allowed_roles, sort_order
) VALUES (
    'AI_ENGLISH_CHAT_PAGE', 'ai_en_chat', 'AI English Conversation', 'AI_CHAT',
    'AI_CHAT_EN', 'ai_en_config', 'ai-chat-en', 'ROLE_USER', 1
);

-- ============================================================
-- 3. ui_metadata: AI_KOREAN_CHAT_PAGE
-- ============================================================

-- 3-1. 데이터 소스
INSERT INTO ui_metadata (
    screen_id, component_id, label_text, component_type,
    action_type, data_sql_key, allowed_roles, sort_order
) VALUES (
    'AI_KOREAN_CHAT_PAGE', 'ai_ko_config', '', 'DATA_SOURCE',
    'AUTO_FETCH', 'ai_korean_chat_config', 'ROLE_USER', 0
);

-- 3-2. AI 채팅 컴포넌트
INSERT INTO ui_metadata (
    screen_id, component_id, label_text, component_type,
    action_type, ref_data_id, css_class, allowed_roles, sort_order
) VALUES (
    'AI_KOREAN_CHAT_PAGE', 'ai_ko_chat', 'AI 한국어 대화', 'AI_CHAT',
    'AI_CHAT_KO', 'ai_ko_config', 'ai-chat-ko', 'ROLE_USER', 1
);
