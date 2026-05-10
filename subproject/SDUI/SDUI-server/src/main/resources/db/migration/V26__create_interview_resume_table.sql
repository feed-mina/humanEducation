-- V26: AI 면접 이력서 업로드 추적 테이블
-- 파일은 S3에 저장되며, 이 테이블은 업로드 이력 및 만료 관리를 위한 메타데이터만 보관
-- 통합 범위: 구 V40

CREATE TABLE IF NOT EXISTS interview_resume (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT       NOT NULL,
    file_key    VARCHAR(512) NOT NULL,          -- S3 object key (e.g. resume/{userId}/{uuid}.pdf)
    file_type   VARCHAR(20)  NOT NULL,          -- 'image' | 'pdf'
    created_at  TIMESTAMP    NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMP    NOT NULL DEFAULT NOW() + INTERVAL '7 days'
);

CREATE INDEX IF NOT EXISTS idx_interview_resume_user_id ON interview_resume (user_id);
CREATE INDEX IF NOT EXISTS idx_interview_resume_expires_at ON interview_resume (expires_at);

DO $$ BEGIN RAISE NOTICE 'V26 완료 - interview_resume 테이블 생성'; END $$;

-- 튜토리얼 벤토 카드 → AI 영어 채팅 전환
-- 벤토 그룹: action_url 변경 + 게스트 숨김
UPDATE ui_metadata
SET action_url = '/view/AI_ENGLISH_CHAT_PAGE',
    allowed_roles = 'ROLE_USER'
WHERE component_id = 'main_bento_tutorial_grp';

-- 타이틀: label_text 변경 + 프리미엄 배지 css_class 추가
UPDATE ui_metadata
SET label_text = 'AI 영어 채팅',
    css_class = TRIM(CONCAT(COALESCE(css_class, ''), ' premium-badge'))
WHERE component_id = 'main_bento_tutorial_title';

-- desc 텍스트 변경
UPDATE ui_metadata
SET label_text = 'AI와 영어 대화 연습을 해보세요.'
WHERE component_id = 'main_bento_tutorial_desc';

-- SIDE_MENU: 튜토리얼 → AI 영어 채팅 (로그인 유저만 노출)
UPDATE ui_metadata
SET label_text = 'AI 영어 채팅',
    action_url = '/view/AI_ENGLISH_CHAT_PAGE',
    allowed_roles = 'ROLE_USER'
WHERE component_id = 'menu_tutorial';
