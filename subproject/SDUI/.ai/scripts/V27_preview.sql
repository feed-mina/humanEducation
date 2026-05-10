-- V27: 카카오 토큰 저장 + 약속 알림 플래그
-- 적용 대상: users 테이블, goal_settings 테이블

-- users 테이블: 서버 측 카카오톡 알림 발송을 위해 토큰 저장
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS kakao_access_token     TEXT,
    ADD COLUMN IF NOT EXISTS kakao_refresh_token    TEXT,
    ADD COLUMN IF NOT EXISTS kakao_token_expires_at TIMESTAMP;

-- goal_settings 테이블: 알림 발송 여부 추적 (중복 발송 방지)
ALTER TABLE goal_settings
    ADD COLUMN IF NOT EXISTS notif_sent_30min  BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS notif_sent_90min  BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS notif_sent_180min BOOLEAN NOT NULL DEFAULT FALSE;
