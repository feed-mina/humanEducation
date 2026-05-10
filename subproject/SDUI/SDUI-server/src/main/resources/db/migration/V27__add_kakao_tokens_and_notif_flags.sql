-- V27: 카카오 토큰 저장 + 약속 알림 플래그
-- users 테이블: 서버 측 카카오톡 알림 발송을 위해 토큰 저장
ALTER TABLE users
    ADD COLUMN kakao_access_token     TEXT,
    ADD COLUMN kakao_refresh_token    TEXT,
    ADD COLUMN kakao_token_expires_at TIMESTAMP;

-- goal_settings 테이블: 알림 발송 여부 추적 (중복 발송 방지)
ALTER TABLE goal_settings
    ADD COLUMN notif_sent_30min  BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN notif_sent_90min  BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN notif_sent_180min BOOLEAN NOT NULL DEFAULT FALSE;

-- 알림 스케줄러 조회 성능 최적화 인덱스 (PostgreSQL 부분 인덱스)
-- status IS NULL + 미발송 조건을 만족하는 행만 인덱싱 → 풀스캔 방지
CREATE INDEX idx_goal_pending_30min
    ON goal_settings (target_time)
    WHERE status IS NULL AND notif_sent_30min = false;

CREATE INDEX idx_goal_pending_90min
    ON goal_settings (target_time)
    WHERE status IS NULL AND notif_sent_90min = false;

CREATE INDEX idx_goal_pending_180min
    ON goal_settings (target_time)
    WHERE status IS NULL AND notif_sent_180min = false;
