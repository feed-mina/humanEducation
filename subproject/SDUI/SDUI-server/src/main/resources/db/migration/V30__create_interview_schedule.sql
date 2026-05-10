-- V30: 면접 일정 관리 + D-1 Slack 리마인더
-- 사용자가 면접 날짜를 등록하면 전날 09:00 KST Slack 알림 발송
CREATE TABLE IF NOT EXISTS interview_schedule (
    id             SERIAL PRIMARY KEY,
    user_sqno      BIGINT       NOT NULL REFERENCES users(user_sqno) ON DELETE CASCADE,
    interview_date DATE         NOT NULL,
    company        VARCHAR(100),
    notif_sent_d1  BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_interview_schedule_notif
    ON interview_schedule (interview_date)
    WHERE notif_sent_d1 = FALSE;

DO $$ BEGIN RAISE NOTICE 'V30 완료 - interview_schedule 테이블 생성'; END $$;
