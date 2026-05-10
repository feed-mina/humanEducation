-- V2: content 테이블 생성
-- diary 테이블을 대체할 content 테이블을 생성합니다.
-- diary → content 리네이밍: diary_id → content_id, diary_status → content_status, diary_type → content_type

-- 안전장치: diary_backup 테이블 생성 (구조만 복사)
CREATE TABLE IF NOT EXISTS diary_backup (
    diary_id bigint,
    content character varying(255),
    date character varying(255),
    del_dt timestamp(6) without time zone,
    del_yn character varying(255),
    diary_status character varying(255),
    diary_type character varying(255),
    email character varying(255),
    emotion integer,
    frst_reg_ip character varying(255),
    last_updt_dt character varying(255),
    last_updt_ip character varying(255),
    reg_dt timestamp(6) without time zone,
    role_nm character varying(255),
    selected_times jsonb,
    title character varying(255),
    updt_dt timestamp(6) without time zone,
    user_id character varying(255),
    user_sqno bigint,
    daily_slots jsonb,
    day_tag1 character varying(255),
    day_tag2 character varying(255),
    day_tag3 character varying(255),
    frst_dt timestamp(6) without time zone,
    frst_rgst_usps_sqno numeric(38, 0),
    last_updt_usps_sqno numeric(38, 0),
    role_cd character varying(255)
);

-- content 테이블 생성
CREATE TABLE IF NOT EXISTS content (
    content_id bigserial NOT NULL,
    content character varying(255),
    date character varying(255),
    del_dt timestamp(6) without time zone,
    del_yn character varying(255) NOT NULL DEFAULT 'N',
    content_status character varying(255),
    content_type character varying(255),
    email character varying(255),
    emotion integer,
    frst_reg_ip character varying(255),
    last_updt_dt character varying(255),
    last_updt_ip character varying(255),
    reg_dt timestamp(6) without time zone,
    role_nm character varying(255),
    selected_times jsonb,
    title character varying(255),
    updt_dt timestamp(6) without time zone,
    user_id character varying(255),
    user_sqno bigint,
    daily_slots jsonb DEFAULT '{}'::jsonb,
    day_tag1 character varying(255),
    day_tag2 character varying(255),
    day_tag3 character varying(255),
    frst_dt timestamp(6) without time zone,
    frst_rgst_usps_sqno numeric(38, 0),
    last_updt_usps_sqno numeric(38, 0),
    role_cd character varying(255),
    CONSTRAINT content_pkey PRIMARY KEY (content_id),
    CONSTRAINT fkdqvlqxqs75ruipisce1c50xvw FOREIGN KEY (user_sqno)
        REFERENCES users (user_sqno) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_content_user_sqno ON content(user_sqno);
CREATE INDEX IF NOT EXISTS idx_content_date ON content(date);
CREATE INDEX IF NOT EXISTS idx_content_del_yn ON content(del_yn);

-- 시퀀스 생성 (bigserial로 자동 생성되지만 명시적으로 확인)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = 'content_content_id_seq') THEN
        CREATE SEQUENCE content_content_id_seq OWNED BY content.content_id;
    END IF;
END $$;
