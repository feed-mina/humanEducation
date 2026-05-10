-- V3: diary → content 데이터 마이그레이션
-- diary 테이블의 모든 데이터를 content 테이블로 복사합니다.

-- diary_backup 테이블 생성 (diary와 동일한 구조)
CREATE TABLE IF NOT EXISTS diary_backup (
    diary_id bigserial NOT NULL,
    content character varying(255),
    date character varying(255),
    del_dt timestamp(6) without time zone,
    del_yn character varying(255) DEFAULT 'N',
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
    daily_slots jsonb DEFAULT '{}'::jsonb,
    day_tag1 character varying(255),
    day_tag2 character varying(255),
    day_tag3 character varying(255),
    frst_dt timestamp(6) without time zone,
    frst_rgst_usps_sqno numeric(38, 0),
    last_updt_usps_sqno numeric(38, 0),
    role_cd character varying(255),
    CONSTRAINT diary_backup_pkey PRIMARY KEY (diary_id)
);

-- diary 테이블 존재 시 데이터 마이그레이션 수행
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'diary') THEN
        RAISE NOTICE 'V3: diary 테이블이 존재하므로 데이터 마이그레이션을 시작합니다.';

        -- diary_backup에 전체 데이터 백업
        IF EXISTS (SELECT 1 FROM diary LIMIT 1) THEN
            INSERT INTO diary_backup SELECT * FROM diary;
        END IF;

        -- diary → content 데이터 복사
        INSERT INTO content (
            content_id, content, date, del_dt, del_yn,
            content_status, content_type, email, emotion,
            frst_reg_ip, last_updt_dt, last_updt_ip,
            reg_dt, role_nm, selected_times, title,
            updt_dt, user_id, user_sqno, daily_slots,
            day_tag1, day_tag2, day_tag3, frst_dt,
            frst_rgst_usps_sqno, last_updt_usps_sqno, role_cd
        )
        SELECT
            diary_id, content, date, del_dt, del_yn,
            diary_status, diary_type, email, emotion,
            frst_reg_ip, last_updt_dt, last_updt_ip,
            reg_dt, role_nm, selected_times, title,
            updt_dt, user_id, user_sqno, daily_slots,
            day_tag1, day_tag2, day_tag3, frst_dt,
            frst_rgst_usps_sqno, last_updt_usps_sqno, role_cd
        FROM diary
        WHERE NOT EXISTS (
            SELECT 1 FROM content WHERE content_id = diary.diary_id
        );

        -- 시퀀스 동기화 (content 테이블의 최대값 + 1로 설정)
        PERFORM setval('content_content_id_seq', (SELECT COALESCE(MAX(content_id), 0) + 1 FROM content), false);

    ELSE
        RAISE NOTICE 'V3: diary 테이블이 존재하지 않아 데이터 마이그레이션을 건너뜁니다.';
    END IF;
END $$;

-- 검증: 데이터 개수 확인
DO $$
DECLARE
    diary_count INTEGER := 0;
    content_count INTEGER;
    backup_count INTEGER;
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'diary') THEN
        SELECT COUNT(*) INTO diary_count FROM diary;
    END IF;

    SELECT COUNT(*) INTO content_count FROM content;
    SELECT COUNT(*) INTO backup_count FROM diary_backup;

    RAISE NOTICE 'Migration V3 완료 - diary: %, content: %, backup: %', diary_count, content_count, backup_count;

    IF diary_count > 0 AND content_count = 0 AND backup_count > 0 THEN
        RAISE EXCEPTION '데이터 마이그레이션 실패: diary에 데이터가 있지만 content에 복사되지 않았습니다.';
    END IF;
END $$;

