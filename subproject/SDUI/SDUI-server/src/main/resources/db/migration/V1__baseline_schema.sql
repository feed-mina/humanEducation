-- V1: Baseline Schema DDL
-- 빈 Docker DB에서 기동 시 모든 베이스라인 테이블을 생성합니다.
-- (기존 DB에는 이미 테이블이 있으므로 IF NOT EXISTS로 안전하게 처리)

-- ==========================================
-- TABLE CREATION
-- ==========================================

-- users 테이블
CREATE TABLE IF NOT EXISTS users (
    user_sqno bigserial NOT NULL,
    user_id character varying(50),
    password character varying(255),
    hashed_password character varying(255),
    role character varying(255),
    phone character varying(255),
    email character varying(255),
    del_yn character varying(255) DEFAULT 'N',
    verify_yn character varying(255) DEFAULT 'N',
    social_type character varying(255),
    verification_code character varying(255),
    created_at timestamp(6) without time zone,
    updated_at timestamp(6) without time zone,
    withdraw_at timestamp(6) without time zone,
    verification_expired_at timestamp(6) without time zone,
    time_using_type character varying(255),
    drug_using_type character varying(255),
    zip_code character varying(255),
    road_address character varying(255),
    detail_address character varying(255),
    CONSTRAINT users_pkey PRIMARY KEY (user_sqno)
);

-- diary 테이블 (베이스라인; V3에서 content로 이전, V7에서 삭제)
CREATE TABLE IF NOT EXISTS diary (
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
    CONSTRAINT diary_pkey PRIMARY KEY (diary_id)
);

-- ui_metadata 테이블
CREATE TABLE IF NOT EXISTS ui_metadata (
    ui_id bigserial NOT NULL,
    screen_id character varying(50) NOT NULL,
    component_id character varying(50) NOT NULL,
    label_text character varying(100) NOT NULL,
    component_type character varying(20) NOT NULL,
    sort_order integer NOT NULL,
    is_required boolean DEFAULT false,
    is_readonly boolean DEFAULT true,
    default_value character varying(255),
    placeholder character varying(255),
    created_at timestamp(6) without time zone,
    css_class character varying(100),
    inline_style character varying(500),
    action_type character varying(50),
    action_url character varying(500),
    data_sql_key character varying(50),
    data_api_url character varying(500),
    data_params character varying(50),
    ref_data_id character varying(255),
    group_id character varying(50),
    group_direction character varying(10),
    submit_group_id character varying(50),
    submit_group_order integer,
    submit_group_separator character varying(5),
    parent_group_id character varying(50),
    is_visible character varying(50),
    CONSTRAINT ui_metadata_pkey PRIMARY KEY (ui_id)
);

-- query_master 테이블
CREATE TABLE IF NOT EXISTS query_master (
    sql_key character varying(255) NOT NULL,
    query_text text NOT NULL,
    return_type character varying(255),
    description character varying(255),
    created_at timestamp(6) without time zone,
    updated_at timestamp(6) without time zone,
    CONSTRAINT query_master_pkey PRIMARY KEY (sql_key)
);

-- goal_settings 테이블
CREATE TABLE IF NOT EXISTS goal_settings (
    id bigserial NOT NULL,
    user_sqno bigint NOT NULL,
    user_id character varying(255),
    target_time timestamp(6) without time zone NOT NULL,
    recorded_time timestamp(6) without time zone,
    todays_message character varying(255),
    status character varying(20),
    created_at timestamp(6) without time zone DEFAULT now(),
    CONSTRAINT goal_settings_pkey PRIMARY KEY (id)
);

-- ==========================================
-- INITIAL DATA
-- ==========================================

-- MAIN_PAGE 루트 GROUP (V8 벤토 그리드 마이그레이션에 필요)
INSERT INTO ui_metadata
  (screen_id, component_id, component_type, parent_group_id, label_text, css_class, sort_order, created_at)
VALUES
  ('MAIN_PAGE', 'MAIN_SECTION', 'GROUP', NULL, '메인', 'main-page', 1, NOW());

DO $$ BEGIN RAISE NOTICE 'V1: baseline schema + MAIN_SECTION 생성 완료'; END $$;
