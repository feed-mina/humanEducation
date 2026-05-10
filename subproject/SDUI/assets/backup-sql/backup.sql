--
-- PostgreSQL database dump
--

\restrict Qi7Sd5Eum8QAexLrtIx2OZCWwdhgqxtH0aeybHawFx9Y4OPCULUH18wEy3u1A9L

-- Dumped from database version 15.16 (Debian 15.16-1.pgdg13+1)
-- Dumped by pg_dump version 15.16 (Debian 15.16-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

ALTER TABLE ONLY public.content DROP CONSTRAINT fkdqvlqxqs75ruipisce1c50xvw;
DROP INDEX public.idx_content_user_sqno;
DROP INDEX public.idx_content_del_yn;
DROP INDEX public.idx_content_date;
DROP INDEX public.flyway_schema_history_s_idx;
ALTER TABLE ONLY public.users DROP CONSTRAINT users_pkey;
ALTER TABLE ONLY public.ui_metadata DROP CONSTRAINT ui_metadata_pkey;
ALTER TABLE ONLY public.query_master DROP CONSTRAINT query_master_pkey;
ALTER TABLE ONLY public.goal_settings DROP CONSTRAINT goal_settings_pkey;
ALTER TABLE ONLY public.flyway_schema_history DROP CONSTRAINT flyway_schema_history_pk;
ALTER TABLE ONLY public.content DROP CONSTRAINT content_pkey;
ALTER TABLE public.users ALTER COLUMN user_sqno DROP DEFAULT;
ALTER TABLE public.ui_metadata ALTER COLUMN ui_id DROP DEFAULT;
ALTER TABLE public.goal_settings ALTER COLUMN id DROP DEFAULT;
ALTER TABLE public.content ALTER COLUMN content_id DROP DEFAULT;
DROP SEQUENCE public.users_user_sqno_seq;
DROP TABLE public.users;
DROP SEQUENCE public.ui_metadata_ui_id_seq;
DROP TABLE public.ui_metadata;
DROP TABLE public.query_master;
DROP SEQUENCE public.goal_settings_id_seq;
DROP TABLE public.goal_settings;
DROP TABLE public.flyway_schema_history_backup;
DROP TABLE public.flyway_schema_history;
DROP TABLE public.diary_backup;
DROP SEQUENCE public.content_content_id_seq;
DROP TABLE public.content;
SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: content; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.content (
    content_id bigint NOT NULL,
    content character varying(255),
    date character varying(255),
    del_dt timestamp(6) without time zone,
    del_yn character varying(255) DEFAULT 'N'::character varying NOT NULL,
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
    frst_rgst_usps_sqno numeric(38,0),
    last_updt_usps_sqno numeric(38,0),
    role_cd character varying(255)
);


--
-- Name: content_content_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.content_content_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: content_content_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.content_content_id_seq OWNED BY public.content.content_id;


--
-- Name: diary_backup; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.diary_backup (
    diary_id bigint,
    content character varying(255),
    daily_slots jsonb,
    date character varying(255),
    day_tag1 character varying(255),
    day_tag2 character varying(255),
    day_tag3 character varying(255),
    del_dt timestamp(6) without time zone,
    del_yn character varying(255),
    diary_status character varying(255),
    diary_type character varying(255),
    email character varying(255),
    emotion integer,
    frst_dt timestamp(6) without time zone,
    frst_reg_ip character varying(255),
    frst_rgst_usps_sqno numeric(38,0),
    last_updt_dt character varying(255),
    last_updt_ip character varying(255),
    last_updt_usps_sqno numeric(38,0),
    reg_dt timestamp(6) without time zone,
    role_cd character varying(255),
    role_nm character varying(255),
    selected_times jsonb,
    title character varying(255),
    updt_dt timestamp(6) without time zone,
    user_id character varying(255),
    user_sqno bigint
);


--
-- Name: flyway_schema_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.flyway_schema_history (
    installed_rank integer NOT NULL,
    version character varying(50),
    description character varying(200) NOT NULL,
    type character varying(20) NOT NULL,
    script character varying(1000) NOT NULL,
    checksum integer,
    installed_by character varying(100) NOT NULL,
    installed_on timestamp without time zone DEFAULT now() NOT NULL,
    execution_time integer NOT NULL,
    success boolean NOT NULL
);


--
-- Name: flyway_schema_history_backup; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.flyway_schema_history_backup (
    installed_rank integer,
    version character varying(50),
    description character varying(200),
    type character varying(20),
    script character varying(1000),
    checksum integer,
    installed_by character varying(100),
    installed_on timestamp without time zone,
    execution_time integer,
    success boolean
);


--
-- Name: goal_settings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.goal_settings (
    id bigint NOT NULL,
    created_at timestamp(6) without time zone,
    recorded_time timestamp(6) without time zone,
    status character varying(20),
    target_time timestamp(6) without time zone NOT NULL,
    todays_message character varying(255),
    user_id character varying(255),
    user_sqno bigint NOT NULL
);


--
-- Name: goal_settings_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.goal_settings_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: goal_settings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.goal_settings_id_seq OWNED BY public.goal_settings.id;


--
-- Name: query_master; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.query_master (
    sql_key character varying(100) NOT NULL,
    query_text text NOT NULL,
    return_type character varying(20),
    description text,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now(),
    required_params character varying(255),
    param_mapping jsonb,
    use_redis_yn character(1) DEFAULT 'N'::bpchar,
    redis_ttl_sec integer DEFAULT 3600
);


--
-- Name: ui_metadata; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ui_metadata (
    ui_id bigint NOT NULL,
    screen_id character varying(50) NOT NULL,
    component_id character varying(50) NOT NULL,
    label_text character varying(100) NOT NULL,
    component_type character varying(20) NOT NULL,
    sort_order integer NOT NULL,
    is_required boolean DEFAULT false,
    is_readonly boolean DEFAULT true,
    default_value text,
    placeholder text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    css_class character varying(100),
    inline_style text,
    action_type character varying(50),
    action_url character varying(255),
    data_sql_key character varying(100),
    data_api_url character varying(255),
    data_params character varying(50),
    ref_data_id character varying(100),
    group_id character varying(50) DEFAULT NULL::character varying,
    group_direction character varying(10) DEFAULT 'COLUMN'::character varying,
    submit_group_id character varying(50) DEFAULT NULL::character varying,
    submit_group_order integer,
    submit_group_separator character varying(5) DEFAULT NULL::character varying,
    parent_group_id character varying(50) DEFAULT NULL::character varying,
    is_visible character varying(50) DEFAULT true,
    component_props jsonb DEFAULT '{}'::jsonb,
    allowed_roles character varying(255) DEFAULT NULL::character varying,
    label_text_overrides jsonb,
    css_class_overrides jsonb
);


--
-- Name: TABLE ui_metadata; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.ui_metadata IS '화면 항목 구성을 위한 메타데이터 테이블';


--
-- Name: COLUMN ui_metadata.ui_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.ui_metadata.ui_id IS '화면 식별 코드';


--
-- Name: COLUMN ui_metadata.screen_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.ui_metadata.screen_id IS '화면 구분 코드 (예: DIARY_WRITE)';


--
-- Name: COLUMN ui_metadata.component_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.ui_metadata.component_id IS 'DB에 저장될 필드명 (예: title)';


--
-- Name: COLUMN ui_metadata.label_text; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.ui_metadata.label_text IS '사용자에게 보여줄 이름 (예: 제목)';


--
-- Name: COLUMN ui_metadata.component_type; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.ui_metadata.component_type IS '입력 도구 종류 (TEXT, TEXTAREA, SELECT, DATE)';


--
-- Name: COLUMN ui_metadata.sort_order; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.ui_metadata.sort_order IS '화면 표시 순서';


--
-- Name: COLUMN ui_metadata.is_required; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.ui_metadata.is_required IS '필수 입력 여부';


--
-- Name: COLUMN ui_metadata.is_readonly; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.ui_metadata.is_readonly IS '입력 가능 여부(상세 화면은 Y, 작성화면은 N)';


--
-- Name: ui_metadata_ui_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ui_metadata_ui_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ui_metadata_ui_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ui_metadata_ui_id_seq OWNED BY public.ui_metadata.ui_id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.users (
    user_sqno bigint NOT NULL,
    created_at timestamp(6) without time zone,
    del_yn character varying(255),
    detail_address character varying(255),
    drug_using_type character varying(255),
    email character varying(255),
    hashed_password character varying(255),
    password character varying(255),
    phone character varying(255),
    road_address character varying(255),
    role character varying(255),
    social_type character varying(255),
    time_using_type character varying(255),
    updated_at timestamp(6) without time zone,
    user_id character varying(50),
    verification_code character varying(255),
    verification_expired_at timestamp(6) without time zone,
    verify_yn character varying(255),
    withdraw_at timestamp(6) without time zone,
    zip_code character varying(255)
);


--
-- Name: users_user_sqno_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.users_user_sqno_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: users_user_sqno_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.users_user_sqno_seq OWNED BY public.users.user_sqno;


--
-- Name: content content_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.content ALTER COLUMN content_id SET DEFAULT nextval('public.content_content_id_seq'::regclass);


--
-- Name: goal_settings id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.goal_settings ALTER COLUMN id SET DEFAULT nextval('public.goal_settings_id_seq'::regclass);


--
-- Name: ui_metadata ui_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ui_metadata ALTER COLUMN ui_id SET DEFAULT nextval('public.ui_metadata_ui_id_seq'::regclass);


--
-- Name: users user_sqno; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users ALTER COLUMN user_sqno SET DEFAULT nextval('public.users_user_sqno_seq'::regclass);


--
-- Data for Name: content; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.content (content_id, content, date, del_dt, del_yn, content_status, content_type, email, emotion, frst_reg_ip, last_updt_dt, last_updt_ip, reg_dt, role_nm, selected_times, title, updt_dt, user_id, user_sqno, daily_slots, day_tag1, day_tag2, day_tag3, frst_dt, frst_rgst_usps_sqno, last_updt_usps_sqno, role_cd) FROM stdin;
1	내용입력란에 라벨보이기	\N	\N	N	\N	\N	\N	2	\N	\N	\N	2026-03-03 04:00:03.139457	\N	[0, 1, 2, 3]	배포테스트 	\N	myelin24	1	\N	\N	\N	\N	\N	\N	\N	\N
\.


--
-- Data for Name: diary_backup; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.diary_backup (diary_id, content, daily_slots, date, day_tag1, day_tag2, day_tag3, del_dt, del_yn, diary_status, diary_type, email, emotion, frst_dt, frst_reg_ip, frst_rgst_usps_sqno, last_updt_dt, last_updt_ip, last_updt_usps_sqno, reg_dt, role_cd, role_nm, selected_times, title, updt_dt, user_id, user_sqno) FROM stdin;
\.


--
-- Data for Name: flyway_schema_history; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.flyway_schema_history (installed_rank, version, description, type, script, checksum, installed_by, installed_on, execution_time, success) FROM stdin;
1	1	baseline schema	SQL	V1__baseline_schema.sql	-559520453	mina	2026-03-02 20:02:55.136846	8	t
2	2	create content table	SQL	V2__create_content_table.sql	-554112740	mina	2026-03-03 00:43:09.729319	148	t
3	7	baseline at version 7 after manual migration	BASELINE	V7__drop_diary_table.sql	\N	mina	2026-03-03 02:04:28.506414	0	t
\.


--
-- Data for Name: flyway_schema_history_backup; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.flyway_schema_history_backup (installed_rank, version, description, type, script, checksum, installed_by, installed_on, execution_time, success) FROM stdin;
1	1	baseline schema	SQL	V1__baseline_schema.sql	-559520453	mina	2026-03-02 20:02:55.136846	8	t
2	2	create content table	SQL	V2__create_content_table.sql	-554112740	mina	2026-03-03 00:43:09.729319	148	t
3	7	baseline at version 7 after manual migration	BASELINE	V7__drop_diary_table.sql	\N	mina	2026-03-03 02:04:28.506414	0	t
\.


--
-- Data for Name: goal_settings; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.goal_settings (id, created_at, recorded_time, status, target_time, todays_message, user_id, user_sqno) FROM stdin;
\.


--
-- Data for Name: query_master; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.query_master (sql_key, query_text, return_type, description, created_at, updated_at, required_params, param_mapping, use_redis_yn, redis_ttl_sec) FROM stdin;
FIND_WITHDRAW_USER	SELECT user_sqno AS "userSqno", user_id AS "userId", email FROM users\r\n      WHERE email = :email AND del_yn = 'Y' AND withdraw_at > NOW() - INTERVAL '7 days'	SINGLE	7일 이내 탈퇴한 회원 조회	2026-01-14 23:11:41.459645	2026-01-14 23:11:41.459645	\N	\N	N	3600
GET_USER_GOAL_TIME	SELECT target_time AS "targetTime"\r\n                  FROM goal_settings\r\n                  WHERE user_sqno = :userSqno\r\n                    AND status IS NULL\r\n                    AND target_time >= CURRENT_DATE  -- ★ 수정됨: 오늘 0시 이후 데이터는 다 가져옴\r\n                  ORDER BY target_time ASC\r\n                  LIMIT 1	SINGLE	메인 화면용: 가장 가까운 미래 목표 1건 조회	2026-01-26 06:54:52.464448	2026-01-31 07:42:39.286013	\N	\N	Y	3600
INSERT_CONTENT	INSERT INTO content (\n    user_sqno, user_id, title, content, emotion, \n    selected_times, daily_slots, day_tag1, day_tag2, day_tag3, \n    reg_dt\n) VALUES (\n    :userSqno, :userId, :title, :content, CAST(:emotion AS INTEGER), \n    CAST(:selected_times AS jsonb), CAST(:daily_slots AS jsonb), :day_tag1, :day_tag2, :day_tag3, \n    NOW()\n)	COMMAND	신규 콘텐츠 작성 (JSONB 타입 및 day_tag 반영)	2026-01-14 23:11:41.459645	2026-03-02 14:16:26.460792	title,content	\N	N	3600
GET_USER_GOAL_LIST	SELECT target_time AS "targetTime"\r\n                  FROM goal_settings\r\n                  WHERE user_sqno = :userSqno\r\n                    AND status IS NULL\r\n                    AND target_time >= CURRENT_DATE  -- ★ 수정됨\r\n                  ORDER BY target_time ASC\r\n                  LIMIT 3 OFFSET 1	MULTI	팝업 리스트용: 메인 목표 이후의 다음 3건 조회	2026-01-31 07:42:53.704614	2026-01-31 07:42:53.704614	\N	\N	Y	3600
UPDATE_GOAL_RESULT	UPDATE goal_settings\r\n                  SET status = :status, recorded_time = :recordedTime\r\n                  WHERE id = (\r\n                      SELECT id\r\n                      FROM goal_settings\r\n                      WHERE user_sqno = :userSqno\r\n                        AND status IS NULL\r\n                        AND target_time >= CURRENT_DATE -- ★ 수정됨\r\n                      ORDER BY target_time ASC\r\n                      LIMIT 1\r\n                  )	COMMAND	가장 가까운 미완료 목표의 도착 결과(성공/실패) 업데이트	2026-01-31 11:25:20.344869	2026-01-31 11:25:20.344869	\N	\N	N	0
INSERT_USER	INSERT INTO users (user_id, password, hashed_password, email, phone, nickname, role, username, created_at, updated_at, social_type)\r\n      VALUES (:userId, :password, :hashedPassword, :email, :phone, COALESCE(:nickname, :userId), :role, :username, NOW(), NOW(), :socialType)	COMMAND	신규 회원 가입 (닉네임 없을 시 아이디로 대체)	2026-01-14 23:10:06.713005	2026-01-14 23:10:06.713005	userId,password,email	{"email": "body.email", "userId": "body.userId", "password": "body.password"}	N	3600
GET_CONTENT_LIST_PAGE	SELECT d.content_id AS "content_id", d.user_sqno AS "user_sqno", d.title AS "title", d.date AS "date", d.emotion AS "emotion", d.day_tag1 AS "tag1", d.day_tag2 AS "tag2", d.day_tag3 AS "tag3", d.content_status AS "content_status", d.reg_dt AS "reg_dt", CASE WHEN u.del_yn = 'Y' THEN CONCAT('del_', d.user_id) ELSE d.user_id END AS "user_id" FROM content d LEFT JOIN users u ON d.user_sqno = u.user_sqno WHERE d.del_yn = 'N' AND (NULLIF(CAST(:filterId AS VARCHAR), '') IS NULL OR d.user_id = CAST(:filterId AS VARCHAR)) ORDER BY d.reg_dt DESC LIMIT CAST(COALESCE(NULLIF(CAST(:pageSize AS VARCHAR), ''), '5') AS INTEGER) OFFSET CAST(COALESCE(NULLIF(CAST(:offset AS VARCHAR), ''), '0') AS INTEGER)	MULTI	콘텐츠 목록 페이징 조회	2026-01-14 23:10:06.713005	2026-03-02 14:16:26.460792	\N	{"offset": "params.offset", "userId": "params.userId", "pageSize": "params.pageSize"}	Y	300
GET_CONTENT_DETAIL	SELECT d.content_id, d.user_id, d.title, d.content, d.date, d.emotion, d.day_tag1, d.day_tag2, d.day_tag3, d.content_status, d.role_nm, d.selected_times, d.daily_slots, d.reg_dt \n                  FROM content d \n                  WHERE d.content_id = CAST(:contentId AS BIGINT) AND d.del_yn = 'N'	SINGLE	콘텐츠 상세 정보 조회 (JSON 데이터 포함)	2026-02-22 09:18:35.850523	2026-03-02 14:16:26.460792	\N	\N	N	3600
COUNT_CONTENT_LIST	SELECT COUNT(*) AS total_count FROM content d WHERE d.del_yn = 'N' AND (:userId IS NULL OR :userId = '' OR d.user_id = :userId)	SINGLE	전체 콘텐츠 개수 조회	2026-01-14 23:11:41.459645	2026-03-02 14:16:26.460792	\N	\N	N	3600
GET_MEMBER_CONTENT_LIST	SELECT content_id AS "content_id", title AS "title", content AS "content", reg_dt AS "reg_dt", img_url AS "img_url", user_id AS "user_id" FROM content WHERE user_sqno = :userSqno AND del_yn = 'N' ORDER BY reg_dt DESC LIMIT :pageSize OFFSET :offset	MULTI	로그인한 사용자의 콘텐츠 목록 조회	2026-01-14 12:34:04.730293	2026-03-02 14:16:26.460792	\N	{"offset": "params.offset", "pageSize": "params.pageSize", "userSqno": "session.userSqno"}	Y	600
UPDATE_CONTENT_DETAIL	UPDATE content \nSET title = :title, \n    content = :content, \n    emotion = CAST(:emotion AS INTEGER), \n    selected_times = CAST(:selected_times AS jsonb), \n    daily_slots = CAST(:daily_slots AS jsonb), \n    day_tag1 = :day_tag1, \n    day_tag2 = :day_tag2, \n    day_tag3 = :day_tag3\nWHERE content_id = CAST(:content_id AS BIGINT) \n  AND user_sqno = :userSqno	COMMAND	콘텐츠 내용 수정 쿼리	2026-02-22 20:52:05.840824	2026-03-02 14:16:26.460792	\N	\N	N	3600
UPDATE_CONTENT_DELETE	UPDATE content\r\n      SET del_yn = 'Y',\r\n          del_dt = NOW(),\r\n          last_updt_ip = :lastUpdtIp,\r\n          last_updt_usps_sqno = :lastUpdtUspsSqno\r\n      WHERE content_id = ANY(:contentIdList)\r\n      AND (:userSqno IS NULL OR user_sqno = :userSqno)	COMMAND	선택한 콘텐츠 일괄 삭제 처리 (본인 확인 포함)	2026-01-14 23:10:06.713005	2026-03-02 14:16:26.460792	contentIdList	\N	N	3600
\.


--
-- Data for Name: ui_metadata; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.ui_metadata (ui_id, screen_id, component_id, label_text, component_type, sort_order, is_required, is_readonly, default_value, placeholder, created_at, css_class, inline_style, action_type, action_url, data_sql_key, data_api_url, data_params, ref_data_id, group_id, group_direction, submit_group_id, submit_group_order, submit_group_separator, parent_group_id, is_visible, component_props, allowed_roles, label_text_overrides, css_class_overrides) FROM stdin;
283	MAIN_PAGE	LOGIN_LEFT_CONTENT	로그인 왼쪽 영역	GROUP	10	f	t	\N	\N	2026-02-22 19:39:59.124085	card-left-area	\N	\N	\N	\N	\N	\N	\N	LOGIN_LEFT_CONTENT	COLUMN	\N	\N	\N	MAIN_LOGIN_CARD	true	{}	\N	\N	\N
284	MAIN_PAGE	TUTORIAL_LEFT_CONTENT	튜토리얼 왼쪽 영역	GROUP	10	f	t	\N	\N	2026-02-22 19:39:59.124085	card-left-area	\N	\N	\N	\N	\N	\N	\N	TUTORIAL_LEFT_CONTENT	COLUMN	\N	\N	\N	MAIN_TUTORIAL_CARD	true	{}	\N	\N	\N
864	CONTENT_DETAIL	CONTENTWRITE_BTN_GROUP	CONTENTWRITE_BTN_GROUP	GROUP	70	f	f	\N	\N	2026-02-28 14:36:41.971575	content_btn_group	\N	\N	\N	\N	\N	\N	\N	DIARYWRITE_BTN_GROUP	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
1010	REGISTER_PAGE	INFO_SECTION	기본 정보 영역	SECTION	1	f	t	\N	\N	2026-02-21 17:33:38.324058	INFO_SECTION	\N	\N	\N	\N	\N	\N	\N	INFO_SECTION	vertical	\N	\N	\N	REG_CONTAINER	true	{"gap": "15px"}	\N	\N	\N
709	CONTENT_WRITE	DAYTAG_SUB_GROUP	하루해쉬태그그룹	GROUP	40	f	f	\N	\N	2026-01-23 22:01:08.909154	write_sub_group	\N	\N	\N	\N	\N	\N	\N	DAYTAG_SUB_GROUP	ROW	\N	\N	\N	DIARYWRITE_SECTION	true	{}	\N	\N	\N
607	CONTENT_LIST	list_item_date	날짜	TEXT	2	f	t	\N	\N	2026-01-18 00:22:38.322161	contentDate	\N	\N	\N	\N	\N	\N	reg_dt	\N	COLUMN	\N	\N	\N	CONTENT_CARD_HEADER	true	{}	\N	\N	\N
1030	REGISTER_PAGE	ACTION_SECTION	버튼 영역	SECTION	3	f	t	\N	\N	2026-02-21 17:33:38.324058	mt-40	\N	\N	\N	\N	\N	\N	\N	ACTION_SECTION	COLUMN	\N	\N	\N	REG_CONTAINER	true	{}	\N	\N	\N
301	MAIN_PAGE	main_header_img	dino_content.png	IMAGE	1	f	f	\N	\N	2026-01-14 06:52:12.260807	contentImg	\N	\N	\N	\N	\N	\N	\N	 	COLUMN	\N	\N	\N	MAIN_TOP_CARD	true	{}	\N	\N	\N
1113	CONTENT_MODIFY	go_modify_btn	수정하기	BUTTON	70	f	f	\N	\N	2026-02-23 10:16:54.957794	content-btn-primary	\N	LINK	/view/MAIN_PAGE	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	DIARYWRITE_BTN_GROUP	false	{}	\N	\N	\N
715	CONTENT_WRITE	title_dayTag2	하루태그2	TEXT	81	f	f	\N	\N	2026-01-23 22:10:02.955892	content-label	\N	\N	\N	\N	\N	\N	\N	GROUP_TAG_ROW2	COLUMN	\N	\N	\N	GROUP_TAG_ROW2	true	{}	\N	\N	\N
1020	REGISTER_PAGE	ADDR_SECTION	주소 정보 영역	SECTION	2	f	f	\N	\N	2026-02-21 17:33:38.324058	mt-30	\N	\N	\N	\N	\N	\N	\N	ADDR_SECTION	COLUMN	\N	\N	\N	REG_CONTAINER	true	{}	\N	\N	\N
1032	REGISTER_PAGE	email_verify_modal	회원 정보 입력	MODAL	99	f	f	\N	\N	2026-02-24 23:26:50.70224	\N	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	ROOT	true	{"content": "인증 메일을 전송했습니다. 인증을 완료하신 후 이 페이지로 돌아와 확인 버튼을 눌러주세요.", "button_text": "확인"}	\N	\N	\N
718	CONTENT_WRITE	title_dayTag3	하루태그3	TEXT	91	f	f	\N	\N	2026-01-23 22:10:02.955892	content-label	\N	\N	\N	\N	\N	\N	\N	GROUP_TAG_ROW3	COLUMN	\N	\N	\N	GROUP_TAG_ROW3	true	{}	\N	\N	\N
1011	REGISTER_PAGE	reg_email	이메일	INPUT	1	t	f	\N	example@email.com	2026-02-21 17:33:38.324058	reg_email	\N	\N	\N	\N	\N	\N	email	\N	COLUMN	\N	\N	\N	INFO_SECTION	true	{}	\N	\N	\N
1114	CONTENT_MODIFY	go_list_btn	수정 완료	BUTTON	71	f	f	\N	\N	2026-02-23 10:16:54.957794	save-button	\N	SUBMIT	/api/execute/UPDATE_CONTENT_DETAIL	UPDATE_CONTENT_DETAIL	\N	\N	\N	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
1000	REGISTER_PAGE	REG_CONTAINER	회원가입 메인 컨테이너	CONTAINER	1	f	t	\N	\N	2026-02-21 17:33:38.324058	REG_CONTAINER	\N	\N	\N	\N	\N	\N	\N	REG_CONTAINER	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
862	CONTENT_DETAIL	go_list_btn	목록으로 돌아가기	BUTTON	93	f	f	\N	\N	2026-02-21 18:50:04.313187	go_list_btn	\N	ROUTE	/view/CONTENT_LIST	\N	\N	\N	\N	DIARYWRITE_BTN_GROUP	COLUMN	\N	\N	\N	DIARYWRITE_BTN_GROUP	true	{}	\N	\N	\N
851	CONTENT_DETAIL	label_contentTitle	제목	TEXT	20	f	f	\N	\N	2026-02-21 18:50:04.313187	content-label	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
1021	REGISTER_PAGE	ADDR_SEARCH_ROW	우편번호 검색줄	ROW	1	f	f	\N	\N	2026-02-21 17:33:38.324058	ADDR_SEARCH_ROW	\N	\N	\N	\N	\N	\N	\N	ADDR_SEARCH_ROW	horizontal	\N	\N	\N	ADDR_SECTION	true	{}	\N	\N	\N
281	MAIN_PAGE	login_card_title	로그인 하러가기	TEXT	5	f	t	\N	\N	2026-02-22 19:39:59.124085	card-title	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	LOGIN_LEFT_CONTENT	false	{}	\N	\N	\N
280	MAIN_PAGE	top_card_title	오늘의 약속 시간은 언제인가요?	TEXT	5	f	t	\N	\N	2026-02-22 19:39:59.124085	card-title	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	TOP_LEFT_CONTENT	false	{}	\N	\N	\N
282	MAIN_PAGE	tutorial_card_title	튜토리얼 보기	TEXT	5	f	t	\N	\N	2026-02-22 19:39:59.124085	card-title	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	TUTORIAL_LEFT_CONTENT	false	{}	\N	\N	\N
608	CONTENT_LIST	list_item_author	작성자	TEXT	2	f	t	\N	\N	2026-01-18 00:22:19.623286	contentContent	\N	\N	\N	\N	\N	\N	user_id	\N	COLUMN	\N	\N	\N	TITLE_AUTHOR_GROUP	true	{}	\N	\N	\N
853	CONTENT_DETAIL	contentContent		TEXTAREA	30	f	t	\N	\N	2026-02-21 18:50:04.313187	contentTextarea	\N	\N	\N	\N	\N	\N	content	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
200	SIDE_MENU	menu_main	메인페이지	MENU_ITEM	1	f	f	\N	\N	2026-01-13 03:13:37.484183	menu_main	\N	LINK	/view/MAIN_PAGE	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
857	CONTENT_DETAIL	contentEmotion	오늘의 감정	EMOTION_SELECT	52	f	t	\N	\N	2026-02-21 18:50:04.313187	emotionSelect	\N	\N	\N	\N	\N	\N	emotion	\N	COLUMN	\N	\N	\N	EMOTION_SUB_GROUP	true	{}	\N	\N	\N
859	CONTENT_DETAIL	dayTag1	태그 1	INPUT	72	f	t	\N	\N	2026-02-21 18:50:04.313187	contentInput	\N	\N	\N	\N	\N	\N	day_tag1	\N	COLUMN	\N	\N	\N	DAYTAG_SUB_GROUP	true	{}	\N	\N	\N
201	SIDE_MENU	menu_tutorial	튜토리얼	MENU_ITEM	2	f	f	\N	\N	2026-01-13 03:13:41.983994	menu_tutorial	\N	LINK	/view/TUTORIAL_PAGE	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
706	CONTENT_WRITE	EMOTION_SUB_GROUP	감정 입력 그룹	GROUP	60	f	f	\N	\N	2026-01-23 21:17:18.186064	write_sub_group	\N	\N	\N	\N	\N	\N	\N	EMOTION_SUB_GROUP	COLUMN	\N	\N	\N	DIARYWRITE_SECTION	true	{}	\N	\N	\N
850	CONTENT_DETAIL	DETAIL_SECTION		GROUP	1	f	f	\N	\N	2026-02-21 18:50:04.313187	write_section1	\N	\N	\N	\N	\N	\N	\N	DETAIL_SECTION	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
860	CONTENT_DETAIL	dayTag2	태그 2	INPUT	82	f	t	\N	\N	2026-02-21 18:50:04.313187	contentInput	\N	\N	\N	\N	\N	\N	day_tag2	\N	COLUMN	\N	\N	\N	DAYTAG_SUB_GROUP	true	{}	\N	\N	\N
1024	REGISTER_PAGE	reg_road_addr	도로명 주소	INPUT	2	f	t	\N	\N	2026-02-21 17:33:38.324058	reg_road_addr	\N	\N	\N	\N	\N	\N	roadAddress	\N	COLUMN	\N	\N	\N	ADDR_SECTION	true	{}	\N	\N	\N
1022	REGISTER_PAGE	reg_zipcode	우편번호	INPUT	1	t	t	\N	\N	2026-02-21 17:33:38.324058	reg_zipcode	\N	\N	\N	\N	\N	\N	zipCode	\N	COLUMN	\N	\N	\N	ADDR_SEARCH_ROW	true	{}	\N	\N	\N
861	CONTENT_DETAIL	dayTag3	태그 3	INPUT	92	f	t	\N	\N	2026-02-21 18:50:04.313187	contentInput	\N	\N	\N	\N	\N	\N	day_tag3	\N	COLUMN	\N	\N	\N	DAYTAG_SUB_GROUP	true	{}	\N	\N	\N
1012	REGISTER_PAGE	reg_pw	비밀번호	INPUT	2	t	f	\N	\N	2026-02-21 17:33:38.324058	reg_pw	\N	\N	\N	\N	\N	\N	password	\N	COLUMN	\N	\N	\N	INFO_SECTION	true	{"type": "password"}	\N	\N	\N
1013	REGISTER_PAGE	reg_phone	핸드폰 번호	INPUT	3	f	f	\N	010-0000-0000	2026-02-21 17:33:38.324058	reg_phone	\N	\N	\N	\N	\N	\N	phone	\N	COLUMN	\N	\N	\N	INFO_SECTION	true	{}	\N	\N	\N
1031	REGISTER_PAGE	reg_submit	회원 가입 완료	BUTTON	1	f	f	\N	\N	2026-02-21 17:33:38.324058	reg_submit	\N	REGISTER_SUBMIT	/api/auth/register	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	ACTION_SECTION	true	{"variant": "primary", "fullWidth": true}	\N	\N	\N
1023	REGISTER_PAGE	reg_addr_btn	주소 찾기	BUTTON	2	f	f	\N	\N	2026-02-21 17:33:38.324058	reg_addr_btn	\N	OPEN_POSTCODE	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	ADDR_SEARCH_ROW	true	{}	\N	\N	\N
601	CONTENT_LIST	go_write_btn	새 콘텐츠 쓰기	BUTTON	0	f	f	\N	\N	2026-02-03 04:38:57.542742	write-btn	\N	LINK	/view/CONTENT_WRITE	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	LIST_SECTION	true	{}	\N	\N	\N
1033	CONTENT_DETAIL	content_detail_source	상세 데이터 소스	DATA_SOURCE	0	f	t	\N	\N	2026-02-22 09:18:35.850523	content_detail_source	\N	AUTO_FETCH	\N	GET_CONTENT_DETAIL	/api/execute/GET_DIARY_DETAIL	\N	\N	\N	COLUMN	\N	\N	\N	\N	false	{}	\N	\N	\N
852	CONTENT_DETAIL	contentTitle	제목	INPUT	21	f	t	\N	\N	2026-02-21 18:50:04.313187	contentInput	\N	\N	\N	\N	\N	\N	title	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
1200	VERIFY_CODE_PAGE	VERIFY_CONTAINER	이메일 인증	CONTAINER	1	f	t	\N	\N	2026-02-24 16:44:46.38388	verify-container	\N	\N	\N	\N	\N	\N	\N	VERIFY_CONTAINER	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
606	CONTENT_LIST	list_item_title	제목	TEXT	1	f	t	\N	\N	2026-01-18 00:22:34.120928	contentTitle	\N	\N	\N	\N	\N	\N	title	\N	COLUMN	\N	\N	\N	TITLE_AUTHOR_GROUP	true	{}	\N	\N	\N
605	CONTENT_LIST	TITLE_AUTHOR_GROUP	제목작성자묶음	GROUP	1	f	t	\N	\N	2026-02-15 16:50:42.817815	title-author-wrapper	\N	\N	\N	\N	\N	\N	\N	TITLE_AUTHOR_GROUP	ROW	\N	\N	\N	CONTENT_CARD_HEADER	true	{}	\N	\N	\N
310	MAIN_PAGE	MAIN_TOP_CARD	메인탑카드	GROUP	20	f	t	\N	\N	2026-02-22 18:13:25.54212	main-card-item top-full	\N	\N	\N	\N	\N	\N	\N	MAIN_TOP_CARD	COLUMN	\N	\N	\N	MAIN_SECTION	true	{}	\N	\N	\N
311	MAIN_PAGE	MAIN_LOGIN_CARD	로그인카드	GROUP	30	f	t	\N	\N	2026-02-22 18:13:25.54212	main-card-item sub-half	\N	\N	\N	\N	\N	\N	\N	MAIN_LOGIN_CARD	COLUMN	\N	\N	\N	MAIN_SECTION	true	{}	\N	\N	\N
312	MAIN_PAGE	MAIN_TUTORIAL_CARD	튜토리얼카드	GROUP	40	f	t	\N	\N	2026-02-22 18:13:25.54212	main-card-item sub-half	\N	\N	\N	\N	\N	\N	\N	MAIN_TUTORIAL_CARD	COLUMN	\N	\N	\N	MAIN_SECTION	true	{}	\N	\N	\N
1201	VERIFY_CODE_PAGE	reg_email	가입 이메일	INPUT	1	f	t	\N	가입하신 이메일입니다	2026-02-24 16:44:46.38388	verify-input-readonly	\N	\N	\N	\N	\N	\N	email	\N	COLUMN	\N	\N	\N	VERIFY_CONTAINER	true	{}	\N	\N	\N
1202	VERIFY_CODE_PAGE	reg_code	인증 번호	INPUT	2	f	f	\N	메일로 발송된 6자리 번호를 입력하세요	2026-02-24 16:44:46.38388	verify-input-active	\N	\N	\N	\N	\N	\N	code	\N	COLUMN	\N	\N	\N	VERIFY_CONTAINER	true	{}	\N	\N	\N
906	SET_TIME_PAGE	back_btn	취소	BUTTON	7	f	f	\N	\N	2026-01-26 12:32:28.91547	cancel-button	\N	NAVIGATE	/view/MAIN_PAGE	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	SET_TIME_SECTION	true	{}	\N	\N	\N
610	CONTENT_LIST	content_pagination	페이지네이션	PAGINATION	100	f	t	\N	\N	2026-02-19 16:45:05.2768	content-pagination	\N	\N	\N	\N	\N	\N	content_total_count	\N	COLUMN	\N	\N	\N	LIST_SECTION	true	{}	\N	\N	\N
1025	REGISTER_PAGE	reg_detail_addr	상세 주소	INPUT	3	f	f	\N	나머지 주소를 입력하세요	2026-02-21 17:33:38.324058	reg_detail_addr	\N	\N	\N	\N	\N	\N	detailAddress	\N	COLUMN	\N	\N	\N	ADDR_SECTION	true	{}	\N	\N	\N
1204	VERIFY_CODE_PAGE	resend_btn	인증 번호 재발송	LINK	4	f	f	\N	\N	2026-02-24 16:44:46.38388	resend-link	\N	RESEND_CODE	/api/auth/resend-code	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	VERIFY_CONTAINER	true	{}	\N	\N	\N
1203	VERIFY_CODE_PAGE	verify_submit	인증 완료	BUTTON	3	f	f	\N	\N	2026-02-24 16:44:46.38388	verify-submit-btn	\N	VERIFY_CODE	/api/auth/verify-code	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	VERIFY_CONTAINER	true	{}	\N	\N	\N
905	SET_TIME_PAGE	save_time_btn	시간정하기	BUTTON	6	f	f	\N	\N	2026-01-26 12:32:23.775178	save-button	\N	SUBMIT	/api/goalTime/save	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	SET_TIME_SECTION	true	{"next_url": "/view/MAIN_PAGE"}	\N	\N	\N
858	CONTENT_DETAIL	DAYTAG_SUB_GROUP		GROUP	40	f	f	\N	\N	2026-02-21 18:50:04.313187	write_sub_group	\N	\N	\N	\N	\N	\N	\N	DAYTAG_SUB_GROUP	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
856	CONTENT_DETAIL	EMOTION_SUB_GROUP		GROUP	60	f	f	\N	\N	2026-02-21 18:50:04.313187	write_sub_group	\N	\N	\N	\N	\N	\N	\N	EMOTION_SUB_GROUP	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
854	CONTENT_DETAIL	sleep_time_select	수면 시간	TIME_SELECT	10	f	t	\N	\N	2026-02-21 18:50:04.313187	time-select-wrapper	\N	\N	\N	\N	\N	\N	selected_times	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
1036	ADDITIONAL_INFO_PAGE	HEADER_TEXT	추가 정보를 입력해주세요	TEXT	10	f	t	\N	\N	2026-03-01 13:06:30.627043	text-xl font-bold mb-6 text-center	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
1037	ADDITIONAL_INFO_PAGE	PHONE_INPUT	전화번호	INPUT	20	t	f	\N	010-1234-5678	2026-03-01 13:06:30.627043	mb-4	\N	\N	\N	\N	\N	\N	phone	\N	COLUMN	\N	\N	\N	\N	true	{"type": "tel"}	\N	\N	\N
1038	ADDITIONAL_INFO_PAGE	ADDRESS_GROUP	주소	ADDRESS_SEARCH_GROUP	30	t	f	\N	\N	2026-03-01 13:06:30.627043	mb-4	\N	\N	\N	\N	\N	\N	ADDRESS_GROUP	\N	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
1039	ADDITIONAL_INFO_PAGE	SUBMIT_BTN	제출하기	BUTTON	40	f	f	\N	\N	2026-03-01 13:06:30.627043	btn-primary w-full mt-6	\N	SUBMIT_ADDITIONAL_INFO	/api/auth/update-profile	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	\N	true	{"variant": "primary"}	\N	\N	\N
1040	CONTENT_LIST	ADMIN_DELETE_ALL_BTN	전체 삭제	BUTTON	999	f	t	\N	\N	2026-03-01 13:08:08.183416	btn-danger	\N	DELETE_ALL_CONTENTS	/api/content/delete-all	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	\N	true	{}	ROLE_ADMIN	\N	\N
855	CONTENT_DETAIL	daily_routine_section	일과 기록	TIME_SLOT_RECORD	50	f	t	\N	\N	2026-02-21 18:50:04.313187	time-slot-container	\N	\N	\N	\N	\N	\N	daily_slots	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
500	RECORD_TIME_COMPONENT	MAIN_CLOCK_SECTION	지각방지 섹션	GROUP	1	f	t	\N	\N	2026-01-26 06:54:14.013873	time-record-container	\N	\N	\N	\N	\N	\N	\N	MAIN_CLOCK_SECTION	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
502	RECORD_TIME_COMPONENT	ACTIVE_INFO_GROUP	실시간 정보 그룹	GROUP	2	f	t	\N	\N	2026-01-26 06:54:24.681945	active-display-row	\N	\N	\N	\N	\N	\N	\N	ACTIVE_INFO_GROUP	ROW	\N	\N	\N	MAIN_CLOCK_SECTION	false	{}	\N	\N	\N
863	CONTENT_DETAIL	go_modify_btn	수정하기	BUTTON	71	f	f	\N	\N	2026-02-22 20:47:11.801275	go_modify_btn	\N	ROUTE_MODIFY	/view/CONTENT_MODIFY	\N	\N	\N	\N	 	COLUMN	\N	\N	\N	DIARYWRITE_BTN_GROUP	true	{}	\N	\N	\N
300	MAIN_PAGE	MAIN_SECTION	메인 전체 섹션	GROUP	0	f	t	\N	\N	2026-01-17 14:37:18.986831	main-responsive-grid	\N	\N	\N	\N	\N	\N	\N	MAIN_SECTION	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
320	MAIN_PAGE	TOP_LEFT_CONTENT	왼쪽	GROUP	10	f	t	\N	\N	2026-02-22 19:02:28.175262	card-left-area	\N	\N	\N	\N	\N	\N	\N	TOP_LEFT_CONTENT	COLUMN	\N	\N	\N	MAIN_TOP_CARD	true	{}	\N	\N	\N
702	CONTENT_WRITE	contentTitle	제목	INPUT	21	f	f	\N	제목을 입력하세요	2026-01-22 06:39:44.17085	contentInput	\N	\N	\N	\N	\N	\N	title	\N	COLUMN	\N	\N	\N	DIARYWRITE_SECTION	true	{}	\N	\N	\N
713	CONTENT_WRITE	day_tag1	태그 1	INPUT	72	f	f	\N	#tag1	2026-01-23 22:01:08.909154	contentInput	\N	\N	\N	\N	\N	\N	day_tag1	GROUP_TAGS_ONELINE	ROW	\N	\N	\N	GROUP_TAGS_ONELINE	true	{}	\N	\N	\N
716	CONTENT_WRITE	day_tag2	태그 2	INPUT	82	f	f	\N	#tag2	2026-01-23 22:10:02.955892	contentInput	\N	\N	\N	\N	\N	\N	day_tag2	GROUP_TAGS_ONELINE	ROW	\N	\N	\N	GROUP_TAGS_ONELINE	true	{}	\N	\N	\N
719	CONTENT_WRITE	day_tag3	태그 3	INPUT	92	f	f	\N	#tag3	2026-01-23 22:10:02.955892	contentInput	\N	\N	\N	\N	\N	\N	day_tag3	GROUP_TAGS_ONELINE	ROW	\N	\N	\N	GROUP_TAGS_ONELINE	true	{}	\N	\N	\N
1043	CONTENT_LIST	content_list_container	콘텐츠리스트	GROUP	10	f	t	\N	\N	2026-03-02 17:00:47.273442	content-list-container	\N	\N	\N	\N	\N	\N	content_list_source	content_list_group	COLUMN	\N	\N	\N	content_list_group	true	{}	\N	\N	\N
903	SET_TIME_PAGE	targetTime	목표 시간	DATETIME_PICKER	4	t	f	\N	목표시간 입력 yyyy-MM-dd HH:mm:ss	2026-01-31 14:25:53.841138	targetTime	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	SET_TIME_SECTION	true	{}	\N	\N	\N
609	CONTENT_LIST	content_total_count	전체 개수 조회	DATA_SOURCE	99	f	t	\N	\N	2026-01-20 03:16:29.470008	content_total_count	\N	AUTO_FETCH	\N	COUNT_CONTENT_LIST	/api/execute/COUNT_CONTENT_LIST	{}	\N	\N	COLUMN	\N	\N	\N	LIST_SECTION	true	{}	\N	\N	\N
405	LOGIN_PAGE	PW_SUB_GROUP	비밀번호 섹션	GROUP	5	f	t	\N	\N	2026-01-17 10:20:06.873916	PW_SUB_GROUP	\N	\N	\N	\N	\N	\N	\N	PW_SUB_GROUP	COLUMN	\N	\N	\N	LOGIN_SECTION	true	{}	\N	\N	\N
410	LOGIN_PAGE	SNS_SECTION	SNS 로그인	GROUP	10	f	t	\N	\N	2026-01-17 08:02:00.186332	SNS_SECTION	\N	\N	\N	\N	\N	\N	\N	SNS_SECTION	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
705	CONTENT_WRITE	daily_slots	오늘의 흐름	TIME_SLOT_RECORD	50	f	f	\N	\N	2026-02-18 11:39:34.194682	daily_routine_section	\N	\N	\N	\N	\N	\N	daily_slots	\N	COLUMN	\N	\N	\N	DIARYWRITE_SECTION	true	{"title": "나의 하루 기록", "description": "아침, 점심, 저녁 기록", "placeholders": {"lunch": "점심 식사 이후", "evening": "마무리", "morning": "기상 직후 "}}	\N	\N	\N
900	SET_TIME_PAGE	SET_TIME_SECTION	시간 설정 전체 섹션	GROUP	1	f	f	\N	\N	2026-01-26 12:31:57.674292	SET_TIME_SECTION	\N	\N	\N	\N	\N	\N	\N	SET_TIME_SECTION	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
1109	CONTENT_MODIFY	DAYTAG_SUB_GROUP		GROUP	40	f	f	\N	\N	2026-02-23 10:16:54.957794	write_sub_group	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
1107	CONTENT_MODIFY	EMOTION_SUB_GROUP		GROUP	60	f	f	\N	\N	2026-02-23 10:16:54.957794	write_sub_group	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
1108	CONTENT_MODIFY	emotion	오늘의 감정	EMOTION_SELECT	52	f	f	\N	\N	2026-02-23 10:16:54.957794	emotionSelect	\N	\N	\N	\N	\N	\N	emotion	\N	COLUMN	\N	\N	\N	EMOTION_SUB_GROUP	true	{}	\N	\N	\N
901	SET_TIME_PAGE	set_time_title	퇴근/약속 목표 시간 설정	TEXT	2	f	t	\N	\N	2026-01-26 12:32:05.163861	page-title	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	SET_TIME_SECTION	true	{}	\N	\N	\N
902	SET_TIME_PAGE	set_time_desc	오늘의 목표 시간을 선택해 주세요.	TEXT	3	f	t	\N	\N	2026-01-26 12:32:10.688515	page-desc	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	SET_TIME_SECTION	true	{}	\N	\N	\N
904	SET_TIME_PAGE	messageInput	오늘의 메모	INPUT	5	f	f	\N	오늘의 각오 한마디	2026-01-26 12:32:18.602056	time-input	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	SET_TIME_SECTION	true	{}	\N	\N	\N
1105	CONTENT_MODIFY	selected_times	수면 시간	TIME_SELECT	10	f	f	\N	\N	2026-02-23 10:16:54.957794	time-select-wrapper	\N	\N	\N	\N	\N	\N	selected_times	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
1106	CONTENT_MODIFY	daily_slots	일과 기록	TIME_SLOT_RECORD	50	f	f	\N	\N	2026-02-23 10:16:54.957794	time-slot-container	\N	\N	\N	\N	\N	\N	daily_slots	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
1101	CONTENT_MODIFY	DETAIL_SECTION		GROUP	1	f	f	\N	\N	2026-02-23 10:16:54.957794	write_section1	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
1100	CONTENT_MODIFY	content_detail_source	상세 데이터 소스	DATA_SOURCE	0	f	t	\N	\N	2026-02-23 10:16:54.957794	content_detail_source	\N	AUTO_FETCH	\N	GET_CONTENT_DETAIL	\N	\N	\N	\N	COLUMN	\N	\N	\N	\N	false	{}	\N	\N	\N
1103	CONTENT_MODIFY	contentTitle	제목	INPUT	21	f	f	\N	\N	2026-02-23 10:16:54.957794	contentInput	\N	\N	\N	\N	\N	\N	title	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
1104	CONTENT_MODIFY	content		TEXTAREA	30	f	f	\N	\N	2026-02-23 10:16:54.957794	contentTextarea	\N	\N	\N	\N	\N	\N	content	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
1110	CONTENT_MODIFY	day_tag1	태그 1	INPUT	72	f	f	\N	\N	2026-02-23 10:16:54.957794	contentInput	\N	\N	\N	\N	\N	\N	day_tag1	\N	COLUMN	\N	\N	\N	DAYTAG_SUB_GROUP	true	{}	\N	\N	\N
1111	CONTENT_MODIFY	day_tag2	태그 2	INPUT	82	f	f	\N	\N	2026-02-23 10:16:54.957794	contentInput	\N	\N	\N	\N	\N	\N	day_tag2	\N	COLUMN	\N	\N	\N	DAYTAG_SUB_GROUP	true	{}	\N	\N	\N
1112	CONTENT_MODIFY	day_tag3	태그 3	INPUT	92	f	f	\N	\N	2026-02-23 10:16:54.957794	contentInput	\N	\N	\N	\N	\N	\N	day_tag3	\N	COLUMN	\N	\N	\N	DAYTAG_SUB_GROUP	true	{}	\N	\N	\N
1102	CONTENT_MODIFY	label_contentTitle	제목	TEXT	20	f	f	\N	\N	2026-02-23 10:16:54.957794	content-label	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	DETAIL_SECTION	true	{}	\N	\N	\N
406	LOGIN_PAGE	label_pw	Password	TEXT	6	f	t	\N	\N	2026-01-17 04:58:38.911604	label_pw	\N	\N	\N	\N	\N	\N	\N	\N	ROW	\N	\N	\N	PW_SUB_GROUP	false	{}	\N	\N	\N
402	LOGIN_PAGE	label_email	Email	TEXT	2	f	t	\N	\N	2026-01-17 04:58:49.78615	label_email	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	EMAIL_SUB_GROUP	false	{}	\N	\N	\N
701	CONTENT_WRITE	label_contentTitle	제목	TEXT	20	f	f	\N	\N	2026-01-23 20:37:02.741491	content-label	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	DIARYWRITE_SECTION	true	{}	\N	\N	\N
703	CONTENT_WRITE	content	내용	TEXTAREA	30	f	f	\N	내용을 입력하세요	2026-01-22 06:39:58.895244	contentTextarea	\N	\N	\N	\N	\N	\N	content	\N	COLUMN	\N	\N	\N	DIARYWRITE_SECTION	true	{}	\N	\N	\N
409	LOGIN_PAGE	login_btn	로그인	BUTTON	9	f	f	\N	\N	2026-01-13 03:12:22.419475	login_form_button	\N	LOGIN_SUBMIT	/api/auth/login	\N	/api/auth/login	\N	\N	LOGIN_BTN_GROUP	COLUMN	\N	\N	\N	LOGIN_SECTION	true	{}	\N	\N	\N
302	MAIN_PAGE	main_illust_img	content_body.png	IMAGE	1	f	t	\N	\N	2026-01-14 06:52:17.379511	contentImg	\N	\N	\N	\N	\N	\N	\N	 	COLUMN	\N	\N	\N	MAIN_TOP_CARD	true	{}	\N	\N	\N
305	MAIN_PAGE	go_content_btn	콘텐츠 작성하기	BUTTON	30	f	f	\N	\N	2026-01-14 10:07:48.424413	content-nav1	\N	LINK	/view/CONTENT_WRITE	GET_MEMBER_CONTENT_LIST	/api/common/fetch-data	\N	\N	 	COLUMN	\N	\N	\N	TOP_LEFT_CONTENT	true	{}	\N	\N	\N
303	MAIN_PAGE	go_login_btn	로그인 하러가기	BUTTON	20	f	f	\N	\N	2026-01-17 14:31:17.619384	content-btn-primary	\N	LINK	/view/LOGIN_PAGE	\N	\N	\N	\N	 	COLUMN	\N	\N	\N	LOGIN_LEFT_CONTENT	true	{}	\N	\N	\N
304	MAIN_PAGE	go_tutorial_btn	튜토리얼 보기	BUTTON	20	f	f	\N	\N	2026-01-17 14:31:21.489734	content-btn-secondary	\N	LINK	/view/TUTORIAL_PAGE	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	TUTORIAL_LEFT_CONTENT	true	{}	\N	\N	\N
306	MAIN_PAGE	view_content_list_btn	콘텐츠 보러가기	BUTTON	20	f	f	\N	\N	2026-01-18 00:12:24.454526	content-nav1	\N	LINK	/view/CONTENT_LIST	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	TOP_LEFT_CONTENT	true	{}	\N	\N	\N
712	CONTENT_WRITE	title_dayTag1	하루태그1	TEXT	71	f	f	\N	\N	2026-01-23 22:10:02.955892	content-label	\N	\N	\N	\N	\N	\N	\N	GROUP_TAG_ROW1	COLUMN	\N	\N	\N	GROUP_TAG_ROW1	true	{}	\N	\N	\N
104	GLOBAL_HEADER	header_kakao_logout	카카오 로그아웃	SNS_BUTTON	14	f	f	\N	\N	2026-02-16 13:56:38.530441	kakao_logout_button	\N	KAKAO_LOGOUT	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	HEADER_SECTION	true	{}	\N	\N	\N
403	LOGIN_PAGE	user_email	ID	INPUT	3	t	f	\N	아이디 입력	2026-01-13 03:12:12.706028	login_form-input	\N	\N	\N	\N	\N	\N	\N	\N	ROW	\N	\N	\N	EMAIL_SUB_GROUP	true	{}	\N	\N	\N
103	GLOBAL_HEADER	header_general_logout	로그아웃	BUTTON	13	f	t	\N	\N	2026-02-16 13:56:32.593884	logout_button	\N	LOGOUT	/api/auth/logout	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	HEADER_SECTION	true	{}	\N	\N	\N
400	LOGIN_PAGE	LOGIN_SECTION	로그인 섹션	GROUP	0	f	t	\N	\N	2026-01-17 08:08:24.637217	LOGIN_SECTION	\N	\N	\N	\N	\N	\N	\N	LOGIN_SECTION	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
401	LOGIN_PAGE	EMAIL_SUB_GROUP	이메일 입력 그룹	GROUP	1	f	t	\N	\N	2026-01-17 08:02:04.332134	EMAIL_SUB_GROUP	\N	\N	\N	\N	\N	\N	\N	EMAIL_SUB_GROUP	ROW	\N	\N	\N	LOGIN_SECTION	true	{}	\N	\N	\N
404	LOGIN_PAGE	user_email_domain	@	EMAIL_SELECT	4	t	t	\N	\N	2026-01-17 04:32:18.542272	user_email_domain	\N	\N	\N	\N	\N	\N	\N	\N	ROW	\N	\N	\N	EMAIL_SUB_GROUP	true	{}	\N	\N	\N
407	LOGIN_PAGE	user_pw	Password	PASSWORD	7	t	f	\N	비밀번호를 입력하세요	2026-01-13 03:12:18.54615	login_form-input	\N	\N	\N	\N	\N	\N	\N	\N	ROW	\N	\N	\N	PW_SUB_GROUP	true	{}	\N	\N	\N
408	LOGIN_PAGE	pw_toggle_btn	보이기	BUTTON	8	f	t	\N	\N	2026-01-17 04:58:34.801551	pw_toggle_btn	\N	TOGGLE_PW	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	PW_SUB_GROUP	true	{}	\N	\N	\N
501	RECORD_TIME_COMPONENT	no_goal_msg	오늘의 약속 시간은 언제인가요?	TEXT	1	f	t	\N	\N	2026-01-26 12:03:21.48682	no-goal-container	\N	NAVIGATE	/view/SET_TIME_PAGE	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	MAIN_CLOCK_SECTION	true	{}	\N	\N	\N
100	GLOBAL_HEADER	HEADER_SECTION	메뉴 수정	GROUP	0	f	t	\N	\N	2026-02-16 13:56:25.750512	HEADER_SECTION	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
101	GLOBAL_HEADER	header_logo	JustSaying	LINK	1	f	t	\N	\N	2026-02-16 13:56:49.840582	header_logo	\N	ROUTE	/view/MAIN_PAGE	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	HEADER_SECTION	true	{}	\N	\N	\N
102	GLOBAL_HEADER	header_login_btn	로그인	LINK	12	f	t	\N	\N	2026-02-16 14:42:13.687615	header_login_btn	\N	ROUTE	/view/LOGIN_PAGE	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	HEADER_SECTION	true	{}	\N	\N	\N
704	CONTENT_WRITE	selected_times	수면 시간 기록	TIME_SELECT	10	f	f	\N	\N	2026-02-17 11:22:24.682063	sleep_time_select	\N	\N	\N	\N	\N	\N	selected_times	\N	COLUMN	\N	\N	\N	DIARYWRITE_SECTION	true	{"endHour": 24, "startHour": 0, "slidesPerView": 6}	\N	\N	\N
412	LOGIN_PAGE	join_btn	회원가입	BUTTON	12	f	f	\N	\N	2026-01-13 03:13:23.080997	signup-nav	\N	LINK	/view/REGISTER_PAGE	\N	\N	\N	\N	JOIN_GROUP	COLUMN	\N	\N	\N	LOGIN_SECTION	true	{}	\N	\N	\N
503	RECORD_TIME_COMPONENT	remainTimeCountdown	남은 시간	COUNTDOWN	1	f	t	\N	\N	2026-01-26 06:54:29.835625	countdown-timer	\N	\N	\N	\N	/api/time/remain	\N	\N	\N	COLUMN	\N	\N	\N	ACTIVE_INFO_GROUP	true	{}	\N	\N	\N
504	RECORD_TIME_COMPONENT	ACTION_BTN_GROUP	버튼 그룹	GROUP	2	f	t	\N	\N	2026-01-26 06:54:34.518264	action-button-box	\N	\N	\N	\N	\N	\N	\N	ACTION_BTN_GROUP	COLUMN	\N	\N	\N	ACTIVE_INFO_GROUP	true	{}	\N	\N	\N
505	RECORD_TIME_COMPONENT	arrival_btn	도착 완료	BUTTON	1	f	t	\N	\N	2026-01-26 06:54:40.956846	arrival-button	\N	SUBMIT	/api/arrival/check	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	ACTION_BTN_GROUP	true	{}	\N	\N	\N
506	RECORD_TIME_COMPONENT	list_more_btn	...	BUTTON	2	f	t	\N	\N	2026-01-26 06:54:45.888398	more-button	\N	TOGGLE_LIST	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	ACTION_BTN_GROUP	true	{}	\N	\N	\N
604	CONTENT_LIST	CONTENT_CARD_HEADER	카드 상단 영역	GROUP	1	f	t	\N	\N	2026-02-04 15:31:58.565112	content-card-header	\N	\N	\N	\N	\N	\N	\N	CONTENT_CARD_HEADER	ROW	\N	\N	\N	CONTENT_CARD	true	{}	\N	\N	\N
602	CONTENT_LIST	content_list_source	콘텐츠 목록 데이터	DATA_SOURCE	0	f	t	\N	\N	2026-01-18 00:21:35.43999	content_list_source	\N	AUTO_FETCH	\N	GET_CONTENT_LIST_PAGE	/api/execute/GET_CONTENT_LIST_PAGE	{"pageSize": 5, "offset": 0, "filterId": ""}	\N	\N	COLUMN	\N	\N	\N	LIST_SECTION	true	{}	\N	\N	\N
700	CONTENT_WRITE	CONTENTWRITE_SECTION	콘텐츠쓰기 섹션	GROUP	1	f	f	\N	\N	2026-01-23 21:00:59.185022	write_section1	\N	\N	\N	\N	\N	\N	\N	DIARYWRITE_SECTION	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
720	CONTENT_WRITE	save_btn	저장하기	BUTTON	70	f	f	\N	\N	2026-01-22 06:40:11.564646	save-button	\N	SUBMIT	/api/execute/INSERT_CONTENT	INSERT_CONTENT	\N	\N	\N	DIARYWRITE_BTN_GROUP	COLUMN	\N	\N	\N	DIARYWRITE_SECTION	true	{}	\N	\N	\N
717	CONTENT_WRITE	GROUP_TAG_ROW3	태그3행	GROUP	90	f	t	\N	\N	2026-01-24 10:20:53.40285	GROUP_TAG_ROW3	\N	\N	\N	\N	\N	\N	\N	\N	ROW	\N	\N	\N	DAYTAG_SUB_GROUP	true	{}	\N	\N	\N
710	CONTENT_WRITE	GROUP_TAGS_ONELINE	태그한줄그룹	GROUP	61	f	f	\N	\N	2026-01-24 10:50:57.243018	GROUP_TAGS_ONELINE	\N	\N	\N	\N	\N	\N	\N	\N	ROW	\N	\N	\N	DAYTAG_SUB_GROUP	true	{}	\N	\N	\N
707	CONTENT_WRITE	label_contentEmotion	오늘의 감정은?	TEXT	51	f	f	\N	\N	2026-01-23 21:15:20.131199	content-label	\N	\N	\N	\N	\N	\N	\N	\N	COLUMN	\N	\N	\N	EMOTION_SUB_GROUP	true	{}	\N	\N	\N
711	CONTENT_WRITE	GROUP_TAG_ROW1	태그1행	GROUP	70	f	f	\N	\N	2026-01-24 10:20:53.40285	GROUP_TAG_ROW1	\N	\N	\N	\N	\N	\N	\N	\N	ROW	\N	\N	\N	DAYTAG_SUB_GROUP	true	{}	\N	\N	\N
714	CONTENT_WRITE	GROUP_TAG_ROW2	태그2행	GROUP	80	f	t	\N	\N	2026-01-24 10:20:53.40285	GROUP_TAG_ROW2	\N	\N	\N	\N	\N	\N	\N	\N	ROW	\N	\N	\N	DAYTAG_SUB_GROUP	true	{}	\N	\N	\N
708	CONTENT_WRITE	emotion	감정	EMOTION_SELECT	52	f	f	\N	감정값이 필요합니다.	2026-01-23 21:13:13.733535	emotionSelect	\N	\N	\N	\N	\N	\N	emotion	\N	ROW	\N	\N	\N	EMOTION_SUB_GROUP	true	{}	\N	\N	\N
603	CONTENT_LIST	CONTENT_CARD	콘텐츠 카드	GROUP	2	f	t	\N	\N	2026-01-18 00:22:11.602885	content-post	\N	ROUTE_DETAIL	/view/CONTENT_DETAIL	\N	\N	\N	content_list_source	CONTENT_CARD	COLUMN	\N	\N	\N	LIST_SECTION	true	{}	\N	\N	\N
600	CONTENT_LIST	LIST_SECTION	리스트 전체 섹션	GROUP	1	f	t	\N	\N	2026-01-18 00:21:58.578459	LIST_SECTION	\N	\N	\N	\N	\N	\N	\N	LIST_SECTION	COLUMN	\N	\N	\N	\N	true	{}	\N	\N	\N
411	LOGIN_PAGE	kakao_login_btn	Login with Kakao	SNS_BUTTON	11	f	f	\N	\N	2026-01-13 03:13:18.890374	kakao-button	\N	LINK	https://kauth.kakao.com/oauth/authorize?client_id=2d22c7fa1d59eb77a5162a3948a0b6fe&redirect_uri=https://yerin.duckdns.org/api/kakao/callback&response_type=code	\N	\N	\N	\N	SNS_GROUP	COLUMN	\N	\N	\N	SNS_SECTION	true	{}	\N	\N	\N
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.users (user_sqno, created_at, del_yn, detail_address, drug_using_type, email, hashed_password, password, phone, road_address, role, social_type, time_using_type, updated_at, user_id, verification_code, verification_expired_at, verify_yn, withdraw_at, zip_code) FROM stdin;
1	2026-03-03 03:58:31.322572	N	103동 804호	\N	myelin24@naver.com	4e0579712ed7cf89c57f35ea84d4ee71f8582d036c998fd08eb4cfc908c5c37a	qawsedrf12!@	01046412537	서울 관악구 승방길 66	ROLE_USER	N	\N	2026-03-03 03:59:37.450531	myelin24	1006940	\N	Y	\N	08807
3	2026-03-03 05:35:48.459212	N	103동804호	\N	myeliln12@naver.com			01046412537	서울 관악구 승방길 66	ROLE_USER	K	\N	2026-03-03 07:34:25.680636	myeliln12	\N	\N	Y	\N	08807
\.


--
-- Name: content_content_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.content_content_id_seq', 1, true);


--
-- Name: goal_settings_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.goal_settings_id_seq', 1, false);


--
-- Name: ui_metadata_ui_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.ui_metadata_ui_id_seq', 1043, true);


--
-- Name: users_user_sqno_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.users_user_sqno_seq', 3, true);


--
-- Name: content content_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.content
    ADD CONSTRAINT content_pkey PRIMARY KEY (content_id);


--
-- Name: flyway_schema_history flyway_schema_history_pk; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.flyway_schema_history
    ADD CONSTRAINT flyway_schema_history_pk PRIMARY KEY (installed_rank);


--
-- Name: goal_settings goal_settings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.goal_settings
    ADD CONSTRAINT goal_settings_pkey PRIMARY KEY (id);


--
-- Name: query_master query_master_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.query_master
    ADD CONSTRAINT query_master_pkey PRIMARY KEY (sql_key);


--
-- Name: ui_metadata ui_metadata_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ui_metadata
    ADD CONSTRAINT ui_metadata_pkey PRIMARY KEY (ui_id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (user_sqno);


--
-- Name: flyway_schema_history_s_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX flyway_schema_history_s_idx ON public.flyway_schema_history USING btree (success);


--
-- Name: idx_content_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_content_date ON public.content USING btree (date);


--
-- Name: idx_content_del_yn; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_content_del_yn ON public.content USING btree (del_yn);


--
-- Name: idx_content_user_sqno; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_content_user_sqno ON public.content USING btree (user_sqno);


--
-- Name: content fkdqvlqxqs75ruipisce1c50xvw; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.content
    ADD CONSTRAINT fkdqvlqxqs75ruipisce1c50xvw FOREIGN KEY (user_sqno) REFERENCES public.users(user_sqno);


--
-- PostgreSQL database dump complete
--

\unrestrict Qi7Sd5Eum8QAexLrtIx2OZCWwdhgqxtH0aeybHawFx9Y4OPCULUH18wEy3u1A9L

