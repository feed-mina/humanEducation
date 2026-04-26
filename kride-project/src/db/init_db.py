"""
init_db.py
===========
PostgreSQL + PostGIS 스키마 초기화

[ 사전 준비 ]
  pip install psycopg2-binary sqlalchemy python-dotenv

[ 환경 변수 ]
  DATABASE_URL=postgresql://user:password@host/kride   (Neon 등)

[ 실행 ]
  python src/db/init_db.py
  python src/db/init_db.py --drop   # 기존 테이블 삭제 후 재생성 (주의!)
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("❌ psycopg2 없음: pip install psycopg2-binary")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# ── DDL ────────────────────────────────────────────────────────────────────────

DDL_EXTENSIONS = """
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
"""

DDL_DROP = """
DROP TABLE IF EXISTS weather_daily      CASCADE;
DROP TABLE IF EXISTS district_danger    CASCADE;
DROP TABLE IF EXISTS road_segment       CASCADE;
DROP TABLE IF EXISTS trail              CASCADE;
DROP TABLE IF EXISTS course_template    CASCADE;
DROP TABLE IF EXISTS poi                CASCADE;
"""

DDL_POI = """
CREATE TABLE IF NOT EXISTS poi (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    name_en         VARCHAR(200),
    category        VARCHAR(30)  NOT NULL,
    -- food / kculture / tourism / nature / trail / facility
    sub_category    VARCHAR(50),
    sido            VARCHAR(20),
    sigungu         VARCHAR(30),
    address         VARCHAR(300),
    geom            GEOMETRY(POINT, 4326),
    description     TEXT,
    description_en  TEXT,
    source          VARCHAR(50),
    score           DECIMAL(3,2),
    visit_count     INTEGER     DEFAULT 0,
    image_url       VARCHAR(500),
    tags            TEXT[],
    phone           VARCHAR(30),
    url             VARCHAR(500),
    is_24h          BOOLEAN     DEFAULT FALSE,
    raw_data        JSONB,
    created_at      TIMESTAMP   DEFAULT NOW(),
    updated_at      TIMESTAMP   DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_poi_geom     ON poi USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_poi_category ON poi(category);
CREATE INDEX IF NOT EXISTS idx_poi_sido     ON poi(sido);
CREATE INDEX IF NOT EXISTS idx_poi_tags     ON poi USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_poi_name_trgm ON poi USING GIN(name gin_trgm_ops);
"""

DDL_COURSE = """
CREATE TABLE IF NOT EXISTS course_template (
    id              SERIAL PRIMARY KEY,
    title           VARCHAR(200),
    title_en        VARCHAR(200),
    duration_days   SMALLINT,
    category        VARCHAR(30),
    transport       VARCHAR(20),   -- walk/bike/public/car
    estimated_cost  INTEGER,       -- 원
    poi_ids         INTEGER[],
    route_geom      GEOMETRY(LINESTRING, 4326),
    description     TEXT,
    description_en  TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_course_geom ON course_template USING GIST(route_geom);
"""

DDL_TRAIL = """
CREATE TABLE IF NOT EXISTS trail (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200),
    trail_type      VARCHAR(20),   -- dullegil/bike_road/forest
    sido            VARCHAR(20),
    sigungu         VARCHAR(30),
    length_km       DECIMAL(8,3),
    width_m         DECIMAL(6,2),
    difficulty      SMALLINT,      -- 1=쉬움 2=보통 3=어려움
    has_stamp       BOOLEAN DEFAULT FALSE,
    is_official     BOOLEAN DEFAULT FALSE,
    safety_index    DECIMAL(6,4),
    road_type       VARCHAR(50),
    start_lat       DECIMAL(10,7),
    start_lon       DECIMAL(10,7),
    end_lat         DECIMAL(10,7),
    end_lon         DECIMAL(10,7),
    geom            GEOMETRY(LINESTRING, 4326),
    source          VARCHAR(50),
    road_id         VARCHAR(200),
    created_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trail_geom ON trail USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_trail_sido ON trail(sido);
"""

DDL_DISTRICT_DANGER = """
CREATE TABLE IF NOT EXISTS district_danger (
    id              SERIAL PRIMARY KEY,
    sido            VARCHAR(20)  NOT NULL,
    sigungu         VARCHAR(30)  NOT NULL,
    crash_count     INTEGER      DEFAULT 0,
    death_count     INTEGER      DEFAULT 0,
    severe_count    INTEGER      DEFAULT 0,
    injury_count    INTEGER      DEFAULT 0,
    raw_score       DECIMAL(10,2),
    danger_score    DECIMAL(6,4),  -- MinMaxScaler 정규화 0~1
    geom            GEOMETRY(POINT, 4326),
    data_year       SMALLINT,
    source          VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (sido, sigungu, data_year)
);
CREATE INDEX IF NOT EXISTS idx_danger_geom   ON district_danger USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_danger_sido   ON district_danger(sido);
CREATE INDEX IF NOT EXISTS idx_danger_score  ON district_danger(danger_score DESC);
"""

DDL_WEATHER = """
CREATE TABLE IF NOT EXISTS weather_daily (
    id              SERIAL PRIMARY KEY,
    stn_id          VARCHAR(10)  NOT NULL,
    stn_name        VARCHAR(30),
    sido            VARCHAR(20),
    obs_date        DATE         NOT NULL,
    avg_temp        DECIMAL(5,1),
    min_temp        DECIMAL(5,1),
    max_temp        DECIMAL(5,1),
    precipitation   DECIMAL(7,1),
    avg_wind_spd    DECIMAL(5,1),
    avg_humidity    DECIMAL(5,1),
    sunshine_hr     DECIMAL(5,1),
    geom            GEOMETRY(POINT, 4326),
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (stn_id, obs_date)
);
CREATE INDEX IF NOT EXISTS idx_weather_date ON weather_daily(obs_date);
CREATE INDEX IF NOT EXISTS idx_weather_stn  ON weather_daily(stn_id);
"""

ALL_DDL = [
    DDL_POI,
    DDL_COURSE,
    DDL_TRAIL,
    DDL_DISTRICT_DANGER,
    DDL_WEATHER,
]


def get_conn(url: str):
    conn = psycopg2.connect(url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return conn


def init_schema(url: str, drop_first: bool = False) -> None:
    print("=" * 60)
    print("PostgreSQL + PostGIS 스키마 초기화")
    print("=" * 60)

    conn = get_conn(url)
    cur = conn.cursor()

    # PostGIS 확장
    print("\n[1] 확장 설치 (postgis, pg_trgm)")
    cur.execute(DDL_EXTENSIONS)
    print("  ✅ 확장 설치 완료")

    # DROP (옵션)
    if drop_first:
        print("\n[2] 기존 테이블 삭제 (--drop 옵션)")
        cur.execute(DDL_DROP)
        print("  ✅ 테이블 삭제 완료")

    # CREATE
    print("\n[3] 테이블 생성")
    table_names = ["poi", "course_template", "trail", "district_danger", "weather_daily"]
    for ddl, name in zip(ALL_DDL, table_names):
        cur.execute(ddl)
        print(f"  ✅ {name}")

    cur.close()
    conn.close()

    print("\n" + "=" * 60)
    print("✅ 스키마 초기화 완료")
    print("   다음 단계: python src/db/load_data.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Ride DB 스키마 초기화")
    parser.add_argument("--drop", action="store_true",
                        help="기존 테이블을 삭제하고 재생성 (데이터 전체 삭제 주의)")
    parser.add_argument("--url", default="", help="DATABASE_URL 직접 지정 (env 우선)")
    args = parser.parse_args()

    db_url = DATABASE_URL or args.url
    if not db_url:
        print("❌ DATABASE_URL 환경변수가 없습니다.")
        print("   export DATABASE_URL=postgresql://user:pass@host/kride")
        sys.exit(1)

    init_schema(db_url, drop_first=args.drop)
