"""
load_data.py
=============
기존 CSV 파일들을 PostgreSQL + PostGIS DB에 로드

[ 환경 변수 ]
  DATABASE_URL=postgresql://user:password@host/kride

[ 실행 ]
  python src/db/load_data.py                   # 전체 로드
  python src/db/load_data.py --table poi       # poi 테이블만
  python src/db/load_data.py --table trail
  python src/db/load_data.py --table danger
  python src/db/load_data.py --table weather

[ 로드 대상 CSV ]
  data/raw_ml/tour_poi.csv                → poi (category=tourism/food/nature)
  data/raw_ml/facility_clean.csv          → poi (category=facility)
  data/raw_ml/facility_clean_nationwide.csv → poi (category=facility)
  data/raw_ml/road_clean_nationwide.csv   → trail (trail_type=bike_road)
  data/raw_ml/district_danger_nationwide.csv → district_danger
  data/dl/kma_weather_raw/weather_asos_daily_nationwide.csv → weather_daily
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

try:
    import psycopg2
    import psycopg2.extras
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("❌ psycopg2 없음: pip install psycopg2-binary")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass

# ── 경로 ────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw_ml")
WEATHER_DIR = os.path.join(BASE_DIR, "data", "dl", "kma_weather_raw")

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def get_conn(url: str):
    conn = psycopg2.connect(url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return conn


# ── 1. tour_poi.csv → poi ───────────────────────────────────────────────────────

TOUR_CATEGORY_MAP = {
    "12": "tourism",   # 관광지
    "14": "tourism",   # 문화시설
    "15": "kculture",  # 축제공연행사
    "25": "nature",    # 여행코스
    "28": "tourism",   # 레포츠
    "32": "food",      # 숙박
    "38": "food",      # 쇼핑
    "39": "food",      # 음식점
}

def load_tour_poi(cur, path: str) -> int:
    if not os.path.exists(path):
        print(f"  ⚠️  파일 없음: {path}")
        return 0

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  tour_poi.csv: {len(df):,}행")

    rows = []
    for _, r in df.iterrows():
        lat = _float(r.get("mapy") or r.get("lat"))
        lon = _float(r.get("mapx") or r.get("lon"))
        ct  = str(r.get("contenttypeid", "12"))
        rows.append({
            "name":         str(r.get("title", ""))[:200],
            "name_en":      None,
            "category":     TOUR_CATEGORY_MAP.get(ct, "tourism"),
            "sub_category": str(r.get("cat2", ""))[:50] if pd.notna(r.get("cat2")) else None,
            "sido":         str(r.get("areacode", ""))[:20] if pd.notna(r.get("areacode")) else None,
            "sigungu":      str(r.get("sigungucode", ""))[:30] if pd.notna(r.get("sigungucode")) else None,
            "address":      str(r.get("addr1", ""))[:300] if pd.notna(r.get("addr1")) else None,
            "geom":         f"SRID=4326;POINT({lon} {lat})" if lat and lon else None,
            "source":       "tourapi",
            "image_url":    str(r.get("firstimage", ""))[:500] if pd.notna(r.get("firstimage")) else None,
        })

    _upsert_poi(cur, rows)
    return len(rows)


# ── 2. facility CSV → poi ───────────────────────────────────────────────────────

def load_facility(cur, path: str, source_tag: str) -> int:
    if not os.path.exists(path):
        print(f"  ⚠️  파일 없음: {path}")
        return 0

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  {os.path.basename(path)}: {len(df):,}행")

    rows = []
    for _, r in df.iterrows():
        lat = _float(r.get("lat") or r.get("y"))
        lon = _float(r.get("lon") or r.get("x"))
        rows.append({
            "name":         str(r.get("place_name", r.get("install_type", "자전거시설")))[:200],
            "category":     "facility",
            "sub_category": str(r.get("install_type", ""))[:50] if pd.notna(r.get("install_type", None)) else None,
            "sido":         str(r.get("sido", ""))[:20] if pd.notna(r.get("sido", None)) else None,
            "sigungu":      str(r.get("sigungu", ""))[:30] if pd.notna(r.get("sigungu", None)) else None,
            "address":      str(r.get("address", ""))[:300] if pd.notna(r.get("address", None)) else None,
            "geom":         f"SRID=4326;POINT({lon} {lat})" if lat and lon else None,
            "source":       source_tag,
            "is_24h":       bool(r.get("is_24h", False)),
        })

    _upsert_poi(cur, rows)
    return len(rows)


# ── 3. road_clean_nationwide.csv → trail ─────────────────────────────────────────

def load_trail(cur, path: str) -> int:
    if not os.path.exists(path):
        print(f"  ⚠️  파일 없음: {path}")
        return 0

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  road_clean_nationwide.csv: {len(df):,}행")

    sql = """
    INSERT INTO trail (
        name, trail_type, sido, sigungu,
        length_km, width_m, is_official, safety_index, road_type,
        start_lat, start_lon, end_lat, end_lon,
        geom, source, road_id
    ) VALUES (
        %(name)s, 'bike_road', %(sido)s, %(sigungu)s,
        %(length_km)s, %(width_m)s, %(is_official)s, %(safety_index)s, %(road_type)s,
        %(start_lat)s, %(start_lon)s, %(end_lat)s, %(end_lon)s,
        CASE WHEN %(start_lon)s IS NOT NULL AND %(start_lat)s IS NOT NULL
             THEN ST_SetSRID(ST_MakeLine(
                     ST_MakePoint(%(start_lon)s::double precision, %(start_lat)s::double precision),
                     ST_MakePoint(COALESCE(%(end_lon)s::double precision, %(start_lon)s::double precision),
                                  COALESCE(%(end_lat)s::double precision, %(start_lat)s::double precision))
                  ), 4326)
             ELSE NULL
        END,
        'road_csv', %(road_id)s
    )
    ON CONFLICT DO NOTHING
    """

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "name":         str(r.get("route_name", ""))[:200] if pd.notna(r.get("route_name", None)) else "unknown",
            "sido":         str(r.get("sido", ""))[:20] if pd.notna(r.get("sido", None)) else None,
            "sigungu":      str(r.get("sigungu", ""))[:30] if pd.notna(r.get("sigungu", None)) else None,
            "length_km":    _float(r.get("length_km")),
            "width_m":      _float(r.get("width_m")),
            "is_official":  bool(r.get("is_official", False)),
            "safety_index": _float(r.get("safety_index")),
            "road_type":    str(r.get("road_type", ""))[:50] if pd.notna(r.get("road_type", None)) else None,
            "start_lat":    _float(r.get("start_lat")),
            "start_lon":    _float(r.get("start_lon")),
            "end_lat":      _float(r.get("end_lat")),
            "end_lon":      _float(r.get("end_lon")),
            "road_id":      str(r.get("road_id", ""))[:200] if pd.notna(r.get("road_id", None)) else None,
        })

    psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    return len(rows)


# ── 4. district_danger_nationwide.csv → district_danger ─────────────────────────

def load_danger(cur, path: str) -> int:
    if not os.path.exists(path):
        print(f"  ⚠️  파일 없음: {path}")
        return 0

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  district_danger_nationwide.csv: {len(df):,}행")

    sql = """
    INSERT INTO district_danger (
        sido, sigungu, crash_count, death_count,
        severe_count, injury_count, raw_score, danger_score, data_year, source
    ) VALUES (
        %(sido)s, %(sigungu)s, %(crash_count)s, %(death_count)s,
        %(severe_count)s, %(injury_count)s, %(raw_score)s, %(danger_score)s,
        %(data_year)s, 'taas_csv'
    )
    ON CONFLICT (sido, sigungu, data_year) DO UPDATE SET
        crash_count  = EXCLUDED.crash_count,
        death_count  = EXCLUDED.death_count,
        severe_count = EXCLUDED.severe_count,
        injury_count = EXCLUDED.injury_count,
        raw_score    = EXCLUDED.raw_score,
        danger_score = EXCLUDED.danger_score
    """

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "sido":         str(r.get("sido", ""))[:20],
            "sigungu":      str(r.get("sigungu", ""))[:30],
            "crash_count":  int(r.get("crash_count", 0) or 0),
            "death_count":  int(r.get("death_count", 0) or 0),
            "severe_count": int(r.get("severe_count", 0) or 0),
            "injury_count": int(r.get("injury_count", 0) or 0),
            "raw_score":    _float(r.get("raw_score")),
            "danger_score": _float(r.get("danger_score")),
            "data_year":    int(r.get("data_year", 2024) or 2024),
        })

    psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    return len(rows)


# ── 5. weather CSV → weather_daily ───────────────────────────────────────────────

def load_weather(cur, path: str) -> int:
    if not os.path.exists(path):
        print(f"  ⚠️  파일 없음: {path}")
        return 0

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  weather CSV: {len(df):,}행")

    sql = """
    INSERT INTO weather_daily (
        stn_id, stn_name, sido, obs_date,
        avg_temp, min_temp, max_temp,
        precipitation, avg_wind_spd, avg_humidity, sunshine_hr
    ) VALUES (
        %(stn_id)s, %(stn_name)s, %(sido)s, %(obs_date)s,
        %(avg_temp)s, %(min_temp)s, %(max_temp)s,
        %(precipitation)s, %(avg_wind_spd)s, %(avg_humidity)s, %(sunshine_hr)s
    )
    ON CONFLICT (stn_id, obs_date) DO NOTHING
    """

    col = lambda c: df.columns[df.columns.str.lower() == c.lower()].tolist()

    def _colval(r, *candidates):
        for c in candidates:
            if c in r.index and pd.notna(r[c]):
                return r[c]
        return None

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "stn_id":       str(_colval(r, "stn_id", "지점번호", "지점") or ""),
            "stn_name":     str(_colval(r, "stn_name", "지점명") or ""),
            "sido":         str(_colval(r, "sido", "시도") or ""),
            "obs_date":     str(_colval(r, "obs_date", "일시", "date") or ""),
            "avg_temp":     _float(_colval(r, "avg_temp", "평균기온(°C)")),
            "min_temp":     _float(_colval(r, "min_temp", "최저기온(°C)")),
            "max_temp":     _float(_colval(r, "max_temp", "최고기온(°C)")),
            "precipitation":_float(_colval(r, "precipitation", "일강수량(mm)")),
            "avg_wind_spd": _float(_colval(r, "avg_wind_spd", "평균풍속(m/s)")),
            "avg_humidity": _float(_colval(r, "avg_humidity", "평균상대습도(%)")),
            "sunshine_hr":  _float(_colval(r, "sunshine_hr", "합계일조시간(hr)")),
        })

    psycopg2.extras.execute_batch(cur, sql, rows, page_size=1000)
    return len(rows)


# ── 공통 유틸 ────────────────────────────────────────────────────────────────────

def _float(v) -> float | None:
    try:
        f = float(v)
        return None if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return None


def _upsert_poi(cur, rows: list[dict]) -> None:
    sql = """
    INSERT INTO poi (
        name, name_en, category, sub_category,
        sido, sigungu, address, geom, source,
        image_url, is_24h
    ) VALUES (
        %(name)s, %(name_en)s, %(category)s, %(sub_category)s,
        %(sido)s, %(sigungu)s, %(address)s,
        %(geom)s::geometry,
        %(source)s, %(image_url)s, %(is_24h)s
    )
    ON CONFLICT DO NOTHING
    """
    # geom이 None이면 NULL로
    for r in rows:
        if r.get("geom") is None:
            sql_row = sql.replace("%(geom)s::geometry", "NULL")
        if "name_en" not in r:     r["name_en"]     = None
        if "sub_category" not in r: r["sub_category"] = None
        if "sido" not in r:        r["sido"]         = None
        if "sigungu" not in r:     r["sigungu"]      = None
        if "address" not in r:     r["address"]      = None
        if "image_url" not in r:   r["image_url"]    = None
        if "is_24h" not in r:      r["is_24h"]       = False

    # geom None 처리: 두 그룹으로 분리
    rows_with_geom    = [r for r in rows if r.get("geom")]
    rows_without_geom = [r for r in rows if not r.get("geom")]

    sql_with = """
    INSERT INTO poi (name, name_en, category, sub_category, sido, sigungu, address, geom, source, image_url, is_24h)
    VALUES (%(name)s, %(name_en)s, %(category)s, %(sub_category)s, %(sido)s, %(sigungu)s, %(address)s,
            ST_GeomFromEWKT(%(geom)s), %(source)s, %(image_url)s, %(is_24h)s)
    ON CONFLICT DO NOTHING
    """
    sql_without = """
    INSERT INTO poi (name, name_en, category, sub_category, sido, sigungu, address, geom, source, image_url, is_24h)
    VALUES (%(name)s, %(name_en)s, %(category)s, %(sub_category)s, %(sido)s, %(sigungu)s, %(address)s,
            NULL, %(source)s, %(image_url)s, %(is_24h)s)
    ON CONFLICT DO NOTHING
    """
    if rows_with_geom:
        psycopg2.extras.execute_batch(cur, sql_with, rows_with_geom, page_size=500)
    if rows_without_geom:
        psycopg2.extras.execute_batch(cur, sql_without, rows_without_geom, page_size=500)


# ── 메인 ────────────────────────────────────────────────────────────────────────

def main(url: str, target: str | None) -> None:
    conn = get_conn(url)
    cur  = conn.cursor()

    tasks = {
        "poi": [
            ("tour_poi",     lambda: load_tour_poi(cur, os.path.join(RAW_DIR, "tour_poi.csv"))),
            ("facility",     lambda: load_facility(cur, os.path.join(RAW_DIR, "facility_clean.csv"), "kakao_legacy")),
            ("facility_nw",  lambda: load_facility(cur, os.path.join(RAW_DIR, "facility_clean_nationwide.csv"), "kakao")),
        ],
        "trail": [
            ("trail",        lambda: load_trail(cur, os.path.join(RAW_DIR, "road_clean_nationwide.csv"))),
        ],
        "danger": [
            ("danger",       lambda: load_danger(cur, os.path.join(RAW_DIR, "district_danger_nationwide.csv"))),
        ],
        "weather": [
            ("weather",      lambda: load_weather(cur, os.path.join(WEATHER_DIR, "weather_asos_daily_nationwide.csv"))),
        ],
    }

    run_all = not target or target == "all"
    total   = 0

    for group, loaders in tasks.items():
        if not run_all and target != group:
            continue
        print(f"\n{'='*60}")
        print(f"[{group.upper()}] 로드 시작")
        print(f"{'='*60}")
        for name, fn in loaders:
            print(f"\n  [{name}]")
            try:
                n = fn()
                print(f"  → {n:,}행 처리")
                total += n or 0
            except Exception as e:
                print(f"  ❌ {name} 실패: {e}")

    cur.close()
    conn.close()

    print(f"\n{'='*60}")
    print(f"✅ 전체 로드 완료 — 총 {total:,}행 처리")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Ride CSV → PostgreSQL 로더")
    parser.add_argument("--table", choices=["poi", "trail", "danger", "weather", "all"],
                        default="all", help="로드할 테이블 그룹 (기본: all)")
    parser.add_argument("--url", default="", help="DATABASE_URL 직접 지정")
    args = parser.parse_args()

    db_url = DATABASE_URL or args.url
    if not db_url:
        print("❌ DATABASE_URL 환경변수가 없습니다.")
        print("   export DATABASE_URL=postgresql://user:pass@host/kride")
        sys.exit(1)

    main(db_url, args.table)
