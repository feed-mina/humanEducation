"""
collect_tourapi_food.py
=======================
한국관광공사 TourAPI → 음식점(contentTypeId=39) 전국 수집

[ 수집 전략 ]
  TourAPI areaBasedList2 — contentTypeId=39(음식점) 전국 17개 시도 × 전체 페이지
  소상공인 803K 건과 별개로 관광공사 등록 식당(인증/추천 위주) 수집.
  기존 food_poi에 source='tourapi_food' 로 병합 가능.

[ 출력 ]
  data/raw_ml/tourapi_food_nationwide.csv
    컬럼: contentid, name, address, sido, lat, lon, sub_category,
           cat1, cat2, cat3, tel, image_url, source

[ 실행 ]
  python src/data_collect/collect_tourapi_food.py
  python src/data_collect/collect_tourapi_food.py --db     # DB 즉시 적재
  python src/data_collect/collect_tourapi_food.py --sido 서울 경기   # 특정 시도만
"""

from __future__ import annotations

import argparse
import os
import random
import time

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass

# ── 경로 설정 ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw_ml")
os.makedirs(RAW_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(RAW_DIR, "tourapi_food_nationwide.csv")

# ── API 설정 ─────────────────────────────────────────────────────────────────
_FALLBACK_KEY = "1982f962b3451ad1a449051bf6266ac560540e346678c51933ae885bc2b4a95e"
API_KEY = (
    os.environ.get("TOUR_API_KEY")
    or os.environ.get("KMA_API_KEY")
    or _FALLBACK_KEY
)

AREA_URL      = "https://apis.data.go.kr/B551011/KorService2/areaBasedList2"
TIMEOUT       = 60
MAX_RETRY     = 3
ROWS_PER_PAGE = 1000

# ── 전국 areaCode ────────────────────────────────────────────────────────────
AREA_CODES: dict[str, int] = {
    "서울":  1, "인천":  2, "대전":  3, "대구":  4, "광주":  5,
    "부산":  6, "울산":  7, "세종":  8, "경기": 31, "강원": 32,
    "충북": 33, "충남": 34, "경북": 35, "경남": 36, "전북": 37,
    "전남": 38, "제주": 39,
}

# TourAPI cat3 → sub_category 매핑 (음식점, A05 계열)
CAT3_MAP: dict[str, str] = {
    "A05020100": "한식",
    "A05020200": "서양식",
    "A05020300": "일식",
    "A05020400": "중식",
    "A05020600": "이색음식점",
    "A05020700": "체험음식",
    "A05020900": "카페·디저트",
    "A05021000": "뷔페",
    "A05030100": "카페",
    "A05030200": "전통찻집",
}


# ── API 유틸 ─────────────────────────────────────────────────────────────────
def _api_call(url: str, params: dict) -> dict | None:
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"    ⚠️ 타임아웃 (시도 {attempt}/{MAX_RETRY}) — {wait:.1f}초 후 재시도")
            if attempt < MAX_RETRY:
                time.sleep(wait)
        except requests.exceptions.ConnectionError:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"    ⚠️ 연결 오류 (시도 {attempt}/{MAX_RETRY}) — {wait:.1f}초 후 재시도")
            if attempt < MAX_RETRY:
                time.sleep(wait)
        except Exception as e:
            print(f"    ❌ 오류: {e}")
            return None
    return None


def _extract_items(data: dict | None) -> tuple[list, int]:
    if not data:
        return [], 0
    body  = data.get("response", {}).get("body", {})
    total = int(body.get("totalCount", 0))
    items_raw = body.get("items", {})
    if not items_raw:
        return [], total
    item_list = items_raw.get("item", [])
    if isinstance(item_list, dict):
        item_list = [item_list]
    return item_list, total


# ── 수집 ─────────────────────────────────────────────────────────────────────
def collect_food(target_areas: dict[str, int]) -> list[dict]:
    all_items: list[dict] = []

    for sido, area_code in target_areas.items():
        print(f"\n  📍 {sido}(areaCode={area_code}) 음식점 수집 중...", end=" ")
        params = {
            "serviceKey":    API_KEY,
            "numOfRows":     ROWS_PER_PAGE,
            "pageNo":        1,
            "MobileOS":      "ETC",
            "MobileApp":     "KRide",
            "_type":         "json",
            "areaCode":      area_code,
            "contentTypeId": 39,
            "arrange":       "A",
        }

        data         = _api_call(AREA_URL, params)
        items, total = _extract_items(data)

        if not items:
            print("0건")
            continue

        all_items.extend(items)
        total_pages = max(1, (total + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
        print(f"총 {total:,}건 / {total_pages}페이지")

        for page in range(2, total_pages + 1):
            time.sleep(0.3 + random.uniform(0, 0.2))
            params["pageNo"] = page
            pdata            = _api_call(AREA_URL, params)
            pitems, _        = _extract_items(pdata)
            if pitems:
                all_items.extend(pitems)
            print(f"    page {page}/{total_pages} (+{len(pitems) if pitems else 0}건)", end="\r")

        print(f"    {sido} 완료 ✅")
        time.sleep(0.4)

    return all_items


# ── 전처리 ───────────────────────────────────────────────────────────────────
def build_dataframe(items: list) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)

    KEEP = ["contentid", "contenttypeid", "title", "addr1", "addr2",
            "mapx", "mapy", "cat1", "cat2", "cat3", "firstimage", "tel"]
    df = df[[c for c in KEEP if c in df.columns]].copy()

    df["mapx"] = pd.to_numeric(df["mapx"], errors="coerce")
    df["mapy"] = pd.to_numeric(df["mapy"], errors="coerce")
    df = df.drop_duplicates(subset=["contentid"])

    # 한국 좌표 범위 필터
    mask = df["mapy"].between(33.0, 38.7) & df["mapx"].between(124.6, 131.9)
    df   = df[mask].copy()

    # sub_category (cat3 기반)
    df["sub_category"] = (
        df["cat3"].map(CAT3_MAP)
        if "cat3" in df.columns
        else "기타음식점"
    )
    df["sub_category"] = df["sub_category"].fillna("기타음식점")

    # 시도 추출
    def _sido(addr: str | float) -> str | None:
        if not addr or pd.isna(addr):
            return None
        tokens = str(addr).split()
        return tokens[0] if tokens else None

    df["sido"] = df["addr1"].apply(_sido) if "addr1" in df.columns else None

    df = df.rename(columns={
        "title":      "name",
        "addr1":      "address",
        "mapy":       "lat",
        "mapx":       "lon",
        "firstimage": "image_url",
    })
    df["source"] = "tourapi_food"

    return df


# ── DB 적재 ──────────────────────────────────────────────────────────────────
def load_to_db(df: pd.DataFrame) -> int:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("  ❌ psycopg2 없음 — pip install psycopg2")
        return 0

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("  ❌ DATABASE_URL 미설정")
        return 0

    sql = """
    INSERT INTO poi (name, name_en, category, sub_category, sido, sigungu,
                     address, geom, source, image_url, is_24h, raw_data)
    VALUES (%(name)s, NULL, 'food', %(sub_category)s, %(sido)s, NULL,
            %(address)s,
            CASE WHEN %(lon)s IS NOT NULL AND %(lat)s IS NOT NULL
                 THEN ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)
                 ELSE NULL END,
            'tourapi_food', %(image_url)s, FALSE, NULL)
    ON CONFLICT DO NOTHING
    """

    rows = [
        {
            "name":         str(r.get("name", ""))[:200],
            "sub_category": str(r.get("sub_category", ""))[:50],
            "sido":         str(r["sido"])[:20] if r.get("sido") else None,
            "address":      str(r["address"])[:300] if r.get("address") else None,
            "lat":          r.get("lat"),
            "lon":          r.get("lon"),
            "image_url":    str(r["image_url"])[:500] if r.get("image_url") else None,
        }
        for r in df.to_dict("records")
    ]

    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cur = conn.cursor()
    psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    cur.close()
    conn.close()
    return len(rows)


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="TourAPI 음식점 전국 수집")
    parser.add_argument("--db",   action="store_true", help="수집 후 DB 즉시 적재")
    parser.add_argument(
        "--sido", nargs="+", default=["all"],
        help="수집 시도 (기본: 전국). 예: --sido 서울 경기 부산"
    )
    args = parser.parse_args()

    if args.sido == ["all"]:
        target = AREA_CODES
    else:
        target = {s: AREA_CODES[s] for s in args.sido if s in AREA_CODES}
        unknown = [s for s in args.sido if s not in AREA_CODES]
        if unknown:
            print(f"⚠️ 알 수 없는 시도: {unknown}")

    print("=" * 65)
    print("TourAPI 음식점(contentTypeId=39) 전국 수집")
    print("=" * 65)
    print(f"수집 대상: {list(target.keys())} ({len(target)}개 시도)")
    print(f"API Key  : {'설정됨' if API_KEY != _FALLBACK_KEY else '기본값(KMA_API_KEY)'}")

    # Step 1: 수집
    print(f"\n{'─'*65}")
    print("STEP 1 — areaBasedList2 (contentTypeId=39) 전국 수집")
    print(f"{'─'*65}")
    items = collect_food(target)
    print(f"\n  수집 합계: {len(items):,}건 (중복 포함)")

    # Step 2: 전처리
    print(f"\n{'─'*65}")
    print("STEP 2 — 전처리 + 저장")
    print(f"{'─'*65}")
    df = build_dataframe(items)

    if df.empty:
        print("  ❌ 수집된 데이터 없음")
        return

    print(f"  최종 행수: {len(df):,}행")

    print("  sub_category 분포:")
    for cat, cnt in df["sub_category"].value_counts().items():
        print(f"    {cat}: {cnt:,}건")

    print("  시도 분포 (상위 10):")
    for sido, cnt in df["sido"].value_counts().head(10).items():
        print(f"    {sido}: {cnt:,}건")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] {OUTPUT_PATH}")
    print(f"       {len(df):,}행 저장 완료")

    # Step 3: DB 적재 (옵션)
    if args.db:
        print(f"\n{'─'*65}")
        print("STEP 3 — DB 적재")
        print(f"{'─'*65}")
        n = load_to_db(df)
        print(f"  → {n:,}행 DB 적재 완료")

    print(f"\n{'='*65}")
    print("[DONE] collect_tourapi_food.py 완료")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
