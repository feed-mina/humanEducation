"""
collect_kculture_poi.py
========================
한국관광공사 TourAPI → K-Culture 촬영지 / 성지순례 POI 수집

[ 수집 전략 ]
  1. TourAPI searchKeyword API — 촬영지/드라마/영화/K-pop 등 키워드 검색
  2. TourAPI areaBasedList — contentTypeId=25(여행코스) 전국 수집 후 촬영지 필터
  3. 결과 병합 + 중복 제거 → CSV 저장

[ 출력 ]
  data/raw_ml/kculture_poi_nationwide.csv

[ 실행 ]
  python src/data_collect/collect_kculture_poi.py
  python src/data_collect/collect_kculture_poi.py --db   # DB 즉시 적재
"""

from __future__ import annotations

import argparse
import os
import random
import sys
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

OUTPUT_PATH = os.path.join(RAW_DIR, "kculture_poi_nationwide.csv")

# ── API 설정 ─────────────────────────────────────────────────────────────────
_FALLBACK_KEY = "1982f962b3451ad1a449051bf6266ac560540e346678c51933ae885bc2b4a95e"
API_KEY = (
    os.environ.get("TOUR_API_KEY")
    or os.environ.get("KMA_API_KEY")
    or _FALLBACK_KEY
)
TIMEOUT       = 60
MAX_RETRY     = 3
ROWS_PER_PAGE = 1000

# TourAPI 엔드포인트
SEARCH_URL   = "https://apis.data.go.kr/B551011/KorService2/searchKeyword1"
AREA_URL     = "https://apis.data.go.kr/B551011/KorService2/areaBasedList2"
DETAIL_URL   = "https://apis.data.go.kr/B551011/KorService2/detailCommon1"

# ── 전국 areaCode ────────────────────────────────────────────────────────────
AREA_CODES = {
    "서울":  1, "인천":  2, "대전":  3, "대구":  4, "광주":  5,
    "부산":  6, "울산":  7, "세종":  8, "경기": 31, "강원": 32,
    "충북": 33, "충남": 34, "경북": 35, "경남": 36, "전북": 37,
    "전남": 38, "제주": 39,
}

# ── K-Culture 키워드 ─────────────────────────────────────────────────────────
SEARCH_KEYWORDS = [
    "촬영지", "드라마촬영지", "영화촬영지",
    "K-pop", "케이팝", "한류",
    "BTS", "방탄소년단", "블랙핑크", "BLACKPINK",
    "뮤직비디오", "MV촬영지",
    "성지순례", "팬덤",
    "촬영명소", "로케이션", "세트장",
]

# 관광지(contentTypeId=12) area-based 결과에서 K-Culture 관련성 필터
FILMING_KEYWORDS = [
    "촬영", "드라마", "영화", "뮤직비디오", "mv", "로케이션",
    "세트장", "오픈세트", "kpop", "케이팝", "한류",
    "bts", "방탄", "블랙핑크", "아이돌", "팬", "성지",
]


# ── API 호출 유틸 ────────────────────────────────────────────────────────────
def _api_call(url: str, params: dict) -> dict | None:
    """TourAPI 호출 (재시도 포함)."""
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
    """TourAPI JSON → (아이템 리스트, 전체 건수)."""
    if not data:
        return [], 0
    body = data.get("response", {}).get("body", {})
    total = int(body.get("totalCount", 0))
    items_raw = body.get("items", {})
    if not items_raw:
        return [], total
    item_list = items_raw.get("item", [])
    if isinstance(item_list, dict):
        item_list = [item_list]
    return item_list, total


# ── 1. 키워드 검색 수집 ──────────────────────────────────────────────────────
def collect_by_keyword() -> list[dict]:
    """TourAPI searchKeyword2로 K-Culture 관련 POI 수집."""
    all_items = []

    for kw in SEARCH_KEYWORDS:
        print(f"  🔍 키워드: '{kw}' 검색 중...", end=" ")
        params = {
            "serviceKey":  API_KEY,
            "numOfRows":   ROWS_PER_PAGE,
            "pageNo":      1,
            "MobileOS":    "ETC",
            "MobileApp":   "KRide",
            "_type":       "json",
            "keyword":     kw,
            "arrange":     "A",
        }

        data = _api_call(SEARCH_URL, params)
        items, total = _extract_items(data)

        if not items:
            print(f"0건")
            continue

        all_items.extend(items)
        total_pages = max(1, (total + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
        print(f"{total}건 ({total_pages}페이지)")

        for page in range(2, total_pages + 1):
            time.sleep(0.3 + random.uniform(0, 0.2))
            params["pageNo"] = page
            data = _api_call(SEARCH_URL, params)
            page_items, _ = _extract_items(data)
            if page_items:
                all_items.extend(page_items)

        time.sleep(0.5)

    print(f"  → 키워드 검색 합계: {len(all_items)}건 (중복 포함)")
    return all_items


# ── 2. 관광지(12) 전국 수집 + K-Culture 필터 ─────────────────────────────────
def collect_tourism_spots() -> list[dict]:
    """
    TourAPI areaBasedList — contentTypeId=12(관광지) 전국 수집 후
    제목에 촬영지/드라마/세트장 등 K-Culture 키워드가 포함된 항목만 반환.
    """
    all_items = []

    for sido, area_code in AREA_CODES.items():
        print(f"  📍 {sido} / 관광지(12) 수집 중...", end=" ")
        params = {
            "serviceKey":    API_KEY,
            "numOfRows":     ROWS_PER_PAGE,
            "pageNo":        1,
            "MobileOS":      "ETC",
            "MobileApp":     "KRide",
            "_type":         "json",
            "areaCode":      area_code,
            "contentTypeId": 12,
            "arrange":       "A",
        }

        data         = _api_call(AREA_URL, params)
        items, total = _extract_items(data)

        if not items:
            print("0건")
            continue

        total_pages = max(1, (total + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
        print(f"{total:,}건 ({total_pages}페이지) 수집 후 필터링 중...")

        def _is_kculture(item: dict) -> bool:
            text = (
                str(item.get("title", "")).lower()
                + str(item.get("addr1", "")).lower()
            )
            return any(kw in text for kw in FILMING_KEYWORDS)

        filtered = [it for it in items if _is_kculture(it)]
        all_items.extend(filtered)

        for page in range(2, total_pages + 1):
            time.sleep(0.3 + random.uniform(0, 0.2))
            params["pageNo"] = page
            data             = _api_call(AREA_URL, params)
            page_items, _    = _extract_items(data)
            if page_items:
                filtered_page = [it for it in page_items if _is_kculture(it)]
                all_items.extend(filtered_page)

        time.sleep(0.3)

    print(f"  → 관광지(12) K-Culture 필터 결과: {len(all_items)}건")
    return all_items


# ── 3. 여행코스(25) 전국 수집 ────────────────────────────────────────────────
def collect_travel_courses() -> list[dict]:
    """TourAPI areaBasedList — contentTypeId=25(여행코스) 전국 수집."""
    all_items = []

    for sido, area_code in AREA_CODES.items():
        print(f"  📍 {sido} / 여행코스(25) 수집 중...", end=" ")
        params = {
            "serviceKey":    API_KEY,
            "numOfRows":     ROWS_PER_PAGE,
            "pageNo":        1,
            "MobileOS":      "ETC",
            "MobileApp":     "KRide",
            "_type":         "json",
            "areaCode":      area_code,
            "contentTypeId": 25,
            "arrange":       "A",
        }

        data = _api_call(AREA_URL, params)
        items, total = _extract_items(data)

        if not items:
            print(f"0건")
            continue

        all_items.extend(items)
        total_pages = max(1, (total + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
        print(f"{total}건 ({total_pages}페이지)")

        for page in range(2, total_pages + 1):
            time.sleep(0.3 + random.uniform(0, 0.2))
            params["pageNo"] = page
            data = _api_call(AREA_URL, params)
            page_items, _ = _extract_items(data)
            if page_items:
                all_items.extend(page_items)

        time.sleep(0.3)

    print(f"  → 여행코스 합계: {len(all_items)}건")
    return all_items


# ── 4. 결과 병합 + 정리 ──────────────────────────────────────────────────────
def merge_and_clean(
    keyword_items: list,
    tourism_items: list,
    course_items: list,
) -> pd.DataFrame:
    """수집 결과 병합, 중복 제거, 좌표 필터."""
    all_items = keyword_items + tourism_items + course_items
    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)

    # 필요한 컬럼만
    KEEP = ["contentid", "contenttypeid", "title", "addr1", "addr2",
            "mapx", "mapy", "cat1", "cat2", "cat3", "firstimage", "tel"]
    available = [c for c in KEEP if c in df.columns]
    df = df[available].copy()

    # 좌표 변환
    df["mapx"] = pd.to_numeric(df["mapx"], errors="coerce")
    df["mapy"] = pd.to_numeric(df["mapy"], errors="coerce")

    # contentid 중복 제거
    df = df.drop_duplicates(subset=["contentid"])

    # 좌표 유효 범위 (한국)
    mask = (
        df["mapy"].between(33, 39) &
        df["mapx"].between(124, 132)
    )
    df = df[mask].copy()

    # sub_category 분류
    def classify_sub(row):
        title = str(row.get("title", "")).lower()
        addr  = str(row.get("addr1", "")).lower()
        text  = title + " " + addr

        if any(k in text for k in ["촬영", "로케이션", "세트장", "mv"]):
            return "filming_location"
        elif any(k in text for k in ["bts", "방탄", "블랙핑크", "아이돌", "팬", "성지"]):
            return "kpop_spot"
        elif any(k in text for k in ["박물관", "미술관", "전시"]):
            return "museum"
        elif any(k in text for k in ["한옥", "전통", "문화"]):
            return "traditional_culture"
        elif any(k in text for k in ["테마파크", "놀이공원"]):
            return "theme_park"
        else:
            return "tourism_course"

    df["sub_category"] = df.apply(classify_sub, axis=1)

    # 시도 추출 (주소 첫 토큰)
    def extract_sido(addr):
        if not addr or pd.isna(addr):
            return None
        tokens = str(addr).split()
        return tokens[0] if tokens else None

    df["sido"] = df["addr1"].apply(extract_sido)

    # 출력 스키마
    df = df.rename(columns={
        "title": "name",
        "addr1": "address",
        "mapy":  "lat",
        "mapx":  "lon",
        "firstimage": "image_url",
    })

    return df


# ── 4. DB 적재 ───────────────────────────────────────────────────────────────
def load_to_db(df: pd.DataFrame) -> int:
    """kculture POI를 DB poi 테이블에 적재."""
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("  ❌ psycopg2 없음 — DB 적재 건너뜀")
        return 0

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("  ❌ DATABASE_URL 없음 — DB 적재 건너뜀")
        return 0

    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cur = conn.cursor()

    sql = """
    INSERT INTO poi (name, name_en, category, sub_category, sido, sigungu,
                     address, geom, source, image_url, is_24h, raw_data)
    VALUES (%(name)s, %(name_en)s, 'kculture', %(sub_category)s, %(sido)s, NULL,
            %(address)s,
            CASE WHEN %(lon)s IS NOT NULL AND %(lat)s IS NOT NULL
                 THEN ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)
                 ELSE NULL END,
            'tourapi_kculture', %(image_url)s, FALSE, NULL)
    ON CONFLICT DO NOTHING
    """

    rows = []
    for rec in df.to_dict("records"):
        rows.append({
            "name":         str(rec.get("name", ""))[:200],
            "name_en":      None,
            "sub_category": str(rec.get("sub_category", ""))[:50],
            "sido":         str(rec.get("sido", ""))[:20] if rec.get("sido") else None,
            "address":      str(rec.get("address", ""))[:300] if rec.get("address") else None,
            "lat":          rec.get("lat"),
            "lon":          rec.get("lon"),
            "image_url":    str(rec.get("image_url", ""))[:500] if rec.get("image_url") else None,
        })

    psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    cur.close()
    conn.close()
    return len(rows)


# ── 메인 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="K-Culture POI 수집 (TourAPI)")
    parser.add_argument("--db", action="store_true", help="수집 후 DB 즉시 적재")
    args = parser.parse_args()

    print("=" * 65)
    print("K-Culture 촬영지 / 성지순례 POI 수집 (전국)")
    print("=" * 65)
    print(f"API Key: {'설정됨' if API_KEY != _FALLBACK_KEY else '기본값(KMA_API_KEY)'}")
    print(f"수집 대상 시도: {len(AREA_CODES)}개 전국")

    # Step 1: 키워드 검색
    print(f"\n{'─'*65}")
    print("STEP 1 — TourAPI 키워드 검색 (촬영지/K-pop/BTS 등)")
    print(f"{'─'*65}")
    keyword_items = collect_by_keyword()

    # Step 2: 관광지(12) 전국 + K-Culture 필터
    print(f"\n{'─'*65}")
    print("STEP 2 — TourAPI 관광지(contentTypeId=12) 전국 + K-Culture 필터")
    print(f"{'─'*65}")
    tourism_items = collect_tourism_spots()

    # Step 3: 여행코스(25) 전국 수집
    print(f"\n{'─'*65}")
    print("STEP 3 — TourAPI 여행코스(contentTypeId=25) 전국 수집")
    print(f"{'─'*65}")
    course_items = collect_travel_courses()

    # Step 4: 병합 + 정리
    print(f"\n{'─'*65}")
    print("STEP 4 — 결과 병합 + 정리")
    print(f"{'─'*65}")
    print(f"  키워드 검색: {len(keyword_items):,}건")
    print(f"  관광지 필터: {len(tourism_items):,}건")
    print(f"  여행코스   : {len(course_items):,}건")
    df = merge_and_clean(keyword_items, tourism_items, course_items)

    if df.empty:
        print("  ❌ 수집된 데이터가 없습니다.")
        return

    print(f"  최종 행수: {len(df):,}행")
    print(f"  sub_category 분포:")
    for cat, cnt in df["sub_category"].value_counts().items():
        print(f"    {cat}: {cnt:,}건")
    print(f"  시도 분포 (상위 10):")
    for sido, cnt in df["sido"].value_counts().head(10).items():
        print(f"    {sido}: {cnt:,}건")

    # Step 5: CSV 저장
    print(f"\n{'─'*65}")
    print("STEP 5 — 저장")
    print(f"{'─'*65}")
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"  [OK] {OUTPUT_PATH}")
    print(f"  총 {len(df):,}행")

    # Step 6: DB 적재 (옵션)
    if args.db:
        print(f"\n{'─'*65}")
        print("STEP 6 — DB 적재")
        print(f"{'─'*65}")
        n = load_to_db(df)
        print(f"  → {n:,}행 DB 적재 완료")

    print(f"\n{'='*65}")
    print(f"[DONE] collect_kculture_poi.py 완료")
    print(f"{'='*65}")
    main()
