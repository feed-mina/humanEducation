"""
collect_nationwide_facility.py
================================
카카오 로컬 API → 전국 자전거 편의시설 수집 → facility_clean_nationwide.csv

기존 step1_facility_clean.py (서울시 CSV만 처리) 대비 변경:
  - 서울시 자전거 편의시설.csv 대신 카카오 로컬 API 전국 검색
  - 수집 카테고리: 자전거 대여소 / 자전거 보관함 / 자전거 수리점
  - 출력 컬럼: facility_clean.csv 와 동일 (x, y, distance, install_type, is_24h, has_restricted_hours)
  - 전국 시군구 좌표 기반 반경 3km 검색

[ 실행 방법 ]
  set KAKAO_REST_API_KEY=발급받은_키   (Windows)
  export KAKAO_REST_API_KEY=발급받은_키 (Mac/Linux)

  python src/data_collect/collect_nationwide_facility.py
  python src/data_collect/collect_nationwide_facility.py --sido 서울 부산 제주
  python src/data_collect/collect_nationwide_facility.py --radius 2000 --delay 0.5

[ 출력 파일 ]
  data/raw_ml/facility_clean_nationwide.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import requests

# ── .env 자동 로드 ────────────────────────────────────────────────────────────
def _load_dotenv() -> None:
    for candidate in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]:
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            break

_load_dotenv()

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()

RAW_DIR     = os.path.join(BASE_DIR, "data", "raw_ml")
OUTPUT_PATH = os.path.join(RAW_DIR, "facility_clean_nationwide.csv")

# ── 카카오 API ────────────────────────────────────────────────────────────────
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"

# 수집 키워드 및 install_type 레이블
FACILITY_KEYWORDS = [
    ("자전거 대여소",   "대여소"),
    ("자전거 보관함",   "보관함"),
    ("따릉이",          "공공대여"),
    ("자전거 주차장",   "주차장"),
    ("자전거 수리점",   "수리점"),
    ("바이크 스테이션", "스테이션"),
]

# 전국 시군구 대표 좌표 (시군구명: (위도, 경도))
# 주요 시군구 250여개 - 자전거 시설 밀집 지역 중심
SIGUNGU_COORDS: dict[str, tuple[float, float, str]] = {
    # (위도, 경도, 시도명)
    # 서울
    "종로구":   (37.5730, 126.9794, "서울특별시"),
    "중구":     (37.5640, 126.9979, "서울특별시"),
    "용산구":   (37.5311, 126.9810, "서울특별시"),
    "성동구":   (37.5635, 127.0369, "서울특별시"),
    "광진구":   (37.5385, 127.0823, "서울특별시"),
    "동대문구": (37.5744, 127.0396, "서울특별시"),
    "중랑구":   (37.6063, 127.0926, "서울특별시"),
    "성북구":   (37.5894, 127.0167, "서울특별시"),
    "강북구":   (37.6396, 127.0255, "서울특별시"),
    "도봉구":   (37.6688, 127.0471, "서울특별시"),
    "노원구":   (37.6542, 127.0568, "서울특별시"),
    "은평구":   (37.6026, 126.9291, "서울특별시"),
    "서대문구": (37.5791, 126.9368, "서울특별시"),
    "마포구":   (37.5663, 126.9014, "서울특별시"),
    "양천구":   (37.5170, 126.8664, "서울특별시"),
    "강서구":   (37.5509, 126.8496, "서울특별시"),
    "구로구":   (37.4954, 126.8874, "서울특별시"),
    "금천구":   (37.4569, 126.8953, "서울특별시"),
    "영등포구": (37.5264, 126.8963, "서울특별시"),
    "동작구":   (37.5124, 126.9393, "서울특별시"),
    "관악구":   (37.4784, 126.9516, "서울특별시"),
    "서초구":   (37.4836, 127.0327, "서울특별시"),
    "강남구":   (37.4979, 127.0276, "서울특별시"),
    "송파구":   (37.5145, 127.1059, "서울특별시"),
    "강동구":   (37.5301, 127.1238, "서울특별시"),
    # 부산
    "중구(부산)":   (35.1030, 129.0327, "부산광역시"),
    "서구(부산)":   (35.0975, 129.0245, "부산광역시"),
    "동구(부산)":   (35.1362, 129.0560, "부산광역시"),
    "영도구":       (35.0912, 129.0677, "부산광역시"),
    "부산진구":     (35.1631, 129.0531, "부산광역시"),
    "동래구":       (35.1998, 129.0857, "부산광역시"),
    "남구(부산)":   (35.1367, 129.0844, "부산광역시"),
    "북구(부산)":   (35.1974, 128.9905, "부산광역시"),
    "해운대구":     (35.1631, 129.1636, "부산광역시"),
    "사상구":       (35.1549, 128.9921, "부산광역시"),
    "사하구":       (35.1042, 128.9748, "부산광역시"),
    "금정구":       (35.2424, 129.0920, "부산광역시"),
    "강서구(부산)": (35.2121, 128.9807, "부산광역시"),
    "연제구":       (35.1763, 129.0791, "부산광역시"),
    "수영구":       (35.1453, 129.1135, "부산광역시"),
    # 대구
    "중구(대구)":   (35.8694, 128.6063, "대구광역시"),
    "동구(대구)":   (35.8870, 128.6349, "대구광역시"),
    "서구(대구)":   (35.8713, 128.5598, "대구광역시"),
    "남구(대구)":   (35.8456, 128.5971, "대구광역시"),
    "북구(대구)":   (35.8851, 128.5825, "대구광역시"),
    "수성구":       (35.8583, 128.6306, "대구광역시"),
    "달서구":       (35.8298, 128.5327, "대구광역시"),
    "달성군":       (35.7748, 128.4314, "대구광역시"),
    # 인천
    "계양구":       (37.5375, 126.7377, "인천광역시"),
    "남동구":       (37.4490, 126.7314, "인천광역시"),
    "미추홀구":     (37.4638, 126.6502, "인천광역시"),
    "부평구":       (37.5074, 126.7213, "인천광역시"),
    "서구(인천)":   (37.5450, 126.6761, "인천광역시"),
    "연수구":       (37.4104, 126.6780, "인천광역시"),
    # 광주
    "동구(광주)":   (35.1468, 126.9237, "광주광역시"),
    "서구(광주)":   (35.1516, 126.8910, "광주광역시"),
    "남구(광주)":   (35.1339, 126.9028, "광주광역시"),
    "북구(광주)":   (35.1744, 126.9122, "광주광역시"),
    "광산구":       (35.1395, 126.7935, "광주광역시"),
    # 대전
    "동구(대전)":   (36.3121, 127.4534, "대전광역시"),
    "중구(대전)":   (36.3252, 127.4210, "대전광역시"),
    "서구(대전)":   (36.3550, 127.3832, "대전광역시"),
    "유성구":       (36.3626, 127.3563, "대전광역시"),
    "대덕구":       (36.3465, 127.4159, "대전광역시"),
    # 울산
    "중구(울산)":   (35.5692, 129.3319, "울산광역시"),
    "남구(울산)":   (35.5384, 129.3266, "울산광역시"),
    "동구(울산)":   (35.5046, 129.4167, "울산광역시"),
    "북구(울산)":   (35.5820, 129.3609, "울산광역시"),
    "울주군":       (35.5228, 129.2413, "울산광역시"),
    # 세종
    "세종시":       (36.4800, 127.2890, "세종특별자치시"),
    # 경기
    "수원시":       (37.2636, 127.0286, "경기도"),
    "성남시":       (37.4200, 127.1264, "경기도"),
    "의정부시":     (37.7382, 127.0337, "경기도"),
    "안양시":       (37.3943, 126.9568, "경기도"),
    "부천시":       (37.5035, 126.7660, "경기도"),
    "광명시":       (37.4784, 126.8644, "경기도"),
    "평택시":       (37.0000, 127.1125, "경기도"),
    "동두천시":     (37.9036, 127.0604, "경기도"),
    "안산시":       (37.3236, 126.8318, "경기도"),
    "고양시":       (37.6584, 126.8320, "경기도"),
    "과천시":       (37.4292, 126.9876, "경기도"),
    "구리시":       (37.5943, 127.1296, "경기도"),
    "남양주시":     (37.6360, 127.2165, "경기도"),
    "오산시":       (37.1498, 127.0777, "경기도"),
    "시흥시":       (37.3799, 126.8031, "경기도"),
    "군포시":       (37.3613, 126.9353, "경기도"),
    "의왕시":       (37.3447, 126.9688, "경기도"),
    "하남시":       (37.5392, 127.2148, "경기도"),
    "용인시":       (37.2411, 127.1776, "경기도"),
    "파주시":       (37.7601, 126.7798, "경기도"),
    "이천시":       (37.2720, 127.4345, "경기도"),
    "안성시":       (37.0078, 127.2797, "경기도"),
    "김포시":       (37.6152, 126.7157, "경기도"),
    "화성시":       (37.1994, 126.8312, "경기도"),
    "광주시(경기)": (37.4296, 127.2557, "경기도"),
    "양주시":       (37.7855, 127.0457, "경기도"),
    "포천시":       (37.8946, 127.2003, "경기도"),
    "여주시":       (37.2985, 127.6378, "경기도"),
    # 강원
    "춘천시":   (37.8813, 127.7298, "강원특별자치도"),
    "원주시":   (37.3422, 127.9201, "강원특별자치도"),
    "강릉시":   (37.7519, 128.8761, "강원특별자치도"),
    "동해시":   (37.5247, 129.1144, "강원특별자치도"),
    "태백시":   (37.1600, 128.9854, "강원특별자치도"),
    "속초시":   (38.2049, 128.5914, "강원특별자치도"),
    "삼척시":   (37.4500, 129.1653, "강원특별자치도"),
    "홍천군":   (37.6972, 127.8886, "강원특별자치도"),
    "횡성군":   (37.4916, 127.9845, "강원특별자치도"),
    "영월군":   (37.1837, 128.4614, "강원특별자치도"),
    "평창군":   (37.3704, 128.3914, "강원특별자치도"),
    # 충북
    "청주시":   (36.6424, 127.4890, "충청북도"),
    "충주시":   (36.9910, 127.9259, "충청북도"),
    "제천시":   (37.1326, 128.1908, "충청북도"),
    "보은군":   (36.4895, 127.7295, "충청북도"),
    "옥천군":   (36.3060, 127.5710, "충청북도"),
    # 충남
    "천안시":   (36.8151, 127.1139, "충청남도"),
    "공주시":   (36.4465, 127.1189, "충청남도"),
    "보령시":   (36.3330, 126.6127, "충청남도"),
    "아산시":   (36.7898, 127.0021, "충청남도"),
    "서산시":   (36.7848, 126.4503, "충청남도"),
    "논산시":   (36.1872, 127.0987, "충청남도"),
    "계룡시":   (36.2744, 127.2487, "충청남도"),
    "당진시":   (36.8895, 126.6456, "충청남도"),
    # 전북
    "전주시":   (35.8242, 127.1480, "전라북도"),
    "군산시":   (35.9676, 126.7368, "전라북도"),
    "익산시":   (35.9483, 126.9575, "전라북도"),
    "정읍시":   (35.5700, 126.8562, "전라북도"),
    "남원시":   (35.4165, 127.3905, "전라북도"),
    "김제시":   (35.8036, 126.8807, "전라북도"),
    "완주군":   (35.9067, 127.1618, "전라북도"),
    # 전남
    "목포시":   (34.8118, 126.3922, "전라남도"),
    "여수시":   (34.7604, 127.6622, "전라남도"),
    "순천시":   (34.9506, 127.4874, "전라남도"),
    "나주시":   (35.0160, 126.7108, "전라남도"),
    "광양시":   (34.9407, 127.6956, "전라남도"),
    "담양군":   (35.3214, 126.9884, "전라남도"),
    "영광군":   (35.2770, 126.5120, "전라남도"),
    # 경북
    "포항시":   (36.0190, 129.3435, "경상북도"),
    "경주시":   (35.8562, 129.2247, "경상북도"),
    "김천시":   (36.1398, 128.1136, "경상북도"),
    "안동시":   (36.5684, 128.7294, "경상북도"),
    "구미시":   (36.1195, 128.3446, "경상북도"),
    "영주시":   (36.8058, 128.6240, "경상북도"),
    "영천시":   (35.9734, 128.9387, "경상북도"),
    "상주시":   (36.4108, 128.1589, "경상북도"),
    "문경시":   (36.5860, 128.1866, "경상북도"),
    "경산시":   (35.8254, 128.7415, "경상북도"),
    "칠곡군":   (35.9943, 128.4015, "경상북도"),
    # 경남
    "창원시":   (35.2279, 128.6817, "경상남도"),
    "진주시":   (35.1799, 128.1076, "경상남도"),
    "통영시":   (34.8544, 128.4330, "경상남도"),
    "사천시":   (35.0037, 128.0645, "경상남도"),
    "김해시":   (35.2285, 128.8891, "경상남도"),
    "밀양시":   (35.5035, 128.7460, "경상남도"),
    "거제시":   (34.8804, 128.6211, "경상남도"),
    "양산시":   (35.3350, 129.0378, "경상남도"),
    "의령군":   (35.3222, 128.2620, "경상남도"),
    "함안군":   (35.2726, 128.4066, "경상남도"),
    "합천군":   (35.5665, 128.1655, "경상남도"),
    "남해군":   (34.8376, 127.8924, "경상남도"),
    # 제주
    "제주시":   (33.4996, 126.5312, "제주특별자치도"),
    "서귀포시": (33.2541, 126.5600, "제주특별자치도"),
}


# ══════════════════════════════════════════════════════════════════════════════
# 카카오 키워드 검색
# ══════════════════════════════════════════════════════════════════════════════
def search_kakao_keyword(
    api_key: str,
    query: str,
    lat: float,
    lon: float,
    radius: int = 3000,
    max_pages: int = 3,
) -> list[dict]:
    headers = {"Authorization": f"KakaoAK {api_key}"}
    results = []
    for page in range(1, max_pages + 1):
        params = {
            "query": query,
            "y": str(lat),
            "x": str(lon),
            "radius": str(radius),
            "page": str(page),
            "size": "15",
            "sort": "distance",
        }
        try:
            resp = requests.get(KAKAO_KEYWORD_URL, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("documents", [])
            if not docs:
                break
            results.extend(docs)
            if data.get("meta", {}).get("is_end", True):
                break
        except Exception as e:
            print(f"    ⚠ 카카오 API 오류 ({query}, {page}p): {e}")
            break
        time.sleep(0.1)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 레코드 → facility_clean.csv 포맷 변환
# ══════════════════════════════════════════════════════════════════════════════
def parse_facility(doc: dict, install_type: str, sigungu: str, sido: str) -> dict:
    name = doc.get("place_name", "")
    x = float(doc.get("x", 0) or 0)   # 경도
    y = float(doc.get("y", 0) or 0)   # 위도
    distance = float(doc.get("distance", 0) or 0)
    place_url = doc.get("place_url", "")
    category_name = doc.get("category_name", "")

    is_24h = False
    has_restricted_hours = False

    # 이름/카테고리에서 운영시간 힌트 추출
    text = f"{name} {category_name}".lower()
    if "24시" in text or "24h" in text or "무인" in text:
        is_24h = True
    if any(kw in text for kw in ["예약", "제한", "이용불가", "임시"]):
        has_restricted_hours = True

    return {
        "x":                   x,
        "y":                   y,
        "distance":            distance,
        "install_type":        install_type,
        "is_24h":              is_24h,
        "has_restricted_hours": has_restricted_hours,
        # 전국 모델에서 지역 필터용 추가 컬럼
        "sido":                sido,
        "sigungu":             sigungu,
        "place_name":          name,
        "address":             doc.get("road_address_name", doc.get("address_name", "")),
        "lat":                 y,
        "lon":                 x,
        "source":              "kakao",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 전체 수집
# ══════════════════════════════════════════════════════════════════════════════
def collect(
    api_key: str,
    target_sidos: list[str] | None,
    radius: int,
    delay: float,
) -> pd.DataFrame:
    os.makedirs(RAW_DIR, exist_ok=True)

    # 시도 필터
    if target_sidos:
        coords = {k: v for k, v in SIGUNGU_COORDS.items()
                  if any(s in v[2] for s in target_sidos)}
        if not coords:
            print(f"  ❌ 해당 시도 시군구 없음: {target_sidos}")
            sys.exit(1)
    else:
        coords = SIGUNGU_COORDS

    total_sigungu = len(coords)
    total_keywords = len(FACILITY_KEYWORDS)
    print(f"  수집 시군구  : {total_sigungu}개")
    print(f"  수집 키워드  : {total_keywords}개 → {', '.join(kw for kw, _ in FACILITY_KEYWORDS)}")
    print(f"  검색 반경    : {radius}m")
    print(f"  예상 API 호출: ~{total_sigungu * total_keywords * 2:,}건\n")

    all_rows: list[dict] = []
    seen_ids: set[str] = set()  # 중복 제거용 (place_id)

    for idx, (sigungu, (lat, lon, sido)) in enumerate(coords.items(), 1):
        print(f"  [{idx:3d}/{total_sigungu}] {sido} {sigungu}...", end="", flush=True)
        loc_count = 0

        for keyword, install_type in FACILITY_KEYWORDS:
            docs = search_kakao_keyword(api_key, keyword, lat, lon, radius=radius)
            for doc in docs:
                place_id = doc.get("id", "")
                if place_id and place_id in seen_ids:
                    continue
                if place_id:
                    seen_ids.add(place_id)
                row = parse_facility(doc, install_type, sigungu, sido)
                # 좌표 유효성 (한반도 범위)
                if not (33.0 <= row["lat"] <= 38.6 and 124.5 <= row["lon"] <= 130.0):
                    continue
                all_rows.append(row)
                loc_count += 1
            time.sleep(delay)

        print(f" {loc_count}건")

    if not all_rows:
        print("\n  ❌ 수집된 데이터 없음. API 키와 네트워크를 확인하세요.")
        sys.exit(1)

    return pd.DataFrame(all_rows)


# ══════════════════════════════════════════════════════════════════════════════
# 저장
# ══════════════════════════════════════════════════════════════════════════════
def save(df: pd.DataFrame) -> None:
    # facility_clean.csv 와 동일한 핵심 컬럼 순서 유지
    core_cols = ["x", "y", "distance", "install_type", "is_24h", "has_restricted_hours"]
    extra_cols = ["sido", "sigungu", "place_name", "address", "lat", "lon", "source"]
    ordered_cols = core_cols + [c for c in extra_cols if c in df.columns]
    df = df[[c for c in ordered_cols if c in df.columns]]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n  ✅ 저장 완료: {OUTPUT_PATH}")
    print(f"     총 {len(df):,}행 × {len(df.columns)}컬럼")
    print(f"\n     install_type 분포:")
    print(df["install_type"].value_counts().to_string())
    if "sido" in df.columns:
        print(f"\n     시도별 분포:")
        print(df["sido"].value_counts().to_string())
    print(f"\n     is_24h     : {df['is_24h'].sum():,}건 ({df['is_24h'].mean()*100:.1f}%)")
    print(f"     제한운영   : {df['has_restricted_hours'].sum():,}건")


def save_to_db(df: pd.DataFrame, db_url: str) -> None:
    """수집된 시설 데이터를 PostgreSQL poi 테이블에 저장."""
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("  ❌ psycopg2 없음: pip install psycopg2-binary")
        return

    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cur = conn.cursor()

    sql_with = """
    INSERT INTO poi (
        name, category, sub_category, sido, sigungu,
        address, geom, source, is_24h, raw_data
    ) VALUES (
        %(name)s, 'facility', %(sub_category)s, %(sido)s, %(sigungu)s,
        %(address)s,
        ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326),
        'kakao', %(is_24h)s, %(raw_data)s::jsonb
    )
    ON CONFLICT DO NOTHING
    """
    sql_without = """
    INSERT INTO poi (
        name, category, sub_category, sido, sigungu,
        address, geom, source, is_24h, raw_data
    ) VALUES (
        %(name)s, 'facility', %(sub_category)s, %(sido)s, %(sigungu)s,
        %(address)s, NULL, 'kakao', %(is_24h)s, %(raw_data)s::jsonb
    )
    ON CONFLICT DO NOTHING
    """

    rows_with: list[dict] = []
    rows_without: list[dict] = []

    for _, r in df.iterrows():
        lat = float(r.get("lat") or r.get("y") or 0)
        lon = float(r.get("lon") or r.get("x") or 0)
        has_coord = 33.0 <= lat <= 38.6 and 124.5 <= lon <= 130.0
        row = {
            "name":         str(r.get("place_name") or r.get("install_type") or "자전거시설")[:200],
            "sub_category": str(r.get("install_type", ""))[:50] if r.get("install_type") else None,
            "sido":         str(r.get("sido", ""))[:20] if r.get("sido") else None,
            "sigungu":      str(r.get("sigungu", ""))[:30] if r.get("sigungu") else None,
            "address":      str(r.get("address", ""))[:300] if r.get("address") else None,
            "lat":          lat,
            "lon":          lon,
            "is_24h":       bool(r.get("is_24h", False)),
            "raw_data":     None,
        }
        if has_coord:
            rows_with.append(row)
        else:
            rows_without.append(row)

    if rows_with:
        psycopg2.extras.execute_batch(cur, sql_with, rows_with, page_size=500)
    if rows_without:
        psycopg2.extras.execute_batch(cur, sql_without, rows_without, page_size=500)

    cur.close()
    conn.close()
    print(f"\n  ✅ DB 저장 완료: poi 테이블 +{len(df):,}행 (좌표有={len(rows_with):,})")



# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전국 자전거 편의시설 수집")
    parser.add_argument(
        "--api_key",
        default=os.environ.get("KAKAO_REST_API_KEY", ""),
        help="카카오 REST API 키 (환경변수 KAKAO_REST_API_KEY 권장)",
    )
    parser.add_argument(
        "--sido", nargs="+", default=None, metavar="시도명",
        help="특정 시도만 수집 (예: --sido 서울 부산 제주). 생략 시 전국 전체.",
    )
    parser.add_argument(
        "--radius", type=int, default=3000,
        help="검색 반경 (m, 기본값: 3000)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="API 호출 간 대기 시간 (초, 기본값: 0.3)",
    )
    parser.add_argument(
        "--db", action="store_true",
        help="수집 완료 후 PostgreSQL poi 테이블에도 저장 (DATABASE_URL 환경변수 필요)",
    )
    parser.add_argument(
        "--db_url", default=os.environ.get("DATABASE_URL", ""),
        help="DATABASE_URL 직접 지정 (환경변수 DATABASE_URL 권장)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("❌ 카카오 REST API 키 없음.")
        print("   방법 1: set KAKAO_REST_API_KEY=발급받은_키  (Windows)")
        print("   방법 2: python ... collect_nationwide_facility.py --api_key 키")
        print()
        print("   카카오 Developers (developers.kakao.com) 에서 앱 생성 후")
        print("   REST API 키 발급 (무료, 300,000콜/일)")
        sys.exit(1)

    print("=" * 65)
    print("전국 자전거 편의시설 수집 시작")
    print(f"  기존 step1_facility_clean.py 대비: 서울 CSV → 전국 API 수집")
    print("=" * 65)
    print()

    df = collect(
        api_key=args.api_key,
        target_sidos=args.sido,
        radius=args.radius,
        delay=args.delay,
    )

    print("=" * 65)
    print("CSV 저장")
    print("=" * 65)
    save(df)

    if args.db:
        db_url = args.db_url
        if not db_url:
            print("\n  ❌ --db 옵션 사용 시 DATABASE_URL 환경변수가 필요합니다.")
            print("     export DATABASE_URL=postgresql://user:pass@host/kride")
        else:
            print("\n" + "=" * 65)
            print("DB 저장 (PostgreSQL poi 테이블)")
            print("=" * 65)
            save_to_db(df, db_url)

    print("\n" + "=" * 65)
    print("✅ collect_nationwide_facility.py 완료")
    print("   다음 단계: python src/preprocessing/step4_spatial_join.py")
    print("=" * 65)
