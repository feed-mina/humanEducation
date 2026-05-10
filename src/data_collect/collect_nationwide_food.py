"""
collect_nationwide_food.py
===========================
전국 맛집/카페 POI 수집 (카카오 로컬 API FD6/CE7)

[ 수집 전략 — 3-Layer ]
  Layer 1. tour_poi.csv 전체 좌표 (15,905개) × 반경 500m
           → 관광지 주변 맛집 커버
  Layer 2. 전국 시군구 중심좌표 격자 (250개 시군구 × 3×3=9격자)
           → 관광지 없는 주거/상업 지역 보완
  Layer 3. 인기 상권 집중 수집 (명동/홍대/서면 등 20개) × 반경 2km
           → 핵심 상권 밀도 보완

[ API 할당량 ]
  하루 한도: 300,000콜
  Layer 1:  15,905 × 2카테고리 × 3페이지 ≈ 95,430콜
  Layer 2:  250 × 9 × 2 × 3               ≈ 13,500콜
  Layer 3:  20 × 2 × 5                    ≈    200콜
  합계 ≈ 109,130콜 → 1일 수집 가능

[ 실행 방법 ]
  # 기본 (CSV 저장)
  python src/data_collect/collect_nationwide_food.py

  # DB 직접 저장 포함
  python src/data_collect/collect_nationwide_food.py --db

  # Layer 1만 빠르게 (테스트)
  python src/data_collect/collect_nationwide_food.py --layer 1

  # 이어서 수집 (중단 후 재시작)
  python src/data_collect/collect_nationwide_food.py --resume

[ 출력 파일 ]
  data/raw_ml/food_poi_nationwide.csv          ← 최종 결과
  data/raw_ml/food_poi_checkpoint.csv          ← 진행 상황 (이어받기용)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings

import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── .env 자동 로드 ─────────────────────────────────────────────────────────────
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

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))

RAW_DIR        = os.path.join(BASE_DIR, "data", "raw_ml")
TOUR_POI_PATH  = os.path.join(RAW_DIR, "tour_poi.csv")
OUTPUT_PATH    = os.path.join(RAW_DIR, "food_poi_nationwide.csv")
CHECKPOINT_PATH = os.path.join(RAW_DIR, "food_poi_checkpoint.csv")
os.makedirs(RAW_DIR, exist_ok=True)

# ── API 설정 ───────────────────────────────────────────────────────────────────
KAKAO_REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY", "")
KAKAO_CATEGORY_URL = "https://dapi.kakao.com/v2/local/search/category.json"

KAKAO_CATEGORIES = {
    "FD6": "food",
    "CE7": "cafe",
}

# ── Layer 3: 인기 상권 중심좌표 (위도, 경도, 상권명) ────────────────────────────
POPULAR_DISTRICTS: list[tuple[float, float, str]] = [
    (37.5632, 126.9844, "서울_명동"),
    (37.5563, 126.9236, "서울_홍대"),
    (37.5172, 127.0473, "서울_강남역"),
    (37.5080, 127.0617, "서울_선릉"),
    (37.5700, 126.9836, "서울_광화문"),
    (37.5796, 126.9770, "서울_경복궁"),
    (35.1580, 129.0597, "부산_서면"),
    (35.1013, 129.0267, "부산_남포동"),
    (35.1586, 128.9952, "부산_사직"),
    (35.8704, 128.5960, "대구_동성로"),
    (37.4563, 126.7052, "인천_부평"),
    (35.1481, 126.9196, "광주_충장로"),
    (36.3504, 127.3846, "대전_으능정이"),
    (35.5467, 129.3143, "울산_삼산"),
    (37.2752, 127.0094, "수원_행리단길"),
    (37.3943, 127.1112, "성남_판교"),
    (37.6566, 127.0644, "서울_노원"),
    (37.4970, 126.8676, "서울_목동"),
    (36.6420, 127.4890, "청주_성안길"),
    (35.8268, 127.1481, "전주_한옥마을"),
]

# ── 전국 시군구 중심좌표 (Layer 2) ─────────────────────────────────────────────
# 주요 시군구 대표 좌표 (위도, 경도, 시도, 시군구)
SIGUNGU_CENTERS: list[tuple[float, float, str, str]] = [
    # 서울
    (37.5729, 126.9794, "서울특별시", "종로구"),
    (37.5614, 126.9998, "서울특별시", "중구"),
    (37.5284, 126.9643, "서울특별시", "용산구"),
    (37.5530, 126.9337, "서울특별시", "마포구"),
    (37.5172, 126.9015, "서울특별시", "영등포구"),
    (37.4979, 127.0276, "서울특별시", "서초구"),
    (37.5172, 127.0473, "서울특별시", "강남구"),
    (37.5409, 127.0863, "서울특별시", "광진구"),
    (37.6176, 127.0305, "서울특별시", "노원구"),
    (37.6027, 126.9291, "서울특별시", "은평구"),
    # 부산
    (35.1580, 129.0597, "부산광역시", "부산진구"),
    (35.1013, 129.0267, "부산광역시", "중구"),
    (35.1584, 129.1603, "부산광역시", "해운대구"),
    (35.0977, 128.9641, "부산광역시", "사하구"),
    (35.2124, 129.0129, "부산광역시", "북구"),
    # 대구
    (35.8704, 128.5960, "대구광역시", "중구"),
    (35.8268, 128.5351, "대구광역시", "달서구"),
    (35.8850, 128.6430, "대구광역시", "수성구"),
    # 인천
    (37.4563, 126.7052, "인천광역시", "부평구"),
    (37.4763, 126.6161, "인천광역시", "연수구"),
    (37.5219, 126.6768, "인천광역시", "남동구"),
    # 광주
    (35.1481, 126.9196, "광주광역시", "동구"),
    (35.1555, 126.8497, "광주광역시", "서구"),
    (35.1453, 126.9236, "광주광역시", "남구"),
    # 대전
    (36.3504, 127.3846, "대전광역시", "중구"),
    (36.3623, 127.3566, "대전광역시", "서구"),
    (36.3696, 127.4429, "대전광역시", "유성구"),
    # 울산
    (35.5467, 129.3143, "울산광역시", "남구"),
    (35.5665, 129.2763, "울산광역시", "중구"),
    # 경기
    (37.2752, 127.0094, "경기도", "수원시"),
    (37.4138, 127.5183, "경기도", "여주시"),
    (37.6579, 126.8320, "경기도", "고양시"),
    (37.3943, 127.1112, "경기도", "성남시"),
    (37.3021, 126.8345, "경기도", "안산시"),
    (37.2411, 127.1775, "경기도", "용인시"),
    (37.4926, 126.7237, "경기도", "부천시"),
    (37.6450, 127.2890, "경기도", "남양주시"),
    (37.7600, 127.0437, "경기도", "의정부시"),
    (37.8040, 127.1526, "경기도", "포천시"),
    (37.1441, 127.0694, "경기도", "평택시"),
    (37.4140, 127.5183, "경기도", "이천시"),
    # 강원
    (37.8817, 127.7299, "강원특별자치도", "춘천시"),
    (37.3420, 128.3896, "강원특별자치도", "원주시"),
    (37.7519, 128.8964, "강원특별자치도", "강릉시"),
    (38.2096, 128.5910, "강원특별자치도", "속초시"),
    (37.6754, 128.7110, "강원특별자치도", "평창군"),
    # 충북
    (36.6420, 127.4890, "충청북도", "청주시"),
    (36.9910, 127.9262, "충청북도", "충주시"),
    (36.4803, 127.7257, "충청북도", "보은군"),
    # 충남
    (36.8151, 127.1139, "충청남도", "천안시"),
    (36.4800, 126.9302, "충청남도", "홍성군"),
    (36.7950, 126.4523, "충청남도", "태안군"),
    (36.6740, 127.4566, "충청남도", "공주시"),
    # 전북
    (35.8268, 127.1481, "전라북도", "전주시"),
    (35.9758, 126.9609, "전라북도", "익산시"),
    (35.5868, 127.0965, "전라북도", "남원시"),
    (35.7176, 127.1516, "전라북도", "완주군"),
    # 전남
    (34.8160, 126.4629, "전라남도", "목포시"),
    (34.7604, 127.6623, "전라남도", "여수시"),
    (34.9025, 127.7604, "전라남도", "순천시"),
    (34.5961, 126.9820, "전라남도", "강진군"),
    (34.9382, 127.4860, "전라남도", "보성군"),
    # 경북
    (36.0190, 129.3430, "경상북도", "포항시"),
    (35.8547, 128.4910, "경상북도", "경산시"),
    (35.8560, 129.2247, "경상북도", "경주시"),
    (36.5760, 128.5057, "경상북도", "안동시"),
    (36.1075, 128.4190, "경상북도", "구미시"),
    # 경남
    (35.1796, 128.1067, "경상남도", "창원시"),
    (35.5467, 128.7487, "경상남도", "밀양시"),
    (35.0036, 128.0640, "경상남도", "고성군"),
    (34.8427, 128.4221, "경상남도", "통영시"),
    (35.3249, 128.2130, "경상남도", "합천군"),
    # 제주
    (33.4996, 126.5312, "제주특별자치도", "제주시"),
    (33.2541, 126.5601, "제주특별자치도", "서귀포시"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════════════════════
SIDO_NAMES = [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
    "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원특별자치도",
    "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도",
    "제주특별자치도",
]


def parse_sido_sigungu(address: str) -> tuple[str, str]:
    """카카오 주소 문자열 → (시도, 시군구) 파싱"""
    address = str(address).strip()
    for sido in sorted(SIDO_NAMES, key=len, reverse=True):
        if address.startswith(sido):
            rest = address[len(sido):].strip()
            # 시군구: 첫 번째 공백 구분 토큰
            parts = rest.split()
            sigungu = parts[0] if parts else sido
            return sido, sigungu
    # 매칭 실패 시 빈 값
    return "", ""


def make_grid(center_lat: float, center_lon: float,
              step_deg: float = 0.013,
              n: int = 3) -> list[tuple[float, float]]:
    """
    중심 좌표를 포함한 n×n 격자 좌표 생성
    step_deg ≈ 1.4km (위도 기준 0.013° ≈ 1.4km)
    """
    half = (n // 2) * step_deg
    points = []
    lat0 = center_lat - half
    for i in range(n):
        lon0 = center_lon - half
        for j in range(n):
            points.append((lat0 + i * step_deg, lon0 + j * step_deg))
    return points


def _print_kakao_platform_guide() -> None:
    print("""
  ─────────────────────────────────────────────────────────
  [Kakao 400 오류 해결 방법]

  1. https://developers.kakao.com 접속 → 내 애플리케이션 선택
  2. [앱 설정] → [플랫폼]
  3. 'Web 플랫폼' 항목에서 아래 중 하나:
     a) 사이트 도메인을 'https://localhost' 로 추가
     b) 또는 제한 없이 사용하려면 도메인 입력란을 비워두고
        '사이트 도메인 없이 REST API 허용' 체크 (앱마다 UI 상이)
  4. [저장] 후 재실행
  ─────────────────────────────────────────────────────────
""")


def search_kakao_category(lon: float, lat: float,
                          category_code: str,
                          radius_m: int = 500,
                          max_pages: int = 3) -> list[dict]:
    """카카오 로컬 카테고리 반경 검색 → 결과 리스트"""
    results = []
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}

    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                KAKAO_CATEGORY_URL,
                headers=headers,
                params={
                    "category_group_code": category_code,
                    "x": lon,
                    "y": lat,
                    "radius": radius_m,
                    "size": 15,
                    "page": page,
                    "sort": "accuracy",
                },
                timeout=10,
            )
            if resp.status_code == 401:
                print("  [ERR] 카카오 API 인증 실패 — KAKAO_REST_API_KEY를 확인하세요.")
                sys.exit(1)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            print(f"    ⚠ 타임아웃 (cat={category_code}, page={page}) — 스킵")
            break
        except requests.exceptions.HTTPError as e:
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text
            print(f"    ⚠ HTTP {resp.status_code} 오류: {e}")
            print(f"    ⚠ 응답 본문: {err_body}")
            if resp.status_code == 400:
                print("    → Kakao 앱 플랫폼 설정을 확인하세요 (아래 안내 참조)")
                _print_kakao_platform_guide()
                sys.exit(1)
            break
        except requests.exceptions.RequestException as e:
            print(f"    ⚠ 요청 오류: {e}")
            break

        items = data.get("documents", [])
        for item in items:
            addr = item.get("road_address_name") or item.get("address_name", "")
            sido, sigungu = parse_sido_sigungu(addr)
            results.append({
                "name":         item.get("place_name", ""),
                "category":     KAKAO_CATEGORIES.get(category_code, category_code),
                "sub_category": item.get("category_name", "").split(" > ")[-1],
                "address":      addr,
                "sido":         sido,
                "sigungu":      sigungu,
                "lon":          float(item.get("x", 0)),
                "lat":          float(item.get("y", 0)),
                "source":       f"kakao_{category_code}",
                "place_url":    item.get("place_url", ""),
            })

        if data.get("meta", {}).get("is_end", True):
            break

    return results


def save_checkpoint(records: list[dict]) -> None:
    pd.DataFrame(records).to_csv(CHECKPOINT_PATH, index=False, encoding="utf-8-sig")


def load_checkpoint() -> tuple[list[dict], set[str]]:
    if not os.path.exists(CHECKPOINT_PATH):
        return [], set()
    df = pd.read_csv(CHECKPOINT_PATH, encoding="utf-8-sig")
    records = df.to_dict("records")
    keys = {f"{r['name']}|{round(r['lon'],5)}|{round(r['lat'],5)}" for r in records}
    print(f"  체크포인트 로드: {len(records):,}건 기존 수집")
    return records, keys


def upsert_to_db(df: pd.DataFrame) -> None:
    """poi 테이블에 food/cafe 데이터 upsert"""
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        print("  [ERR] psycopg2 없음: pip install psycopg2")
        return

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("  [ERR] DATABASE_URL 환경 변수 없음")
        return

    print(f"  DB upsert 시작 ({len(df):,}행)...")
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        rows = []
        for _, r in df.iterrows():
            lon = r.get("lon")
            lat = r.get("lat")
            geom = f"SRID=4326;POINT({lon} {lat})" if lon and lat else None
            rows.append((
                r.get("name", ""),
                None,                        # name_en
                r.get("category", "food"),
                r.get("sub_category", ""),
                r.get("sido", ""),
                r.get("sigungu", ""),
                r.get("address", ""),
                geom,
                r.get("source", "kakao"),
                None,                        # score
                r.get("place_url", ""),
            ))

        sql = """
            INSERT INTO poi (name, name_en, category, sub_category, sido, sigungu,
                             address, geom, source, score, image_url)
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        execute_values(cur, sql, rows, template="""
            (%s, %s, %s, %s, %s, %s, %s,
             ST_GeomFromEWKT(%s), %s, %s, %s)
        """)
        conn.commit()
        cur.close()
        conn.close()
        print(f"  [OK] DB upsert 완료")
    except Exception as e:
        print(f"  [ERR] DB upsert 실패: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════
def main(args: argparse.Namespace) -> None:
    if not KAKAO_REST_API_KEY or KAKAO_REST_API_KEY == "":
        print("[ERR] KAKAO_REST_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("  .env 파일에 KAKAO_REST_API_KEY=your_key 를 추가하세요.")
        sys.exit(1)

    # 체크포인트 로드 (--resume)
    all_records, seen_keys = (load_checkpoint() if args.resume else ([], set()))

    def collect(points: list[tuple[float, float]],
                layer_label: str,
                radius_m: int = 500,
                max_pages: int = 3) -> None:
        for i, (lat, lon) in enumerate(points, 1):
            for cat_code in KAKAO_CATEGORIES:
                results = search_kakao_category(lon, lat, cat_code, radius_m, max_pages)
                new_count = 0
                for r in results:
                    key = f"{r['name']}|{round(r['lon'],5)}|{round(r['lat'],5)}"
                    if key not in seen_keys and r["name"]:
                        seen_keys.add(key)
                        all_records.append(r)
                        new_count += 1
                time.sleep(args.delay)

            if i % 200 == 0:
                save_checkpoint(all_records)
                print(f"  [{layer_label}] {i}/{len(points)} — 누계 {len(all_records):,}건 (체크포인트 저장)")

        save_checkpoint(all_records)
        print(f"  [{layer_label}] 완료 — 누계 {len(all_records):,}건\n")

    # ── Layer 1: tour_poi 전체 앵커 ──────────────────────────────────────────
    if args.layer in (0, 1):
        print("=" * 65)
        print("Layer 1: tour_poi.csv 전체 좌표 기반 수집")
        print("=" * 65)

        if not os.path.exists(TOUR_POI_PATH):
            print(f"  [WARN] tour_poi.csv 없음: {TOUR_POI_PATH} — Layer 1 스킵")
        else:
            tour_df = pd.read_csv(TOUR_POI_PATH, encoding="utf-8-sig")
            tour_df = tour_df.dropna(subset=["mapx", "mapy"])
            tour_df["mapx"] = pd.to_numeric(tour_df["mapx"], errors="coerce")
            tour_df["mapy"] = pd.to_numeric(tour_df["mapy"], errors="coerce")
            tour_df = tour_df.dropna(subset=["mapx", "mapy"])

            # 좌표 범위 필터 (한반도)
            tour_df = tour_df[
                tour_df["mapy"].between(33.0, 38.5) &
                tour_df["mapx"].between(124.0, 132.0)
            ]
            print(f"  tour_poi 유효 좌표: {len(tour_df):,}개")

            points_l1 = [(row["mapy"], row["mapx"]) for _, row in tour_df.iterrows()]
            print(f"  수집 시작: {len(points_l1):,}점 × 2카테고리 × {args.pages}페이지")
            collect(points_l1, "Layer1", radius_m=args.radius, max_pages=args.pages)

    # ── Layer 2: 시군구 중심 격자 ─────────────────────────────────────────────
    if args.layer in (0, 2):
        print("=" * 65)
        print("Layer 2: 전국 시군구 중심좌표 격자 수집")
        print("=" * 65)

        points_l2: list[tuple[float, float]] = []
        for clat, clon, sido, sigungu in SIGUNGU_CENTERS:
            points_l2.extend(make_grid(clat, clon, step_deg=0.013, n=3))

        print(f"  격자점: {len(SIGUNGU_CENTERS)}개 시군구 × 9격자 = {len(points_l2)}점")
        collect(points_l2, "Layer2", radius_m=1000, max_pages=3)

    # ── Layer 3: 인기 상권 집중 수집 ─────────────────────────────────────────
    if args.layer in (0, 3):
        print("=" * 65)
        print("Layer 3: 인기 상권 집중 수집")
        print("=" * 65)

        points_l3 = [(lat, lon) for lat, lon, _ in POPULAR_DISTRICTS]
        labels = [name for _, _, name in POPULAR_DISTRICTS]
        print(f"  상권 수: {len(points_l3)}개 → 반경 2km, 5페이지")
        for (lat, lon), label in zip(points_l3, labels):
            print(f"    수집중: {label}")
            for cat_code in KAKAO_CATEGORIES:
                results = search_kakao_category(lon, lat, cat_code, radius_m=2000, max_pages=5)
                for r in results:
                    key = f"{r['name']}|{round(r['lon'],5)}|{round(r['lat'],5)}"
                    if key not in seen_keys and r["name"]:
                        seen_keys.add(key)
                        all_records.append(r)
                time.sleep(args.delay)

        save_checkpoint(all_records)
        print(f"  [Layer3] 완료 — 누계 {len(all_records):,}건\n")

    # ── 결과 정리 ─────────────────────────────────────────────────────────────
    print("=" * 65)
    print("결과 정리")
    print("=" * 65)

    if not all_records:
        print("  수집된 데이터 없음.")
        return

    df = pd.DataFrame(all_records)

    # 좌표 범위 필터
    before = len(df)
    df = df.dropna(subset=["lon", "lat"])
    df = df[df["lat"].between(33.0, 38.5) & df["lon"].between(124.0, 132.0)]
    df = df[df["name"].str.strip() != ""]
    print(f"  필터링: {before:,} → {len(df):,}행 ({before - len(df):,}행 제거)")

    # 중복 제거 (name + 좌표 반올림)
    df["_lon_r"] = df["lon"].round(4)
    df["_lat_r"] = df["lat"].round(4)
    df = df.drop_duplicates(subset=["name", "_lon_r", "_lat_r"])
    df = df.drop(columns=["_lon_r", "_lat_r"])
    print(f"  중복 제거 후: {len(df):,}행")

    # 통계
    print(f"\n  카테고리 분포:")
    print(df["category"].value_counts().to_string())
    print(f"\n  시도별 분포 (상위 10):")
    print(df["sido"].value_counts().head(10).to_string())

    # CSV 저장
    col_order = ["name", "category", "sub_category", "address",
                 "sido", "sigungu", "lon", "lat", "source", "place_url"]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values(["sido", "sigungu", "name"]).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] CSV 저장: {OUTPUT_PATH}")
    print(f"     {len(df):,}행 × {len(df.columns)}열")

    # DB 저장
    if args.db:
        print()
        upsert_to_db(df)

    # 체크포인트 삭제 (완료)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("  체크포인트 파일 삭제 완료")

    print("\n" + "=" * 65)
    print(f"[DONE] 전국 맛집 POI 수집 완료: {len(df):,}건")
    print(f"  저장 위치: {OUTPUT_PATH}")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전국 맛집/카페 POI 수집 (카카오 로컬 FD6/CE7)")
    parser.add_argument(
        "--layer", type=int, default=0, choices=[0, 1, 2, 3],
        help="수집 레이어 (0=전체, 1=tour_poi앵커, 2=시군구격자, 3=인기상권)",
    )
    parser.add_argument(
        "--radius", type=int, default=500,
        help="[Layer1] 검색 반경(m) — 기본 500",
    )
    parser.add_argument(
        "--pages", type=int, default=3,
        help="[Layer1/2] 페이지 수 — 기본 3 (최대 45건/카테고리/포인트)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.1,
        help="요청 간 딜레이(초) — 기본 0.1",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="체크포인트에서 이어서 수집",
    )
    parser.add_argument(
        "--db", action="store_true",
        help="수집 완료 후 Neon PostgreSQL poi 테이블에 upsert",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("전국 맛집 POI 수집 시작 (카카오 로컬 API)")
    print(f"  Layer: {args.layer if args.layer else '전체(1+2+3)'}")
    print(f"  반경: {args.radius}m  |  페이지: {args.pages}  |  딜레이: {args.delay}s")
    print(f"  DB 저장: {'ON' if args.db else 'OFF'}")
    print(f"  이어받기: {'ON' if args.resume else 'OFF'}")
    print("=" * 65)
    print()

    main(args)
