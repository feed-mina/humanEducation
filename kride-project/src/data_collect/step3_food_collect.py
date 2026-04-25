# =============================================================
# step3_food_collect.py
# 카카오 로컬 API → 음식점/카페 POI 수집 (반경 검색 기반)
# Naver 지역 검색은 --provider naver 로 대체 사용 가능
# =============================================================
# 전제 조건 (기본: Kakao):
#   - Kakao 앱 REST API 키 (비즈니스 등록 불필요)
#   - 환경 변수: KAKAO_REST_API_KEY
#     (또는 하단 KAKAO_REST_API_KEY 직접 입력)
#
# 전제 조건 (--provider naver):
#   - 네이버 krider 앱에 '지역(검색)' API 추가 신청 완료
#   - 환경 변수: NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
#
# 수집 전략 (Kakao 기본):
#   - tour_poi.csv 관광지 좌표 중심으로 반경 N km 내 FD6(음식점)/CE7(카페) 수집
#   - 카테고리 코드 기반 → 텍스트 검색보다 정확하고 구조화된 데이터
#   - 무료 한도: 300,000콜/일
#
# 실행 예시:
#   python step3_food_collect.py                            # tour_poi 좌표 기반 (Kakao)
#   python step3_food_collect.py --radius 1000              # 반경 1km (기본 500m)
#   python step3_food_collect.py --max_poi 500              # 상위 500개 관광지 기준
#   python step3_food_collect.py --provider naver           # 네이버 지역 검색 사용
#   python step3_food_collect.py --incremental              # 기존 food_poi.csv에 추가
# =============================================================

import argparse
import os
import re
import time

import pandas as pd
import requests

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

TOUR_POI_PATH = os.path.join(BASE_DIR, "data", "raw_ml", "tour_poi.csv")
OUTPUT_PATH   = os.path.join(BASE_DIR, "data", "raw_ml", "food_poi.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ── API 키 ────────────────────────────────────────────────────────────────────
KAKAO_REST_API_KEY  = os.environ.get("KAKAO_REST_API_KEY",  "여기에_Kakao_REST_API_키_입력")
NAVER_CLIENT_ID     = os.environ.get("NAVER_CLIENT_ID",     "여기에_Client_ID_입력")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "여기에_Client_Secret_입력")

KAKAO_CATEGORY_URL  = "https://dapi.kakao.com/v2/local/search/category.json"
NAVER_LOCAL_URL     = "https://openapi.naver.com/v1/search/local.json"

# ── 카카오 카테고리 코드 ──────────────────────────────────────────────────────
KAKAO_CATEGORIES = {
    "FD6": "음식점",
    "CE7": "카페",
}


# ── HTML 태그 제거 ────────────────────────────────────────────────────────────
def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


# ── Kakao: 카테고리 기반 반경 검색 ───────────────────────────────────────────
def search_kakao_category(lon: float, lat: float,
                          category_code: str,
                          radius_m: int = 500,
                          max_pages: int = 3) -> list[dict]:
    """
    카카오 로컬 카테고리 검색 (반경 내 음식점/카페).
    - 한 페이지 최대 15개, max_pages까지 수집
    - 반환: [{"name", "category", "address", "lon", "lat", "source", "place_url"}, ...]
    """
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
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"    ⚠️ Kakao 요청 오류 (cat={category_code}, page={page}): {e}")
            break

        items = data.get("documents", [])
        for item in items:
            results.append({
                "name":       item.get("place_name", ""),
                "category":   item.get("category_name", "").split(" > ")[-1],
                "address":    item.get("road_address_name") or item.get("address_name", ""),
                "lon":        float(item.get("x", 0)),
                "lat":        float(item.get("y", 0)),
                "source":     f"kakao_{KAKAO_CATEGORIES.get(category_code, category_code)}",
                "place_url":  item.get("place_url", ""),
            })

        # 마지막 페이지 확인
        meta = data.get("meta", {})
        if meta.get("is_end", True):
            break

    return results


# ── Naver: 텍스트 기반 지역 검색 ─────────────────────────────────────────────
def search_naver_local(query: str, display: int = 5) -> list[dict]:
    """
    네이버 지역 검색 API (--provider naver 모드).
    mapx/mapy → KATECH → WGS84 변환 포함.
    """
    try:
        resp = requests.get(
            NAVER_LOCAL_URL,
            headers={
                "X-Naver-Client-Id":     NAVER_CLIENT_ID,
                "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
            },
            params={"query": query, "display": display, "sort": "comment"},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except requests.exceptions.RequestException as e:
        print(f"    ⚠️ Naver 요청 오류 [{query}]: {e}")
        return []

    results = []
    for item in items:
        try:
            lon = int(item.get("mapx", 0)) / 10_000_000.0
            lat = int(item.get("mapy", 0)) / 10_000_000.0
        except (ValueError, TypeError):
            continue
        if not (33.0 <= lat <= 38.5 and 126.0 <= lon <= 130.0):
            continue
        results.append({
            "name":      strip_html(item.get("title", "")),
            "category":  item.get("category", ""),
            "address":   item.get("roadAddress") or item.get("address", ""),
            "lon":       lon,
            "lat":       lat,
            "source":    "naver_local",
            "place_url": item.get("link", ""),
        })
    return results


# ── CLI 파싱 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="음식점/카페 POI 수집 (Kakao/Naver)")
parser.add_argument(
    "--provider", choices=["kakao", "naver"], default="kakao",
    help="API 제공자 (기본: kakao)",
)
parser.add_argument(
    "--max_poi", type=int, default=500,
    help="tour_poi.csv 상위 N개 관광지 기준 (기본 500)",
)
parser.add_argument(
    "--radius", type=int, default=500,
    help="[Kakao] 검색 반경(m) — 기본 500m",
)
parser.add_argument(
    "--max_pages", type=int, default=3,
    help="[Kakao] 페이지당 15개 × max_pages = 최대 45개/POI/카테고리 (기본 3)",
)
parser.add_argument(
    "--display", type=int, default=5,
    help="[Naver] 쿼리당 최대 결과 수 (1~5, 기본 5)",
)
parser.add_argument(
    "--delay", type=float, default=0.1,
    help="요청 간 딜레이(초) — 기본 0.1s",
)
parser.add_argument(
    "--incremental", action="store_true",
    help="기존 food_poi.csv 보유 시 중복 name 제외 후 추가",
)
args = parser.parse_args()

# ── tour_poi.csv 로드 ─────────────────────────────────────────────────────────
if not os.path.exists(TOUR_POI_PATH):
    print(f"❌ tour_poi.csv 없음: {TOUR_POI_PATH}")
    raise SystemExit(1)

tour_df = pd.read_csv(TOUR_POI_PATH, encoding="utf-8-sig")
tour_df = tour_df.dropna(subset=["mapx", "mapy"])
tour_df = tour_df.head(args.max_poi)
print(f"tour_poi.csv: 상위 {len(tour_df):,}개 관광지 (좌표 보유) → {args.provider} API 수집")

# ── 기존 데이터 로드 ──────────────────────────────────────────────────────────
existing_names: set = set()
if args.incremental and os.path.exists(OUTPUT_PATH):
    try:
        existing_df = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
        existing_names = set(existing_df["name"].astype(str))
        print(f"  기존 데이터: {len(existing_names):,}개 장소 로드 → 중복 건너뜀")
    except Exception as e:
        print(f"  ⚠️ 기존 파일 로드 실패: {e}")

# ── 메인 수집 루프 ────────────────────────────────────────────────────────────
all_records: list[dict] = []

for idx, row in tour_df.iterrows():
    lon = float(row["mapx"])
    lat = float(row["mapy"])
    poi_name = row.get("title", f"poi_{idx}")

    if args.provider == "kakao":
        for cat_code in KAKAO_CATEGORIES:
            results = search_kakao_category(lon, lat, cat_code,
                                            radius_m=args.radius,
                                            max_pages=args.max_pages)
            new = [r for r in results if r["name"] not in existing_names]
            for r in new:
                existing_names.add(r["name"])
            all_records.extend(new)
            time.sleep(args.delay)
    else:
        # Naver: POI 이름 + "맛집" / "카페" 쿼리
        for suffix in ("맛집", "카페"):
            query = f"{poi_name} {suffix}"
            results = search_naver_local(query, display=args.display)
            new = [r for r in results if r["name"] not in existing_names]
            for r in new:
                existing_names.add(r["name"])
            all_records.extend(new)
            time.sleep(args.delay)

    if (tour_df.index.get_loc(idx) + 1) % 50 == 0:
        print(f"  진행: {tour_df.index.get_loc(idx) + 1}/{len(tour_df)} "
              f"— 신규 수집 {len(all_records):,}건")

# ── 결과 정리 ─────────────────────────────────────────────────────────────────
print(f"\n▶ 신규 수집 완료: {len(all_records):,}건")

if not all_records:
    print("❌ 수집된 데이터가 없습니다.")
    if args.provider == "kakao":
        print("   → KAKAO_REST_API_KEY 환경 변수를 확인하세요.")
    else:
        print("   → NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 및 지역 검색 API 활성화를 확인하세요.")
else:
    df_new = pd.DataFrame(all_records)

    # 좌표 결측 및 범위 이상 제거
    before = len(df_new)
    df_new = df_new.dropna(subset=["lon", "lat"])
    df_new = df_new[
        (df_new["lat"].between(33.0, 38.5)) &
        (df_new["lon"].between(126.0, 130.0))
    ]
    df_new = df_new.drop_duplicates(subset=["name"])
    print(f"  결측/이상/중복 제거: {before - len(df_new)}행 → {len(df_new):,}행")

    # 소스/카테고리 분포
    print("\n  출처 분포:")
    print(df_new["source"].value_counts().to_string())
    print("\n  카테고리 상위 10:")
    print(df_new["category"].value_counts().head(10).to_string())

    # 증분 업데이트
    if args.incremental and os.path.exists(OUTPUT_PATH):
        try:
            df_existing = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final = df_final.drop_duplicates(subset=["name"])
            print(f"\n  증분 업데이트: 기존 {len(df_existing):,} + 신규 {len(df_new):,}"
                  f" → 합계 {len(df_final):,}행")
        except Exception:
            df_final = df_new
    else:
        df_final = df_new

    df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 저장 완료 → {OUTPUT_PATH}")
    print(f"   최종 shape: {df_final.shape}")
    print(f"   파일 크기: {os.path.getsize(OUTPUT_PATH):,} bytes")
    print(f"\n  샘플 (상위 5행):")
    print(df_final[["name", "category", "address", "source"]].head(5).to_string(index=False))
