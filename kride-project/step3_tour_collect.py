# =============================================================
# Step 3: 한국관광공사 TourAPI → 서울+경기 관광지 POI 수집
# =============================================================
# API   : 한국관광공사 국문 관광정보 서비스_GW (areaBasedList2)
# 입력  : 없음 (API 직접 호출)
# 출력  : data/raw_ml/tour_poi.csv
#         - 컬럼: contentid, contentTypeId, title, addr1, mapx, mapy, cat1, cat2, cat3
#
# 수집 대상:
#   지역: 서울특별시(areaCode=1) + 경기도(areaCode=31)
#   유형: 관광지(12), 문화시설(14), 레저스포츠(28)
#
# Spatial Join 에서의 활용:
#   자전거도로 경로 1km 반경 내 POI 수 집계
#   → tourist_count, cultural_count, leisure_count 피처 생성
# =============================================================

import requests
import pandas as pd
import os
import time

# ── 경로 설정 (Jupyter 호환) ──────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw_ml", "tour_poi.csv")

# ── API 설정 ──────────────────────────────────────────────
API_KEY   = "1982f962b3451ad1a449051bf6266ac560540e346678c51933ae885bc2b4a95e"
BASE_URL  = "https://apis.data.go.kr/B551011/KorService2/areaBasedList2"

# 수집 대상 설정
AREA_CODES = {
    1:  "서울특별시",
    31: "경기도",
}

CONTENT_TYPES = {
    12: "관광지",
    14: "문화시설",
    28: "레저스포츠",
}

ROWS_PER_PAGE = 1000  # API 1회 최대 요청 수

# =============================================================
# 함수: 단일 API 페이지 호출
# =============================================================
def fetch_page(area_code, content_type_id, page_no):
    """
    areaBasedList2 API 1페이지 호출 후 items 반환.
    실패 시 None 반환.
    """
    params = {
        "serviceKey"   : API_KEY,
        "numOfRows"    : ROWS_PER_PAGE,
        "pageNo"       : page_no,
        "MobileOS"     : "ETC",
        "MobileApp"    : "KRide",
        "_type"        : "json",          # JSON 응답 요청
        "areaCode"     : area_code,
        "contentTypeId": content_type_id,
        "arrange"      : "A",             # 제목순 정렬
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # 응답 구조: response → body → items → item (리스트)
        body      = data.get("response", {}).get("body", {})
        total_cnt = body.get("totalCount", 0)
        items     = body.get("items", {})

        if not items:
            return [], total_cnt

        item_list = items.get("item", [])
        # item이 딕셔너리(단일)인 경우 리스트로 변환
        if isinstance(item_list, dict):
            item_list = [item_list]

        return item_list, total_cnt

    except requests.exceptions.Timeout:
        print(f"    ⚠️ 타임아웃 (area={area_code}, type={content_type_id}, page={page_no})")
        return None, 0
    except Exception as e:
        print(f"    ❌ 오류: {e}")
        return None, 0


# =============================================================
# 메인: 전체 수집 루프
# =============================================================
all_records = []  # 전체 수집 결과 저장

for area_code, area_name in AREA_CODES.items():
    for ct_id, ct_name in CONTENT_TYPES.items():

        print(f"\n▶ {area_name} / {ct_name}(contentTypeId={ct_id}) 수집 중...")

        # ── 1페이지 먼저 호출해서 전체 건수 확인
        items, total_cnt = fetch_page(area_code, ct_id, page_no=1)

        if items is None:
            print(f"   → 첫 페이지 호출 실패, 건너뜀")
            continue

        print(f"   전체 {total_cnt:,}건 → {(total_cnt // ROWS_PER_PAGE) + 1}페이지")
        all_records.extend(items)

        # ── 2페이지 이후 수집
        total_pages = (total_cnt // ROWS_PER_PAGE) + 1

        for page in range(2, total_pages + 1):
            time.sleep(0.3)  # API 과부하 방지 (0.3초 딜레이)
            items_page, _ = fetch_page(area_code, ct_id, page)

            if items_page is None:
                print(f"   ⚠️ {page}페이지 실패, 건너뜀")
                continue

            all_records.extend(items_page)
            print(f"   {page}/{total_pages} 페이지 완료 (+{len(items_page)}건)")

        print(f"   ✅ {area_name}/{ct_name} 수집 완료")


# =============================================================
# 결과 정리 및 저장
# =============================================================
print(f"\n▶ 전체 수집 완료: {len(all_records):,}건")

if not all_records:
    print("❌ 수집된 데이터가 없습니다. API 키 또는 네트워크를 확인하세요.")
else:
    df_tour = pd.DataFrame(all_records)

    print(f"  DataFrame 컬럼: {list(df_tour.columns)}")

    # ── 필요한 컬럼만 선택 (존재하는 경우에만)
    KEEP = ["contentid", "contenttypeid", "title", "addr1", "mapx", "mapy",
            "cat1", "cat2", "cat3"]
    available = [c for c in KEEP if c in df_tour.columns]
    df_tour = df_tour[available].copy()

    # ── 컬럼명 통일
    df_tour = df_tour.rename(columns={"contenttypeid": "contentTypeId"})

    # ── 좌표를 숫자형으로 변환
    df_tour["mapx"] = pd.to_numeric(df_tour["mapx"], errors="coerce")
    df_tour["mapy"] = pd.to_numeric(df_tour["mapy"], errors="coerce")

    # ── 좌표 없는 행 제거 (Spatial Join 불가)
    before = len(df_tour)
    df_tour = df_tour.dropna(subset=["mapx", "mapy"])
    after  = len(df_tour)
    print(f"  좌표 결측 제거: {before - after}행 제거 → {after:,}행 남음")

    # ── 중복 제거 (동일 POI가 여러 페이지에서 중복될 수 있음)
    df_tour = df_tour.drop_duplicates(subset=["contentid"])
    print(f"  중복 제거 후   : {len(df_tour):,}행")

    # ── contentTypeId 별 분포
    print("\n  contentTypeId 분포:")
    type_map = {12: "관광지", 14: "문화시설", 28: "레저스포츠"}
    for tid, cnt in df_tour["contentTypeId"].astype(int).value_counts().items():
        print(f"    {tid}({type_map.get(int(tid), '기타')}) : {cnt:,}건")

    # ── 샘플 출력
    print("\n  샘플 (상위 5행):")
    print(df_tour[["title", "addr1", "mapx", "mapy", "contentTypeId"]].head(5).to_string())

    # ── CSV 저장
    df_tour.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    # index=False → 행 번호 미포함
    # utf-8-sig   → Excel 한글 깨짐 방지

    print(f"\n✅ 저장 완료 → {OUTPUT_PATH}")
    print(f"   파일 크기 : {os.path.getsize(OUTPUT_PATH):,} bytes")
    print(f"   최종 shape: {df_tour.shape}")
