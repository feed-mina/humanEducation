# =============================================================
# step3_tour_collect_v2.py
# 한국관광공사 TourAPI → 관광지 POI 수집 (v2 개선판)
# =============================================================
# v1 대비 개선:
#   1. 타임아웃 10s → 60s (경기도 레저스포츠 타임아웃 해결)
#   2. 실패 페이지 재시도 (max_retry=3, 간격 랜덤화)
#   3. 수집 대상 확장 — 전국 광역시도 선택적 수집 가능 (--sido 인자)
#   4. contentTypeId=32(숙박) 추가 옵션
#   5. 기존 tour_poi.csv 보유 시 증분 업데이트 (--incremental)
#   6. 오류 요약 보고서 출력
#
# 실행 예시:
#   python step3_tour_collect_v2.py                         # 서울+경기 기본
#   python step3_tour_collect_v2.py --sido 서울 경기 인천   # 지역 선택
#   python step3_tour_collect_v2.py --sido all              # 전국
#   python step3_tour_collect_v2.py --include_lodging       # 숙박 타입 포함
#   python step3_tour_collect_v2.py --incremental           # 기존 데이터에 추가만
# =============================================================

import argparse
import os
import random
import time

import pandas as pd
import requests

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw_ml", "tour_poi.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ── API 설정 ──────────────────────────────────────────────────────────────────
API_KEY  = "1982f962b3451ad1a449051bf6266ac560540e346678c51933ae885bc2b4a95e"
BASE_URL = "https://apis.data.go.kr/B551011/KorService2/areaBasedList2"
ROWS_PER_PAGE = 1000

# ── 전국 광역시도 areaCode 매핑 ───────────────────────────────────────────────
AREA_CODES = {
    "서울":   1,
    "인천":   2,
    "대전":   3,
    "대구":   4,
    "광주":   5,
    "부산":   6,
    "울산":   7,
    "세종":   8,
    "경기":  31,
    "강원":  32,
    "충북":  33,
    "충남":  34,
    "경북":  35,
    "경남":  36,
    "전북":  37,
    "전남":  38,
    "제주":  39,
}

# ── 기본 수집 대상 (v1과 동일: 서울+경기) ─────────────────────────────────────
DEFAULT_AREAS = ["서울", "경기"]

# ── contentTypeId 매핑 ────────────────────────────────────────────────────────
CONTENT_TYPES_BASE = {
    12: "관광지",
    14: "문화시설",
    28: "레저스포츠",
}
CONTENT_TYPE_LODGING = {32: "숙박"}

# ── CLI 파싱 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="TourAPI POI 수집 v2")
parser.add_argument(
    "--sido", nargs="+", default=DEFAULT_AREAS,
    help=f"수집할 시도명 목록 (기본: {DEFAULT_AREAS}) 또는 'all' 입력 시 전국",
)
parser.add_argument(
    "--include_lodging", action="store_true",
    help="contentTypeId=32(숙박) 포함 수집",
)
parser.add_argument(
    "--timeout", type=int, default=60,
    help="API 요청 타임아웃(초) — 기본 60 (v1: 10초)",
)
parser.add_argument(
    "--max_retry", type=int, default=3,
    help="페이지 실패 시 최대 재시도 횟수 (기본 3)",
)
parser.add_argument(
    "--incremental", action="store_true",
    help="기존 tour_poi.csv 보유 시 contentid 중복 제외 후 추가",
)
args = parser.parse_args()

# 전국 수집 시 all_areas 사용
if args.sido == ["all"]:
    target_areas = list(AREA_CODES.keys())
else:
    target_areas = [s for s in args.sido if s in AREA_CODES]
    unknown = [s for s in args.sido if s not in AREA_CODES]
    if unknown:
        print(f"  ⚠️ 알 수 없는 시도: {unknown} — 건너뜀")

content_types = dict(CONTENT_TYPES_BASE)
if args.include_lodging:
    content_types.update(CONTENT_TYPE_LODGING)

print(f"수집 대상 시도: {target_areas}")
print(f"수집 대상 유형: {list(content_types.values())}")
print(f"타임아웃: {args.timeout}초 / 최대 재시도: {args.max_retry}회")


# ── 기존 데이터 로드 (증분 업데이트용) ───────────────────────────────────────
existing_ids: set = set()
if args.incremental and os.path.exists(OUTPUT_PATH):
    try:
        existing_df = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
        existing_ids = set(existing_df["contentid"].astype(str))
        print(f"  기존 데이터: {len(existing_ids):,}개 contentid 로드 → 중복 건너뜀")
    except Exception as e:
        print(f"  ⚠️ 기존 파일 로드 실패: {e}")


# ── 단일 API 페이지 호출 (재시도 포함) ───────────────────────────────────────
def fetch_page(area_code: int, content_type_id: int, page_no: int,
               timeout: int, max_retry: int) -> tuple[list, int]:
    """
    TourAPI areaBasedList2 1페이지 호출.
    실패 시 max_retry 횟수까지 지수 백오프로 재시도.
    반환: (item_list, total_count)
    """
    params = {
        "serviceKey":    API_KEY,
        "numOfRows":     ROWS_PER_PAGE,
        "pageNo":        page_no,
        "MobileOS":      "ETC",
        "MobileApp":     "KRide",
        "_type":         "json",
        "areaCode":      area_code,
        "contentTypeId": content_type_id,
        "arrange":       "A",
    }

    for attempt in range(1, max_retry + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            body      = data.get("response", {}).get("body", {})
            total_cnt = int(body.get("totalCount", 0))
            items_raw = body.get("items", {})

            if not items_raw:
                return [], total_cnt

            item_list = items_raw.get("item", [])
            if isinstance(item_list, dict):
                item_list = [item_list]

            return item_list, total_cnt

        except requests.exceptions.Timeout:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"    ⚠️ 타임아웃 (시도 {attempt}/{max_retry}, "
                  f"area={area_code}, type={content_type_id}, page={page_no})"
                  f" — {wait:.1f}초 후 재시도")
            if attempt < max_retry:
                time.sleep(wait)
            else:
                print(f"    ❌ 최대 재시도 초과 — 건너뜀")
                return None, 0

        except requests.exceptions.ConnectionError:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"    ⚠️ 연결 오류 (시도 {attempt}/{max_retry}) — {wait:.1f}초 후 재시도")
            if attempt < max_retry:
                time.sleep(wait)
            else:
                return None, 0

        except Exception as e:
            print(f"    ❌ 오류: {e}")
            return None, 0

    return None, 0


# ── 메인 수집 루프 ─────────────────────────────────────────────────────────────
all_records = []
error_log   = []  # (시도명, 유형, 페이지) 실패 목록

for sido_name in target_areas:
    area_code = AREA_CODES[sido_name]

    for ct_id, ct_name in content_types.items():
        print(f"\n▶ {sido_name} / {ct_name}(contentTypeId={ct_id}) 수집 중...")

        # 1페이지 호출 → 전체 건수 확인
        items, total_cnt = fetch_page(area_code, ct_id, 1, args.timeout, args.max_retry)

        if items is None:
            print(f"   → 첫 페이지 호출 실패, 건너뜀")
            error_log.append((sido_name, ct_name, 1))
            continue

        total_pages = max(1, (total_cnt + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
        print(f"   전체 {total_cnt:,}건 → {total_pages}페이지")

        # 중복 필터 적용
        new_items = [it for it in items
                     if str(it.get("contentid", "")) not in existing_ids]
        all_records.extend(new_items)

        for page in range(2, total_pages + 1):
            delay = 0.3 + random.uniform(0, 0.2)  # 0.3~0.5초 랜덤 딜레이
            time.sleep(delay)

            items_page, _ = fetch_page(area_code, ct_id, page, args.timeout, args.max_retry)

            if items_page is None:
                error_log.append((sido_name, ct_name, page))
                print(f"   ⚠️ {page}/{total_pages} 실패 → 오류 로그 기록 후 계속")
                continue

            new_page_items = [it for it in items_page
                              if str(it.get("contentid", "")) not in existing_ids]
            all_records.extend(new_page_items)
            print(f"   {page}/{total_pages} 완료 (+{len(new_page_items)}건 신규)")

        print(f"   ✅ {sido_name}/{ct_name} 수집 완료")


# ── 결과 정리 ─────────────────────────────────────────────────────────────────
print(f"\n▶ 신규 수집 완료: {len(all_records):,}건")

if not all_records:
    print("❌ 수집된 신규 데이터가 없습니다.")
    if error_log:
        print(f"   실패 목록: {error_log}")
else:
    df_new = pd.DataFrame(all_records)

    KEEP = ["contentid", "contenttypeid", "title", "addr1", "mapx", "mapy",
            "cat1", "cat2", "cat3"]
    available = [c for c in KEEP if c in df_new.columns]
    df_new = df_new[available].copy()
    df_new = df_new.rename(columns={"contenttypeid": "contentTypeId"})

    df_new["mapx"] = pd.to_numeric(df_new["mapx"], errors="coerce")
    df_new["mapy"] = pd.to_numeric(df_new["mapy"], errors="coerce")

    before = len(df_new)
    df_new = df_new.dropna(subset=["mapx", "mapy"])
    df_new = df_new.drop_duplicates(subset=["contentid"])
    print(f"  좌표 결측 제거 + 중복 제거: {before - len(df_new)}행 → {len(df_new):,}행")

    # 증분 업데이트: 기존 데이터에 신규 추가
    if args.incremental and os.path.exists(OUTPUT_PATH):
        try:
            df_existing = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final = df_final.drop_duplicates(subset=["contentid"])
            print(f"  증분 업데이트: 기존 {len(df_existing):,} + 신규 {len(df_new):,}"
                  f" → 합계 {len(df_final):,}행")
        except Exception:
            df_final = df_new
    else:
        df_final = df_new

    # contentTypeId 분포 출력
    type_map = {12: "관광지", 14: "문화시설", 28: "레저스포츠", 32: "숙박"}
    print("\n  contentTypeId 분포:")
    for tid, cnt in df_final["contentTypeId"].astype(float).astype(int).value_counts().items():
        print(f"    {tid}({type_map.get(int(tid), '기타')}) : {cnt:,}건")

    # 시도별 분포
    if "addr1" in df_final.columns:
        print("\n  샘플 (상위 5행):")
        print(df_final[["title", "addr1", "contentTypeId"]].head(5).to_string())

    df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 저장 완료 → {OUTPUT_PATH}")
    print(f"   최종 shape: {df_final.shape}")
    print(f"   파일 크기: {os.path.getsize(OUTPUT_PATH):,} bytes")

# ── 오류 요약 ─────────────────────────────────────────────────────────────────
if error_log:
    print(f"\n⚠️ 수집 실패 ({len(error_log)}건): ")
    for sido, ct, page in error_log:
        print(f"   - {sido} / {ct} / {page}페이지")
    print("   → 나중에 --incremental 모드로 재실행하면 실패 건만 추가 수집됩니다.")
else:
    print("\n✅ 수집 오류 없음")
