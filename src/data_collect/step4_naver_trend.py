# =============================================================
# step4_naver_trend.py
# 네이버 DataLab 검색어 트렌드 → POI별 sns_mention_norm 산출
# =============================================================
# 전제 조건:
#   - 네이버 krider 앱에 DataLab 검색어 트렌드 API 활성화 완료 ✅
#   - 환경 변수: NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
#     (또는 하단 직접 입력 — 공개 저장소엔 올리지 말 것)
#
# 수집 전략:
#   - tour_poi.csv 제목 기준 1회 5개 키워드 배치 처리
#   - 1,000콜/일 × 5개 = 5,000 POI/일 처리 가능
#   - 전국 15,905건 → 약 4일 배치 (--offset으로 구간 지정)
#
# 실행 예시:
#   python step4_naver_trend.py                         # 전체 (1일 처리 가능 분량)
#   python step4_naver_trend.py --offset 0 --limit 5000 # 1일차: 0~4999
#   python step4_naver_trend.py --offset 5000           # 2일차: 5000~
#   python step4_naver_trend.py --max_calls 500         # 오늘 500콜만 사용
# =============================================================

import argparse
import os
import time

import pandas as pd
import requests

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

TOUR_POI_PATH = os.path.join(BASE_DIR, "data", "raw_ml", "tour_poi.csv")
OUTPUT_PATH   = os.path.join(BASE_DIR, "data", "raw_ml", "poi_trend.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ── API 키 (환경 변수 우선, 없으면 직접 입력) ─────────────────────────────────
NAVER_CLIENT_ID     = os.environ.get("NAVER_CLIENT_ID",     "여기에_Client_ID_입력")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "여기에_Client_Secret_입력")

DATALAB_URL = "https://openapi.naver.com/v1/datalab/search"

# ── 트렌드 수집 기간 ──────────────────────────────────────────────────────────
TREND_START = "2024-01-01"
TREND_END   = "2024-12-31"
TIME_UNIT   = "month"   # "date" | "week" | "month"

# ── 핵심 함수 ─────────────────────────────────────────────────────────────────
def get_search_trend(keyword_groups: list[dict],
                     start: str = TREND_START,
                     end: str   = TREND_END,
                     time_unit: str = TIME_UNIT) -> dict[str, float]:
    """
    네이버 DataLab 검색어 트렌드 1회 호출.

    keyword_groups: [{"groupName": "한강공원", "keywords": ["한강공원", "한강 공원"]}]
                    최대 5개 그룹
    반환: {groupName: avg_ratio (0~100)}  — 0이면 검색 데이터 없음
    """
    try:
        resp = requests.post(
            DATALAB_URL,
            headers={
                "X-Naver-Client-Id":     NAVER_CLIENT_ID,
                "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
                "Content-Type":          "application/json",
            },
            json={
                "startDate":     start,
                "endDate":       end,
                "timeUnit":      time_unit,
                "keywordGroups": keyword_groups,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        # 401: 키 오류 / 403: API 미활성화 / 429: 일일 한도 초과
        print(f"    ❌ HTTP {resp.status_code}: {e}")
        if resp.status_code == 429:
            print("    → 일일 한도(1,000콜) 초과. 내일 --offset으로 이어서 실행하세요.")
        return None   # None = 한도 초과 또는 치명 오류
    except requests.exceptions.RequestException as e:
        print(f"    ⚠️ 요청 오류: {e}")
        return {}

    result = {}
    for item in data.get("results", []):
        ratios = [p["ratio"] for p in item.get("data", []) if p.get("ratio", 0) > 0]
        result[item["title"]] = round(sum(ratios) / len(ratios), 4) if ratios else 0.0
    return result


def build_keyword_groups(titles: list[str]) -> list[dict]:
    """
    POI 제목 목록 → DataLab keywordGroups 형식 변환.
    같은 POI의 띄어쓰기 변형을 keywords 배열에 포함.
    """
    groups = []
    for t in titles:
        # 원래 제목 + 공백 없는 변형 (최대 2개 키워드)
        kws = list(dict.fromkeys([t, t.replace(" ", "")]))[:5]
        groups.append({"groupName": t, "keywords": kws})
    return groups


# ── CLI 파싱 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="네이버 DataLab POI 트렌드 수집")
parser.add_argument(
    "--offset", type=int, default=0,
    help="tour_poi.csv 시작 행 오프셋 (1일차=0, 2일차=5000 등)",
)
parser.add_argument(
    "--limit", type=int, default=5000,
    help="이번 실행에서 처리할 최대 POI 수 (기본 5000 = 1,000콜/일 분량)",
)
parser.add_argument(
    "--max_calls", type=int, default=1000,
    help="이번 실행의 최대 API 콜 수 (기본 1,000 = 일일 한도)",
)
parser.add_argument(
    "--batch_size", type=int, default=5,
    help="1회 API 콜당 키워드 그룹 수 (최대 5, 기본 5)",
)
parser.add_argument(
    "--delay", type=float, default=0.12,
    help="API 요청 간 딜레이(초) — 기본 0.12s",
)
parser.add_argument(
    "--start", type=str, default=TREND_START,
    help=f"트렌드 수집 시작일 (기본: {TREND_START})",
)
parser.add_argument(
    "--end", type=str, default=TREND_END,
    help=f"트렌드 수집 종료일 (기본: {TREND_END})",
)
args = parser.parse_args()

batch_size = min(args.batch_size, 5)  # DataLab 최대 5개

# ── tour_poi.csv 로드 ─────────────────────────────────────────────────────────
if not os.path.exists(TOUR_POI_PATH):
    print(f"❌ tour_poi.csv 없음: {TOUR_POI_PATH}")
    raise SystemExit(1)

tour_df = pd.read_csv(TOUR_POI_PATH, encoding="utf-8-sig")
all_titles = tour_df["title"].dropna().unique().tolist()
target_titles = all_titles[args.offset: args.offset + args.limit]
print(f"tour_poi.csv: 전체 {len(all_titles):,}개 POI")
print(f"이번 처리 범위: {args.offset} ~ {args.offset + len(target_titles) - 1} "
      f"({len(target_titles):,}개)")
print(f"예상 API 콜: {-(-len(target_titles)//batch_size)}콜 "
      f"(일일 한도 {args.max_calls}콜)")

# ── 기존 트렌드 데이터 로드 (증분 방지) ───────────────────────────────────────
existing_trends: dict[str, float] = {}
if os.path.exists(OUTPUT_PATH):
    try:
        exist_df = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
        existing_trends = dict(zip(exist_df["title"], exist_df["trend_avg"]))
        print(f"  기존 트렌드 데이터: {len(existing_trends):,}개 → 중복 건너뜀")
    except Exception as e:
        print(f"  ⚠️ 기존 파일 로드 실패: {e}")

# 이미 수집된 title 제외
target_titles = [t for t in target_titles if t not in existing_trends]
print(f"  미수집 POI: {len(target_titles):,}개")

# ── 메인 수집 루프 ────────────────────────────────────────────────────────────
new_trends: dict[str, float] = {}
call_count  = 0
stop_reason = "완료"

for i in range(0, len(target_titles), batch_size):
    if call_count >= args.max_calls:
        stop_reason = f"일일 한도 {args.max_calls}콜 도달"
        break

    batch    = target_titles[i: i + batch_size]
    groups   = build_keyword_groups(batch)
    result   = get_search_trend(groups, start=args.start, end=args.end)
    call_count += 1

    if result is None:
        # 429 한도 초과 → 즉시 중단
        stop_reason = "일일 한도 초과 (429)"
        break

    new_trends.update(result)

    if call_count % 100 == 0:
        next_offset = args.offset + i + batch_size
        print(f"  {call_count}콜 완료 — 누적 {len(new_trends):,}개 "
              f"(다음 실행 시 --offset {next_offset})")

    time.sleep(args.delay)

print(f"\n▶ 이번 수집: {len(new_trends):,}개 POI 트렌드 ({call_count}콜 사용)")
print(f"  종료 이유: {stop_reason}")

# ── 결과 저장 ─────────────────────────────────────────────────────────────────
if new_trends:
    df_new = pd.DataFrame([
        {"title": t, "trend_avg": v} for t, v in new_trends.items()
    ])

    # 기존 데이터와 병합
    all_trends = {**existing_trends, **new_trends}
    df_final = pd.DataFrame([
        {"title": t, "trend_avg": v} for t, v in all_trends.items()
    ])
    # sns_mention_norm: trend_avg 를 0~1 min-max 정규화
    t_min = df_final["trend_avg"].min()
    t_max = df_final["trend_avg"].max()
    if t_max > t_min:
        df_final["sns_mention_norm"] = (
            (df_final["trend_avg"] - t_min) / (t_max - t_min)
        ).round(6)
    else:
        df_final["sns_mention_norm"] = 0.0

    df_final = df_final.sort_values("trend_avg", ascending=False).reset_index(drop=True)
    df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n✅ 저장 완료 → {OUTPUT_PATH}")
    print(f"   전체 누적: {len(df_final):,}개 / 전국 {len(all_titles):,}개 중 "
          f"{len(df_final)/len(all_titles)*100:.1f}% 완료")
    print(f"   trend_avg 분포: min={df_final['trend_avg'].min():.2f} "
          f"/ mean={df_final['trend_avg'].mean():.2f} "
          f"/ max={df_final['trend_avg'].max():.2f}")
    print(f"\n  상위 10 POI (검색 트렌드):")
    print(df_final[["title", "trend_avg", "sns_mention_norm"]].head(10).to_string(index=False))

    # 다음 실행 안내
    remaining = len(all_titles) - len(df_final)
    if remaining > 0:
        next_offset = args.offset + len(new_trends)
        print(f"\n  📌 남은 POI: {remaining:,}개")
        print(f"     다음 실행: python step4_naver_trend.py --offset {next_offset}")
    else:
        print("\n  🎉 전체 POI 트렌드 수집 완료!")
        print("     build_tourism_score_v2.py 실행 시 poi_trend.csv 자동 연동됩니다.")
else:
    print("❌ 수집된 트렌드 데이터가 없습니다.")
    print("   → NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경 변수를 확인하세요.")
    print("   → krider 앱에서 DataLab 검색어 트렌드 API가 활성화되었는지 확인하세요.")
