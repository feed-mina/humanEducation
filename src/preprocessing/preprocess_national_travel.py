"""
preprocess_national_travel.py
==============================
2024년 국민여행조사 국내여행 RAWDATA → ConsumeTabNet v3 학습 전처리

[ 입력 ]
  (프로젝트 루트)
    국민여행조사_국내여행_2024_데이터.txt          ← 주 입력 (cp949 탭구분)
    2024년 국민여행조사 국내여행 RAWDATA.xlsx      ← txt 없을 때 폴백
    202504-202603_데이터랩_다운로드2/
      20260427184625_방문자수 히트맵.csv           ← --with_datalab 옵션

[ 전처리 전략 ]
  1. D_TRA1_COST > 0 & notna() 필터 → 26,342행
       COST 있는 행은 S_Day/NUM/CASE/SPOT도 모두 완전 — 추가 결측 처리 불필요
  2. D_TRA1_1_SPOT 앞 2자리 → 표준 시도코드 → sido_enc (label 정수)
  3. MON_EXP_1~5 이진 합산 (==1) → income_score 0~5 → income_tier 0/1/2
  4. SA1_1~5 그대로 label encoding (코드북 없어도 정수값 그대로 사용 가능)
  5. D_TRA1_SMONTH → season 1=봄 2=여름 3=가을 4=겨울
  6. D_TRA1_ONE_COST 이상치 제거 (1%~99% 분위)
  7. 타겟: log1p(D_TRA1_ONE_COST) → log_one_cost

[ 출력 컬럼 ]
  travel_days, companion_cnt, trip_type,
  sido_enc, sido_name,
  sa1_1~sa1_5, income_score, income_tier, season,
  [popularity_ratio — --with_datalab 시 추가],
  log_one_cost (타겟), one_cost_raw, total_cost_raw

[ 실행 ]
  python src/preprocessing/preprocess_national_travel.py
  python src/preprocessing/preprocess_national_travel.py --with_datalab
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Windows cp949 터미널에서 한글/특수문자 출력 오류 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw_ml")
os.makedirs(RAW_DIR, exist_ok=True)

TXT_PATH  = os.path.join(BASE_DIR, "국민여행조사_국내여행_2024_데이터.txt")
XLSX_PATH = os.path.join(BASE_DIR, "2024년 국민여행조사 국내여행 RAWDATA.xlsx")
DATALAB_DIR     = os.path.join(BASE_DIR, "202504-202603_데이터랩_다운로드2")
VISITOR_PATH    = os.path.join(DATALAB_DIR, "20260427184625_방문자수 히트맵.csv")
OUTPUT_PATH     = os.path.join(RAW_DIR, "national_travel_consume.csv")

# ── 시도 코드 매핑 ─────────────────────────────────────────────────────────────
# SPOT 코드 앞 2자리 = 표준 행정구역 코드 (int(spot) // 1000)
SIDO_CODE_MAP: dict[int, str] = {
    11: "서울특별시",    21: "부산광역시",    22: "대구광역시",
    23: "인천광역시",    24: "광주광역시",    25: "대전광역시",
    26: "울산광역시",    29: "세종특별자치시",
    31: "경기도",        32: "강원특별자치도", 33: "충청북도",
    34: "충청남도",      35: "전북특별자치도", 36: "전라남도",
    37: "경상북도",      38: "경상남도",      39: "제주특별자치도",
}

MONTH_TO_SEASON: dict[int, int] = {
    1: 4, 2: 4,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 2,
    9: 3, 10: 3, 11: 3,
    12: 4,
}


# ── 로드 ───────────────────────────────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    """txt 우선, 없으면 xlsx 폴백"""
    if os.path.exists(TXT_PATH):
        print(f"  로드: {os.path.basename(TXT_PATH)}")
        for enc in ("cp949", "utf-8-sig", "utf-8"):
            try:
                df = pd.read_csv(TXT_PATH, sep="\t", encoding=enc, low_memory=False)
                print(f"  → shape={df.shape}, encoding={enc}")
                return df
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"인코딩 실패: {TXT_PATH}")

    if os.path.exists(XLSX_PATH):
        print(f"  로드: {os.path.basename(XLSX_PATH)} (xlsx, 시간이 걸릴 수 있음)")
        df = pd.read_excel(XLSX_PATH, sheet_name=0)
        print(f"  → shape={df.shape}")
        return df

    raise FileNotFoundError(
        f"입력 파일 없음:\n  {TXT_PATH}\n  {XLSX_PATH}"
    )


# ── 유틸 ───────────────────────────────────────────────────────────────────────
def _spot_to_sido_code(val) -> int | None:
    try:
        code = int(float(val)) // 1000
        return code if code in SIDO_CODE_MAP else None
    except (ValueError, TypeError):
        return None


def _load_visitor_ratio() -> dict[str, float]:
    """데이터랩 방문자 히트맵 → 시도별 방문자 비율 딕셔너리"""
    if not os.path.exists(VISITOR_PATH):
        print(f"  ⚠️ 방문자 히트맵 파일 없음: {VISITOR_PATH}")
        return {}
    vdf = pd.read_csv(VISITOR_PATH, encoding="utf-8-sig")
    vdf.columns = ["sido_name", "visitor_count"]
    vdf["visitor_count"] = pd.to_numeric(vdf["visitor_count"], errors="coerce")
    total = vdf["visitor_count"].sum()
    return dict(zip(vdf["sido_name"], vdf["visitor_count"] / total))


# ── 전처리 ─────────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, with_datalab: bool) -> pd.DataFrame:

    # ── 1. 소비값 유효 행 필터 ───────────────────────────────────────────────
    cost_num = pd.to_numeric(df["D_TRA1_COST"], errors="coerce")
    mask     = cost_num.notna() & (cost_num > 0)
    valid    = df[mask].copy()
    print(f"  소비값 필터 (D_TRA1_COST > 0): {len(df):,} → {len(valid):,}행")

    # ── 2. 타겟 변수 ────────────────────────────────────────────────────────
    valid["one_cost_raw"]   = pd.to_numeric(valid["D_TRA1_ONE_COST"], errors="coerce")
    valid["total_cost_raw"] = pd.to_numeric(valid["D_TRA1_COST"],     errors="coerce")

    # one_cost 없는 행은 total / NUM 으로 추산
    has_num = pd.to_numeric(valid["D_TRA1_NUM"], errors="coerce").fillna(1).clip(lower=1)
    valid["one_cost_raw"] = valid["one_cost_raw"].fillna(valid["total_cost_raw"] / has_num)

    # ── 3. 이상치 제거 (1%~99%) ──────────────────────────────────────────────
    q01 = valid["one_cost_raw"].quantile(0.01)
    q99 = valid["one_cost_raw"].quantile(0.99)
    before = len(valid)
    valid = valid[valid["one_cost_raw"].between(q01, q99)].copy()
    print(f"  이상치 제거 (1%~99%): {before:,} → {len(valid):,}행"
          f"  (₩{q01:,.0f} ~ ₩{q99:,.0f})")

    valid["log_one_cost"] = np.log1p(valid["one_cost_raw"])

    # ── 4. 여행 속성 피처 ────────────────────────────────────────────────────
    valid["travel_days"]   = pd.to_numeric(valid["D_TRA1_S_Day"], errors="coerce").fillna(0).astype(int)
    valid["companion_cnt"] = pd.to_numeric(valid["D_TRA1_NUM"],   errors="coerce").fillna(1).astype(int)
    valid["trip_type"]     = pd.to_numeric(valid["D_TRA1_CASE"],  errors="coerce").fillna(1).astype(int)

    # ── 5. SPOT 코드 → 시도 인코딩 ──────────────────────────────────────────
    valid["sido_code"] = valid["D_TRA1_1_SPOT"].apply(_spot_to_sido_code)
    sorted_sido = sorted(c for c in SIDO_CODE_MAP if c in valid["sido_code"].dropna().unique())
    sido_enc_map = {code: idx for idx, code in enumerate(sorted_sido)}
    valid["sido_enc"]  = valid["sido_code"].map(sido_enc_map).fillna(-1).astype(int)
    valid["sido_name"] = valid["sido_code"].map(SIDO_CODE_MAP).fillna("기타")
    unmapped = (valid["sido_enc"] == -1).sum()
    print(f"  SPOT → sido_enc: {len(sorted_sido)}개 시도 매핑  (미매핑 {unmapped}건)")

    # ── 6. MON_EXP → income_score / income_tier ──────────────────────────────
    mon_cols = [c for c in ["MON_EXP_1","MON_EXP_2","MON_EXP_3","MON_EXP_4","MON_EXP_5"]
                if c in valid.columns]
    valid["income_score"] = sum(
        (pd.to_numeric(valid[c], errors="coerce") == 1).astype(int)
        for c in mon_cols
    )
    # 0~1점=저지출(0), 2~3점=보통(1), 4~5점=고지출(2)
    valid["income_tier"] = pd.cut(
        valid["income_score"],
        bins=[-1, 1, 3, 5],
        labels=[0, 1, 2],
    ).astype(int)
    print(f"  income_tier: {dict(valid['income_tier'].value_counts().sort_index())}")

    # ── 7. SA1 계열 — 인구통계 (코드 그대로 사용) ────────────────────────────
    for i, col in enumerate(["SA1_1","SA1_2","SA1_3","SA1_4","SA1_5"], start=1):
        if col in valid.columns:
            valid[f"sa1_{i}"] = pd.to_numeric(valid[col], errors="coerce").fillna(0).astype(int)
        else:
            valid[f"sa1_{i}"] = 0

    # ── 8. 계절 파생 ──────────────────────────────────────────────────────────
    smonth_col = next(
        (c for c in ["D_TRA1_SMONTH", "D_TRA1_1_SMONTH"] if c in valid.columns),
        None,
    )
    if smonth_col:
        valid["season"] = (
            pd.to_numeric(valid[smonth_col], errors="coerce")
            .map(MONTH_TO_SEASON)
            .fillna(0)
            .astype(int)
        )
    else:
        valid["season"] = 0
    print(f"  season 분포: {dict(valid['season'].value_counts().sort_index())}")

    # ── 9. 데이터랩 방문자 비율 (선택) ──────────────────────────────────────
    if with_datalab:
        ratio_map = _load_visitor_ratio()
        if ratio_map:
            valid["popularity_ratio"] = valid["sido_name"].map(ratio_map).fillna(0.0)
            joined = (valid["popularity_ratio"] > 0).sum()
            print(f"  데이터랩 방문자 비율 조인: {joined:,}건")
        else:
            valid["popularity_ratio"] = 0.0
    else:
        valid["popularity_ratio"] = None

    return valid


# ── 저장 ───────────────────────────────────────────────────────────────────────
def save(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "travel_days", "companion_cnt", "trip_type",
        "sido_enc", "sido_name",
        "sa1_1", "sa1_2", "sa1_3", "sa1_4", "sa1_5",
        "income_score", "income_tier", "season",
    ]
    if df["popularity_ratio"].notna().any():
        feature_cols.append("popularity_ratio")

    target_cols = ["log_one_cost", "one_cost_raw", "total_cost_raw"]
    out = df[[c for c in feature_cols + target_cols if c in df.columns]].copy()
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    return out


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="국민여행조사 2024 전처리")
    parser.add_argument("--with_datalab", action="store_true",
                        help="데이터랩 방문자 비율 피처 추가")
    args = parser.parse_args()

    print("=" * 65)
    print("국민여행조사 2024 전처리 → national_travel_consume.csv")
    print("=" * 65)

    print(f"\n{'─'*65}")
    print("STEP 1 — 원본 데이터 로드")
    print(f"{'─'*65}")
    df = load_raw()

    print(f"\n{'─'*65}")
    print("STEP 2 — 피처 추출 + 전처리")
    print(f"{'─'*65}")
    processed = preprocess(df, args.with_datalab)

    print(f"\n{'─'*65}")
    print("STEP 3 — 저장")
    print(f"{'─'*65}")
    result = save(processed)

    # ── 요약 리포트 ───────────────────────────────────────────────────────────
    print(f"\n  최종 행수 : {len(result):,}행")
    print(f"  컬럼수   : {len(result.columns)}개 ({list(result.columns)})")
    print(f"\n  타겟(log_one_cost) 통계:")
    s = result["log_one_cost"]
    print(f"    mean={s.mean():.4f}  std={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}")
    print(f"    (역변환 중앙값: ₩{int(np.expm1(s.median())):,}원)")

    print(f"\n  income_tier 분포:")
    for tier, cnt in result["income_tier"].value_counts().sort_index().items():
        label = {0: "저지출", 1: "보통", 2: "고지출"}.get(int(tier), "?")
        print(f"    {tier}({label}): {cnt:,}건")

    print(f"\n  sido_name 분포 (상위 10):")
    for sido, cnt in result["sido_name"].value_counts().head(10).items():
        print(f"    {sido}: {cnt:,}건")

    print(f"\n  travel_days 분포:")
    for day, cnt in result["travel_days"].value_counts().sort_index().head(6).items():
        label = {0: "당일", 1: "1박2일", 2: "2박3일", 3: "3박4일"}.get(int(day), f"{day}박")
        print(f"    {label}: {cnt:,}건")

    print(f"\n  [OK] {OUTPUT_PATH}")

    print(f"\n{'='*65}")
    print("[DONE] preprocess_national_travel.py 완료")
    print(f"{'='*65}")
    print("\n다음 단계:")
    print("  python src/dl/build_consume_model_v3.py")


if __name__ == "__main__":
    main()
