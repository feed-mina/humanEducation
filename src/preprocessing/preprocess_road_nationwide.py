"""
preprocess_road_nationwide.py
==============================
전국자전거도로표준데이터.csv → 전국 17개 시도 전처리 → road_clean_nationwide.csv

기존 preprocess_road.py 대비 변경:
  - 서울+경기 필터링 제거 → 전국 20,262행 전체 처리
  - sido 컬럼 유지 (지역별 모델 학습 및 필터링용)
  - 모델 비교 그래프는 선택적으로 생성 (--plot 옵션)
  - road_clean_v2.csv 와 동일한 컬럼 구조 유지

[ 실행 방법 ]
  python src/preprocessing/preprocess_road_nationwide.py
  python src/preprocessing/preprocess_road_nationwide.py --plot
  python src/preprocessing/preprocess_road_nationwide.py --sido 강원도 경상북도

[ 입력 파일 ]
  data/raw_ml/전국자전거도로표준데이터.csv  (EUC-KR, 20,262행)

[ 출력 파일 ]
  data/raw_ml/road_clean_nationwide.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── 경로 설정 (kride-project 루트 기준) ──────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # kride-project/
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()

RAW_DIR     = os.path.join(BASE_DIR, "data", "raw_ml")
INPUT_PATH  = os.path.join(RAW_DIR, "전국자전거도로표준데이터.csv")
OUTPUT_PATH = os.path.join(RAW_DIR, "road_clean_nationwide.csv")


# ── 컬럼 매핑 (preprocess_road.py 와 동일 구조) ──────────────────────────────
COL_MAP = {
    "노선명":               "route_name",
    "시도명":               "sido",
    "시군구명":             "sigungu",
    "기점지번주소":         "start_addr_jibun",
    "기점도로명주소":       "start_addr_road",
    "기점위도":             "start_lat",
    "기점경도":             "start_lon",
    "종점위도":             "end_lat",
    "종점경도":             "end_lon",
    "총길이(km)":           "length_km",
    "자전거전용도로너비(m)": "width_m",
    "자전거전용도로종류":   "road_type",
    "자전거전용도로이용가능여부": "is_official",
    # v2 호환 fallback 컬럼명
    "자전거도로너비(m)":    "width_m",
    "자전거도로종류":       "road_type",
    "자전거도로고시유무":   "is_official",
}

# 전국 17개 시도 표준명
ALL_SIDOS = [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
    "대전광역시", "울산광역시", "세종특별자치시", "경기도",
    "강원특별자치도", "충청북도", "충청남도", "전라북도", "전라남도",
    "경상북도", "경상남도", "제주특별자치도",
]

# 시도명 정규화 (약칭 → 표준명)
SIDO_NORMALIZE = {
    "강원도": "강원특별자치도",
    "전북특별자치도": "전라북도",
    "전북": "전라북도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "충북": "충청북도",
    "충남": "충청남도",
    "제주": "제주특별자치도",
    "서울": "서울특별시",
    "부산": "부산광역시",
    "대구": "대구광역시",
    "인천": "인천광역시",
    "광주": "광주광역시",
    "대전": "대전광역시",
    "울산": "울산광역시",
    "세종": "세종특별자치시",
    "경기": "경기도",
}


def normalize_sido(sido: str) -> str:
    sido = sido.strip()
    return SIDO_NORMALIZE.get(sido, sido)


def main(target_sidos: list[str] | None, make_plot: bool) -> None:
    # ── STEP 1: 원본 로드 ──────────────────────────────────────────────────────
    print("=" * 65)
    print("STEP 1: 원본 CSV 로드")
    print("=" * 65)

    if not os.path.exists(INPUT_PATH):
        print(f"  ❌ 입력 파일 없음: {INPUT_PATH}")
        sys.exit(1)

    try:
        df = pd.read_csv(INPUT_PATH, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    print(f"  원본 shape : {df.shape}")
    df["시도명"] = df["시도명"].str.strip().apply(normalize_sido)
    print(f"\n  전체 시도별 행 수:")
    print(df["시도명"].value_counts().to_string())
    print()

    # ── STEP 2: 시도 필터 (옵션) ───────────────────────────────────────────────
    print("=" * 65)
    print("STEP 2: 시도 필터링")
    print("=" * 65)

    if target_sidos:
        normalized_targets = [normalize_sido(s) for s in target_sidos]
        df_target = df[df["시도명"].isin(normalized_targets)].copy()
        print(f"  필터: {normalized_targets}")
        print(f"  전체 {len(df):,}행  →  필터 후 {len(df_target):,}행")
    else:
        df_target = df.copy()
        print(f"  필터 없음 (전국 전체): {len(df_target):,}행")

    if df_target.empty:
        print("  ❌ 필터 결과가 비어있습니다.")
        sys.exit(1)

    # 복합 PK (route_name + sigungu)
    if "노선명" in df_target.columns and "시군구명" in df_target.columns:
        df_target["road_id"] = (
            df_target["노선명"].fillna("unknown") + "_" +
            df_target["시군구명"].fillna("unknown")
        )

    # ── STEP 3: 컬럼 선택 및 이름 정리 ────────────────────────────────────────
    print("=" * 65)
    print("STEP 3: 컬럼 선택 및 타입 변환")
    print("=" * 65)

    # 중복 매핑 방지: 먼저 존재하는 컬럼만 선택
    existing_map: dict[str, str] = {}
    for ko, en in COL_MAP.items():
        if ko in df_target.columns and en not in existing_map.values():
            existing_map[ko] = en

    cols_to_select = list(existing_map.keys())
    if "road_id" in df_target.columns:
        cols_to_select.append("road_id")

    df_clean = df_target[cols_to_select].rename(columns=existing_map)

    # width_m 대체 컬럼 처리
    if "width_m" not in df_clean.columns:
        for alt in ["일반도로너비(m)", "자전거도로너비(m)"]:
            if alt in df_target.columns:
                df_clean["width_m"] = df_target[alt].values
                print(f"  ⚠️  width_m → '{alt}' 대체 사용")
                break
        else:
            df_clean["width_m"] = float("nan")
            print(f"  ⚠️  너비 컬럼 없음 → width_m = NaN")

    # 숫자 변환
    num_cols = ["width_m", "length_km", "start_lat", "start_lon", "end_lat", "end_lon"]
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    print(f"  정제 후 shape : {df_clean.shape}")
    print(f"  결측값 현황:")
    print(df_clean.isnull().sum().to_string())
    print()

    # ── STEP 4: 파생 피처 생성 ─────────────────────────────────────────────────
    print("=" * 65)
    print("STEP 4: 파생 피처 생성")
    print("=" * 65)

    # is_wide_road
    if "width_m" in df_clean.columns:
        df_clean["is_wide_road"] = (df_clean["width_m"] >= 2.0).astype(int)
    else:
        df_clean["is_wide_road"] = 0

    # is_official 정규화
    if "is_official" in df_clean.columns:
        df_clean["is_official"] = df_clean["is_official"].apply(
            lambda x: 1 if str(x).strip() in ["가능", "Y", "1", "True"] else 0
        )

    # safety_index (너비×0.7 + 길이×0.3, MinMaxScaler)
    valid_mask = df_clean[["width_m", "length_km"]].notna().all(axis=1)
    df_clean["safety_index"] = float("nan")
    if valid_mask.sum() > 0:
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(
            df_clean.loc[valid_mask, ["width_m", "length_km"]]
        )
        df_clean.loc[valid_mask, "safety_index"] = (
            normalized[:, 0] * 0.7 + normalized[:, 1] * 0.3
        ).round(4)

    wide = df_clean["is_wide_road"].sum()
    si_mean = df_clean["safety_index"].mean()
    print(f"  is_wide_road : 넓음(1)={wide:,}개 ({wide/len(df_clean)*100:.1f}%)")
    print(f"  safety_index : 평균={si_mean:.3f}  결측={df_clean['safety_index'].isna().sum()}")

    has_coord = df_clean[["start_lat", "start_lon"]].notna().all(axis=1)
    print(f"  좌표 있는 행 : {has_coord.sum():,} / {len(df_clean):,} ({has_coord.mean()*100:.1f}%)")
    print()

    # ── STEP 5: 시도별 분포 출력 ────────────────────────────────────────────────
    print("=" * 65)
    print("STEP 5: 전국 시도별 분포")
    print("=" * 65)

    if "sido" in df_clean.columns:
        sido_stats = df_clean.groupby("sido").agg(
            행수=("road_id" if "road_id" in df_clean.columns else "length_km", "count"),
            평균너비=("width_m", "mean"),
            평균길이=("length_km", "mean"),
            평균안전지수=("safety_index", "mean"),
        ).round(3)
        print(sido_stats.to_string())
    print()

    # ── STEP 6: 저장 ───────────────────────────────────────────────────────────
    print("=" * 65)
    print("STEP 6: CSV 저장")
    print("=" * 65)

    df_clean.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"  ✅ 저장 완료 → {OUTPUT_PATH}")
    print(f"  최종 shape  : {df_clean.shape}")
    print(f"  컬럼        : {list(df_clean.columns)}")

    # ── STEP 7: 모델 비교 그래프 (옵션) ────────────────────────────────────────
    if make_plot:
        _make_plot(df_clean, has_coord)


def _make_plot(df_clean: pd.DataFrame, has_coord: pd.Series) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
    except ImportError as e:
        print(f"\n  ⚠ 그래프 생성 스킵 (패키지 없음): {e}")
        return

    print("=" * 65)
    print("STEP 7: 모델 비교 (safety_index 예측)")
    print("=" * 65)

    df_model = df_clean[has_coord & df_clean["safety_index"].notna()].copy()
    print(f"  모델 학습 대상 : {len(df_model):,}행")

    if len(df_model) < 50:
        print("  ⚠️  학습 데이터 부족 (50행 미만) — 모델 비교 생략")
        return

    feats = ["width_m", "length_km", "start_lat", "start_lon"]
    feats = [f for f in feats if f in df_model.columns]
    X = df_model[feats].fillna(df_model[feats].median())
    y = df_model["safety_index"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    lr = LinearRegression().fit(X_tr, y_tr)
    results["LinearRegression"] = r2_score(y_te, lr.predict(X_te))

    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    results["RandomForest"] = r2_score(y_te, rf.predict(X_te))

    print("  모델 R² 비교:")
    for name, r2 in results.items():
        print(f"    {name:<25}: R²={r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, model) in zip(axes, [("LinearRegression", lr), ("RandomForest", rf)]):
        pred = model.predict(X_te)
        ax.scatter(y_te, pred, alpha=0.3, s=8)
        lims = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_title(f"{name}\nR²={results[name]:.4f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    plt.tight_layout()
    plot_path = os.path.join(RAW_DIR, "road_nationwide_model_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  그래프 저장 → {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전국 자전거도로 데이터 전처리")
    parser.add_argument(
        "--sido", nargs="+", default=None, metavar="시도명",
        help="특정 시도만 처리 (예: --sido 강원도 경상북도). 생략 시 전국 전체.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="모델 비교 그래프 생성 (sklearn 필요)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("전국 자전거도로 전처리 시작")
    print(f"  기존 preprocess_road.py 대비: 서울+경기 → 전국 전체")
    print("=" * 65)
    print()

    main(target_sidos=args.sido, make_plot=args.plot)

    print("\n" + "=" * 65)
    print("✅ preprocess_road_nationwide.py 완료")
    print("   다음 단계: python src/ml/build_safety_model.py")
    print("=" * 65)
