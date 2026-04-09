"""
build_attraction_model.py
=========================
AI Hub 여행로그 방문지 정보 → TabNet POI 매력도 예측 모델 학습

[ 목적 ]
  현재 tourism_score는 POI 밀도 기반 규칙 점수.
  이 스크립트는 실제 방문자 만족도(DGSTFN, REVISIT_INTENTION, RCMDTN_INTENTION)를
  타겟으로 TabNet을 학습해 POI별 '매력도 점수'를 도출한다.

  tourism_score_v2 = 0.7 × tourism_score + 0.3 × attraction_mean

[ 데이터 ]
  AI Hub 국내 여행로그(수도권) 2023
  경로: data/ai-hub/국내 여행로그 수도권_2023/02.라벨링데이터/
  필요 파일:
    tn_visit_area_info_방문지정보_E.csv  (21,384행 × 23컬럼)

[ 실행 ]
  python kride-project/build_attraction_model.py
  python kride-project/build_attraction_model.py --use_dummy   ← 데이터 없을 때

[ 출력 ]
  models/attraction_regressor.zip   ← TabNetRegressor 가중치
  models/attraction_scaler.pkl      ← StandardScaler
  models/attraction_meta.json       ← 피처 목록 + 성능 메타
  data/raw_ml/poi_attraction.csv    ← POI별 평균 매력도 (Spatial Join용)

[ 입력 피처 (7개) ]
  visit_area_type_cd, residence_time_min, sgg_cd,
  visit_chc_reason_cd, visit_order, x_coord, y_coord

[ 타겟 ]
  attraction_score = DGSTFN×0.4 + REVISIT_INTENTION×0.3 + RCMDTN_INTENTION×0.3
  (0~5 범위, 높을수록 방문자 만족도 높음)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()

RAW_ML_DIR = os.path.join(BASE_DIR, "data", "raw_ml")
MODELS_DIR = os.path.join(BASE_DIR, "models")
AIHUB_DIR  = os.path.join(
    BASE_DIR, "data", "ai-hub",
    "국내 여행로그 수도권_2023", "02.라벨링데이터"
)
os.makedirs(RAW_ML_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 피처 / 타겟 정의 ──────────────────────────────────────────────────────────
FEATURE_COLS = [
    "visit_area_type_cd",   # 방문지 유형 코드 (카테고리)
    "residence_time_min",   # 체류 시간(분) — 연속형
    "sgg_cd",               # 시군구 코드 (카테고리)
    "visit_chc_reason_cd",  # 방문 선택 이유 코드 (카테고리)
    "visit_order",          # 여행 내 방문 순서 — 연속형
    "x_coord",              # 경도 — 연속형
    "y_coord",              # 위도 — 연속형
]
TARGET_COL = "attraction_score"

# 주거지 유형 코드 (방문지로 부적절)
RESIDENTIAL_TYPE_CD = 21


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
def _read_csv(path: str | None) -> pd.DataFrame | None:
    """인코딩 자동 감지 CSV 로드"""
    if path is None or not os.path.exists(path):
        return None
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"  로드: {os.path.basename(path)}  shape={df.shape}")
            return df
        except UnicodeDecodeError:
            continue
    print(f"  ❌ 인코딩 실패: {path}")
    return None


def _find_csv(directory: str, keyword: str) -> str | None:
    """디렉토리(+하위)에서 keyword 포함 첫 번째 CSV 경로 반환"""
    if not os.path.isdir(directory):
        return None
    for fname in os.listdir(directory):
        if keyword.lower() in fname.lower() and fname.endswith(".csv"):
            return os.path.join(directory, fname)
    matches = glob.glob(os.path.join(directory, "**", f"*{keyword}*"), recursive=True)
    csv_matches = [m for m in matches if m.endswith(".csv")]
    return csv_matches[0] if csv_matches else None


def load_visit_area(data_dir: str) -> pd.DataFrame:
    """tn_visit_area_info CSV 로드"""
    path = _find_csv(data_dir, "visit_area")
    df = _read_csv(path)
    if df is None:
        print("  ❌ tn_visit_area_info CSV 없음")
        return pd.DataFrame()
    print(f"  방문지 정보: {df.shape[0]:,}행 × {df.shape[1]}열")
    return df


def make_dummy_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    AI Hub 데이터가 없을 때 사용하는 합성 방문지 데이터.
    실제 컬럼 구조와 동일하게 생성.
    """
    rng = np.random.default_rng(seed)

    # 서울·경기 대표 시군구 코드
    sgg_pool = [
        "11110", "11140", "11170", "11200", "11215",  # 서울
        "41110", "41130", "41150", "41280", "41310",  # 경기
    ]
    # 방문지 유형 (21=주거지는 제외)
    type_cd_pool = [1, 2, 3, 4, 5, 10, 11, 12, 20, 22, 30]

    # 서울·수도권 위도/경도 범위
    x = rng.uniform(126.7, 127.5, n)  # 경도
    y = rng.uniform(37.4, 37.7, n)    # 위도

    # 만족도 / 재방문 / 추천 (1~5 정수)
    dgstfn   = rng.integers(1, 6, n).astype(float)
    revisit  = rng.integers(1, 6, n).astype(float)
    rcmdtn   = rng.integers(1, 6, n).astype(float)

    df = pd.DataFrame({
        "VISIT_AREA_NM":       [f"POI_{i}" for i in range(n)],
        "VISIT_AREA_TYPE_CD":  rng.choice(type_cd_pool, n),
        "RESIDENCE_TIME_MIN":  rng.integers(10, 180, n).astype(float),
        "SGG_CD":              rng.choice(sgg_pool, n),
        "VISIT_CHC_REASON_CD": rng.integers(1, 10, n).astype(float),
        "VISIT_ORDER":         rng.integers(1, 8, n).astype(float),
        "X_COORD":             x,
        "Y_COORD":             y,
        "DGSTFN":              dgstfn,
        "REVISIT_INTENTION":   revisit,
        "RCMDTN_INTENTION":    rcmdtn,
        "POI_ID":              [f"P{i:05d}" for i in range(n)],
    })
    print(f"  합성 더미 데이터 생성: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 전처리
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """방문지 정보 → TabNet 입력 피처 + 매력도 타겟 생성"""

    # ── 컬럼명 정규화 (대소문자 통일) ─────────────────────────────────────────
    df = df.copy()
    df.columns = [c.upper().strip() for c in df.columns]

    # ── 주거지 제거 ────────────────────────────────────────────────────────────
    if "VISIT_AREA_TYPE_CD" in df.columns:
        df = df[df["VISIT_AREA_TYPE_CD"] != RESIDENTIAL_TYPE_CD]

    # ── 좌표 없는 행 제거 ──────────────────────────────────────────────────────
    for coord_col in ("X_COORD", "Y_COORD"):
        if coord_col in df.columns:
            df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce")
    df = df.dropna(subset=["X_COORD", "Y_COORD"])

    # ── 타겟: 매력도 점수 (0~5) ────────────────────────────────────────────────
    # DGSTFN·REVISIT_INTENTION·RCMDTN_INTENTION: 1~5 정수 척도
    for col in ("DGSTFN", "REVISIT_INTENTION", "RCMDTN_INTENTION"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    dgstfn  = df.get("DGSTFN",              pd.Series(3.0, index=df.index)).fillna(3.0)
    revisit = df.get("REVISIT_INTENTION",   pd.Series(3.0, index=df.index)).fillna(3.0)
    rcmdtn  = df.get("RCMDTN_INTENTION",    pd.Series(3.0, index=df.index)).fillna(3.0)

    df[TARGET_COL] = (dgstfn * 0.4 + revisit * 0.3 + rcmdtn * 0.3).round(4)

    # 타겟이 유효한 행만 유지 (1.0~5.0 범위)
    df = df[(df[TARGET_COL] >= 1.0) & (df[TARGET_COL] <= 5.0)]

    # ── 피처 매핑 ─────────────────────────────────────────────────────────────
    col_map = {
        "VISIT_AREA_TYPE_CD":  "visit_area_type_cd",
        "RESIDENCE_TIME_MIN":  "residence_time_min",
        "SGG_CD":              "sgg_cd",
        "VISIT_CHC_REASON_CD": "visit_chc_reason_cd",
        "VISIT_ORDER":         "visit_order",
        "X_COORD":             "x_coord",
        "Y_COORD":             "y_coord",
    }
    df = df.rename(columns=col_map)

    # ── 카테고리 인코딩 ────────────────────────────────────────────────────────
    for cat_col in ("visit_area_type_cd", "sgg_cd", "visit_chc_reason_cd"):
        if cat_col in df.columns:
            le = LabelEncoder()
            df[cat_col] = le.fit_transform(df[cat_col].astype(str))
        else:
            df[cat_col] = 0

    # ── 연속형 피처 정제 ──────────────────────────────────────────────────────
    for col in ("residence_time_min", "visit_order", "x_coord", "y_coord"):
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    df["residence_time_min"] = df["residence_time_min"].clip(lower=0, upper=480)
    df["visit_order"]        = df["visit_order"].clip(lower=1, upper=20)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    # ── 원본 좌표·POI명 보존 (poi_attraction.csv 생성용) ─────────────────────
    # (이미 rename되었으므로 x_coord/y_coord 사용)
    for keep_col in ("VISIT_AREA_NM", "POI_ID"):
        if keep_col in df.columns:
            pass  # 남겨둠

    print(f"  전처리 완료: {df.shape[0]:,}행 사용 가능")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: TabNet 학습
# ══════════════════════════════════════════════════════════════════════════════
def train_tabnet(df: pd.DataFrame):
    """TabNetRegressor 학습 및 저장"""
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
    except ImportError:
        print("  ❌ pytorch-tabnet 없음: pip install pytorch-tabnet")
        sys.exit(1)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_tr)
    X_val_sc  = scaler.transform(X_val)

    scaler_path = os.path.join(MODELS_DIR, "attraction_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✅ attraction_scaler.pkl 저장")

    import torch
    model = TabNetRegressor(
        n_d=16, n_a=16, n_steps=3,
        gamma=1.3, n_independent=2, n_shared=2,
        momentum=0.02, clip_value=2.0,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 2e-3, "weight_decay": 1e-5},
        verbose=10,
        seed=42,
    )

    model.fit(
        X_train=X_tr_sc, y_train=y_tr,
        eval_set=[(X_val_sc, y_val)],
        eval_metric=["mae"],
        max_epochs=100,
        patience=15,
        batch_size=256,
        virtual_batch_size=128,
    )

    # 성능 평가 (원래 스케일 — 1~5)
    pred   = model.predict(X_val_sc).flatten()
    actual = y_val.flatten()
    pred   = np.clip(pred, 1.0, 5.0)
    mae    = mean_absolute_error(actual, pred)
    r2     = r2_score(actual, pred)
    print(f"\n  MAE: {mae:.4f}  |  R²: {r2:.4f}")
    print(f"  예측 범위: {pred.min():.2f} ~ {pred.max():.2f} (타겟 1~5)")

    save_path = os.path.join(MODELS_DIR, "attraction_regressor")
    model.save_model(save_path)
    print(f"  ✅ attraction_regressor.zip 저장 → {save_path}.zip")

    meta = {
        "feature_cols":  FEATURE_COLS,
        "target_col":    TARGET_COL,
        "target_range":  [1.0, 5.0],
        "target_formula": "DGSTFN×0.4 + REVISIT_INTENTION×0.3 + RCMDTN_INTENTION×0.3",
        "log_transform": False,
        "mae":           round(float(mae), 4),
        "r2":            round(float(r2), 4),
        "n_train":       len(X_tr),
        "n_val":         len(X_val),
    }
    meta_path = os.path.join(MODELS_DIR, "attraction_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  ✅ attraction_meta.json 저장")

    return model, scaler, meta


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: POI 매력도 CSV 생성 (Spatial Join용)
# ══════════════════════════════════════════════════════════════════════════════
def export_poi_attraction(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    전체 데이터에 대한 예측 매력도를 계산하여 POI별 평균으로 집계.

    출력 컬럼:
      x_coord, y_coord, visit_area_nm, sgg_cd_raw,
      attraction_score_mean, attraction_score_std, visit_count
    """
    X_all = scaler.transform(df[FEATURE_COLS].values.astype(np.float32))
    pred  = np.clip(model.predict(X_all).flatten(), 1.0, 5.0)
    df    = df.copy()
    df["attraction_pred"] = pred

    # POI 식별: VISIT_AREA_NM 있으면 사용, 없으면 좌표 반올림
    group_col = "VISIT_AREA_NM" if "VISIT_AREA_NM" in df.columns else None

    if group_col:
        agg = (
            df.groupby(group_col, as_index=False)
            .agg(
                x_coord=("x_coord", "mean"),
                y_coord=("y_coord", "mean"),
                attraction_score_mean=("attraction_pred", "mean"),
                attraction_score_std=("attraction_pred", "std"),
                visit_count=(group_col, "count"),
            )
        )
        agg.rename(columns={group_col: "visit_area_nm"}, inplace=True)
    else:
        df["coord_key"] = (
            df["x_coord"].round(4).astype(str) + "_" +
            df["y_coord"].round(4).astype(str)
        )
        agg = (
            df.groupby("coord_key", as_index=False)
            .agg(
                x_coord=("x_coord", "mean"),
                y_coord=("y_coord", "mean"),
                attraction_score_mean=("attraction_pred", "mean"),
                attraction_score_std=("attraction_pred", "std"),
                visit_count=("coord_key", "count"),
            )
        )
        agg["visit_area_nm"] = agg["coord_key"]
        agg.drop(columns=["coord_key"], inplace=True)

    # 점수 정규화: 1~5 → 0~1 (tourism_score와 스케일 맞춤)
    mn = agg["attraction_score_mean"].min()
    mx = agg["attraction_score_mean"].max()
    agg["attraction_score_norm"] = (
        (agg["attraction_score_mean"] - mn) / (mx - mn + 1e-9)
    ).round(4)

    agg["attraction_score_mean"] = agg["attraction_score_mean"].round(4)
    agg["attraction_score_std"]  = agg["attraction_score_std"].fillna(0).round(4)

    out_path = os.path.join(RAW_ML_DIR, "poi_attraction.csv")
    agg.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ poi_attraction.csv 저장 → {out_path}  ({len(agg):,}개 POI)")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# 추론 함수 (Streamlit / FastAPI에서 import해서 사용)
# ══════════════════════════════════════════════════════════════════════════════
def get_attraction_score_for_route(
    x_coords: list[float],
    y_coords: list[float],
    radius_deg: float = 0.005,   # 약 500m
) -> float:
    """
    경로 좌표 리스트 주변 POI의 평균 매력도 반환.

    poi_attraction.csv를 로드해 반경 내 POI를 집계.
    poi_attraction.csv 없으면 None 반환.

    반환: 0~1 정규화된 attraction_score_norm 평균값
    """
    poi_path = os.path.join(RAW_ML_DIR, "poi_attraction.csv")
    if not os.path.exists(poi_path):
        return None

    poi_df = pd.read_csv(poi_path, encoding="utf-8-sig")
    if poi_df.empty:
        return None

    scores = []
    for rx, ry in zip(x_coords, y_coords):
        nearby = poi_df[
            (poi_df["x_coord"].between(rx - radius_deg, rx + radius_deg)) &
            (poi_df["y_coord"].between(ry - radius_deg, ry + radius_deg))
        ]
        if not nearby.empty:
            scores.append(nearby["attraction_score_norm"].mean())

    if not scores:
        return None
    return round(float(np.mean(scores)), 4)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="POI 매력도 TabNet 학습")
    parser.add_argument(
        "--use_dummy", action="store_true",
        help="AI Hub 데이터 없을 때 합성 더미 데이터로 대체"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  POI 매력도 TabNet 모델 학습")
    print("=" * 60)

    # ── STEP 1: 데이터 로드 ───────────────────────────────────────────────────
    print("\n[STEP 1] 데이터 로드")
    if args.use_dummy:
        print("  더미 모드 사용")
        raw_df = make_dummy_data()
    else:
        raw_df = load_visit_area(AIHUB_DIR)
        if raw_df.empty:
            print("  ⚠️ AI Hub 데이터 없음 → --use_dummy 플래그로 재시도하세요")
            sys.exit(1)

    # ── STEP 2: 전처리 ────────────────────────────────────────────────────────
    print("\n[STEP 2] 전처리")
    df = preprocess(raw_df)
    if len(df) < 100:
        print(f"  ❌ 유효 데이터 부족: {len(df)}행 (최소 100행 필요)")
        sys.exit(1)

    print(f"\n  타겟 분포:")
    print(f"    평균={df[TARGET_COL].mean():.3f}  "
          f"중앙값={df[TARGET_COL].median():.3f}  "
          f"std={df[TARGET_COL].std():.3f}")
    print(f"    범위: {df[TARGET_COL].min():.2f} ~ {df[TARGET_COL].max():.2f}")

    # ── STEP 3: 학습 ──────────────────────────────────────────────────────────
    print("\n[STEP 3] TabNet 학습")
    model, scaler, meta = train_tabnet(df)

    # ── STEP 4: POI 매력도 CSV 저장 ───────────────────────────────────────────
    print("\n[STEP 4] POI 매력도 CSV 생성")
    poi_df = export_poi_attraction(df, model, scaler)

    # ── 완료 요약 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  학습 완료")
    print(f"  MAE: {meta['mae']:.4f}  |  R²: {meta['r2']:.4f}")
    print(f"  학습 샘플: {meta['n_train']:,}  |  검증 샘플: {meta['n_val']:,}")
    print(f"  POI 집계: {len(poi_df):,}개")
    print()
    print("  [ 출력 파일 ]")
    print(f"    models/attraction_regressor.zip")
    print(f"    models/attraction_scaler.pkl")
    print(f"    models/attraction_meta.json")
    print(f"    data/raw_ml/poi_attraction.csv")
    print()
    print("  [ 다음 단계 ]")
    print("    1. poi_attraction.csv → road_features.csv Spatial Join")
    print("       → build_tourism_model.py에서 tourism_score_v2 생성")
    print("    2. route_graph.pkl 재빌드 (tourism_score_v2 반영)")
    print("=" * 60)


if __name__ == "__main__":
    main()
