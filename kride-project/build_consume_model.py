"""
build_consume_model.py
======================
여행로그 소비 데이터 → TabNet 소비 예측 모델 학습

[ 데이터 준비 ]
  AI Hub 여행로그(수도권) TL_csv.zip (2.89MB) 다운
  → data/dl/travel_log/ 에 압축 해제
  핵심 테이블: TN_ACTIVITY_CONSUME_HIS.csv

  ※ 데이터 없을 경우 --use_dummy 플래그로 합성 데이터 사용 가능

[ 실행 ]
  python kride-project/build_consume_model.py
  python kride-project/build_consume_model.py --data_dir data/dl/travel_log
  python kride-project/build_consume_model.py --use_dummy   ← 데이터 없을 때

[ 출력 ]
  models/consume_regressor.zip   ← TabNetRegressor 가중치
  models/consume_scaler.pkl      ← 입력 피처 StandardScaler
  models/consume_meta.json       ← 피처 목록 + 성능 메타

[ 입력 피처 ]
  sgg_code, travel_duration_h, distance_km,
  companion_cnt, season, day_of_week, has_lodging

[ 타겟 ]
  활동 소비 금액 (원)
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

DL_DATA_DIR = os.path.join(BASE_DIR, "data", "dl")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(DL_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

FEATURE_COLS = [
    "sgg_code",           # 시군구 코드 (인코딩)
    "travel_duration_h",  # 여행 시간 (시간)
    "distance_km",        # 이동 거리 (km)
    "companion_cnt",      # 동반자 수
    "season",             # 계절 레이블 (1=봄, 2=여름, 3=가을, 4=겨울)
    "day_of_week",        # 요일 (0=월요일 ~ 6=일요일)
    "has_lodging",        # 숙박 여부 (0/1)
]
TARGET_COL = "consume_amt"   # 소비 금액 (원)

# 계절 매핑 (월 → 계절 레이블)
MONTH_TO_SEASON = {
    1: 4, 2: 4,          # 겨울
    3: 1, 4: 1, 5: 1,   # 봄
    6: 2, 7: 2, 8: 2,   # 여름
    9: 3, 10: 3, 11: 3, # 가을
    12: 4,               # 겨울
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_travel_log(data_dir: str) -> pd.DataFrame:
    """AI Hub 여행로그 TN_ACTIVITY_CONSUME_HIS 파싱"""
    patterns = [
        os.path.join(data_dir, "**", "TN_ACTIVITY_CONSUME_HIS*.csv"),
        os.path.join(data_dir, "**", "CONSUME*.csv"),
        os.path.join(data_dir, "**", "*소비*.csv"),
        os.path.join(data_dir, "*.csv"),
    ]
    files = []
    for p in patterns:
        files += glob.glob(p, recursive=True)
    files = list(dict.fromkeys(files))   # 중복 제거

    if not files:
        return pd.DataFrame()

    dfs = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath, encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(fpath, encoding="cp949")
            except Exception:
                continue
        dfs.append(df)
        print(f"  로드: {os.path.basename(fpath)}  shape={df.shape}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def make_dummy_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    AI Hub 데이터가 없을 때 사용하는 합성 여행로그 데이터.
    실제 데이터와 동일한 컬럼명으로 생성하여 preprocess()를 그대로 통과.
    """
    rng = np.random.default_rng(seed)

    # 시작일 생성 (2022~2024 무작위)
    start_days = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 3, n), unit="D"
    )
    end_days = start_days + pd.to_timedelta(rng.integers(1, 4, n), unit="D")

    # 시군구 코드: 서울(11xxx) + 경기(41xxx) 대표 코드
    sgg_pool = [
        "11110", "11140", "11170", "11200", "11215", "11230",  # 서울 주요 구
        "41110", "41130", "41150", "41280", "41310", "41430",  # 경기 주요 시
    ]

    df = pd.DataFrame({
        "TC_SGG_CD":        rng.choice(sgg_pool, n),
        "CONSUM_AMT":       rng.integers(3_000, 80_000, n),
        "TRAVEL_START_YMD": start_days.strftime("%Y-%m-%d"),
        "TRAVEL_END_YMD":   end_days.strftime("%Y-%m-%d"),
        "ACCOMPANY_CNT":    rng.integers(1, 5, n),
        "LODGING_YN":       rng.choice(["Y", "N"], n, p=[0.3, 0.7]),
        "distance_km":      rng.uniform(5, 40, n).round(1),
    })
    print(f"  합성 더미 데이터 생성: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 전처리
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """AI Hub 컬럼명 → 표준 피처명 매핑 + 파생 변수 생성"""

    # ── 컬럼명 매핑 ────────────────────────────────────────────────────────────
    col_map = {}
    candidates = {
        "sgg_code":      ["TC_SGG_CD", "SGG_CD", "시군구코드", "sgg_cd"],
        TARGET_COL:      ["CONSUM_AMT", "CONSUME_AMT", "소비금액", "활동소비금액"],
        "travel_start":  ["TRAVEL_START_YMD", "START_DT", "여행시작일"],
        "travel_end":    ["TRAVEL_END_YMD",   "END_DT",   "여행종료일"],
        "companion_cnt": ["ACCOMPANY_CNT", "COMPANION_CNT", "동반자수"],
        "has_lodging":   ["LODGING_YN", "숙박여부"],
    }
    for key, names in candidates.items():
        for n in names:
            if n in df.columns:
                col_map[n] = key
                break

    df = df.rename(columns=col_map)

    # ── 소비 금액 타겟 ─────────────────────────────────────────────────────────
    if TARGET_COL not in df.columns:
        amt_col = next(
            (c for c in df.columns if any(k in c.upper() for k in ["AMT", "AMOUNT", "금액"])),
            None,
        )
        if amt_col:
            df[TARGET_COL] = df[amt_col]
        else:
            print("  ⚠️ 소비금액 컬럼 없음 → 더미 금액 사용")
            df[TARGET_COL] = np.random.randint(3000, 50000, len(df))

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0)
    df = df[df[TARGET_COL] > 0].copy()

    # ── 날짜 파생 변수 ─────────────────────────────────────────────────────────
    if "travel_start" in df.columns and "travel_end" in df.columns:
        df["travel_start"] = pd.to_datetime(df["travel_start"], errors="coerce")
        df["travel_end"]   = pd.to_datetime(df["travel_end"],   errors="coerce")

        diff = (df["travel_end"] - df["travel_start"]).dt.total_seconds() / 3600
        df["travel_duration_h"] = diff.clip(lower=0).fillna(8.0)

        # 요일 (0=월 ~ 6=일), NaT는 0으로
        df["day_of_week"] = df["travel_start"].dt.dayofweek.fillna(0).astype(int)

        # 계절: 월 → MONTH_TO_SEASON 딕셔너리 매핑 (봄=1, 여름=2, 가을=3, 겨울=4)
        month_series = df["travel_start"].dt.month
        df["season"] = month_series.map(MONTH_TO_SEASON).fillna(1).astype(int)
    else:
        df["travel_duration_h"] = 8.0
        df["day_of_week"]       = 0
        df["season"]            = 1

    # ── 거리 ───────────────────────────────────────────────────────────────────
    if "distance_km" not in df.columns:
        df["distance_km"] = 10.0
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").fillna(10.0)

    # ── 동반자 수 ──────────────────────────────────────────────────────────────
    if "companion_cnt" not in df.columns:
        df["companion_cnt"] = 1
    df["companion_cnt"] = pd.to_numeric(df["companion_cnt"], errors="coerce").fillna(1).astype(int)

    # ── 숙박 여부 ──────────────────────────────────────────────────────────────
    if "has_lodging" in df.columns:
        df["has_lodging"] = df["has_lodging"].map({"Y": 1, "N": 0, 1: 1, 0: 0}).fillna(0).astype(int)
    else:
        df["has_lodging"] = 0

    # ── 시군구 코드 인코딩 ─────────────────────────────────────────────────────
    if "sgg_code" not in df.columns:
        df["sgg_code"] = "0"
    le = LabelEncoder()
    df["sgg_code"] = le.fit_transform(df["sgg_code"].astype(str))

    # ── 수치 변환 & 결측 제거 ──────────────────────────────────────────────────
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
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
    y = df[TARGET_COL].values.astype(np.float32)

    # 로그 스케일 타겟 (소비 금액 분포 개선)
    y_log = np.log1p(y).reshape(-1, 1)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)

    # StandardScaler를 pickle로 저장 (joblib 없어도 동작)
    scaler_path = os.path.join(MODELS_DIR, "consume_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✅ consume_scaler.pkl 저장")

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

    # 성능 평가 (원래 스케일로 역변환)
    pred_log = model.predict(X_val_sc)
    pred     = np.expm1(pred_log).flatten()
    actual   = np.expm1(y_val).flatten()
    mae      = mean_absolute_error(actual, pred)
    r2       = r2_score(actual, pred)
    print(f"\n  MAE: {mae:,.0f}원  |  R²: {r2:.4f}")

    # TabNet 모델 저장 (.zip 자동 생성)
    save_path = os.path.join(MODELS_DIR, "consume_regressor")
    model.save_model(save_path)   # → consume_regressor.zip 생성
    print(f"  ✅ consume_regressor.zip 저장 → {save_path}.zip")

    # 메타 저장
    meta = {
        "feature_cols":   FEATURE_COLS,
        "target_col":     TARGET_COL,
        "log_transform":  True,
        "season_label":   {"1": "봄", "2": "여름", "3": "가을", "4": "겨울"},
        "mae_krw":        round(mae),
        "r2":             round(r2, 4),
        "n_train":        len(X_tr),
        "n_val":          len(X_val),
    }
    meta_path = os.path.join(MODELS_DIR, "consume_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  ✅ consume_meta.json 저장")

    return model, scaler, meta


# ══════════════════════════════════════════════════════════════════════════════
# 추론 함수 (FastAPI / Streamlit에서 import해서 사용)
# ══════════════════════════════════════════════════════════════════════════════
def predict_consume(
    sgg_code: int = 0,
    travel_duration_h: float = 8.0,
    distance_km: float = 10.0,
    companion_cnt: int = 1,
    season: int = 1,
    day_of_week: int = 0,
    has_lodging: int = 0,
) -> dict:
    """
    입력 피처 → 예상 소비 금액 (원) 반환

    season 값:  1=봄, 2=여름, 3=가을, 4=겨울

    반환:
      { "estimated_cost_krw": int, "model_mae_krw": int, "model_r2": float }
    """
    model_zip   = os.path.join(MODELS_DIR, "consume_regressor.zip")
    scaler_path = os.path.join(MODELS_DIR, "consume_scaler.pkl")
    meta_path   = os.path.join(MODELS_DIR, "consume_meta.json")

    if not os.path.exists(model_zip):
        return {
            "estimated_cost_krw": 10000,
            "note": "모델 없음 — build_consume_model.py 실행 필요 (기본값 10,000원)",
        }

    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
    except ImportError:
        return {"estimated_cost_krw": 10000, "note": "pytorch-tabnet 미설치"}

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    model = TabNetRegressor()
    model.load_model(model_zip)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    X = np.array([[
        sgg_code, travel_duration_h, distance_km,
        companion_cnt, season, day_of_week, has_lodging,
    ]], dtype=np.float32)
    X_sc  = scaler.transform(X)
    y_log = model.predict(X_sc)
    cost  = int(np.expm1(float(y_log[0])))

    return {
        "estimated_cost_krw": max(cost, 0),
        "model_mae_krw":      meta.get("mae_krw", 0),
        "model_r2":           meta.get("r2", 0.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI 진입점
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="소비 TabNet 학습 스크립트")
    parser.add_argument(
        "--data_dir",
        default=os.path.join(DL_DATA_DIR, "travel_log"),
        help="AI Hub 여행로그 CSV 디렉토리",
    )
    parser.add_argument(
        "--use_dummy",
        action="store_true",
        help="AI Hub 데이터 없을 때 합성 데이터로 학습 (테스트 용도)",
    )
    parser.add_argument(
        "--dummy_n",
        type=int,
        default=2000,
        help="더미 데이터 행 수 (기본: 2000)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("STEP 1: 여행로그 소비 데이터 로드")
    print("=" * 65)

    if args.use_dummy:
        print("  [더미 모드] 합성 데이터 생성 중...")
        df_raw = make_dummy_data(n=args.dummy_n)
    else:
        df_raw = load_travel_log(args.data_dir)
        if df_raw.empty:
            print(f"  ❌ 데이터 없음: {args.data_dir}")
            print("  → AI Hub 여행로그(수도권) TL_csv.zip 다운 후 압축 해제하거나,")
            print("    --use_dummy 플래그로 합성 데이터 학습을 실행하세요.")
            sys.exit(1)

    print(f"  로드 shape: {df_raw.shape}")

    print("\n" + "=" * 65)
    print("STEP 2: 전처리")
    print("=" * 65)

    df = preprocess(df_raw)
    print(f"  전처리 후: {df.shape}")

    season_names = {1: "봄", 2: "여름", 3: "가을", 4: "겨울"}
    print(f"  계절 분포:\n{df['season'].map(season_names).value_counts().to_string()}")
    print(f"\n  소비금액 통계:\n{df[TARGET_COL].describe().round(0).to_string()}\n")

    if len(df) < 50:
        print("  ⚠️ 학습 데이터가 50행 미만입니다. --dummy_n 값을 늘리거나 실제 데이터를 사용하세요.")
        sys.exit(1)

    print("=" * 65)
    print("STEP 3: TabNet 학습")
    print("=" * 65)

    train_tabnet(df)

    print("\n" + "=" * 65)
    print("✅ build_consume_model.py 완료")
    print("  출력: models/consume_regressor.zip")
    print("        models/consume_scaler.pkl")
    print("        models/consume_meta.json")
    print("=" * 65)
