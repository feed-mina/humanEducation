"""
build_consume_model_v3.py
=========================
국민여행조사 2024 전국 소비 데이터 → TabNet 소비 예측 모델 v3

[ v2 대비 개선 사항 ]
  1. 수도권 2,508행(AI Hub) → 전국 25,893행(국민여행조사 2024), 10배 이상
  2. 시군구 TargetEncoding 제거 → sido_enc (17개 시도 label int) 직접 사용
  3. income_tier(편향 극심) 대신 income_score(0~5 연속) 사용
  4. 인구통계 피처 추가: sa1_1~sa1_5 (성별·연령·직업·학력·혼인 코드)
  5. 계절 균등 분포(봄 6,597/여름 6,408/가을 6,576/겨울 6,312) → 가중치 불필요
  6. 단일 CSV 입력 → 데이터 로드 단순화

[ 입력 ]
  data/raw_ml/national_travel_consume.csv  ← preprocess_national_travel.py 산출물

[ 출력 ]
  models/consume_regressor_v3.zip
  models/consume_scaler_v3.pkl
  models/consume_meta_v3.json

[ 실행 ]
  python src/dl/build_consume_model_v3.py
  python src/dl/build_consume_model_v3.py --epochs 200 --n_d 64
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))   # kride-project/
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
DATA_PATH   = os.path.join(PROJECT_DIR, "data", "raw_ml", "national_travel_consume.csv")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 피처 / 타겟 ────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "travel_days",    # 0=당일, 1=1박2일, ...
    "companion_cnt",  # 동반자 수(본인 포함)
    "trip_type",      # 여행 유형 1~5
    "sido_enc",       # 시도 label int 0~16
    "sa1_1",          # 성별 코드
    "sa1_2",          # 연령대 코드
    "sa1_3",          # 직업 코드
    "sa1_4",          # 학력 코드
    "sa1_5",          # 혼인 상태 코드
    "income_score",   # MON_EXP 합산 0~5 (연속)
    "season",         # 1=봄 2=여름 3=가을 4=겨울
]
TARGET_COL = "log_one_cost"   # log1p(1인 지출 비용)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"입력 파일 없음: {DATA_PATH}\n"
            "  먼저 실행: python src/preprocessing/preprocess_national_travel.py"
        )
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"  로드: national_travel_consume.csv  shape={df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 전처리
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    required = FEATURE_COLS + [TARGET_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 없음: {missing}")

    df = df[required].copy()

    # 결측 보완
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).copy()
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df[col].median())

    print(f"  결측 제거: {before - len(df)}행 → {len(df):,}행 잔존")

    # 정수 컬럼 캐스팅
    int_cols = ["travel_days", "companion_cnt", "trip_type", "sido_enc",
                "sa1_1", "sa1_2", "sa1_3", "sa1_4", "sa1_5", "season"]
    for col in int_cols:
        df[col] = df[col].round().astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: TabNet 학습
# ══════════════════════════════════════════════════════════════════════════════
def train_tabnet(df: pd.DataFrame, df_raw: pd.DataFrame,
                 n_d: int = 32, n_a: int = 32,
                 n_steps: int = 5, max_epochs: int = 150):
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
        import torch
    except ImportError:
        print("  pytorch-tabnet 없음: pip install pytorch-tabnet")
        sys.exit(1)

    # ── 70 / 15 / 15 분할 ────────────────────────────────────────────────────
    df_train, df_tmp = train_test_split(df, test_size=0.30, random_state=42)
    df_val,   df_test = train_test_split(df_tmp, test_size=0.50, random_state=42)
    print(f"  분할: train={len(df_train):,} / val={len(df_val):,} / test={len(df_test):,}")

    feat_cols = [f for f in FEATURE_COLS if f in df_train.columns]

    X_tr  = df_train[feat_cols].values.astype(np.float32)
    X_val = df_val[feat_cols].values.astype(np.float32)
    X_te  = df_test[feat_cols].values.astype(np.float32)
    y_tr  = df_train[TARGET_COL].values.astype(np.float32).reshape(-1, 1)
    y_val = df_val[TARGET_COL].values.astype(np.float32).reshape(-1, 1)
    y_te  = df_test[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

    # ── StandardScaler ────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_te_sc  = scaler.transform(X_te)

    scaler_path = os.path.join(MODELS_DIR, "consume_scaler_v3.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  consume_scaler_v3.pkl 저장")

    # ── TabNet 학습 ───────────────────────────────────────────────────────────
    model = TabNetRegressor(
        n_d=n_d, n_a=n_a, n_steps=n_steps,
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
        max_epochs=max_epochs,
        patience=25,
        batch_size=512,
        virtual_batch_size=256,
    )

    # ── 성능 평가 ─────────────────────────────────────────────────────────────
    def eval_split(X_sc, y_log, name):
        pred_log = model.predict(X_sc)
        pred     = np.expm1(pred_log).flatten()
        actual   = np.expm1(y_log).flatten()
        mae_v = mean_absolute_error(actual, pred)
        r2_v  = r2_score(actual, pred)
        print(f"  [{name}] MAE={mae_v:,.0f}원  R2={r2_v:.4f}")
        return mae_v, r2_v

    val_mae,  val_r2  = eval_split(X_val_sc, y_val, "val ")
    test_mae, test_r2 = eval_split(X_te_sc,  y_te,  "test")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(MODELS_DIR, "consume_regressor_v3")
    model.save_model(save_path)
    print(f"  consume_regressor_v3.zip 저장")

    # feature importance
    importances = dict(zip(feat_cols, model.feature_importances_.tolist()))

    # sido_name → sido_enc 역매핑 (FastAPI 추론 시 지역명 → 정수 변환용)
    sido_name_to_enc = (
        df_raw.groupby("sido_name")["sido_enc"].first()
        .astype(int)
        .sort_values()
        .apply(lambda x: int(x))
        .to_dict()
    ) if "sido_name" in df_raw.columns else {}

    meta = {
        "feature_cols":      feat_cols,
        "target_col":        TARGET_COL,
        "log_transform":     True,
        "data_source":       "국민여행조사 2024 전국",
        "n_rows_total":      len(df),
        "season_label":      {"1": "봄", "2": "여름", "3": "가을", "4": "겨울"},
        "sido_name_to_enc":  sido_name_to_enc,   # "경기도" -> 8 등
        "sido_enc_range":    "0~16 (17개 시도)",
        "income_score_range": "0~5 (MON_EXP 합산, 연속)",
        "tabnet_params":  {"n_d": n_d, "n_a": n_a, "n_steps": n_steps},
        "feature_importances": importances,
        "val_mae_krw":    round(val_mae),
        "val_r2":         round(val_r2, 4),
        "test_mae_krw":   round(test_mae),
        "test_r2":        round(test_r2, 4),
        "n_train":        len(df_train),
        "n_val":          len(df_val),
        "n_test":         len(df_test),
        "model_version":  "v3",
    }
    meta_path = os.path.join(MODELS_DIR, "consume_meta_v3.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  consume_meta_v3.json 저장")

    return model, scaler, meta


# ══════════════════════════════════════════════════════════════════════════════
# 추론 함수 (FastAPI / Streamlit에서 import)
# ══════════════════════════════════════════════════════════════════════════════
def predict_consume_v3(
    travel_days: int     = 1,
    companion_cnt: int   = 2,
    trip_type: int       = 1,
    sido_enc: int        = 0,        # 0=서울, ..., 16=제주
    sido_name: str       = "",       # 지역명 지정 시 sido_enc 자동 변환 (우선)
    sa1_1: int           = 1,        # 성별 코드
    sa1_2: int           = 3,        # 연령대 코드
    sa1_3: int           = 1,        # 직업 코드
    sa1_4: int           = 2,        # 학력 코드
    sa1_5: int           = 1,        # 혼인 상태 코드
    income_score: float  = 1.0,      # 0~5 (연속)
    season: int          = 1,        # 1=봄 2=여름 3=가을 4=겨울
) -> dict:
    model_zip   = os.path.join(MODELS_DIR, "consume_regressor_v3.zip")
    scaler_path = os.path.join(MODELS_DIR, "consume_scaler_v3.pkl")
    meta_path   = os.path.join(MODELS_DIR, "consume_meta_v3.json")

    if not os.path.exists(model_zip):
        return {
            "estimated_cost_krw": 78500,
            "note": "v3 모델 없음 — build_consume_model_v3.py 실행 필요 (중앙값 기본값)",
            "model_version": "v3_missing",
        }

    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
    except ImportError:
        return {"estimated_cost_krw": 78500, "note": "pytorch-tabnet 미설치"}

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = TabNetRegressor()
    model.load_model(model_zip)

    # sido_name 지정 시 메타의 역매핑으로 sido_enc 자동 결정
    if sido_name:
        name_to_enc = meta.get("sido_name_to_enc", {})
        sido_enc = name_to_enc.get(sido_name, sido_enc)

    feat_order = meta["feature_cols"]
    feat_values = {
        "travel_days":   travel_days,
        "companion_cnt": companion_cnt,
        "trip_type":     trip_type,
        "sido_enc":      sido_enc,
        "sa1_1":         sa1_1,
        "sa1_2":         sa1_2,
        "sa1_3":         sa1_3,
        "sa1_4":         sa1_4,
        "sa1_5":         sa1_5,
        "income_score":  income_score,
        "season":        season,
    }
    X    = np.array([[feat_values.get(f, 0) for f in feat_order]], dtype=np.float32)
    X_sc = scaler.transform(X)
    cost = int(np.expm1(float(model.predict(X_sc)[0])))

    season_labels = {1: "봄", 2: "여름", 3: "가을", 4: "겨울"}
    return {
        "estimated_cost_krw": max(cost, 0),
        "season_label":       season_labels.get(season, "봄"),
        "model_mae_krw":      meta.get("test_mae_krw", 0),
        "model_r2":           meta.get("test_r2", 0.0),
        "model_version":      "v3",
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="소비 TabNet v3 학습 — 국민여행조사 2024 전국")
    parser.add_argument("--epochs", type=int, default=150, help="최대 에포크 (기본: 150)")
    parser.add_argument("--n_d",    type=int, default=32,  help="TabNet n_d (기본: 32)")
    parser.add_argument("--n_a",    type=int, default=32,  help="TabNet n_a (기본: 32)")
    parser.add_argument("--n_steps",type=int, default=5,   help="TabNet n_steps (기본: 5)")
    args = parser.parse_args()

    print("=" * 65)
    print("ConsumeTabNet v3 — 국민여행조사 2024 전국 25,893행")
    print("=" * 65)

    print(f"\n{'─'*65}")
    print("STEP 1 -- 데이터 로드")
    print(f"{'─'*65}")
    df_raw = load_data()

    print(f"\n{'─'*65}")
    print("STEP 2 -- 전처리")
    print(f"{'─'*65}")
    df = preprocess(df_raw)

    season_names = {1: "봄", 2: "여름", 3: "가을", 4: "겨울"}
    print(f"\n  계절 분포: {dict(df['season'].map(season_names).value_counts().sort_index())}")
    print(f"  income_score 분포: mean={df['income_score'].mean():.2f}  "
          f"std={df['income_score'].std():.2f}  "
          f"max={df['income_score'].max()}")
    print(f"  sido_enc unique: {df['sido_enc'].nunique()}개 시도")
    s = np.expm1(df[TARGET_COL])
    print(f"  1인 지출 통계: median=₩{int(s.median()):,}  "
          f"mean=₩{int(s.mean()):,}  "
          f"std=₩{int(s.std()):,}")

    print(f"\n{'─'*65}")
    print(f"STEP 3 -- TabNet v3 학습 (n_d={args.n_d}, n_steps={args.n_steps}, "
          f"epochs={args.epochs})")
    print(f"{'─'*65}")
    _, _, meta = train_tabnet(df, df_raw, n_d=args.n_d, n_a=args.n_a,
                              n_steps=args.n_steps, max_epochs=args.epochs)

    print(f"\n{'='*65}")
    print("[DONE] build_consume_model_v3.py 완료")
    print(f"  test MAE : ₩{meta['test_mae_krw']:,}원")
    print(f"  test R2  : {meta['test_r2']}")
    print(f"  출력     : models/consume_regressor_v3.zip")
    print(f"             models/consume_scaler_v3.pkl")
    print(f"             models/consume_meta_v3.json")
    print(f"{'='*65}")
    print("\n다음 단계:")
    print("  python src/report/report_step3_consume_tabnet.py")
