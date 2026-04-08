"""
build_consume_model.py
======================
여행로그 소비 데이터 → TabNet 소비 예측 모델 학습

[ 데이터 준비 ]
  AI Hub 여행로그(수도권) TL_csv.zip (2.89MB) 다운
  → data/dl/travel_log/ 에 압축 해제
  핵심 테이블: TN_ACTIVITY_CONSUME_HIS.csv

[ 실행 ]
  python kride-project/build_consume_model.py
  python kride-project/build_consume_model.py --data_dir data/dl/travel_log

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
import sys
import warnings

import joblib
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

DL_DATA_DIR   = os.path.join(BASE_DIR, "data", "dl")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
os.makedirs(DL_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

FEATURE_COLS = [
    "sgg_code",          # 시군구 코드 (인코딩)
    "travel_duration_h", # 여행 시간 (시간)
    "distance_km",       # 이동 거리 (km)
    "companion_cnt",     # 동반자 수
    "season",            # 계절 (1~4)
    "day_of_week",       # 요일 (0~6)
    "has_lodging",       # 숙박 여부 (0/1)
]
TARGET_COL = "consume_amt"   # 소비 금액 (원)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_travel_log(data_dir: str) -> pd.DataFrame:
    """AI Hub 여행로그 TN_ACTIVITY_CONSUME_HIS 파싱"""
    # 후보 파일명 패턴
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
        print(f"  로드: {os.path.basename(fpath)}  {df.shape}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 전처리
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """AI Hub 컬럼명 → 표준 피처명 매핑 + 파생 변수 생성"""
    col_map = {}
    candidates = {
        "sgg_code":     ["TC_SGG_CD", "SGG_CD", "시군구코드", "sgg_cd"],
        TARGET_COL:     ["CONSUM_AMT", "CONSUME_AMT", "소비금액", "활동소비금액"],
        "travel_start": ["TRAVEL_START_YMD", "START_DT", "여행시작일"],
        "travel_end":   ["TRAVEL_END_YMD",   "END_DT",   "여행종료일"],
        "companion_cnt":["ACCOMPANY_CNT", "COMPANION_CNT", "동반자수"],
        "has_lodging":  ["LODGING_YN", "숙박여부"],
    }
    for key, names in candidates.items():
        for n in names:
            if n in df.columns:
                col_map[n] = key
                break

    df = df.rename(columns=col_map)

    # 소비 금액 타겟
    if TARGET_COL not in df.columns:
        # 금액 관련 컬럼 자동 탐색
        amt_col = next((c for c in df.columns if any(k in c.upper() for k in ["AMT", "AMOUNT", "금액"])), None)
        if amt_col:
            df[TARGET_COL] = df[amt_col]
        else:
            print("  ⚠️ 소비금액 컬럼 없음 → 더미 데이터로 대체")
            df[TARGET_COL] = np.random.randint(3000, 50000, len(df))

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0)
    df = df[df[TARGET_COL] > 0]   # 0원 제거

    # 여행 기간 (시간 단위)
    if "travel_start" in df.columns and "travel_end" in df.columns:
        df["travel_start"] = pd.to_datetime(df["travel_start"], errors="coerce")
        df["travel_end"]   = pd.to_datetime(df["travel_end"],   errors="coerce")
        diff = (df["travel_end"] - df["travel_start"]).dt.total_seconds() / 3600
        df["travel_duration_h"] = diff.clip(lower=0).fillna(8.0)
        df["day_of_week"]       = df["travel_start"].dt.dayofweek.fillna(0)
        df["season"]            = df["travel_start"].dt.month.map(
            lambda m: 1 if m in [3,4,5] else 2 if m in [6,7,8] else 3 if m in [9,10,11] else 4
        ).fillna(1)
    else:
        df["travel_duration_h"] = 8.0
        df["day_of_week"]       = 0
        df["season"]            = 1

    # 거리 (없으면 기본값)
    if "distance_km" not in df.columns:
        df["distance_km"] = 10.0

    # 동반자 수
    if "companion_cnt" not in df.columns:
        df["companion_cnt"] = 1
    df["companion_cnt"] = pd.to_numeric(df["companion_cnt"], errors="coerce").fillna(1)

    # 숙박 여부
    if "has_lodging" in df.columns:
        df["has_lodging"] = df["has_lodging"].map({"Y": 1, "N": 0, 1: 1, 0: 0}).fillna(0)
    else:
        df["has_lodging"] = 0

    # 시군구 코드 인코딩
    if "sgg_code" not in df.columns:
        df["sgg_code"] = 0
    df["sgg_code"] = LabelEncoder().fit_transform(df["sgg_code"].astype(str))

    # 수치 변환
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
    y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

    # 로그 스케일 타겟 (소비 금액 분포 개선)
    y_log = np.log1p(y)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)

    scaler_path = os.path.join(MODELS_DIR, "consume_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ consume_scaler.pkl 저장\n")

    model = TabNetRegressor(
        n_d=16, n_a=16, n_steps=3,
        gamma=1.3, n_independent=2, n_shared=2,
        momentum=0.02, clip_value=2.0,
        optimizer_fn=__import__("torch").optim.Adam,
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

    # 성능 평가 (원래 스케일)
    pred_log = model.predict(X_val_sc)
    pred     = np.expm1(pred_log)
    actual   = np.expm1(y_val)
    mae      = mean_absolute_error(actual, pred)
    r2       = r2_score(actual, pred)
    print(f"\n  MAE: {mae:,.0f}원  |  R²: {r2:.4f}")

    # 저장
    save_path = os.path.join(MODELS_DIR, "consume_regressor.zip")
    model.save_model(save_path.replace(".zip", ""))   # TabNet은 .zip 자동 추가
    print(f"  ✅ consume_regressor.zip 저장 → {save_path}")

    meta = {
        "feature_cols": FEATURE_COLS,
        "target_col":   TARGET_COL,
        "log_transform": True,
        "mae_krw": round(mae),
        "r2":      round(r2, 4),
        "n_train":  len(X_tr),
        "n_val":    len(X_val),
    }
    meta_path = os.path.join(MODELS_DIR, "consume_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  ✅ consume_meta.json 저장")
    return model, scaler, meta


# ══════════════════════════════════════════════════════════════════════════════
# 추론 함수 (FastAPI에서 import해서 사용)
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

    반환:
      { "estimated_cost_krw": int, "note": str }
    """
    model_zip = os.path.join(MODELS_DIR, "consume_regressor.zip")
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
    scaler = joblib.load(scaler_path)

    X = np.array([[
        sgg_code, travel_duration_h, distance_km,
        companion_cnt, season, day_of_week, has_lodging,
    ]], dtype=np.float32)
    X_sc  = scaler.transform(X)
    y_log = model.predict(X_sc)
    cost  = int(np.expm1(y_log[0][0]))

    return {
        "estimated_cost_krw": max(cost, 0),
        "model_mae_krw":      meta.get("mae_krw", 0),
        "model_r2":           meta.get("r2", 0.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI 진입점
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=os.path.join(DL_DATA_DIR, "travel_log"),
        help="AI Hub 여행로그 CSV 디렉토리",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("STEP 1: 여행로그 소비 데이터 로드")
    print("=" * 65)

    df_raw = load_travel_log(args.data_dir)

    if df_raw.empty:
        print(f"  ❌ 데이터 없음: {args.data_dir}")
        print("  AI Hub 여행로그(수도권) TL_csv.zip 다운 후 압축 해제하세요.")
        sys.exit(1)

    print(f"  병합 shape: {df_raw.shape}")

    print("\n" + "=" * 65)
    print("STEP 2: 전처리")
    print("=" * 65)

    df = preprocess(df_raw)
    print(f"  전처리 후: {df.shape}")
    print(f"  소비금액 통계:\n{df[TARGET_COL].describe().round(0).to_string()}\n")

    print("=" * 65)
    print("STEP 3: TabNet 학습")
    print("=" * 65)

    train_tabnet(df)

    print("\n" + "=" * 65)
    print("✅ build_consume_model.py 완료")
    print("=" * 65)
