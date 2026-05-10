"""
build_consume_model_v2.py
=========================
여행로그 소비 데이터 → TabNet 소비 예측 모델 v2

[ v1 대비 개선 사항 ]
  1. 소득 분위 3단계 구간화 (income_tier: 0=짠순이, 1=보통, 2=호캉스)
  2. traveller_master 연동 → income_tier, gender, age_grp, travel_purpose, travel_styl_avg 추가
  3. 소비 타겟 재정의: 활동+숙박+이동수단+사전 소비 4개 테이블 합산
  4. 이상치 제거 (q01~q99) + log1p 변환
  5. sgg_code → TargetEncoding (시군구별 평균 소비 반영)
  6. 70 / 15 / 15 분할 (train / val / test)
  7. 계절 불균형 보정 (여름 가중치 0.5)

[ 실행 ]
  python kride-project/build_consume_model_v2.py
  python kride-project/build_consume_model_v2.py --use_dummy

[ 출력 ]
  models/consume_regressor_v2.zip
  models/consume_scaler_v2.pkl
  models/consume_meta_v2.json
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()

MODELS_DIR = os.path.join(BASE_DIR, "models")
AIHUB_DIR  = os.path.join(
    BASE_DIR, "data", "ai-hub",
    "국내 여행로그 수도권_2023", "02.라벨링데이터"
)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 피처 목록 ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "sgg_enc",          # 시군구 TargetEncoding (시군구별 평균 소비)
    "travel_duration_h",
    "distance_km",
    "companion_cnt",
    "season",           # 1=봄, 2=여름, 3=가을, 4=겨울
    "day_of_week",
    "has_lodging",
    "income_tier",      # 0=짠순이(1~3), 1=보통(4~6), 2=호캉스(7~8)
    "age_grp_enc",      # 연령대 인코딩
    "gender_enc",       # 성별 (0=여, 1=남)
    "travel_purpose_enc",
    "travel_styl_avg",  # TRAVEL_STYL_1~8 평균
]
TARGET_COL = "log_total_consume"  # log1p(총소비금액)

MONTH_TO_SEASON = {
    1: 4, 2: 4,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 2,
    9: 3, 10: 3, 11: 3,
    12: 4,
}


# ══════════════════════════════════════════════════════════════════════════════
# 유틸 함수
# ══════════════════════════════════════════════════════════════════════════════
def _read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"  ⚠️  파일 없음: {os.path.basename(path)}")
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
    if not os.path.isdir(directory):
        return None
    for fname in os.listdir(directory):
        if keyword.lower() in fname.lower() and fname.endswith(".csv"):
            return os.path.join(directory, fname)
    matches = glob.glob(os.path.join(directory, "**", f"*{keyword}*"), recursive=True)
    csv_matches = [m for m in matches if m.endswith(".csv")]
    return csv_matches[0] if csv_matches else None


def map_income_tier(income) -> int:
    """
    소득 분위(1~8) → 소비 성향 3단계
      0 = 짠순이 (1~3분위): 저소득, 비용 최소화
      1 = 보통   (4~6분위): 중간 소득, 평균적 소비
      2 = 호캉스 (7~8분위): 고소득, 숙박·식음료 지출 많음
    """
    try:
        v = int(income)
    except (ValueError, TypeError):
        return 1  # 결측 → 보통으로 처리
    if v <= 3:
        return 0
    elif v <= 6:
        return 1
    else:
        return 2


def target_encode(train_series: pd.Series, train_target: pd.Series,
                  test_series: pd.Series, smoothing: float = 10.0) -> tuple[pd.Series, pd.Series]:
    """
    시군구 코드 → 타겟(소비금액) 기반 인코딩
    smoothing: 카테고리 빈도 낮을 때 전체 평균으로 수렴하는 강도
    """
    global_mean = train_target.mean()
    stats = (
        pd.DataFrame({"cat": train_series, "target": train_target})
        .groupby("cat")["target"]
        .agg(["count", "mean"])
    )
    stats["encoded"] = (
        (stats["count"] * stats["mean"] + smoothing * global_mean)
        / (stats["count"] + smoothing)
    )
    enc_map = stats["encoded"].to_dict()

    train_enc = train_series.map(enc_map).fillna(global_mean)
    test_enc  = test_series.map(enc_map).fillna(global_mean)
    return train_enc, test_enc, enc_map, global_mean


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드 — 4개 소비 테이블 합산 + traveller_master
# ══════════════════════════════════════════════════════════════════════════════
def load_aihub_data(data_dir: str) -> pd.DataFrame:
    """
    AI Hub 여행로그 다중 테이블 병합

    머지 순서:
      tn_travel  ← 여행 기간/목적/TRAVELER_ID
        + tn_traveller_master ← income, gender, age_grp, travel_styl
        + 4개 소비 테이블 합산 (활동+숙박+이동+사전)
        + tn_companion ← 동반자 수
        + tn_visit_area ← 시군구, 체류시간
    """
    travel    = _read_csv(_find_csv(data_dir, "tn_travel_"))
    traveller = _read_csv(_find_csv(data_dir, "traveller_master"))
    activity  = _read_csv(_find_csv(data_dir, "activity_consume"))
    lodge     = _read_csv(_find_csv(data_dir, "lodge_consume"))
    mvmn      = _read_csv(_find_csv(data_dir, "mvmn_consume"))
    adv       = _read_csv(_find_csv(data_dir, "adv_consume"))
    companion = _read_csv(_find_csv(data_dir, "companion"))
    visit     = _read_csv(_find_csv(data_dir, "visit_area"))

    if travel is None or activity is None:
        print("  ❌ 필수 테이블(tn_travel, activity_consume) 없음")
        return pd.DataFrame()

    # ── 4개 소비 테이블 합산 ──────────────────────────────────────────────────
    consume_parts = []
    for df_c, label in [(activity, "활동"), (lodge, "숙박"),
                        (mvmn, "이동"), (adv, "사전")]:
        if df_c is not None and "PAYMENT_AMT_WON" in df_c.columns:
            part = (
                df_c.groupby("TRAVEL_ID")["PAYMENT_AMT_WON"]
                .sum()
                .reset_index(name=f"consume_{label}")
            )
            consume_parts.append(part)

    if not consume_parts:
        print("  ❌ 소비 테이블 없음")
        return pd.DataFrame()

    # 전체 합산 → total_consume
    from functools import reduce
    total_consume = reduce(
        lambda a, b: a.merge(b, on="TRAVEL_ID", how="outer"),
        consume_parts
    ).fillna(0)
    consume_cols = [c for c in total_consume.columns if c.startswith("consume_")]
    total_consume["total_consume"] = total_consume[consume_cols].sum(axis=1)
    print(f"  소비 합산: {total_consume.shape} | 평균={total_consume['total_consume'].mean():,.0f}원")

    # ── traveller_master → income_tier, gender, age_grp, travel_styl ─────────
    if traveller is not None:
        styl_cols = [c for c in traveller.columns if c.startswith("TRAVEL_STYL_")]
        traveller["travel_styl_avg"] = traveller[styl_cols].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
        keep_master = ["TRAVELER_ID", "INCOME", "GENDER", "AGE_GRP", "travel_styl_avg"]
        keep_master = [c for c in keep_master if c in traveller.columns]
        traveller = traveller[keep_master].copy()

    # ── 동반자 수 ─────────────────────────────────────────────────────────────
    if companion is not None:
        companion_agg = (
            companion.groupby("TRAVEL_ID").size().reset_index(name="COMPANION_CNT")
        )
    else:
        companion_agg = None

    # ── 방문지: SGG_CD(첫 번째) + 체류시간 합계 ───────────────────────────────
    if visit is not None:
        sgg_col = next((c for c in visit.columns
                        if "SGG" in c.upper() or "시군구" in c), None)
        time_col = next((c for c in visit.columns
                         if "RESIDENCE_TIME" in c.upper() or "체류" in c), None)
        agg_dict = {}
        if sgg_col:
            agg_dict["SGG_CD"] = (sgg_col, "first")
        if time_col:
            agg_dict["RESIDENCE_TIME_MIN"] = (time_col, "sum")
        if agg_dict:
            visit_agg = visit.groupby("TRAVEL_ID").agg(**agg_dict).reset_index()
        else:
            visit_agg = None
    else:
        visit_agg = None

    # ── 여행 날짜 처리 ────────────────────────────────────────────────────────
    travel = travel.copy()
    travel["TRAVEL_START_YMD"] = pd.to_datetime(travel["TRAVEL_START_YMD"], errors="coerce")
    travel["TRAVEL_END_YMD"]   = pd.to_datetime(travel["TRAVEL_END_YMD"],   errors="coerce")
    travel["LODGING_YN"] = (
        (travel["TRAVEL_END_YMD"] - travel["TRAVEL_START_YMD"]).dt.days >= 1
    ).map({True: "Y", False: "N"})

    # ── 머지 ────────────────────────────────────────────────────────────────
    merged = travel.merge(total_consume[["TRAVEL_ID", "total_consume"]],
                          on="TRAVEL_ID", how="inner")

    if traveller is not None and "TRAVELER_ID" in travel.columns:
        merged = merged.merge(traveller, on="TRAVELER_ID", how="left")

    if companion_agg is not None:
        merged = merged.merge(companion_agg, on="TRAVEL_ID", how="left")

    if visit_agg is not None:
        merged = merged.merge(visit_agg, on="TRAVEL_ID", how="left")

    print(f"  최종 머지: {merged.shape[0]:,}행 / {merged.shape[1]}열")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 더미 데이터 (AI Hub 없을 때)
# ══════════════════════════════════════════════════════════════════════════════
def make_dummy_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_days = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 3, n), unit="D"
    )
    end_days = start_days + pd.to_timedelta(rng.integers(1, 4, n), unit="D")
    sgg_pool = ["11110","11140","11170","11200","11215","11230",
                "41110","41130","41150","41280","41310","41430"]

    df = pd.DataFrame({
        "SGG_CD":           rng.choice(sgg_pool, n),
        "total_consume":    rng.integers(5_000, 200_000, n),
        "TRAVEL_START_YMD": start_days.strftime("%Y-%m-%d"),
        "TRAVEL_END_YMD":   end_days.strftime("%Y-%m-%d"),
        "COMPANION_CNT":    rng.integers(1, 5, n),
        "LODGING_YN":       rng.choice(["Y", "N"], n, p=[0.3, 0.7]),
        "distance_km":      rng.uniform(5, 40, n).round(1),
        "INCOME":           rng.integers(1, 9, n),
        "GENDER":           rng.choice(["남", "여"], n),
        "AGE_GRP":          rng.choice([20, 30, 40, 50, 60], n),
        "TRAVEL_PURPOSE":   rng.integers(1, 30, n),
        "travel_styl_avg":  rng.uniform(1, 5, n).round(2),
    })
    print(f"  합성 더미 데이터: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: 전처리
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # ── 컬럼명 매핑 ───────────────────────────────────────────────────────────
    col_map = {
        "SGG_CD": "sgg_code", "TC_SGG_CD": "sgg_code",
        "COMPANION_CNT": "companion_cnt", "ACCOMPANY_CNT": "companion_cnt",
        "LODGING_YN": "has_lodging",
        "RESIDENCE_TIME_MIN": "residence_min",
        "INCOME": "income_raw",
        "GENDER": "gender_raw",
        "AGE_GRP": "age_grp_raw",
        "TRAVEL_PURPOSE": "travel_purpose_raw",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # ── 소비 금액 이상치 제거 (q01~q99) ─────────────────────────────────────
    df["total_consume"] = pd.to_numeric(df["total_consume"], errors="coerce").fillna(0)
    df = df[df["total_consume"] > 0].copy()
    q01 = df["total_consume"].quantile(0.01)
    q99 = df["total_consume"].quantile(0.99)
    before = len(df)
    df = df[(df["total_consume"] >= q01) & (df["total_consume"] <= q99)].copy()
    print(f"  이상치 제거: {before - len(df)}행 제거 → {len(df)}행 잔존"
          f"  (범위: {q01:,.0f}~{q99:,.0f}원)")

    # ── 로그 변환 타겟 ────────────────────────────────────────────────────────
    df[TARGET_COL] = np.log1p(df["total_consume"])

    # ── 날짜 파생 ─────────────────────────────────────────────────────────────
    if "TRAVEL_START_YMD" in df.columns:
        df["TRAVEL_START_YMD"] = pd.to_datetime(df["TRAVEL_START_YMD"], errors="coerce")
        df["TRAVEL_END_YMD"]   = pd.to_datetime(df.get("TRAVEL_END_YMD", df["TRAVEL_START_YMD"]),
                                                  errors="coerce")
        diff = (df["TRAVEL_END_YMD"] - df["TRAVEL_START_YMD"]).dt.total_seconds() / 3600
        df["travel_duration_h"] = diff.clip(lower=0).fillna(8.0)
        df["day_of_week"] = df["TRAVEL_START_YMD"].dt.dayofweek.fillna(0).astype(int)
        df["season"] = df["TRAVEL_START_YMD"].dt.month.map(MONTH_TO_SEASON).fillna(1).astype(int)
    else:
        df["travel_duration_h"] = 8.0
        df["day_of_week"] = 0
        df["season"] = 1

    # ── 거리 km ───────────────────────────────────────────────────────────────
    if "distance_km" not in df.columns:
        if "residence_min" in df.columns:
            df["distance_km"] = (
                pd.to_numeric(df["residence_min"], errors="coerce") / 60 * 15
            ).clip(lower=1.0).fillna(10.0)
        else:
            df["distance_km"] = 10.0
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").fillna(10.0)

    # ── 동반자 수 ─────────────────────────────────────────────────────────────
    if "companion_cnt" not in df.columns:
        df["companion_cnt"] = 1
    df["companion_cnt"] = pd.to_numeric(df["companion_cnt"], errors="coerce").fillna(1).astype(int)

    # ── 숙박 여부 ─────────────────────────────────────────────────────────────
    if "has_lodging" in df.columns:
        df["has_lodging"] = df["has_lodging"].map({"Y": 1, "N": 0, 1: 1, 0: 0}).fillna(0).astype(int)
    else:
        df["has_lodging"] = 0

    # ── 시군구 코드 (TargetEncoding은 분할 후 적용) ───────────────────────────
    if "sgg_code" not in df.columns:
        df["sgg_code"] = "0"
    df["sgg_code"] = df["sgg_code"].astype(str).str.strip()

    # ── 소득 분위 → 3단계 income_tier ────────────────────────────────────────
    if "income_raw" in df.columns:
        df["income_tier"] = df["income_raw"].apply(map_income_tier)
    else:
        df["income_tier"] = 1  # 보통

    # ── 성별 인코딩 ──────────────────────────────────────────────────────────
    if "gender_raw" in df.columns:
        gender_map = {"남": 1, "M": 1, "1": 1, "여": 0, "F": 0, "2": 0}
        df["gender_enc"] = df["gender_raw"].astype(str).map(gender_map).fillna(0).astype(int)
    else:
        df["gender_enc"] = 0

    # ── 연령대 인코딩 ─────────────────────────────────────────────────────────
    if "age_grp_raw" in df.columns:
        df["age_grp_enc"] = pd.to_numeric(
            df["age_grp_raw"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        ).fillna(30).astype(int) // 10  # 20대=2, 30대=3, ...
    else:
        df["age_grp_enc"] = 3

    # ── 여행 목적 인코딩 ──────────────────────────────────────────────────────
    if "travel_purpose_raw" in df.columns:
        df["travel_purpose_enc"] = pd.to_numeric(
            df["travel_purpose_raw"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        df["travel_purpose_enc"] = 0

    # ── 여행 스타일 평균 ──────────────────────────────────────────────────────
    if "travel_styl_avg" not in df.columns:
        styl_cols = [c for c in df.columns if "TRAVEL_STYL" in c.upper()]
        if styl_cols:
            df["travel_styl_avg"] = df[styl_cols].apply(
                pd.to_numeric, errors="coerce"
            ).mean(axis=1).fillna(3.0)
        else:
            df["travel_styl_avg"] = 3.0
    df["travel_styl_avg"] = pd.to_numeric(df["travel_styl_avg"], errors="coerce").fillna(3.0)

    df = df.dropna(subset=["sgg_code", TARGET_COL])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: TabNet 학습
# ══════════════════════════════════════════════════════════════════════════════
def train_tabnet(df: pd.DataFrame):
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
        import torch
    except ImportError:
        print("  ❌ pytorch-tabnet 없음: pip install pytorch-tabnet")
        sys.exit(1)

    # ── 70 / 15 / 15 분할 ────────────────────────────────────────────────────
    df_train, df_tmp = train_test_split(df, test_size=0.30, random_state=42)
    df_val,   df_test = train_test_split(df_tmp, test_size=0.50, random_state=42)
    print(f"  분할: train={len(df_train):,} / val={len(df_val):,} / test={len(df_test):,}")

    # ── TargetEncoding (train 기준 학습, val/test 적용) ───────────────────────
    train_enc, val_enc, enc_map, global_mean = target_encode(
        df_train["sgg_code"], df_train[TARGET_COL],
        df_val["sgg_code"]
    )
    _, test_enc, _, _ = target_encode(
        df_train["sgg_code"], df_train[TARGET_COL],
        df_test["sgg_code"]
    )
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_test  = df_test.copy()
    df_train["sgg_enc"] = train_enc.values
    df_val["sgg_enc"]   = val_enc.values
    df_test["sgg_enc"]  = test_enc.values

    # ── 피처 / 타겟 배열 ──────────────────────────────────────────────────────
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

    scaler_path = os.path.join(MODELS_DIR, "consume_scaler_v2.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✅ consume_scaler_v2.pkl 저장")

    # ── 계절 가중치 (여름=0.5, 나머지=1.5) ────────────────────────────────────
    summer_mask = df_train["season"] == 2
    sample_weights = np.where(summer_mask, 0.5, 1.5).astype(np.float32)

    # ── TabNet 학습 ───────────────────────────────────────────────────────────
    model = TabNetRegressor(
        n_d=32, n_a=32, n_steps=4,
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
        max_epochs=150,
        patience=20,
        batch_size=256,
        virtual_batch_size=128,
        weights=sample_weights,
    )

    # ── 성능 평가 (원래 스케일 역변환) ───────────────────────────────────────
    def eval_set(X_sc, y_log, split_name):
        pred_log = model.predict(X_sc)
        pred     = np.expm1(pred_log).flatten()
        actual   = np.expm1(y_log).flatten()
        mae_v = mean_absolute_error(actual, pred)
        r2_v  = r2_score(actual, pred)
        print(f"  [{split_name}] MAE={mae_v:,.0f}원  R²={r2_v:.4f}")
        return mae_v, r2_v

    val_mae,  val_r2  = eval_set(X_val_sc, y_val, "val ")
    test_mae, test_r2 = eval_set(X_te_sc,  y_te,  "test")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(MODELS_DIR, "consume_regressor_v2")
    model.save_model(save_path)
    print(f"  ✅ consume_regressor_v2.zip 저장")

    meta = {
        "feature_cols":    feat_cols,
        "target_col":      TARGET_COL,
        "log_transform":   True,
        "income_tier_map": {"0": "짠순이(1~3분위)", "1": "보통(4~6분위)", "2": "호캉스(7~8분위)"},
        "season_label":    {"1": "봄", "2": "여름", "3": "가을", "4": "겨울"},
        "sgg_enc_global_mean": float(global_mean),
        "val_mae_krw":     round(val_mae),
        "val_r2":          round(val_r2, 4),
        "test_mae_krw":    round(test_mae),
        "test_r2":         round(test_r2, 4),
        "n_train":         len(df_train),
        "n_val":           len(df_val),
        "n_test":          len(df_test),
        "model_version":   "v2",
    }
    meta_path = os.path.join(MODELS_DIR, "consume_meta_v2.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  ✅ consume_meta_v2.json 저장")

    # TargetEncoding 맵 별도 저장 (추론 시 필요)
    enc_path = os.path.join(MODELS_DIR, "consume_target_enc_v2.pkl")
    with open(enc_path, "wb") as f:
        pickle.dump({"enc_map": enc_map, "global_mean": float(global_mean)}, f)
    print(f"  ✅ consume_target_enc_v2.pkl 저장")

    return model, scaler, meta


# ══════════════════════════════════════════════════════════════════════════════
# 추론 함수 (FastAPI / Streamlit에서 import)
# ══════════════════════════════════════════════════════════════════════════════
def predict_consume_v2(
    sgg_code: str = "11110",
    travel_duration_h: float = 8.0,
    distance_km: float = 15.0,
    companion_cnt: int = 1,
    season: int = 1,
    day_of_week: int = 0,
    has_lodging: int = 0,
    income_tier: int = 1,    # 0=짠순이, 1=보통, 2=호캉스
    age_grp_enc: int = 3,    # 30대
    gender_enc: int = 0,
    travel_purpose_enc: int = 0,
    travel_styl_avg: float = 3.0,
) -> dict:
    """
    v2 소비 예측 추론 함수

    income_tier: 0=짠순이(1~3분위), 1=보통(4~6분위), 2=호캉스(7~8분위)
    """
    model_zip  = os.path.join(MODELS_DIR, "consume_regressor_v2.zip")
    scaler_path = os.path.join(MODELS_DIR, "consume_scaler_v2.pkl")
    meta_path  = os.path.join(MODELS_DIR, "consume_meta_v2.json")
    enc_path   = os.path.join(MODELS_DIR, "consume_target_enc_v2.pkl")

    if not os.path.exists(model_zip):
        return {
            "estimated_cost_krw": 15000,
            "note": "v2 모델 없음 — build_consume_model_v2.py 실행 필요 (기본값)",
            "model_version": "v2_missing",
        }

    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
    except ImportError:
        return {"estimated_cost_krw": 15000, "note": "pytorch-tabnet 미설치"}

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    with open(enc_path, "rb") as f:
        enc_data = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    sgg_enc = enc_data["enc_map"].get(
        str(sgg_code), enc_data["global_mean"]
    )

    model = TabNetRegressor()
    model.load_model(model_zip)

    feat_order = meta["feature_cols"]
    feat_values = {
        "sgg_enc": sgg_enc,
        "travel_duration_h": travel_duration_h,
        "distance_km": distance_km,
        "companion_cnt": companion_cnt,
        "season": season,
        "day_of_week": day_of_week,
        "has_lodging": has_lodging,
        "income_tier": income_tier,
        "age_grp_enc": age_grp_enc,
        "gender_enc": gender_enc,
        "travel_purpose_enc": travel_purpose_enc,
        "travel_styl_avg": travel_styl_avg,
    }
    X = np.array([[feat_values.get(f, 0) for f in feat_order]], dtype=np.float32)
    X_sc  = scaler.transform(X)
    y_log = model.predict(X_sc)
    cost  = int(np.expm1(float(y_log[0])))

    tier_labels = {0: "짠순이", 1: "보통", 2: "호캉스"}
    return {
        "estimated_cost_krw": max(cost, 0),
        "income_tier_label":  tier_labels.get(income_tier, "보통"),
        "model_mae_krw":      meta.get("test_mae_krw", 0),
        "model_r2":           meta.get("test_r2", 0.0),
        "model_version":      "v2",
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="소비 TabNet v2 학습 스크립트")
    parser.add_argument("--data_dir", default=AIHUB_DIR,
                        help="AI Hub 여행로그 라벨링 데이터 디렉토리")
    parser.add_argument("--use_dummy", action="store_true",
                        help="합성 데이터로 학습 (AI Hub 없을 때)")
    parser.add_argument("--dummy_n", type=int, default=2000,
                        help="더미 데이터 행 수 (기본: 2000)")
    args = parser.parse_args()

    print("=" * 65)
    print("STEP 1: 데이터 로드 (4개 소비 테이블 합산 + traveller_master)")
    print("=" * 65)

    if args.use_dummy:
        print("  [더미 모드] 합성 데이터 생성 중...")
        df_raw = make_dummy_data(n=args.dummy_n)
    else:
        print(f"  AI Hub 경로: {args.data_dir}")
        df_raw = load_aihub_data(args.data_dir)
        if df_raw.empty:
            print("  ❌ 데이터 없음. --use_dummy 플래그를 사용하세요.")
            sys.exit(1)

    print(f"  로드 shape: {df_raw.shape}")

    print("\n" + "=" * 65)
    print("STEP 2: 전처리 (이상치 제거 + 피처 엔지니어링)")
    print("=" * 65)
    df = preprocess(df_raw)
    print(f"  전처리 후: {df.shape}")

    income_tier_map = {0: "짠순이", 1: "보통", 2: "호캉스"}
    print(f"\n  소득 분위 분포:\n{df['income_tier'].map(income_tier_map).value_counts().to_string()}")
    season_names = {1: "봄", 2: "여름", 3: "가을", 4: "겨울"}
    print(f"\n  계절 분포:\n{df['season'].map(season_names).value_counts().to_string()}")
    print(f"\n  총소비 통계:\n{df['total_consume'].describe().round(0).to_string()}\n")

    if len(df) < 50:
        print("  ⚠️ 학습 데이터 50행 미만. --dummy_n을 늘리거나 실제 데이터를 사용하세요.")
        sys.exit(1)

    print("=" * 65)
    print("STEP 3: TabNet v2 학습 (70/15/15 분할, 계절 가중치 적용)")
    print("=" * 65)
    train_tabnet(df)

    print("\n" + "=" * 65)
    print("✅ build_consume_model_v2.py 완료")
    print("  출력: models/consume_regressor_v2.zip")
    print("        models/consume_scaler_v2.pkl")
    print("        models/consume_target_enc_v2.pkl")
    print("        models/consume_meta_v2.json")
    print("=" * 65)
