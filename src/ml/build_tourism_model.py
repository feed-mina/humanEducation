"""
build_tourism_model.py
======================
수도권 자전거도로 관광점수 모델 파이프라인

[ 실행 순서 ]
  1. python kride-project/preprocess_road.py      ← road_clean_v2.csv 생성
  2. python kride-project/build_safety_model.py   ← safety pkl 생성
  3. python kride-project/build_tourism_model.py  ← 이 파일

[ 단계 요약 ]
  STEP 1 : road_features.csv 로드 (tourist_count, cultural_count 등)
  STEP 2 : 관광점수(tourism_score) 계산
             └ raw = tourist_count×0.5 + cultural_count×0.3 + leisure_count×0.2
             └ tourism_score = MinMaxScaler(raw) + facility_bonus (cap 0.1)
  STEP 3 : safety_score 계산 (safety_regressor.pkl 적용)
  STEP 4 : final_score = safety_score×0.6 + tourism_score×0.4
  STEP 5 : 결과 저장

[ 출력 파일 ]
  kride-project/models/tourism_scaler.pkl      — MinMaxScaler (관광 raw score용)
  kride-project/data/raw_ml/road_scored.csv   — 전체 점수 포함 최종 데이터셋
"""

import os
import sys
import warnings

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()

RAW_DIR    = os.path.join(BASE_DIR, "data", "raw_ml")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURES_PATH    = os.path.join(RAW_DIR,    "road_features.csv")
REGRESSOR_PATH   = os.path.join(MODELS_DIR, "safety_regressor.pkl")
SCALER_PATH      = os.path.join(MODELS_DIR, "safety_scaler.pkl")
META_PATH        = os.path.join(MODELS_DIR, "safety_meta.pkl")
TOURISM_SCL_PATH = os.path.join(MODELS_DIR, "tourism_scaler.pkl")
OUTPUT_PATH      = os.path.join(RAW_DIR,    "road_scored.csv")

W_SAFETY  = 0.6   # final_score 안전 가중치
W_TOURISM = 0.4   # final_score 관광 가중치


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: road_features.csv 로드
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: road_features.csv 로드")
print("=" * 65)

if not os.path.exists(FEATURES_PATH):
    print(f"  ❌ {FEATURES_PATH} 없음. Spatial Join 결과 파일이 필요합니다.")
    sys.exit(1)

df = pd.read_csv(FEATURES_PATH, encoding="utf-8-sig")
print(f"  shape: {df.shape}")
print(f"  컬럼: {list(df.columns)}\n")

# 관광 피처 컬럼 확인
TOURISM_COLS = ["tourist_count", "cultural_count", "leisure_count", "facility_count"]
missing_cols = [c for c in TOURISM_COLS if c not in df.columns]
if missing_cols:
    print(f"  ⚠️  없는 컬럼: {missing_cols} → 0으로 대체")
    for c in missing_cols:
        df[c] = 0

for c in TOURISM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

print(f"  관광 피처 통계:")
print(df[TOURISM_COLS].describe().round(3).to_string())
print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 관광점수(tourism_score) 계산
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 2: 관광점수(tourism_score) 계산")
print("=" * 65)

# 가중합 raw score
df["tourism_raw"] = (
    df["tourist_count"]  * 0.5
    + df["cultural_count"] * 0.3
    + df["leisure_count"]  * 0.2
)

# MinMaxScaler 정규화
tourism_scaler = MinMaxScaler()
df["tourism_score"] = tourism_scaler.fit_transform(df[["tourism_raw"]])

# facility_count 보너스 (+0.1 상한)
fac_norm = MinMaxScaler().fit_transform(df[["facility_count"]])
df["facility_bonus"] = (fac_norm * 0.1).clip(max=0.1)
df["tourism_score"] = (df["tourism_score"] + df["facility_bonus"]).clip(upper=1.0)
df.drop(columns=["facility_bonus"], inplace=True)

print(f"  tourism_raw  : 평균={df['tourism_raw'].mean():.3f}, 최대={df['tourism_raw'].max():.1f}")
print(f"  tourism_score: 평균={df['tourism_score'].mean():.3f}, 최대={df['tourism_score'].max():.3f}")
print(f"  관광점수 0인 세그먼트: {(df['tourism_score'] == 0).sum():,}개 ({(df['tourism_score']==0).mean()*100:.1f}%)\n")

joblib.dump(tourism_scaler, TOURISM_SCL_PATH)
print(f"  ✅ tourism_scaler.pkl 저장 → {TOURISM_SCL_PATH}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: safety_score 계산 (safety_regressor.pkl 적용)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 3: safety_score 계산 (안전 모델 적용)")
print("=" * 65)

if not os.path.exists(REGRESSOR_PATH):
    print(f"  ⚠️  {REGRESSOR_PATH} 없음 → safety_score = 0.5 (기본값)")
    df["safety_score"] = 0.5
else:
    rf_reg       = joblib.load(REGRESSOR_PATH)
    feat_scaler  = joblib.load(SCALER_PATH)
    meta         = joblib.load(META_PATH)
    FEATURES     = meta["features"]

    print(f"  안전 모델 피처: {FEATURES}")
    print(f"  모델 성능 (학습 시): R²={meta['r2_regressor']}, F1={meta['f1_classifier']}\n")

    # road_features.csv에는 district_danger, road_attr_score가 없을 수 있음
    # → 필요 피처 보완
    if "district_danger" not in df.columns:
        district_path = os.path.join(RAW_DIR, "district_danger.csv")
        if os.path.exists(district_path):
            df_danger = pd.read_csv(district_path)
            danger_map = dict(zip(df_danger["sigungu"], df_danger["district_danger"]))
            # sigungu 컬럼 확인
            sigungu_col = next((c for c in ["sigungu", "시군구명"] if c in df.columns), None)
            if sigungu_col:
                df["district_danger"] = df[sigungu_col].map(danger_map).fillna(df_danger["district_danger"].median())
            else:
                df["district_danger"] = df_danger["district_danger"].median()
            print(f"  district_danger 매핑 완료 (district_danger.csv 사용)")
        else:
            df["district_danger"] = 0.5
            print(f"  ⚠️  district_danger.csv 없음 → 0.5 기본값")

    if "road_attr_score" not in df.columns:
        for col in ["width_m", "length_km"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        valid = df[["width_m", "length_km"]].notna().all(axis=1)
        if valid.sum() > 0:
            norm = MinMaxScaler().fit_transform(df.loc[valid, ["width_m", "length_km"]])
            df.loc[valid, "road_attr_score"] = norm[:, 0] * 0.7 + norm[:, 1] * 0.3
        df["road_attr_score"] = df.get("road_attr_score", pd.Series(0.5, index=df.index)).fillna(0.5)

    # 피처 행렬 구성 (없는 컬럼은 0으로)
    X_list = []
    for f in FEATURES:
        if f in df.columns:
            X_list.append(pd.to_numeric(df[f], errors="coerce").fillna(0))
        else:
            print(f"  ⚠️  피처 '{f}' 없음 → 0 대체")
            X_list.append(pd.Series(0.0, index=df.index))

    X = pd.concat(X_list, axis=1)
    X.columns = FEATURES
    X_scaled = feat_scaler.transform(X)

    df["safety_score"] = rf_reg.predict(X_scaled).clip(0, 1)
    print(f"  safety_score: 평균={df['safety_score'].mean():.3f}, 최대={df['safety_score'].max():.3f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: final_score 계산
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print(f"STEP 4: final_score = safety×{W_SAFETY} + tourism×{W_TOURISM}")
print("=" * 65)

df["final_score"] = (
    df["safety_score"]  * W_SAFETY
    + df["tourism_score"] * W_TOURISM
).clip(0, 1)

print(f"  final_score: 평균={df['final_score'].mean():.3f}, 최대={df['final_score'].max():.3f}")
print(f"\n  상위 10개 세그먼트:")

display_cols = [c for c in ["route_name", "sigungu", "safety_score", "tourism_score", "final_score"] if c in df.columns]
if not display_cols:
    display_cols = [c for c in ["노선명", "시군구명", "safety_score", "tourism_score", "final_score"] if c in df.columns]
print(df.nlargest(10, "final_score")[display_cols].to_string(index=False))
print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: 결과 저장
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 5: 결과 저장")
print("=" * 65)

# 불필요한 중간 컬럼 제거
df.drop(columns=["tourism_raw"], inplace=True, errors="ignore")

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"  ✅ road_scored.csv → {OUTPUT_PATH}")
print(f"     shape: {df.shape}")
print(f"     컬럼: {list(df.columns)}")

print("\n" + "=" * 65)
print("✅ build_tourism_model.py 완료")
print(f"   safety_score  평균: {df['safety_score'].mean():.3f}")
print(f"   tourism_score 평균: {df['tourism_score'].mean():.3f}")
print(f"   final_score   평균: {df['final_score'].mean():.3f}")
print("=" * 65)
