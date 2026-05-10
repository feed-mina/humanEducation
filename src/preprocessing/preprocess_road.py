"""
preprocess_road.py
==================
전국자전거도로표준데이터.csv → 서울+경기 재전처리 → road_clean_v2.csv

변경사항 (v2):
  - 필터 기준: 시도명 → 시군구명 기반 (더 세밀한 지역 분류)
  - PK: 노선명+시군구명 복합키 (노선명 단독 PK 불가 확인)
  - 좌표 결측 행에 기점지번주소 보관 (후속 geocoding 대비)
  - 파생 피처: is_wide_road, safety_index (너비×0.7 + 길이×0.3)
  - 모델 3개 비교: LinearRegression / PolynomialRegression / RandomForest

실행:
  python kride-project/preprocess_road.py

출력:
  kride-project/data/raw_ml/road_clean_v2.csv
"""

import os
import sys
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI 없는 환경에서 그래프 저장용
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────
# Jupyter에서는 __file__이 없으므로 fallback 처리
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Jupyter 실행 시: 현재 작업 디렉토리 기준 kride-project 폴더
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()  # kride-project 안에서 실행하는 경우
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw_ml")
INPUT_PATH = os.path.join(RAW_DIR, "전국자전거도로표준데이터.csv")
OUTPUT_PATH = os.path.join(RAW_DIR, "road_clean_v2.csv")

# 서울+경기 시군구명 목록 (시도명 대신 시군구명으로 필터)
SEOUL_GU = [
    "종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구",
    "강북구","도봉구","노원구","은평구","서대문구","마포구","양천구","강서구",
    "구로구","금천구","영등포구","동작구","관악구","서초구","강남구","송파구","강동구",
]
GYEONGGI_SI_GUN = [
    "수원시","성남시","의정부시","안양시","부천시","광명시","평택시","동두천시",
    "안산시","고양시","과천시","구리시","남양주시","오산시","시흥시","군포시",
    "의왕시","하남시","용인시","파주시","이천시","안성시","김포시","화성시",
    "광주시","양주시","포천시","여주시","연천군","가평군","양평군",
]
TARGET_SIGUNGU = SEOUL_GU + GYEONGGI_SI_GUN


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: 원본 로드
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: 원본 CSV 로드")
print("=" * 60)

try:
    df = pd.read_csv(INPUT_PATH, encoding="cp949")
except UnicodeDecodeError:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
print(f"  원본 shape : {df.shape}")
print(f"  전체 컬럼 :")
for c in df.columns:
    print(f"    - {repr(c)}")
print()

# 노선명 PK 가능 여부 확인
n_unique = df["노선명"].nunique()
print(f"  노선명 유일값: {n_unique:,} / 전체: {len(df):,}")
print(f"  → 노선명 단독 PK {'가능' if n_unique == len(df) else '불가 (복합키 사용)'}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 서울+경기 필터링 (시군구명 기준)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 2: 서울+경기 필터링 (시군구명 기준)")
print("=" * 60)

df_target = df[df["시군구명"].isin(TARGET_SIGUNGU)].copy()
print(f"  전체 {len(df):,}행  →  서울+경기 {len(df_target):,}행")
print(f"  시군구별 분포:\n{df_target['시군구명'].value_counts().head(15).to_string()}\n")

# 복합 PK 생성
df_target["road_id"] = df_target["노선명"] + "_" + df_target["시군구명"]
print(f"  복합키(road_id) 유일값: {df_target['road_id'].nunique():,}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: 컬럼 선택 및 이름 정리
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 3: 컬럼 선택 및 타입 변환")
print("=" * 60)

# 원본 컬럼명 → 영문 매핑
col_map = {
    "노선명"              : "route_name",
    "시도명"              : "sido",
    "시군구명"            : "sigungu",
    "기점지번주소"        : "start_addr_jibun",      # geocoding 대비 보존
    "기점도로명주소"      : "start_addr_road",
    "기점위도"            : "start_lat",
    "기점경도"            : "start_lon",
    "종점위도"            : "end_lat",
    "종점경도"            : "end_lon",
    "총길이(km)"          : "length_km",
    "자전거전용도로너비(m)": "width_m",
    "자전거전용도로종류"  : "road_type",
    "자전거전용도로이용가능여부": "is_official",
}

# 실제 존재하는 컬럼만 선택
existing_cols = {k: v for k, v in col_map.items() if k in df_target.columns}
df_clean = df_target[list(existing_cols.keys()) + ["road_id"]].rename(columns=existing_cols)

# width_m: 자전거전용도로너비가 없으면 일반도로너비(m)로 대체
if "width_m" not in df_clean.columns:
    alt_width = "일반도로너비(m)"
    if alt_width in df_target.columns:
        df_clean["width_m"] = df_target[alt_width].values
        print(f"  ⚠️  자전거전용도로너비 없음 → '{alt_width}' 대체 사용")
    else:
        df_clean["width_m"] = float("nan")
        print(f"  ⚠️  너비 컬럼 없음 → width_m = NaN")

# 숫자 변환 (존재하는 컬럼만)
for col in ["width_m", "length_km", "start_lat", "start_lon", "end_lat", "end_lon"]:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

print(f"  정제 후 shape : {df_clean.shape}")
print(f"  결측값 현황:\n{df_clean.isnull().sum().to_string()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: 파생 피처 생성
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 4: 파생 피처 생성")
print("=" * 60)

# is_wide_road: 너비 2.0m 이상 → 1 (width_m 없으면 0)
if "width_m" in df_clean.columns:
    df_clean["is_wide_road"] = (df_clean["width_m"] >= 2.0).astype(int)
else:
    df_clean["is_wide_road"] = 0

# is_official: 이용가능여부 → 1/0
if "is_official" in df_clean.columns:
    df_clean["is_official"] = df_clean["is_official"].apply(
        lambda x: 1 if str(x).strip() in ["가능", "Y", "1", "True"] else 0
    )

# safety_index: 너비(0.7) + 길이(0.3) → MinMaxScaler 정규화 후 가중합
valid_mask = df_clean[["width_m", "length_km"]].notna().all(axis=1)
scaler_si = MinMaxScaler()
normalized = scaler_si.fit_transform(df_clean.loc[valid_mask, ["width_m", "length_km"]])
df_clean.loc[valid_mask, "safety_index"] = normalized[:, 0] * 0.7 + normalized[:, 1] * 0.3

wide = df_clean["is_wide_road"].sum()
si_mean = df_clean["safety_index"].mean()
print(f"  is_wide_road : 넓음(1)={wide:,}개 ({wide/len(df_clean)*100:.1f}%)")
print(f"  safety_index : 평균={si_mean:.3f}  결측={df_clean['safety_index'].isna().sum()}\n")

# 좌표 유무 구분
has_coord = df_clean[["start_lat", "start_lon"]].notna().all(axis=1)
print(f"  좌표 있는 행 : {has_coord.sum():,} / {len(df_clean):,} ({has_coord.mean()*100:.1f}%)")
print(f"  좌표 없는 행 : {(~has_coord).sum():,}  (기점지번주소 컬럼에 보존 → 추후 geocoding)\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: CSV 저장
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 5: 저장")
print("=" * 60)

df_clean.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"  ✅ 저장 완료 → {OUTPUT_PATH}")
print(f"  최종 shape  : {df_clean.shape}")
print(f"  컬럼        : {list(df_clean.columns)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: 모델 3개 비교 (좌표 있는 행만 사용)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 6: 모델 3개 비교 (타겟: safety_index)")
print("=" * 60)

df_model = df_clean[has_coord & df_clean["safety_index"].notna()].copy()
print(f"  모델 학습 대상 : {len(df_model):,}행\n")

if len(df_model) < 30:
    print("  ⚠️  학습 데이터 부족 (30행 미만) — 모델 비교 생략")
    sys.exit(0)

# 피처 / 타겟
FEATURES_SIMPLE = ["width_m", "length_km"]
FEATURES_MULTI  = ["width_m", "length_km", "start_lat", "start_lon"]
TARGET = "safety_index"

X_s = df_model[FEATURES_SIMPLE].fillna(df_model[FEATURES_SIMPLE].median())
X_m = df_model[FEATURES_MULTI].fillna(df_model[FEATURES_MULTI].median())
y   = df_model[TARGET]

X_s_tr, X_s_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=42)
X_m_tr, X_m_te, _,   _     = train_test_split(X_m, y, test_size=0.2, random_state=42)

results = {}

# ── 모델 1: LinearRegression (단순: 너비+길이) ──
lr = LinearRegression()
lr.fit(X_s_tr, y_tr)
y_pred_lr = lr.predict(X_s_te)
results["LinearRegression"] = {
    "R²" : round(r2_score(y_te, y_pred_lr), 4),
    "MSE": round(mean_squared_error(y_te, y_pred_lr), 4),
    "피처": FEATURES_SIMPLE,
}
print(f"  [1] LinearRegression     R²={results['LinearRegression']['R²']:.4f}")

# ── 모델 2: PolynomialRegression (degree=2, 너비+길이) ──
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_tr = poly.fit_transform(X_s_tr)
X_poly_te = poly.transform(X_s_te)
lr_poly = LinearRegression()
lr_poly.fit(X_poly_tr, y_tr)
y_pred_poly = lr_poly.predict(X_poly_te)
results["PolynomialRegression(d=2)"] = {
    "R²" : round(r2_score(y_te, y_pred_poly), 4),
    "MSE": round(mean_squared_error(y_te, y_pred_poly), 4),
    "피처": FEATURES_SIMPLE + ["(degree=2 확장)"],
}
print(f"  [2] PolynomialRegression R²={results['PolynomialRegression(d=2)']['R²']:.4f}")

# ── 모델 3: RandomForestRegressor (너비+길이+위경도) ──
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_m_tr, y_tr)
y_pred_rf = rf.predict(X_m_te)
results["RandomForest"] = {
    "R²" : round(r2_score(y_te, y_pred_rf), 4),
    "MSE": round(mean_squared_error(y_te, y_pred_rf), 4),
    "피처": FEATURES_MULTI,
}
print(f"  [3] RandomForest         R²={results['RandomForest']['R²']:.4f}")

# 피처 중요도 출력
importances = rf.feature_importances_
print("\n  RandomForest 피처 중요도:")
for feat, imp in sorted(zip(FEATURES_MULTI, importances), key=lambda x: -x[1]):
    print(f"    {feat:<20}: {imp:.4f}")

# 결과 요약
print("\n  ── 모델 비교 요약 ──")
print(f"  {'모델':<30} {'R²':>8} {'MSE':>10}")
print(f"  {'-'*50}")
best_model = max(results, key=lambda k: results[k]["R²"])
for name, res in results.items():
    marker = " ← 최고" if name == best_model else ""
    print(f"  {name:<30} {res['R²']:>8.4f} {res['MSE']:>10.4f}{marker}")


# ── 그래프 저장 ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
preds = [y_pred_lr, y_pred_poly, y_pred_rf]
titles = ["LinearRegression", "Polynomial(d=2)", "RandomForest"]

for ax, pred, title, res in zip(axes, preds, titles, results.values()):
    ax.scatter(y_te, pred, alpha=0.4, s=15)
    lims = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}\nR²={res['R²']:.4f}")

plt.tight_layout()
plot_path = os.path.join(BASE_DIR, "data", "raw_ml", "model_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n  그래프 저장 → {plot_path}")
print("\n✅ preprocess_road.py 완료")
