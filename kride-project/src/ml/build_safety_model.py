"""
build_safety_model.py
=====================
수도권 자전거도로 안전점수 모델 파이프라인

[ 실행 순서 ]
  1. python kride-project/preprocess_road.py   ← 먼저 실행 (road_clean_v2.csv 생성)
  2. python kride-project/build_safety_model.py ← 이 파일

[ 단계 요약 ]
  STEP 1 : road_clean_v2.csv 로드 (없으면 road_clean.csv fallback)
  STEP 2 : 사고다발지_서울.xlsx -> 구별 위험도(district_danger) 계산
  STEP 3 : 도로 데이터 + 구별 위험도 병합 (sigungu 기준)
  STEP 4 : 피처 생성 + safety_index_v2 계산
             └ safety_index_v2 = (1 - district_danger) × 0.6 + road_attr_score × 0.4
  STEP 5 : 회귀 모델 학습 -> safety_score 연속값 (0~1) 예측
  STEP 6 : 분류 모델 학습 -> 위험등급 3단계 (0=안전/1=보통/2=위험) 예측
  STEP 7 : 평가 출력 (R², F1-macro)
  STEP 8 : 모델 저장

[ 출력 파일 ]
  kride-project/models/safety_regressor.pkl   -- RandomForestRegressor
  kride-project/models/safety_classifier.pkl  -- RandomForestClassifier
  kride-project/models/safety_scaler.pkl      -- MinMaxScaler (추론 시 피처 정규화용)
  kride-project/data/raw_ml/district_danger.csv -- 구별 위험도 테이블 (참고용)
"""

import os
import re
import sys
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # src/ml -> src -> kride-project
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw_ml")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ROAD_NATIONWIDE_PATH     = os.path.join(RAW_DIR, "road_clean_nationwide.csv")
ROAD_V2_PATH             = os.path.join(RAW_DIR, "road_clean_v2.csv")
ROAD_V1_PATH             = os.path.join(RAW_DIR, "road_clean.csv")
DISTRICT_NATIONWIDE_PATH = os.path.join(RAW_DIR, "district_danger_nationwide.csv")  # ← 전국
ACCIDENT_PATH            = os.path.join(RAW_DIR, "다발지분석-24년 자전거 교통사고 다발지역_서울.xlsx")  # fallback
DISTRICT_OUT             = os.path.join(RAW_DIR, "district_danger.csv")

REGRESSOR_PATH  = os.path.join(MODELS_DIR, "safety_regressor.pkl")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "safety_classifier.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "safety_scaler.pkl")

RANDOM_STATE = 42


# ==============================================================================
# STEP 1: 도로 데이터 로드
# ==============================================================================
print("=" * 65)
print("STEP 1: 도로 데이터 로드")
print("=" * 65)

if os.path.exists(ROAD_NATIONWIDE_PATH):
    df_road = pd.read_csv(ROAD_NATIONWIDE_PATH, encoding="utf-8-sig")
    print(f"  [OK] road_clean_nationwide.csv -- {df_road.shape}")
elif os.path.exists(ROAD_V2_PATH):
    df_road = pd.read_csv(ROAD_V2_PATH, encoding="utf-8-sig")
    print(f"  [OK] road_clean_v2.csv -- {df_road.shape}")
elif os.path.exists(ROAD_V1_PATH):
    df_road = pd.read_csv(ROAD_V1_PATH, encoding="utf-8-sig")
    print(f"  [WARN] nationwide/v2 not found -> road_clean.csv fallback -- {df_road.shape}")
else:
    print("  [ERR] road CSV not found. Run preprocess_road_nationwide.py first.")
    sys.exit(1)

# sigungu 컬럼 확인 (v1은 한글, v2는 영문)
if "sigungu" in df_road.columns:
    sigungu_col = "sigungu"
elif "시군구명" in df_road.columns:
    df_road = df_road.rename(columns={"시군구명": "sigungu"})
    sigungu_col = "sigungu"
else:
    print("  [ERR] sigungu 또는 시군구명 컬럼이 없습니다. 파일을 확인하세요.")
    sys.exit(1)

# width_m, length_km 컬럼 통일 (v1에서는 한글 컬럼명일 수 있음)
rename_map = {
    "자전거전용도로너비(m)": "width_m",
    "총길이(km)": "length_km",
    "기점위도": "start_lat",
    "기점경도": "start_lon",
}
df_road = df_road.rename(columns={k: v for k, v in rename_map.items() if k in df_road.columns})

# 필수 컬럼 숫자 변환
for col in ["width_m", "length_km", "start_lat", "start_lon"]:
    if col in df_road.columns:
        df_road[col] = pd.to_numeric(df_road[col], errors="coerce")

print(f"  컬럼 목록: {list(df_road.columns)}")
print(f"  시군구 목록 ({df_road[sigungu_col].nunique()}개): {sorted(df_road[sigungu_col].unique())[:10]} ...\n")


# ==============================================================================
# STEP 2: 사고다발지 데이터 -> 구별 위험도 계산
# ==============================================================================
print("=" * 65)
print("STEP 2: 사고다발지 데이터 -> 구별 위험도(district_danger) 계산")
print("=" * 65)

# STEP 2: 사고 위험도 로드
# ==============================================================================
print("=" * 65)
print("STEP 2: 사고다발지 데이터 -> 시군구별 위험도(district_danger) 계산")
print("=" * 65)

if os.path.exists(DISTRICT_NATIONWIDE_PATH):
    # ── 전국 district_danger_nationwide.csv 우선 사용 ──────────────────────
    print(f"  [전국] district_danger_nationwide.csv 로드")
    df_d = pd.read_csv(DISTRICT_NATIONWIDE_PATH, encoding="utf-8-sig")
    print(f"  shape: {df_d.shape}  |  시군구 수: {df_d['sigungu'].nunique()}개")

    # danger_score 컬럼이 이미 0~1로 정규화됨
    if "danger_score" in df_d.columns:
        df_danger = df_d[["sigungu", "danger_score"]].copy()
        df_danger = df_danger.rename(columns={"danger_score": "district_danger"})
    else:
        # raw_score -> MinMaxScaler
        from sklearn.preprocessing import MinMaxScaler as _MMS
        _sc = _MMS()
        df_d["district_danger"] = _sc.fit_transform(df_d[["danger_score"]])
        df_danger = df_d[["sigungu", "district_danger"]].copy()

    print(f"  위험도 범위: {df_danger['district_danger'].min():.4f} ~ {df_danger['district_danger'].max():.4f}")
    print(f"  위험도 상위 10 시군구:")
    print(df_danger.nlargest(10, "district_danger")[["sigungu", "district_danger"]].to_string(index=False))

elif os.path.exists(ACCIDENT_PATH):
    # ── 서울 XLSX fallback ─────────────────────────────────────────────────
    print(f"  [fallback] 서울 XLSX 사용: {ACCIDENT_PATH}")
    df_acc = pd.read_excel(ACCIDENT_PATH, engine="openpyxl")
    print(f"  사고다발지 shape: {df_acc.shape}")
    print(f"  컬럼: {list(df_acc.columns)}")
    print(f"  지점 예시:\n{df_acc.iloc[:5, 0].to_string()}\n")

    jijum_col = df_acc.columns[0]

    def extract_sigungu(text):
        text = str(text)
        m = re.search(r"([가-힣]+구)", text)
        if m:
            return m.group(1)
        m = re.search(r"([가-힣]+(시|군))", text)
        if m:
            return m.group(1)
        return None

    df_acc["sigungu"] = df_acc[jijum_col].apply(extract_sigungu)
    fail_cnt = df_acc["sigungu"].isna().sum()
    print(f"  구 이름 추출: 성공={len(df_acc) - fail_cnt}, 실패={fail_cnt}")
    print(f"  추출된 구 목록: {sorted(df_acc['sigungu'].dropna().unique())}\n")

    col_aliases = {
        "발생건수": ["발생건수", "사고건수"],
        "사망자수": ["사망자수", "사망자"],
        "중상자수": ["중상자수", "중상자"],
        "부상자수": ["부상자수", "부상자"],
    }

    def find_col(df, aliases):
        for a in aliases:
            if a in df.columns:
                return a
        return None

    c_occur = find_col(df_acc, col_aliases["발생건수"])
    c_dead  = find_col(df_acc, col_aliases["사망자수"])
    c_heavy = find_col(df_acc, col_aliases["중상자수"])
    c_injur = find_col(df_acc, col_aliases["부상자수"])

    print(f"  사용 컬럼: 발생건수={c_occur}, 사망자수={c_dead}, 중상자수={c_heavy}, 부상자수={c_injur}")

    df_acc["danger_raw"] = 0.0
    if c_occur: df_acc["danger_raw"] += df_acc[c_occur].fillna(0) * 1.0
    if c_dead:  df_acc["danger_raw"] += df_acc[c_dead].fillna(0)  * 5.0
    if c_heavy: df_acc["danger_raw"] += df_acc[c_heavy].fillna(0) * 2.0
    if c_injur: df_acc["danger_raw"] += df_acc[c_injur].fillna(0) * 1.0

    df_tmp = (
        df_acc.dropna(subset=["sigungu"])
        .groupby("sigungu", as_index=False)["danger_raw"]
        .sum()
        .rename(columns={"danger_raw": "danger_score"})
    )
    scaler_d = MinMaxScaler()
    df_tmp["district_danger"] = scaler_d.fit_transform(df_tmp[["danger_score"]])
    df_danger = df_tmp[["sigungu", "district_danger"]].copy()

    print(f"\n  구별 위험도 테이블 ({len(df_danger)}개 구):")
    print(df_tmp.sort_values("district_danger", ascending=False).to_string(index=False))

else:
    print("  [WARN] 사고 데이터 없음 -> district_danger = 0")
    df_danger = pd.DataFrame(columns=["sigungu", "district_danger"])

df_danger.to_csv(DISTRICT_OUT, index=False, encoding="utf-8-sig")
print(f"\n  저장 -> {DISTRICT_OUT}")


# ==============================================================================
# STEP 3: 도로 데이터 + 구별 위험도 병합
# ==============================================================================
print("\n" + "=" * 65)
print("STEP 3: 도로 + 구별 위험도 병합 (sigungu 기준)")
print("=" * 65)

if len(df_danger) > 0:
    danger_map = dict(zip(df_danger["sigungu"], df_danger["district_danger"]))
    df_road["district_danger"] = df_road[sigungu_col].map(danger_map)

    matched = df_road["district_danger"].notna().sum()
    total   = len(df_road)
    print(f"  위험도 매핑 성공: {matched:,}행 / 전체 {total:,}행 ({matched/total*100:.1f}%)")

    # 매핑 안 된 행(경기도 등) = 중앙값으로 대체
    median_danger = df_danger["district_danger"].median()
    df_road["district_danger"] = df_road["district_danger"].fillna(median_danger)
    print(f"  미매핑 행 -> district_danger 중앙값 대체 ({median_danger:.4f})")
else:
    df_road["district_danger"] = 0.0
    print("  사고 데이터 없음 -> district_danger = 0 (전체)")


# ==============================================================================
# STEP 4: 피처 생성 + safety_index_v2 계산
# ==============================================================================
print("\n" + "=" * 65)
print("STEP 4: 피처 생성 + safety_index_v2 계산")
print("=" * 65)

# 학습 대상: width_m, length_km 결측 없는 행
df_model = df_road.dropna(subset=["width_m", "length_km"]).copy()
print(f"  width_m/length_km 유효 행: {len(df_model):,}행 (전체 {len(df_road):,}행 중)")

# 도로 속성 정규화 (width × 0.7 + length × 0.3)
scaler_road = MinMaxScaler()
road_norm = scaler_road.fit_transform(df_model[["width_m", "length_km"]])
df_model["road_attr_score"] = road_norm[:, 0] * 0.7 + road_norm[:, 1] * 0.3

# 통합 안전지수 v2
# 위험도(district_danger)가 높을수록 unsafe -> (1 - danger)로 반전
df_model["safety_index_v2"] = (
    (1 - df_model["district_danger"]) * 0.6
    + df_model["road_attr_score"] * 0.4
)

print(f"  safety_index_v2 통계:")
print(f"    평균={df_model['safety_index_v2'].mean():.4f}")
print(f"    최소={df_model['safety_index_v2'].min():.4f}")
print(f"    최대={df_model['safety_index_v2'].max():.4f}\n")

# 위험등급 라벨 생성 (삼분위 기준)
# safety_index_v2 높을수록 안전 -> 낮은 구간이 위험
q33 = df_model["safety_index_v2"].quantile(0.33)
q66 = df_model["safety_index_v2"].quantile(0.66)

def assign_danger_level(score):
    if score >= q66:
        return 0   # 안전
    elif score >= q33:
        return 1   # 보통
    else:
        return 2   # 위험

df_model["danger_level"] = df_model["safety_index_v2"].apply(assign_danger_level)
level_dist = df_model["danger_level"].value_counts().sort_index()
print(f"  위험등급 분포 (삼분위 기준: q33={q33:.4f}, q66={q66:.4f}):")
for lvl, cnt in level_dist.items():
    label = {0: "안전", 1: "보통", 2: "위험"}[lvl]
    print(f"    {lvl}({label}): {cnt:,}행 ({cnt/len(df_model)*100:.1f}%)")


# ==============================================================================
# STEP 5 & 6: 피처 / 타겟 준비 + 학습
# ==============================================================================
print("\n" + "=" * 65)
print("STEP 5+6: 모델 학습 (회귀 + 분류)")
print("=" * 65)

# 피처 선택
FEATURES = ["width_m", "length_km", "district_danger", "road_attr_score"]

# 위경도 있으면 추가 (공간 정보 보강)
if "start_lat" in df_model.columns and df_model["start_lat"].notna().sum() > 100:
    df_model_coord = df_model.dropna(subset=["start_lat", "start_lon"])
    FEATURES_WITH_COORD = FEATURES + ["start_lat", "start_lon"]
    print(f"  좌표 포함 행: {len(df_model_coord):,}행 -> 좌표 포함 피처로 학습")
    X_df = df_model_coord[FEATURES_WITH_COORD].copy()
    y_reg = df_model_coord["safety_index_v2"].copy()
    y_cls = df_model_coord["danger_level"].copy()
    FINAL_FEATURES = FEATURES_WITH_COORD
else:
    X_df = df_model[FEATURES].copy()
    y_reg = df_model["safety_index_v2"].copy()
    y_cls = df_model["danger_level"].copy()
    FINAL_FEATURES = FEATURES
    print(f"  좌표 없음 -> 기본 피처만 사용")

print(f"  최종 피처: {FINAL_FEATURES}")
print(f"  학습 대상: {len(X_df):,}행\n")

if len(X_df) < 30:
    print("  [ERR] 학습 데이터 부족 (30행 미만). 데이터 확인 후 재실행하세요.")
    sys.exit(1)

# 피처 스케일링
scaler_feat = MinMaxScaler()
X_scaled = scaler_feat.fit_transform(X_df)

X_tr, X_te, y_reg_tr, y_reg_te, y_cls_tr, y_cls_te = train_test_split(
    X_scaled, y_reg, y_cls,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_cls,     # 분류 비율 유지
)

print(f"  학습: {len(X_tr):,}행 / 테스트: {len(X_te):,}행")


# ── 회귀 모델 ────────────────────────────────────────────────────────────────
print("\n  [ 회귀: RandomForestRegressor ]")
rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_reg.fit(X_tr, y_reg_tr)
y_pred_reg = rf_reg.predict(X_te)

r2  = r2_score(y_reg_te, y_pred_reg)
mse = mean_squared_error(y_reg_te, y_pred_reg)
print(f"  R²  = {r2:.4f}")
print(f"  MSE = {mse:.6f}")

print("\n  피처 중요도 (회귀):")
for feat, imp in sorted(zip(FINAL_FEATURES, rf_reg.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat:<22}: {imp:.4f}")


# ── 분류 모델 ────────────────────────────────────────────────────────────────
print("\n  [ 분류: RandomForestClassifier ]")
rf_cls = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=3,
    class_weight="balanced",   # 클래스 불균형 대응
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_cls.fit(X_tr, y_cls_tr)
y_pred_cls = rf_cls.predict(X_te)

f1 = f1_score(y_cls_te, y_pred_cls, average="macro")
print(f"  F1-macro = {f1:.4f}")
print("\n  분류 리포트:")
print(classification_report(y_cls_te, y_pred_cls, target_names=["안전(0)", "보통(1)", "위험(2)"]))

print("\n  피처 중요도 (분류):")
for feat, imp in sorted(zip(FINAL_FEATURES, rf_cls.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat:<22}: {imp:.4f}")


# ==============================================================================
# STEP 7: 추론 인터페이스 확인 (샘플 테스트)
# ==============================================================================
print("\n" + "=" * 65)
print("STEP 7: 샘플 추론 테스트")
print("=" * 65)

# 추론 시 입력 예시
sample_input = {
    "width_m"         : 3.0,    # 너비 3m
    "length_km"       : 5.0,    # 길이 5km
    "district_danger" : 0.3,    # 구 위험도 30%
    "road_attr_score" : 0.6,    # 도로 속성 60점
}
# 위경도 포함 피처인 경우 추가
if "start_lat" in FINAL_FEATURES:
    sample_input["start_lat"] = 37.55
    sample_input["start_lon"] = 127.05

sample_df  = pd.DataFrame([[sample_input[f] for f in FINAL_FEATURES]], columns=FINAL_FEATURES)
sample_scaled = scaler_feat.transform(sample_df)

pred_score = rf_reg.predict(sample_scaled)[0]
pred_level = rf_cls.predict(sample_scaled)[0]
level_label = {0: "안전", 1: "보통", 2: "위험"}[pred_level]

print(f"  입력: width_m=3.0, length_km=5.0, district_danger=0.3")
print(f"  -> safety_score (회귀): {pred_score:.4f}")
print(f"  -> 위험등급  (분류): {pred_level} ({level_label})")


# ==============================================================================
# STEP 8: 모델 저장
# ==============================================================================
print("\n" + "=" * 65)
print("STEP 8: 모델 저장")
print("=" * 65)

joblib.dump(rf_reg,      REGRESSOR_PATH)
joblib.dump(rf_cls,      CLASSIFIER_PATH)
joblib.dump(scaler_feat, SCALER_PATH)

# 추론 시 필요한 메타 정보도 함께 저장
meta = {
    "features"    : FINAL_FEATURES,
    "danger_level": {0: "안전", 1: "보통", 2: "위험"},
    "q33"         : q33,
    "q66"         : q66,
    "r2_regressor": round(r2, 4),
    "f1_classifier": round(f1, 4),
}
joblib.dump(meta, os.path.join(MODELS_DIR, "safety_meta.pkl"))

print(f"  [OK] safety_regressor.pkl  -> {REGRESSOR_PATH}")
print(f"  [OK] safety_classifier.pkl -> {CLASSIFIER_PATH}")
print(f"  [OK] safety_scaler.pkl     -> {SCALER_PATH}")
print(f"  [OK] safety_meta.pkl       -> {os.path.join(MODELS_DIR, 'safety_meta.pkl')}")
print(f"  [OK] district_danger.csv   -> {DISTRICT_OUT}")

print("\n" + "=" * 65)
print("[OK] build_safety_model.py 완료")
print(f"   회귀 R²      : {r2:.4f}")
print(f"   분류 F1-macro: {f1:.4f}")
print("=" * 65)
