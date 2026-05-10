"""
visualize_safety_model.py
==========================
SafetyRegressor / SafetyClassifier 시각화 보고서 생성

[ 출력 파일 ]
  report/figures/safety_regressor_pred_vs_actual.png
  report/figures/safety_regressor_feature_importance.png
  report/figures/safety_classifier_confusion_matrix.png
  report/figures/safety_classifier_feature_importance.png
  report/tables/safety_performance_summary.json
  report/tables/safety_classifier_report.csv

[ 실행 ]
  python src/ml/visualize_safety_model.py
"""

import json
import os
import sys
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 ────────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))

RAW_DIR     = os.path.join(BASE_DIR, "data", "raw_ml")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "report", "figures")
TABLES_DIR  = os.path.join(BASE_DIR, "report", "tables")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

ROAD_PATH     = os.path.join(RAW_DIR, "road_clean_nationwide.csv")
DISTRICT_PATH = os.path.join(RAW_DIR, "district_danger_nationwide.csv")

RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# 1. 모델 및 메타 로드
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: 모델 로드")
print("=" * 65)

for path, label in [
    (os.path.join(MODELS_DIR, "safety_regressor.pkl"),  "safety_regressor.pkl"),
    (os.path.join(MODELS_DIR, "safety_classifier.pkl"), "safety_classifier.pkl"),
    (os.path.join(MODELS_DIR, "safety_scaler.pkl"),     "safety_scaler.pkl"),
    (os.path.join(MODELS_DIR, "safety_meta.pkl"),       "safety_meta.pkl"),
]:
    if not os.path.exists(path):
        print(f"  [ERR] {label} 없음 -> python src/ml/build_safety_model.py 먼저 실행")
        sys.exit(1)

rf_reg   = joblib.load(os.path.join(MODELS_DIR, "safety_regressor.pkl"))
rf_cls   = joblib.load(os.path.join(MODELS_DIR, "safety_classifier.pkl"))
scaler   = joblib.load(os.path.join(MODELS_DIR, "safety_scaler.pkl"))
meta     = joblib.load(os.path.join(MODELS_DIR, "safety_meta.pkl"))

FEATURES     = meta["features"]
q33          = meta["q33"]
q66          = meta["q66"]
LEVEL_NAMES  = ["안전(0)", "보통(1)", "위험(2)"]

print(f"  피처: {FEATURES}")
print(f"  q33={q33:.4f}, q66={q66:.4f}")
print(f"  저장된 R²={meta['r2_regressor']}, F1={meta['f1_classifier']}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 테스트 데이터 재구성 (build_safety_model.py 와 동일한 로직)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 2: 테스트 데이터 재구성")
print("=" * 65)

if not os.path.exists(ROAD_PATH):
    print(f"  [ERR] {ROAD_PATH} 없음")
    sys.exit(1)

df_road = pd.read_csv(ROAD_PATH, encoding="utf-8-sig")
sigungu_col = "sigungu" if "sigungu" in df_road.columns else "시군구명"
if "시군구명" in df_road.columns:
    df_road = df_road.rename(columns={"시군구명": "sigungu"})
    sigungu_col = "sigungu"

for col in ["width_m", "length_km", "start_lat", "start_lon"]:
    if col in df_road.columns:
        df_road[col] = pd.to_numeric(df_road[col], errors="coerce")

# 사고 위험도 병합
df_d = pd.read_csv(DISTRICT_PATH, encoding="utf-8-sig")
danger_map = dict(zip(df_d["sigungu"], df_d["danger_score"]))
df_road["district_danger"] = df_road[sigungu_col].map(danger_map)
median_danger = df_d["danger_score"].median()
df_road["district_danger"] = df_road["district_danger"].fillna(median_danger)

# road_attr_score 재계산
from sklearn.preprocessing import MinMaxScaler

df_model = df_road.dropna(subset=["width_m", "length_km"]).copy()
scaler_road = MinMaxScaler()
road_norm = scaler_road.fit_transform(df_model[["width_m", "length_km"]])
df_model["road_attr_score"] = road_norm[:, 0] * 0.7 + road_norm[:, 1] * 0.3
df_model["safety_index_v2"] = (
    (1 - df_model["district_danger"]) * 0.6 + df_model["road_attr_score"] * 0.4
)

def assign_danger_level(score):
    if score >= q66:
        return 0
    elif score >= q33:
        return 1
    else:
        return 2

df_model["danger_level"] = df_model["safety_index_v2"].apply(assign_danger_level)

# 피처 선택 (모델 학습 당시와 동일)
if "start_lat" in FEATURES:
    df_model = df_model.dropna(subset=["start_lat", "start_lon"])

X_df  = df_model[FEATURES].copy()
y_reg = df_model["safety_index_v2"].copy()
y_cls = df_model["danger_level"].copy()

X_scaled = scaler.transform(X_df)

_, X_te, _, y_reg_te, _, y_cls_te = train_test_split(
    X_scaled, y_reg, y_cls,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_cls,
)

y_pred_reg = rf_reg.predict(X_te)
y_pred_cls = rf_cls.predict(X_te)

r2  = r2_score(y_reg_te, y_pred_reg)
mse = mean_squared_error(y_reg_te, y_pred_reg)
f1_report = classification_report(y_cls_te, y_pred_cls,
                                   target_names=LEVEL_NAMES, output_dict=True)
print(f"  테스트 R²={r2:.4f}, MSE={mse:.6f}")
print(f"  테스트 F1-macro={f1_report['macro avg']['f1-score']:.4f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 시각화
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 3: 시각화 저장")
print("=" * 65)

# ── 3-1. 회귀: 예측 vs 실제 scatter plot ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_reg_te, y_pred_reg, alpha=0.3, s=10, color="#2196F3", label="테스트 데이터")
lim_min = min(y_reg_te.min(), y_pred_reg.min()) - 0.02
lim_max = max(y_reg_te.max(), y_pred_reg.max()) + 0.02
ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.5, label="완벽 예측선")
ax.set_xlabel("실제 safety_index_v2", fontsize=12)
ax.set_ylabel("예측 safety_index_v2", fontsize=12)
ax.set_title(f"SafetyRegressor: 예측 vs 실제\nR²={r2:.4f}  MSE={mse:.6f}", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
path_scatter = os.path.join(FIGURES_DIR, "safety_regressor_pred_vs_actual.png")
fig.tight_layout()
fig.savefig(path_scatter, dpi=150)
plt.close(fig)
print(f"  [OK] {path_scatter}")

# ── 3-2. 회귀: 피처 중요도 ──────────────────────────────────────────────────
imp_reg = rf_reg.feature_importances_
sorted_idx = np.argsort(imp_reg)
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(
    [FEATURES[i] for i in sorted_idx],
    [imp_reg[i] for i in sorted_idx],
    color="#4CAF50",
)
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_title("SafetyRegressor 피처 중요도 (RandomForest)", fontsize=13)
ax.grid(axis="x", alpha=0.3)
path_imp_reg = os.path.join(FIGURES_DIR, "safety_regressor_feature_importance.png")
fig.tight_layout()
fig.savefig(path_imp_reg, dpi=150)
plt.close(fig)
print(f"  [OK] {path_imp_reg}")

# ── 3-3. 분류: Confusion Matrix ──────────────────────────────────────────────
cm = confusion_matrix(y_cls_te, y_pred_cls)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LEVEL_NAMES)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("SafetyClassifier Confusion Matrix", fontsize=13)
f1_macro = f1_report["macro avg"]["f1-score"]
ax.set_xlabel(f"예측값  (F1-macro={f1_macro:.4f})", fontsize=11)
path_cm = os.path.join(FIGURES_DIR, "safety_classifier_confusion_matrix.png")
fig.tight_layout()
fig.savefig(path_cm, dpi=150)
plt.close(fig)
print(f"  [OK] {path_cm}")

# ── 3-4. 분류: 피처 중요도 ──────────────────────────────────────────────────
imp_cls = rf_cls.feature_importances_
sorted_idx_c = np.argsort(imp_cls)
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(
    [FEATURES[i] for i in sorted_idx_c],
    [imp_cls[i] for i in sorted_idx_c],
    color="#FF9800",
)
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_title("SafetyClassifier 피처 중요도 (RandomForest)", fontsize=13)
ax.grid(axis="x", alpha=0.3)
path_imp_cls = os.path.join(FIGURES_DIR, "safety_classifier_feature_importance.png")
fig.tight_layout()
fig.savefig(path_imp_cls, dpi=150)
plt.close(fig)
print(f"  [OK] {path_imp_cls}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 테이블 저장
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: 테이블 저장")
print("=" * 65)

# 성능 요약 JSON
summary = {
    "model": "SafetyModel (RandomForest)",
    "data_scope": "전국 20,262행 (도로) + 381 시군구 (사고)",
    "mapping_rate": "89.1%",
    "features": FEATURES,
    "regressor": {
        "algorithm": "RandomForestRegressor",
        "r2": round(r2, 4),
        "mse": round(mse, 6),
    },
    "classifier": {
        "algorithm": "RandomForestClassifier",
        "f1_macro": round(f1_report["macro avg"]["f1-score"], 4),
        "accuracy": round(f1_report["accuracy"], 4),
        "classes": LEVEL_NAMES,
    },
    "feature_importance_regressor": {
        feat: round(float(imp), 4) for feat, imp in zip(FEATURES, imp_reg)
    },
    "feature_importance_classifier": {
        feat: round(float(imp), 4) for feat, imp in zip(FEATURES, imp_cls)
    },
}
path_json = os.path.join(TABLES_DIR, "safety_performance_summary.json")
with open(path_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"  [OK] {path_json}")

# 분류 리포트 CSV
df_report = pd.DataFrame(f1_report).T.reset_index().rename(columns={"index": "class"})
path_report = os.path.join(TABLES_DIR, "safety_classifier_report.csv")
df_report.to_csv(path_report, index=False, encoding="utf-8-sig")
print(f"  [OK] {path_report}")


# ══════════════════════════════════════════════════════════════════════════════
# 완료
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("[OK] visualize_safety_model.py 완료")
print(f"  회귀  R²      : {r2:.4f}")
print(f"  분류  F1-macro: {f1_report['macro avg']['f1-score']:.4f}")
print("\n  저장된 파일:")
for p in [path_scatter, path_imp_reg, path_cm, path_imp_cls, path_json, path_report]:
    print(f"    {p}")
print("=" * 65)
