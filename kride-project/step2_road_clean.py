# =============================================================
# Step 2: 자전거도로 데이터 - 서울 + 경기도 필터링 + 정제 CSV 생성
# =============================================================
# 입력 : data/raw_ml/전국자전거도로표준데이터.csv  (EUC-KR, 20,262행, 23컬럼)
# 출력 : data/raw_ml/road_clean.csv
#         - 서울특별시(380행) + 경기도(4,939행) = 5,319행
#         - 원본 컬럼 7개 + 파생 피처 2개 = 총 9컬럼
#
# 시도별 전체 분포 (2026-03-31 실측):
#   경상북도  9,066 (44.7%)  ← 낙동강 자전거길 등 농촌 전용도로 구간 많음
#   경기도    4,939 (24.4%)  ← 한강·임진강 자전거길
#   서울특별시  380  (1.9%)  ← 도시형 차선 위주, 전용도로 공식 등록 적음
#   → 서울+경기 합산: 5,319행 (26.3%) → ML 학습에 충분
#
# ⚠️ 주의: df_clean 대신 df_road 변수명 사용 (notebook 변수 충돌 방지)
# =============================================================

import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# ── 경로 설정 (Jupyter 노트북 호환) ──────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

INPUT_PATH  = os.path.join(BASE_DIR, "data", "raw_ml", "전국자전거도로표준데이터.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw_ml", "road_clean.csv")
print(f"BASE_DIR : {BASE_DIR}\n")

# =============================================================
# STEP ①: 필요한 컬럼만 로드 (usecols → 메모리 절약)
# =============================================================
print("▶ 원본 CSV 로드 중...")

LOAD_COLS = [
    "시도명",            # 필터링용 (서울+경기 후 제거)
    "기점위도",
    "기점경도",
    "종점위도",
    "종점경도",
    "자전거도로너비(m)",
    "총길이(km)",
    "자전거도로종류",    # Categorical 피처 (결측 3.2%)
    "자전거도로고시유무", # Boolean 피처 (결측 2.4%)
]

df = pd.read_csv(INPUT_PATH, encoding="cp949", usecols=LOAD_COLS)
print(f"  전체 shape : {df.shape}  ({df.shape[0]:,}행 × {df.shape[1]}컬럼)")

# 시도명 공백 제거 (안전하게)
df["시도명"] = df["시도명"].str.strip()
print(f"\n  전체 시도별 분포:\n{df['시도명'].value_counts().to_string()}\n")

# =============================================================
# STEP ②: 서울 + 경기도 필터링
# =============================================================
# isin() : SQL의 WHERE 시도명 IN ('서울특별시', '경기도') 와 동일
TARGET_SIDO = ["서울특별시", "경기도"]
df_road = df[df["시도명"].isin(TARGET_SIDO)].copy()
# ⚠️ df_road 사용 (df_clean 이름 충돌 방지)

print(f"▶ 필터링 결과")
print(f"  전체 {len(df):,}행  →  서울+경기 {len(df_road):,}행")
print(f"  시도별 행 수:\n{df_road['시도명'].value_counts()}\n")

# =============================================================
# STEP ③: 수치 컬럼 타입 변환 (문자열 → float)
# =============================================================
# CSV 로드 시 일부 수치 컬럼이 object로 읽히는 경우 방지
NUMERIC_COLS = ["기점위도", "기점경도", "종점위도", "종점경도",
                "자전거도로너비(m)", "총길이(km)"]

for col in NUMERIC_COLS:
    df_road[col] = pd.to_numeric(df_road[col], errors="coerce")

print(f"  수치 컬럼 변환 후 결측:\n{df_road[NUMERIC_COLS].isnull().sum()}\n")

# =============================================================
# STEP ④: 파생 피처 생성
# =============================================================
print("▶ 파생 피처 생성 중...")

# --- A: is_wide_road (Binary) ---
# 국토부 기준: 2.0m 이상 = 양방향 통행 가능 → 충돌 위험 낮음
df_road["is_wide_road"] = (df_road["자전거도로너비(m)"] >= 2.0).astype(int)

wide = df_road["is_wide_road"].sum()
print(f"  is_wide_road  : 넓음(1)={wide:,}개 ({wide/len(df_road)*100:.1f}%)  "
      f"좁음(0)={len(df_road)-wide:,}개")

# --- B: safety_index (MinMaxScaler 정규화 후 가중합) ---
# 너비(0.7) + 길이(0.3) → 0~1 안전지수
scaler = MinMaxScaler()
valid_mask = df_road[["자전거도로너비(m)", "총길이(km)"]].notna().all(axis=1)

normalized = scaler.fit_transform(
    df_road.loc[valid_mask, ["자전거도로너비(m)", "총길이(km)"]]
)
df_road.loc[valid_mask, "safety_index"] = (
    normalized[:, 0] * 0.7 + normalized[:, 1] * 0.3
).round(4)

print(f"  safety_index  : 평균={df_road['safety_index'].mean():.3f}  "
      f"최소={df_road['safety_index'].min():.3f}  "
      f"최대={df_road['safety_index'].max():.3f}\n")

# =============================================================
# STEP ⑤: 컬럼 정리
# =============================================================
# 필터링 끝난 시도명 제거 후 영문 컬럼명으로 통일
df_road = df_road.drop(columns=["시도명"])
df_road = df_road.rename(columns={
    "기점위도"           : "start_lat",
    "기점경도"           : "start_lon",
    "종점위도"           : "end_lat",
    "종점경도"           : "end_lon",
    "자전거도로너비(m)"  : "width_m",
    "총길이(km)"         : "length_km",
    "자전거도로종류"     : "road_type",
    "자전거도로고시유무" : "is_official",
})

# =============================================================
# STEP ⑥: 최종 요약 출력
# =============================================================
print("▶ 최종 정제 결과")
print(f"  shape  : {df_road.shape}")
print(f"  컬럼   : {list(df_road.columns)}\n")

print("데이터 타입:")
print(df_road.dtypes)
print()

print("결측값 현황:")
print(df_road.isnull().sum())
print()

print("기술 통계 (수치 컬럼):")
print(df_road[["width_m", "length_km", "safety_index"]].describe().round(3))
print()

print("road_type 분포:")
print(df_road["road_type"].value_counts())
print()

print("샘플 (상위 5행):")
print(df_road.head(5).to_string())
print()

# =============================================================
# STEP ⑦: CSV 저장
# =============================================================
df_road.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
# index=False  → 행 번호(0,1,2...) 파일에 기록하지 않음
# utf-8-sig    → Excel 한글 깨짐 방지

print(f"✅ 저장 완료 → {OUTPUT_PATH}")
print(f"   파일 크기 : {os.path.getsize(OUTPUT_PATH):,} bytes")
