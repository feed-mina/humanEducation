# =============================================================
# Step 1: 편의시설 데이터 - 유효 컬럼 6개 추출 후 정제 CSV 저장
# =============================================================
# 입력 : data/raw_ml/서울시 자전거 편의시설.csv  (인코딩: EUC-KR, 43컬럼)
# 출력 : data/raw_ml/facility_clean.csv           (6 피처 컬럼, 행 인덱스 없음)
#
# 추출 컬럼:
#   [원본 직접 사용]
#     1. x 좌표            - 편의시설 x좌표 (Spatial Join 기준)
#     2. y 좌표            - 편의시설 y좌표 (Spatial Join 기준)
#     3. 거리              - 자전거도로까지 거리 피처
#     4. 상세정보 값 4     - 설치유형 (Categorical 피처)
#   [파생 생성]
#     5. is_24h            - 24시간 운영 여부 (Boolean)
#     6. has_restricted_hours - 운영 제한 여부 (Boolean)
# =============================================================

import pandas as pd
import os

# ── 경로 설정 ──────────────────────────────────────────────
# __file__ 은 .py 직접 실행 시에만 존재 → Jupyter에서는 os.getcwd() 사용
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()  # Jupyter 노트북 실행 시: 현재 작업 디렉터리

INPUT_PATH  = os.path.join(BASE_DIR, "data", "raw_ml", "서울시 자전거 편의시설.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw_ml", "facility_clean.csv")
print(f"  BASE_DIR : {BASE_DIR}")

# ── 1. 원본 CSV 로드 ───────────────────────────────────────
print("▶ 원본 CSV 로드 중...")
df = pd.read_csv(INPUT_PATH, encoding="cp949")

print(f"  원본 shape : {df.shape}  ({df.shape[0]}행 × {df.shape[1]}컬럼)")
print(f"  컬럼 목록  : {list(df.columns)}\n")

# ── 2. 직접 사용할 원본 컬럼 4개 선택 ─────────────────────
KEEP_COLS = ["x 좌표", "y 좌표", "거리", "상세정보 값 4"]
df_clean = df[KEEP_COLS].copy()

# ── 3. 파생 피처 생성: is_24h ──────────────────────────────
# 상세정보 값 1~5 중 어디에든 "24시간"이라는 텍스트가 있으면 True
detail_value_cols = ["상세정보 값 1", "상세정보 값 2",
                     "상세정보 값 3", "상세정보 값 4", "상세정보 값 5"]

# 원본에 해당 컬럼이 실제로 존재하는지 확인 후 처리
available_detail = [c for c in detail_value_cols if c in df.columns]

def check_24h(row):
    """행의 상세정보 값 컬럼들 중 '24시간' 포함 여부 반환"""
    for col in available_detail:
        val = str(row[col]) if pd.notna(row[col]) else ""
        if "24시간" in val or "24H" in val.upper() or "24h" in val:
            return True
    return False

def check_restricted(row):
    """운영 제한 키워드(예약, 제한, 이용불가 등) 포함 여부 반환"""
    keywords = ["예약", "제한", "이용불가", "운영중단", "폐쇄", "임시"]
    for col in available_detail:
        val = str(row[col]) if pd.notna(row[col]) else ""
        for kw in keywords:
            if kw in val:
                return True
    return False

print("▶ 파생 피처 생성 중 (is_24h, has_restricted_hours)...")
df_clean["is_24h"] = df[available_detail].apply(check_24h, axis=1)
df_clean["has_restricted_hours"] = df[available_detail].apply(check_restricted, axis=1)

# ── 4. 컬럼 이름 정리 (일관된 네이밍) ─────────────────────
df_clean = df_clean.rename(columns={
    "x 좌표"       : "x",
    "y 좌표"       : "y",
    "거리"         : "distance",
    "상세정보 값 4" : "install_type",
})

# ── 5. 기본 통계 확인 ──────────────────────────────────────
print("\n▶ 정제 결과 요약")
print(f"  최종 shape : {df_clean.shape}")
print(f"  컬럼 목록  : {list(df_clean.columns)}")
print()
print(df_clean.dtypes)
print()
print("결측값 현황:")
print(df_clean.isnull().sum())
print()
print("샘플 (상위 3행):")
print(df_clean.head(3))
print()

# is_24h / has_restricted_hours 분포 출력
print("is_24h 분포:")
print(df_clean["is_24h"].value_counts())
print()
print("has_restricted_hours 분포:")
print(df_clean["has_restricted_hours"].value_counts())
print()

# ── 6. 정제 CSV 저장 (index=False → 행 인덱스 미포함) ─────
df_clean.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
# index=False  : 0,1,2... 행 번호를 파일에 쓰지 않음
# encoding utf-8-sig : Excel에서 한글 깨짐 방지 BOM 포함

print(f"✅ 저장 완료 → {OUTPUT_PATH}")
print(f"   파일 크기 : {os.path.getsize(OUTPUT_PATH):,} bytes")
