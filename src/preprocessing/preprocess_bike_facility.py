"""
preprocess_bike_facility.py
============================
자전거보관소정보.csv → bike_facility_nationwide.csv

[ 입력 ]
  kride-project/자전거보관소정보.csv  (cp949, 18,417행, WGS84 좌표 포함)

[ 출력 ]
  data/raw_ml/bike_facility_nationwide.csv

[ 실행 ]
  python src/preprocessing/preprocess_bike_facility.py
"""

import json
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))

IN_PATH  = os.path.join(BASE_DIR, "자전거보관소정보.csv")
OUT_DIR  = os.path.join(BASE_DIR, "data", "raw_ml")
OUT_PATH = os.path.join(OUT_DIR, "bike_facility_nationwide.csv")
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 원본 로드
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: 자전거보관소정보.csv 로드")
print("=" * 65)

if not os.path.exists(IN_PATH):
    print(f"[ERR] 파일 없음: {IN_PATH}")
    sys.exit(1)

df = pd.read_csv(IN_PATH, encoding="cp949", low_memory=False)
print(f"  원본 행수: {len(df):,}행")
print(f"  컬럼: {df.columns.tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 좌표 정리 + 유효 범위 필터
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: 좌표 정리 + 유효 범위 필터")
print("=" * 65)

df = df.rename(columns={"WGS84위도": "lat", "WGS84경도": "lon"})
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

before = len(df)
df = df.dropna(subset=["lat", "lon"])
df = df[(df["lat"] > 33) & (df["lat"] < 39)]
df = df[(df["lon"] > 124) & (df["lon"] < 132)]
print(f"  좌표 결측/범위 제거: {before:,} → {len(df):,}행")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 시도 추출 (도로명주소 첫 토큰)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3: 시도 추출")
print("=" * 65)

df["sido"] = (df["소재지도로명주소"]
              .fillna("")
              .str.split()
              .str[0]
              .replace("", pd.NA))
print("  시도 분포:")
print(df["sido"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 4. 출력 스키마 통일
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: 출력 스키마 통일")
print("=" * 65)

def _clean(v):
    """pandas NaN/NA → None (JSON null)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v

def build_raw(row):
    return json.dumps({
        "보관대수":            _clean(row.get("보관대수")),
        "설치형태":            _clean(row.get("설치형태")),
        "설치연도":            _clean(row.get("설치연도")),
        "차양막설치여부":      _clean(row.get("차양막설치여부")),
        "공기주입기비치여부":  _clean(row.get("공기주입기비치여부")),
        "공기주입기유형명":    _clean(row.get("공기주입기유형명")),
        "수리대설치여부":      _clean(row.get("수리대설치여부")),
        "관리기관명":          _clean(row.get("관리기관명")),
        "관리기관전화번호":    _clean(row.get("관리기관전화번호")),
    }, ensure_ascii=False)

df["name"]         = df["자전거보관소명"].fillna("자전거보관소")
df["category"]     = "facility"
df["sub_category"] = "bike_storage"
df["address"]      = df["소재지도로명주소"].fillna("")
df["source"]       = "public"
df["raw_data"]     = df.apply(build_raw, axis=1)

out = df[["name", "category", "sub_category", "sido",
          "address", "lat", "lon", "source", "raw_data"]].copy()
print(f"  출력 행수: {len(out):,}행")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 저장
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5: 저장")
print("=" * 65)

out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"  [OK] {OUT_PATH}")
print(f"  총 {len(out):,}행 / 시도 {out['sido'].nunique()}개")

print("\n" + "=" * 65)
print("[DONE] preprocess_bike_facility.py 완료")
print("=" * 65)
