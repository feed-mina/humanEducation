"""
preprocess_soho_food.py
=======================
소상공인 + 공공 음식점 데이터 → food_poi_nationwide.csv

[ 입력 ]
  소상공인시장진흥공단_상가(상권)정보_20251231/  (17개 시도 CSV, utf-8-sig)  → soho
  모범음식점정보.csv                              (cp949)                    → is_exemplary 플래그
  식품_관광식당.csv                               (euc-kr)                   → is_tourist_certified 플래그
  식품_일반음식점.csv                             (cp949, 2,271,520행)        → localdata, TM→WGS84
  식품_휴게음식점.csv                             (cp949,   633,423행)        → localdata, TM→WGS84
  식품_관광유흥음식점업.csv                       (cp949,        23행)        → localdata, is_tourist_certified=True

[ 출력 ]
  data/raw_ml/food_poi_nationwide.csv

[ 실행 ]
  python src/preprocessing/preprocess_soho_food.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))

SOHO_DIR             = os.path.join(BASE_DIR, "소상공인시장진흥공단_상가(상권)정보_20251231")
EXEMPLARY_PATH       = os.path.join(BASE_DIR, "모범음식점정보.csv")
TOURIST_PATH         = os.path.join(BASE_DIR, "식품_관광식당.csv")
FOOD_GENERAL_PATH    = os.path.join(BASE_DIR, "식품_일반음식점.csv")
FOOD_SNACK_PATH      = os.path.join(BASE_DIR, "식품_휴게음식점.csv")
FOOD_TOURIST_NIGHT_PATH = os.path.join(BASE_DIR, "식품_관광유흥음식점업.csv")
OUT_DIR              = os.path.join(BASE_DIR, "data", "raw_ml")
OUT_PATH             = os.path.join(OUT_DIR, "food_poi_nationwide.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 필터 기준 ─────────────────────────────────────────────────────────────────
FOOD_MAIN  = "음식"                               # 상권업종대분류명
CAFE_WORDS = ["카페", "제과", "커피", "베이커리"]  # 상권업종중분류명 포함 단어

# ── TM → WGS84 변환 ──────────────────────────────────────────────────────────
def _build_tm_transformer():
    """pyproj TM(EPSG:5174/5186) → WGS84 Transformer. 없으면 None."""
    try:
        from pyproj import Transformer
        ref_x, ref_y = 200820.411, 452168.573  # 서울 종로구 기준점
        for epsg in ["EPSG:5174", "EPSG:5186", "EPSG:2097"]:
            t = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
            lon, lat = t.transform(ref_x, ref_y)
            if 126.0 < lon < 128.5 and 37.0 < lat < 38.5:
                print(f"  [TM→WGS84] 사용 좌표계: {epsg}")
                return t
        return Transformer.from_crs("EPSG:5174", "EPSG:4326", always_xy=True)
    except ImportError:
        print("  [WARN] pyproj 미설치 — pip install pyproj 후 재시도")
        return None


def _tm_chunk_to_wgs84(transformer, chunk: pd.DataFrame) -> pd.DataFrame:
    """청크 단위로 TM 좌표 → WGS84 변환 후 유효 행만 반환 (벡터화)."""
    if transformer is None:
        return chunk.head(0)

    chunk = chunk.copy()
    xs = pd.to_numeric(chunk["좌표정보(X)"], errors="coerce")
    ys = pd.to_numeric(chunk["좌표정보(Y)"], errors="coerce")
    valid = xs.notna() & ys.notna() & (xs != 0) & (ys != 0)

    chunk["lon"] = np.nan
    chunk["lat"] = np.nan

    if valid.sum() > 0:
        lons, lats = transformer.transform(xs[valid].values, ys[valid].values)
        in_korea = (lons > 124) & (lons < 132) & (lats > 33) & (lats < 39)
        idx = xs[valid].index[in_korea]
        chunk.loc[idx, "lon"] = lons[in_korea].round(7)
        chunk.loc[idx, "lat"] = lats[in_korea].round(7)

    return chunk.dropna(subset=["lon", "lat"])


def load_localdata_food_csv(
    path: str,
    sub_cat_default: str,
    enc: str = "cp949",
    tourist_cert: bool = False,
    chunksize: int = 100_000,
) -> pd.DataFrame | None:
    """
    공공데이터포털 localdata 음식점 CSV (좌표정보(X/Y) = TM 좌표계) 로드.
    영업중 필터 + TM→WGS84 변환 후 표준 컬럼 반환.
    """
    if not os.path.exists(path):
        print(f"  [SKIP] 파일 없음: {path}")
        return None

    transformer = _build_tm_transformer()
    load_cols = ["사업장명", "영업상태명", "도로명주소",
                 "좌표정보(X)", "좌표정보(Y)", "업태구분명"]

    result_chunks: list[pd.DataFrame] = []
    total_read = 0

    for chunk in pd.read_csv(
        path, encoding=enc, low_memory=False,
        chunksize=chunksize,
        usecols=lambda c: c in load_cols,
    ):
        total_read += len(chunk)
        chunk = chunk[chunk["영업상태명"].fillna("").str.contains("영업")].copy()
        chunk = _tm_chunk_to_wgs84(transformer, chunk)
        if len(chunk) == 0:
            continue

        sub_col = "업태구분명" if "업태구분명" in chunk.columns else None
        chunk["sub_category"] = (
            chunk[sub_col].fillna(sub_cat_default).str[:50]
            if sub_col else sub_cat_default
        )
        chunk["name"]     = chunk["사업장명"].fillna("알수없음").str[:200]
        chunk["category"] = "food"
        chunk["address"]  = chunk.get("도로명주소", pd.Series(dtype=str)).fillna("")
        chunk["sido"]     = chunk["address"].str.split().str[0].replace("", pd.NA)
        chunk["sigungu"]  = chunk["address"].str.split().str[1].replace("", pd.NA)
        chunk["source"]   = "localdata"
        chunk["is_exemplary"]         = False
        chunk["is_tourist_certified"] = tourist_cert

        result_chunks.append(
            chunk[["name", "category", "sub_category", "sido", "sigungu",
                   "address", "lat", "lon", "source",
                   "is_exemplary", "is_tourist_certified"]]
        )

    if not result_chunks:
        print(f"  [WARN] {os.path.basename(path)}: 유효 데이터 없음")
        return None

    df = pd.concat(result_chunks, ignore_index=True)
    print(f"  {os.path.basename(path)}: 전체 {total_read:,}행 → 유효 {len(df):,}행")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. 소상공인 17개 시도 CSV 로드 + 필터
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: 소상공인 상가 데이터 로드 + 음식/카페 필터")
print("=" * 65)

if not os.path.isdir(SOHO_DIR):
    print(f"[ERR] 폴더 없음: {SOHO_DIR}")
    sys.exit(1)

csv_files = sorted([f for f in os.listdir(SOHO_DIR) if f.endswith(".csv")])
print(f"  CSV 파일 {len(csv_files)}개 발견")

chunks = []
for fname in csv_files:
    sido_name = fname.split("_")[-2] if "_" in fname else fname
    fpath = os.path.join(SOHO_DIR, fname)
    try:
        df = pd.read_csv(fpath, encoding="utf-8-sig", low_memory=False,
                         usecols=["상호명", "상권업종대분류명", "상권업종중분류명",
                                  "상권업종소분류명", "시도명", "시군구명",
                                  "도로명주소", "경도", "위도"])
    except Exception as e:
        print(f"  [WARN] {fname} 읽기 실패: {e}")
        continue

    total = len(df)
    # 음식 대분류 또는 카페/제과 중분류
    is_food = df["상권업종대분류명"] == FOOD_MAIN
    is_cafe = df["상권업종중분류명"].fillna("").str.contains("|".join(CAFE_WORDS))
    df = df[is_food | is_cafe].copy()

    # 좌표 결측 제거
    df = df.dropna(subset=["경도", "위도"])
    df = df[(df["경도"] > 124) & (df["경도"] < 132)]  # 한국 범위
    df = df[(df["위도"] > 33) & (df["위도"] < 39)]

    print(f"  {sido_name}: 전체 {total:,}행 → 음식/카페 {len(df):,}행")
    chunks.append(df)

df_food = pd.concat(chunks, ignore_index=True)
print(f"\n  전체 합계: {len(df_food):,}행")


# ══════════════════════════════════════════════════════════════════════════════
# 1-B. localdata 음식점 CSV 추가 (일반음식점 / 휴게음식점 / 관광유흥음식점업)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 1-B: localdata 음식점 CSV 로드 (TM→WGS84 변환)")
print("=" * 65)

extra_dfs: list[pd.DataFrame] = []

df_general = load_localdata_food_csv(
    FOOD_GENERAL_PATH, sub_cat_default="일반음식점", enc="cp949"
)
if df_general is not None:
    extra_dfs.append(df_general)

df_snack = load_localdata_food_csv(
    FOOD_SNACK_PATH, sub_cat_default="휴게음식점", enc="cp949"
)
if df_snack is not None:
    extra_dfs.append(df_snack)

df_tourist_night = load_localdata_food_csv(
    FOOD_TOURIST_NIGHT_PATH, sub_cat_default="관광유흥음식점",
    enc="cp949", tourist_cert=True,
)
if df_tourist_night is not None:
    extra_dfs.append(df_tourist_night)

if extra_dfs:
    df_extra = pd.concat(extra_dfs, ignore_index=True)
    print(f"\n  localdata 합계: {len(df_extra):,}행")
    df_food = pd.concat([df_food, df_extra], ignore_index=True)
    print(f"  soho + localdata 합산: {len(df_food):,}행")
else:
    print("  [INFO] localdata 추가 데이터 없음 (파일 확인 필요)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 컬럼 정리 + 출력 스키마 통일
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: 컬럼 정리")
print("=" * 65)

df_food = df_food.rename(columns={
    "상호명":          "name",
    "상권업종소분류명": "sub_category",
    "시도명":          "sido",
    "시군구명":        "sigungu",
    "도로명주소":      "address",
    "경도":            "lon",
    "위도":            "lat",
})
df_food["category"] = "food"
df_food["source"]   = "soho"
df_food["is_exemplary"]        = False
df_food["is_tourist_certified"] = False

df_food = df_food[["name", "category", "sub_category", "sido", "sigungu",
                    "address", "lat", "lon", "source",
                    "is_exemplary", "is_tourist_certified"]].drop_duplicates()
print(f"  중복 제거 후: {len(df_food):,}행")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 모범음식점 플래그 매칭 (is_exemplary)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3: 모범음식점 플래그 매칭")
print("=" * 65)

if os.path.exists(EXEMPLARY_PATH):
    df_ex = pd.read_csv(EXEMPLARY_PATH, encoding="cp949", low_memory=False,
                        usecols=["업소명", "도로명주소", "영업상태명"])
    df_ex = df_ex[df_ex["영업상태명"].fillna("").str.contains("영업")].copy()
    df_ex["_key"] = (df_ex["업소명"].fillna("").str.strip() + "|" +
                     df_ex["도로명주소"].fillna("").str[:15].str.strip())
    exemplary_keys = set(df_ex["_key"])

    df_food["_key"] = (df_food["name"].fillna("").str.strip() + "|" +
                       df_food["address"].fillna("").str[:15].str.strip())
    df_food["is_exemplary"] = df_food["_key"].isin(exemplary_keys)
    matched = df_food["is_exemplary"].sum()
    print(f"  모범음식점 영업중: {len(df_ex):,}건 → 매칭 성공: {matched:,}건")
else:
    print(f"  [SKIP] {EXEMPLARY_PATH} 없음")

# ══════════════════════════════════════════════════════════════════════════════
# 4. 관광식당 플래그 매칭 (is_tourist_certified)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: 관광식당 플래그 매칭")
print("=" * 65)

if os.path.exists(TOURIST_PATH):
    df_tc = pd.read_csv(TOURIST_PATH, encoding="euc-kr", low_memory=False,
                        usecols=["사업장명", "도로명주소", "영업상태명"])
    df_tc = df_tc[df_tc["영업상태명"].fillna("").str.contains("영업|정상")].copy()
    df_tc["_key"] = (df_tc["사업장명"].fillna("").str.strip() + "|" +
                     df_tc["도로명주소"].fillna("").str[:15].str.strip())
    tourist_keys = set(df_tc["_key"])

    df_food["is_tourist_certified"] = df_food["_key"].isin(tourist_keys)
    matched_t = df_food["is_tourist_certified"].sum()
    print(f"  관광식당 영업중: {len(df_tc):,}건 → 매칭 성공: {matched_t:,}건")
else:
    print(f"  [SKIP] {TOURIST_PATH} 없음")

df_food = df_food.drop(columns=["_key"], errors="ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 저장
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5: 저장")
print("=" * 65)

df_food.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"  [OK] {OUT_PATH}")
print(f"  총 {len(df_food):,}행 저장 완료")
print(f"  is_exemplary=True     : {df_food['is_exemplary'].sum():,}건")
print(f"  is_tourist_certified=True: {df_food['is_tourist_certified'].sum():,}건")

print("\n" + "=" * 65)
print("[DONE] preprocess_soho_food.py 완료")
print("=" * 65)
