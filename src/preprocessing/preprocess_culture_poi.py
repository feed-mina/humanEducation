"""
preprocess_culture_poi.py
==========================
문화 시설 CSV → culture_poi_nationwide.csv

[ 좌표 전략 ]
  1. 좌표정보(X), 좌표정보(Y) 있음 → pyproj로 TM → WGS84 변환
  2. X,Y 없음 → Vworld API 또는 JUSO API로 도로명주소 지오코딩
     - VWORLD_API_KEY 설정 시 Vworld 우선
     - JUSO_CONFIRM_KEY 설정 시 JUSO 사용
     - 둘 다 없으면 해당 행 제외

[ 입력 ]
  kride-project/문화_박물관 및 미술관.csv   (euc-kr, 1,264행)   → museum
  kride-project/문화_한옥체험업.csv          (cp949,  3,068행)   → hanok
  kride-project/문화_종합테마파크업.csv       (euc-kr,   80행)   → theme_park
  kride-project/문화_테마파크업(기타).csv     (cp949,  6,865행)  → theme_park
  kride-project/문화_종합휴양업.csv           (cp949,    41행)   → resort
  kride-project/문화_국내여행업.csv           (cp949, 20,126행)  → travel_agency
  kride-project/문화_종합여행업.csv           (cp949, 19,265행)  → travel_agency
  kride-project/문화_시내순환관광업.csv        (cp949,   114행)   → tour_bus

[ 출력 ]
  data/raw_ml/culture_poi_nationwide.csv

[ 실행 ]
  python src/preprocessing/preprocess_culture_poi.py
"""

import os
import sys
import time
import warnings

import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── 환경 변수 ─────────────────────────────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))

load_dotenv(os.path.join(BASE_DIR, ".env"))

VWORLD_KEY = os.getenv("VWORLD_API_KEY", "")
JUSO_KEY   = os.getenv("JUSO_CONFIRM_KEY", "")

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
DATA_FILES = [
    {
        "path":         os.path.join(BASE_DIR, "문화_박물관 및 미술관.csv"),
        "enc":          "euc-kr",
        "sub_category": "museum",
        "name_col":     "사업장명",
    },
    {
        "path":         os.path.join(BASE_DIR, "문화_한옥체험업.csv"),
        "enc":          "cp949",
        "sub_category": "hanok",
        "name_col":     "사업장명",
    },
    {
        "path":         os.path.join(BASE_DIR, "문화_종합테마파크업.csv"),
        "enc":          "euc-kr",
        "sub_category": "theme_park",
        "name_col":     "사업장명",
    },
    {
        "path":         os.path.join(BASE_DIR, "문화_테마파크업(기타).csv"),
        "enc":          "cp949",
        "sub_category": "theme_park",
        "name_col":     "사업장명",
    },
    {
        "path":         os.path.join(BASE_DIR, "문화_종합휴양업.csv"),
        "enc":          "cp949",
        "sub_category": "resort",
        "name_col":     "사업장명",
    },
    {
        "path":         os.path.join(BASE_DIR, "문화_국내여행업.csv"),
        "enc":          "cp949",
        "sub_category": "travel_agency",
        "name_col":     "사업장명",
    },
    {
        "path":         os.path.join(BASE_DIR, "문화_종합여행업.csv"),
        "enc":          "cp949",
        "sub_category": "travel_agency",
        "name_col":     "사업장명",
    },
    {
        "path":         os.path.join(BASE_DIR, "문화_시내순환관광업.csv"),
        "enc":          "cp949",
        "sub_category": "tour_bus",
        "name_col":     "사업장명",
    },
]

OUT_DIR  = os.path.join(BASE_DIR, "data", "raw_ml")
OUT_PATH = os.path.join(OUT_DIR, "culture_poi_nationwide.csv")
os.makedirs(OUT_DIR, exist_ok=True)

ACTIVE_FILTER = "영업|정상"   # 영업상태명 포함 패턴

# ── TM 좌표 변환 설정 ─────────────────────────────────────────────────────────
# 공공데이터 좌표정보(X/Y)는 한국 TM 좌표계 (EPSG:5174 또는 EPSG:5186)
# pyproj 설치: pip install pyproj
EPSG_CANDIDATES = ["EPSG:5174", "EPSG:5186", "EPSG:2097"]

def _build_transformer():
    """pyproj Transformer 초기화. 가장 먼저 유효한 EPSG 반환."""
    try:
        from pyproj import Transformer
        # 서울 종로구 창신동 기대값: lon≈127.01, lat≈37.58
        ref_x, ref_y = 200820.411, 452168.573
        for epsg in EPSG_CANDIDATES:
            t = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
            lon, lat = t.transform(ref_x, ref_y)
            if 126.0 < lon < 128.5 and 37.0 < lat < 38.5:
                print(f"  [TM→WGS84] 사용 좌표계: {epsg}  (검증: lon={lon:.4f}, lat={lat:.4f})")
                return t
        print("  [WARN] TM 좌표계 자동 탐지 실패 — EPSG:5174 강제 사용")
        return Transformer.from_crs("EPSG:5174", "EPSG:4326", always_xy=True)
    except ImportError:
        print("  [WARN] pyproj 미설치 — TM 변환 불가. pip install pyproj 실행 후 재시도")
        return None

_transformer = None


def tm_to_wgs84(x, y):
    """TM(x,y) → (lon, lat) WGS84. 실패 시 (None, None)."""
    global _transformer
    if _transformer is None:
        return None, None
    try:
        lon, lat = _transformer.transform(float(x), float(y))
        if 124 < lon < 132 and 33 < lat < 39:
            return round(lon, 7), round(lat, 7)
    except Exception:
        pass
    return None, None


# ── 지오코딩 함수 ─────────────────────────────────────────────────────────────
_geo_cache = {}

def geocode_vworld(address: str) -> tuple:
    """Vworld API로 도로명주소 → (lon, lat). 실패 시 (None, None)."""
    if not VWORLD_KEY or not address:
        return None, None
    if address in _geo_cache:
        return _geo_cache[address]
    try:
        url = "https://api.vworld.kr/req/address"
        params = {
            "service": "address", "request": "getcoord",
            "type": "road", "address": address,
            "key": VWORLD_KEY, "format": "json",
        }
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        if data.get("response", {}).get("status") == "OK":
            pt = data["response"]["result"]["point"]
            lon, lat = float(pt["x"]), float(pt["y"])
            _geo_cache[address] = (lon, lat)
            time.sleep(0.05)
            return lon, lat
    except Exception:
        pass
    _geo_cache[address] = (None, None)
    return None, None


def geocode_juso(address: str) -> tuple:
    """JUSO 도로명주소 API로 (lon, lat). 실패 시 (None, None)."""
    if not JUSO_KEY or not address:
        return None, None
    if address in _geo_cache:
        return _geo_cache[address]
    try:
        url = "https://www.juso.go.kr/addrlink/addrCoordApi.do"
        params = {
            "confmKey": JUSO_KEY, "resultType": "json",
            "keyword": address, "countPerPage": 1,
        }
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        juso_list = data.get("results", {}).get("juso", [])
        if juso_list:
            item = juso_list[0]
            lon = float(item.get("entX", 0) or 0)
            lat = float(item.get("entY", 0) or 0)
            if lon and lat:
                _geo_cache[address] = (lon, lat)
                time.sleep(0.05)
                return lon, lat
    except Exception:
        pass
    _geo_cache[address] = (None, None)
    return None, None


def geocode(address: str) -> tuple:
    """Vworld 우선, 없으면 JUSO, 없으면 (None, None)."""
    if VWORLD_KEY:
        lon, lat = geocode_vworld(address)
        if lon:
            return lon, lat
    if JUSO_KEY:
        return geocode_juso(address)
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("preprocess_culture_poi.py 시작")
print(f"  Vworld API 키: {'설정됨' if VWORLD_KEY else '없음'}")
print(f"  JUSO API 키:   {'설정됨' if JUSO_KEY else '없음'}")
print("=" * 65)

_transformer = _build_transformer()

all_dfs = []

for cfg in DATA_FILES:
    fpath = cfg["path"]
    label = cfg["sub_category"]
    print(f"\n{'=' * 65}")
    print(f"처리: {os.path.basename(fpath)}  →  sub_category={label}")
    print("=" * 65)

    if not os.path.exists(fpath):
        print(f"  [SKIP] 파일 없음: {fpath}")
        continue

    df = pd.read_csv(fpath, encoding=cfg["enc"], low_memory=False)
    total = len(df)

    # 영업 중인 곳만
    if "영업상태명" in df.columns:
        df = df[df["영업상태명"].fillna("").str.contains(ACTIVE_FILTER)].copy()
    print(f"  전체 {total:,}행 → 영업중 {len(df):,}행")

    # 필수 컬럼
    name_col = cfg["name_col"]
    if name_col not in df.columns:
        print(f"  [ERR] 이름 컬럼 없음: {name_col}")
        continue

    df["name"]         = df[name_col].fillna("알수없음")
    df["category"]     = "kculture"
    df["sub_category"] = label
    df["address"]      = df.get("도로명주소", pd.Series(dtype=str)).fillna("")
    df["source"]       = "public"

    # ── 좌표 처리 ──────────────────────────────────────────────────────────
    df["lat"] = pd.NA
    df["lon"] = pd.NA

    has_xy = ("좌표정보(X)" in df.columns) and ("좌표정보(Y)" in df.columns)

    if has_xy and _transformer is not None:
        # TM → WGS84 일괄 변환
        mask_xy = df["좌표정보(X)"].notna() & df["좌표정보(Y)"].notna()
        converted = df[mask_xy].apply(
            lambda r: tm_to_wgs84(r["좌표정보(X)"], r["좌표정보(Y)"]), axis=1
        )
        df.loc[mask_xy, "lon"] = [c[0] for c in converted]
        df.loc[mask_xy, "lat"] = [c[1] for c in converted]
        ok_tm = df["lat"].notna().sum()
        print(f"  TM 변환 완료: {ok_tm:,}건")

    # 좌표 없는 행 → API 지오코딩
    missing_mask = df["lat"].isna()
    missing_cnt  = missing_mask.sum()
    if missing_cnt > 0:
        if VWORLD_KEY or JUSO_KEY:
            print(f"  API 지오코딩 시작: {missing_cnt}건 ({label})")
            results = []
            for i, (idx, row) in enumerate(df[missing_mask].iterrows(), 1):
                lon, lat = geocode(row["address"])
                results.append((idx, lon, lat))
                if i % 50 == 0:
                    print(f"    진행 {i}/{missing_cnt}")
            for idx, lon, lat in results:
                df.at[idx, "lon"] = lon
                df.at[idx, "lat"] = lat
            ok_api = df["lat"].notna().sum() - (len(df) - missing_cnt)
            print(f"  API 지오코딩 성공: {ok_api:,}건")
        else:
            print(f"  [INFO] API 키 없음 → 좌표 없는 {missing_cnt}건 제외")

    # 좌표 없는 행 최종 제거
    df = df.dropna(subset=["lat", "lon"]).copy()
    df = df[(df["lat"] > 33) & (df["lat"] < 39)]
    df = df[(df["lon"] > 124) & (df["lon"] < 132)]
    print(f"  최종 유효 행수: {len(df):,}행")

    # 시도 추출
    df["sido"] = df["address"].str.split().str[0].replace("", pd.NA)

    out_cols = ["name", "category", "sub_category", "sido",
                "address", "lat", "lon", "source"]
    all_dfs.append(df[out_cols])


# ══════════════════════════════════════════════════════════════════════════════
# 저장
# ══════════════════════════════════════════════════════════════════════════════
if not all_dfs:
    print("\n[ERR] 처리된 데이터 없음")
    sys.exit(1)

df_out = pd.concat(all_dfs, ignore_index=True)

print("\n" + "=" * 65)
print("최종 저장")
print("=" * 65)
print(df_out["sub_category"].value_counts().to_string())
df_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n  [OK] {OUT_PATH}")
print(f"  총 {len(df_out):,}행 저장 완료")

print("\n" + "=" * 65)
print("[DONE] preprocess_culture_poi.py 완료")
print("=" * 65)
