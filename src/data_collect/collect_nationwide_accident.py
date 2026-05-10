"""
collect_nationwide_accident.py
================================
전국 교통사고 데이터 → 시도/시군구별 위험도(district_danger) 계산
→ district_danger_nationwide.csv 저장

기존 build_safety_model.py 는 서울 자전거 사고 다발지 XLSX만 사용.
이 스크립트는 전국 단위로 확대:
  1. 기존 보유 데이터 우선 사용
     - data/raw_ml/한국도로교통공단_전국.csv  (902KB, 전국)
     - data/raw_ml/다발지분석-24년 자전거 교통사고 다발지역_서울.xlsx
     - data/raw_ml/다발지분석-24년 보행자 교통사고 다발지역_경기도.xlsx
  2. 공공데이터포털 TAAS API (선택) - 추가 데이터 수집

출력 컬럼 (build_safety_model.py 와 동일):
  sigungu, sido, crash_count, death_count, severe_count, injury_count, danger_score

위험도 계산:
  raw_score = 발생건수×1.0 + 사망자×5.0 + 중상자×2.0 + 부상자×1.0
  danger_score = MinMaxScaler(raw_score)  → 0.0 ~ 1.0

[ 실행 방법 ]
  # 보유 데이터만으로 생성 (API 불필요)
  python src/data_collect/collect_nationwide_accident.py

  # TAAS API 추가 수집 포함 (공공데이터포털 키 필요)
  python src/data_collect/collect_nationwide_accident.py --api_key 인증키

[ 출력 파일 ]
  data/raw_ml/district_danger_nationwide.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── .env 자동 로드 ────────────────────────────────────────────────────────────
def _load_dotenv() -> None:
    for candidate in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]:
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            break

_load_dotenv()

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()

RAW_DIR     = os.path.join(BASE_DIR, "data", "raw_ml")
OUTPUT_PATH = os.path.join(RAW_DIR, "district_danger_nationwide.csv")

# ── 입력 파일 경로 ────────────────────────────────────────────────────────────
# 새로 다운로드한 전국 TAAS 사고다발지 CSV (프로젝트 루트)
BIKE_NATIONWIDE_CSV      = os.path.join(BASE_DIR, "bicycleDataset12_24.csv")
PED_NATIONWIDE_CSV       = os.path.join(BASE_DIR, "pedstriansDataset_19_24.csv")
# 기존 보조 파일
TAAS_NATIONWIDE_CSV      = os.path.join(RAW_DIR, "한국도로교통공단_전국.csv")
BIKE_SEOUL_XLSX          = os.path.join(RAW_DIR, "다발지분석-24년 자전거 교통사고 다발지역_서울.xlsx")
PEDESTRIAN_GYEONG_XLSX  = os.path.join(RAW_DIR, "다발지분석-24년 보행자 교통사고 다발지역_경기도.xlsx")

# TAAS 공공데이터 API (선택)
TAAS_API_URL = "https://apis.data.go.kr/B552061/AccidentInfoService/getSimpleAccidentInfoList"

# 전국 17개 시도 areaCode 매핑 (TAAS API 기준)
SIDO_AREA_CODES: dict[str, str] = {
    "서울특별시":       "11",
    "부산광역시":       "21",
    "대구광역시":       "22",
    "인천광역시":       "23",
    "광주광역시":       "24",
    "대전광역시":       "25",
    "울산광역시":       "26",
    "세종특별자치시":   "29",
    "경기도":           "31",
    "강원특별자치도":   "32",
    "충청북도":         "33",
    "충청남도":         "34",
    "전라북도":         "35",
    "전라남도":         "36",
    "경상북도":         "37",
    "경상남도":         "38",
    "제주특별자치도":   "39",
}


# 전국 17개 시도명 목록 (파싱용)
SIDO_NAMES = [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
    "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원특별자치도",
    "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도",
    "제주특별자치도",
]


def _parse_sido_sigungu(val: str) -> tuple[str, str]:
    """
    '서울특별시 종로구1'  → ('서울특별시', '종로구')
    '경기도 수원시 팔달구1' → ('경기도', '수원시 팔달구')
    끝에 붙는 숫자 제거 후 시도/시군구 분리.
    """
    import re
    val = re.sub(r'\d+$', '', str(val).strip())  # 끝 숫자 제거
    for sido in sorted(SIDO_NAMES, key=len, reverse=True):
        if val.startswith(sido):
            sigungu = val[len(sido):].strip()
            return sido, sigungu if sigungu else sido
    # 매칭 실패 시 공백 분리
    parts = val.split()
    if len(parts) >= 2:
        return parts[0], ' '.join(parts[1:])
    return val, val


# ══════════════════════════════════════════════════════════════════════════════
# 1. 새 TAAS 사고다발지 CSV 파싱 (bicycleDataset12_24 / pedstriansDataset_19_24)
# ══════════════════════════════════════════════════════════════════════════════
def load_taas_hotspot_csv(path: str, label: str = "") -> pd.DataFrame | None:
    """
    TAAS 사고다발지 CSV (전국)
    컬럼: 시도시군구명, 사고건수, 사망자수, 중상자수, 경상자수, 부상신고자수
    → 시도/시군구별 집계 DataFrame 반환
    """
    if not os.path.exists(path):
        print(f"  ⚠ 파일 없음 (스킵): {path}")
        return None

    print(f"  로드: {os.path.basename(path)} ({label})")
    for enc in ["cp949", "euc-kr", "utf-8-sig", "utf-8"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            continue
    else:
        print(f"  ❌ 인코딩 감지 실패: {path}")
        return None

    print(f"     shape: {df.shape},  컬럼: {list(df.columns[:8])}")

    if "시도시군구명" not in df.columns:
        print(f"  ⚠ '시도시군구명' 컬럼 없음")
        return None

    # 수치 변환
    for col in ["사고건수", "사망자수", "중상자수", "경상자수", "부상신고자수"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    rows = []
    for _, r in df.iterrows():
        sido, sigungu = _parse_sido_sigungu(r["시도시군구명"])
        rows.append({
            "sido":         sido,
            "sigungu":      sigungu,
            "crash_count":  float(r.get("사고건수",  0) or 0),
            "death_count":  float(r.get("사망자수",  0) or 0),
            "severe_count": float(r.get("중상자수",  0) or 0),
            "injury_count": float(r.get("경상자수",  0) or 0),
        })

    if not rows:
        return None

    agg = pd.DataFrame(rows).groupby(["sido", "sigungu"], as_index=False).agg(
        crash_count=("crash_count",  "sum"),
        death_count=("death_count",  "sum"),
        severe_count=("severe_count", "sum"),
        injury_count=("injury_count", "sum"),
    )
    print(f"     집계 결과: {len(agg)}개 시군구")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# 2. 기존 보조 데이터 파싱
# ══════════════════════════════════════════════════════════════════════════════
def load_taas_csv(path: str) -> pd.DataFrame | None:
    """한국도로교통공단_전국.csv 파싱 → 시군구별 집계"""
    if not os.path.exists(path):
        print(f"  ⚠ 파일 없음 (스킵): {path}")
        return None

    print(f"  로드: {os.path.basename(path)}")
    for enc in ["cp949", "utf-8-sig", "utf-8"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            continue
    else:
        print(f"  ❌ 인코딩 감지 실패: {path}")
        return None

    print(f"     shape: {df.shape},  컬럼: {list(df.columns[:8])}")

    # 컬럼명 정규화 (다양한 TAAS 포맷 대응)
    col_aliases = {
        "시도": ["시도명", "SIDO_NM", "sido", "광역시도"],
        "시군구": ["시군구명", "SGG_NM", "sigungu", "시군구"],
        "발생건수": ["사고건수", "발생", "OCCR_CNT", "accident_count", "건수"],
        "사망자수": ["사망", "DETH_DNVT_CNT", "death_count", "사망자"],
        "중상자수": ["중상", "SE_DNVT_CNT", "severe_count", "중상자"],
        "경상자수": ["경상", "SL_DNVT_CNT", "injury_count", "경상자", "부상자수", "부상자"],
    }

    rename_map: dict[str, str] = {}
    for std_name, aliases in col_aliases.items():
        for alias in aliases:
            if alias in df.columns and std_name not in rename_map.values():
                rename_map[alias] = std_name
                break

    df = df.rename(columns=rename_map)

    # 필수 컬럼 확인
    required = ["시도", "시군구", "발생건수"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [WARN] 필수 컬럼 없음 {missing} -- 컬럼 목록: {list(df.columns)}")
        # 시도+시군구 없이 시도만 있는 경우 시군구 = 시도로 대체
        if "시도" in df.columns and "시군구" not in df.columns:
            df["시군구"] = df["시도"]
        elif "시군구" in df.columns and "시도" not in df.columns:
            df["시도"] = "전국"
        else:
            return None

    # 숫자 변환
    for col in ["발생건수", "사망자수", "중상자수", "경상자수"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 컬럼 표준화
    if "사망자수" not in df.columns:
        df["사망자수"] = 0
    if "중상자수" not in df.columns:
        df["중상자수"] = 0
    if "경상자수" not in df.columns:
        df["경상자수"] = 0

    # 시군구별 집계
    agg = df.groupby(["시도", "시군구"], as_index=False).agg(
        crash_count=("발생건수", "sum"),
        death_count=("사망자수", "sum"),
        severe_count=("중상자수", "sum"),
        injury_count=("경상자수", "sum"),
    )
    agg = agg.rename(columns={"시도": "sido", "시군구": "sigungu"})
    print(f"     집계 결과: {len(agg)}개 시군구")
    return agg


def load_excel_accident(path: str, sido_default: str = "") -> pd.DataFrame | None:
    """다발지 XLSX 파싱 → 시군구별 집계"""
    if not os.path.exists(path):
        print(f"  ⚠ 파일 없음 (스킵): {path}")
        return None

    print(f"  로드: {os.path.basename(path)}")
    try:
        import openpyxl  # noqa: F401
        df = pd.read_excel(path, engine="openpyxl")
    except ImportError:
        print("  [WARN] openpyxl 없음: pip install openpyxl")
        return None
    except Exception as e:
        print(f"  [ERR] Excel 로드 실패: {e}")
        return None

    print(f"     shape: {df.shape},  컬럼: {list(df.columns[:8])}")

    # 시군구명 컬럼 탐색
    sigungu_col = None
    for c in df.columns:
        if "시군구" in str(c) or "구" in str(c):
            sigungu_col = c
            break

    if sigungu_col is None:
        print("  [WARN] sigungu col not found -- skip")
        return None

    # 수치 컬럼 탐색
    def find_col(keywords: list[str]) -> str | None:
        for kw in keywords:
            for c in df.columns:
                if kw in str(c):
                    return c
        return None

    crash_col  = find_col(["발생건수", "건수", "사고"])
    death_col  = find_col(["사망자", "사망"])
    severe_col = find_col(["중상자", "중상"])
    injury_col = find_col(["경상자", "부상자", "경상"])

    rows = []
    for _, row in df.iterrows():
        sigungu = str(row[sigungu_col]).strip()
        if not sigungu or sigungu in ["nan", "합계", "계"]:
            continue
        rows.append({
            "sido":         sido_default,
            "sigungu":      sigungu,
            "crash_count":  float(row[crash_col])  if crash_col  and pd.notna(row[crash_col])  else 0,
            "death_count":  float(row[death_col])  if death_col  and pd.notna(row[death_col])  else 0,
            "severe_count": float(row[severe_col]) if severe_col and pd.notna(row[severe_col]) else 0,
            "injury_count": float(row[injury_col]) if injury_col and pd.notna(row[injury_col]) else 0,
        })

    if not rows:
        return None

    agg = pd.DataFrame(rows).groupby(["sido", "sigungu"], as_index=False).sum()
    print(f"     집계 결과: {len(agg)}개 시군구")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# 2. TAAS API 수집 (선택)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_taas_api(api_key: str, year: int = 2024) -> pd.DataFrame | None:
    """공공데이터포털 TAAS API → 전국 시도별 사고 통계"""
    print(f"  TAAS API 수집 중 ({year}년)...")
    rows: list[dict] = []

    for sido_name, area_code in SIDO_AREA_CODES.items():
        params = {
            "serviceKey":  api_key,
            "numOfRows":   "1000",
            "pageNo":      "1",
            "type":        "json",
            "year":        str(year),
            "siDo":        area_code,
        }
        try:
            resp = requests.get(TAAS_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            items = (data.get("response", {})
                        .get("body", {})
                        .get("items", {})
                        .get("item", []))
            if isinstance(items, dict):
                items = [items]
            for item in items:
                rows.append({
                    "sido":         sido_name,
                    "sigungu":      item.get("sgg_nm", sido_name),
                    "crash_count":  float(item.get("occrCnt",  0) or 0),
                    "death_count":  float(item.get("dethDnvtCnt", 0) or 0),
                    "severe_count": float(item.get("seDnvtCnt", 0) or 0),
                    "injury_count": float(item.get("slDnvtCnt", 0) or 0),
                })
        except Exception as e:
            print(f"    ⚠ TAAS API 오류 ({sido_name}): {e}")
        time.sleep(0.3)

    if not rows:
        print("  ⚠ TAAS API 결과 없음")
        return None

    df = pd.DataFrame(rows).groupby(["sido", "sigungu"], as_index=False).sum()
    print(f"  TAAS API 집계: {len(df)}개 시군구")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. 위험도 점수 계산
# ══════════════════════════════════════════════════════════════════════════════
def compute_danger_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    build_safety_model.py 의 district_danger 계산 방식과 동일:
      raw_score = 발생×1 + 사망×5 + 중상×2 + 부상×1
      danger_score = MinMaxScaler(raw_score)
    """
    df = df.copy()
    df["raw_score"] = (
        df["crash_count"]  * 1.0 +
        df["death_count"]  * 5.0 +
        df["severe_count"] * 2.0 +
        df["injury_count"] * 1.0
    )

    if df["raw_score"].max() > 0:
        scaler = MinMaxScaler()
        df["danger_score"] = scaler.fit_transform(
            df[["raw_score"]]
        ).round(4)
    else:
        df["danger_score"] = 0.0

    return df.drop(columns=["raw_score"])


# ══════════════════════════════════════════════════════════════════════════════
# 4. 메인
# ══════════════════════════════════════════════════════════════════════════════
def main(api_key: str | None) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)

    frames: list[pd.DataFrame] = []

    print("=" * 65)
    print("STEP 1: 전국 TAAS 사고다발지 CSV 로드")
    print("=" * 65)

    # ① 전국 자전거 사고다발지 (2012-2024) ← 메인 소스
    df_bike = load_taas_hotspot_csv(BIKE_NATIONWIDE_CSV, "자전거 2012-2024")
    if df_bike is not None:
        frames.append(df_bike)

    # ② 전국 보행자 사고다발지 (2019-2024) ← 메인 소스
    df_ped = load_taas_hotspot_csv(PED_NATIONWIDE_CSV, "보행자 2019-2024")
    if df_ped is not None:
        frames.append(df_ped)

    print("\n" + "=" * 65)
    print("STEP 1-B: 기존 보조 데이터 로드 (선택)")
    print("=" * 65)

    # ③ 기존 전국 TAAS CSV (보조)
    df1 = load_taas_csv(TAAS_NATIONWIDE_CSV)
    if df1 is not None:
        frames.append(df1)

    # ④ 서울 자전거 사고 다발지 XLSX (보조)
    df2 = load_excel_accident(BIKE_SEOUL_XLSX, sido_default="서울특별시")
    if df2 is not None:
        frames.append(df2)

    # ⑤ 경기도 보행자 사고 다발지 XLSX (보조)
    df3 = load_excel_accident(PEDESTRIAN_GYEONG_XLSX, sido_default="경기도")
    if df3 is not None:
        frames.append(df3)

    # TAAS API (선택)
    if api_key:
        print("\n" + "=" * 65)
        print("STEP 2: TAAS API 추가 수집")
        print("=" * 65)
        df_api = fetch_taas_api(api_key)
        if df_api is not None:
            frames.append(df_api)

    if not frames:
        print("\n  ❌ 사용 가능한 사고 데이터 없음.")
        print("  필요한 파일:")
        print(f"    1. {BIKE_NATIONWIDE_CSV}")
        print(f"    2. {PED_NATIONWIDE_CSV}")
        print("  또는 --api_key 옵션으로 TAAS API 사용")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("STEP 3: 전국 병합 및 위험도 계산")
    print("=" * 65)

    # 병합 및 중복 집계
    df_all = pd.concat(frames, ignore_index=True)
    df_all["sido"]    = df_all["sido"].str.strip()
    df_all["sigungu"] = df_all["sigungu"].str.strip()

    # 시군구별 합산 (여러 소스에서 중복 가능)
    df_agg = df_all.groupby(["sido", "sigungu"], as_index=False).agg(
        crash_count=("crash_count",  "sum"),
        death_count=("death_count",  "sum"),
        severe_count=("severe_count", "sum"),
        injury_count=("injury_count", "sum"),
    )

    # 위험도 점수 계산
    df_result = compute_danger_score(df_agg)

    # 시도별 통계 출력
    print(f"\n  전국 시군구 수: {len(df_result)}개")
    print(f"  시도별 분포:")
    if "sido" in df_result.columns:
        sido_counts = df_result.groupby("sido").agg(
            시군구수=("sigungu", "count"),
            평균위험도=("danger_score", "mean"),
            최대위험도=("danger_score", "max"),
        ).round(4)
        print(sido_counts.to_string())

    print(f"\n  위험도 상위 20 시군구:")
    top20 = df_result.nlargest(20, "danger_score")[
        ["sido", "sigungu", "crash_count", "death_count", "danger_score"]
    ]
    print(top20.to_string(index=False))

    print("\n" + "=" * 65)
    print("STEP 4: 저장")
    print("=" * 65)

    # 컬럼 순서: build_safety_model.py 의 district_danger.csv 와 호환
    col_order = ["sido", "sigungu", "crash_count", "death_count",
                 "severe_count", "injury_count", "danger_score"]
    df_result = df_result[[c for c in col_order if c in df_result.columns]]
    df_result = df_result.sort_values(["sido", "sigungu"]).reset_index(drop=True)
    df_result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"  [OK] Saved: {OUTPUT_PATH}")
    print(f"     total {len(df_result):,} rows x {len(df_result.columns)} cols")
    print(f"     danger_score: min={df_result['danger_score'].min():.4f}  "
          f"max={df_result['danger_score'].max():.4f}  "
          f"mean={df_result['danger_score'].mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전국 교통사고 위험도 계산")
    parser.add_argument(
        "--api_key",
        default=os.environ.get("TAAS_API_KEY", ""),
        help="공공데이터포털 TAAS API 키 (선택). 없으면 보유 파일만 사용.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("전국 교통사고 위험도 처리 시작")
    print(f"  기존 build_safety_model.py 대비: 서울 → 전국 시군구")
    print("=" * 65)
    print()

    main(api_key=args.api_key if args.api_key else None)

    print("\n" + "=" * 65)
    print("[DONE] collect_nationwide_accident.py complete")
    print("   next: python src/ml/build_safety_model.py")
    print("   note: build_safety_model.py -> district_danger_nationwide.csv")
    print("=" * 65)
