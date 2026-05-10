"""
fetch_weather_data.py
=====================
공공데이터포털 ASOS 일자료 API → 과거 날씨 CSV 수집

API: 기상청_지상(중관,ASOS) 일자료 조회서비스
EndPoint: https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList

[ 실행 방법 ]
  # 환경변수로 키 설정 (권장)
  set ASOS_API_KEY=발급받은_인증키   (Windows)
  export ASOS_API_KEY=발급받은_인증키 (Mac/Linux)
  python kride-project/fetch_weather_data.py

  # 또는 직접 인자 전달
  python kride-project/fetch_weather_data.py --api_key 발급받은_인증키

[ 출력 파일 ]
  kride-project/data/dl/kma_weather_raw/weather_asos_daily.csv

[ 수집 대상 ]
  서울/경기 주요 관측소 5개 × 3년치 (2023-01-01 ~ 2025-12-31)
  관측소: 서울(108), 수원(119), 인천(112), 양평(202), 이천(203)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, date

import pandas as pd
import requests

# ── .env 파일 자동 로드 (있으면 읽음) ────────────────────────────────────────
def _load_dotenv() -> None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))

OUT_DIR  = os.path.join(BASE_DIR, "data", "dl", "kma_weather_raw")
OUT_FILE = os.path.join(OUT_DIR, "weather_asos_daily.csv")

# ── API 설정 ───────────────────────────────────────────────────────────────────
API_URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"

# 수집 기간: 과거 3년
START_DATE = "20230101"
END_DATE   = "20251231"

# 서울/경기 주요 관측소 (지점번호: 지점명)
STATIONS = {
    "108": "서울",
    "119": "수원",
    "112": "인천",
    "202": "양평",
    "203": "이천",
}

# API → CSV 컬럼 매핑 (build_weather_lstm.py의 preprocess()와 맞춤)
# API 응답 필드명 → 저장할 한국어 컬럼명
FIELD_MAP = {
    "tm":     "일시",
    "stnNm":  "지점명",
    "avgTa":  "평균기온(°C)",
    "sumRn":  "일강수량(mm)",
    "avgWs":  "평균풍속(m/s)",
    "avgRhm": "평균상대습도(%)",
    "avgTca": "평균전운량(1/10)",
    "minTa":  "최저기온(°C)",
    "maxTa":  "최고기온(°C)",
}


# ══════════════════════════════════════════════════════════════════════════════
# 월 단위 날짜 구간 생성 (API 부하 분산)
# ══════════════════════════════════════════════════════════════════════════════
def month_ranges(start: str, end: str) -> list[tuple[str, str]]:
    """
    start~end 기간을 월 단위로 분할한 (startDt, endDt) 리스트 반환
    예: ("20230101", "20230201") → 1월치 한 번에 조회
    """
    s = datetime.strptime(start, "%Y%m%d").date()
    e = datetime.strptime(end,   "%Y%m%d").date()
    ranges = []
    cur = s
    while cur <= e:
        # 해당 월의 마지막 날 계산
        if cur.month == 12:
            last = date(cur.year + 1, 1, 1) - timedelta(days=1)
        else:
            last = date(cur.year, cur.month + 1, 1) - timedelta(days=1)
        month_end = min(last, e)
        ranges.append((cur.strftime("%Y%m%d"), month_end.strftime("%Y%m%d")))
        cur = month_end + timedelta(days=1)
    return ranges


# ══════════════════════════════════════════════════════════════════════════════
# API 호출 (관측소 1개, 기간 1개월)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_one(api_key: str, stn_id: str, start_dt: str, end_dt: str) -> list[dict]:
    """
    ASOS 일자료 API 1회 호출 → 레코드 리스트 반환
    한 번에 최대 999건까지 조회 가능 (1개월 ≈ 31건이므로 충분)

    주의: serviceKey는 URL에 직접 붙여야 함.
    requests의 params={}를 사용하면 자동 URL 인코딩이 발생해
    공공데이터포털에서 403 Forbidden 오류가 발생함.
    """
    # serviceKey만 URL에 직접 삽입, 나머지는 params로 전달
    url = f"{API_URL}?serviceKey={api_key}"
    params = {
        "pageNo":    "1",
        "numOfRows": "999",
        "dataType":  "JSON",
        "dataCd":    "ASOS",
        "dateCd":    "DAY",
        "startDt":   start_dt,
        "endDt":     end_dt,
        "stnIds":    stn_id,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        body = resp.json().get("response", {}).get("body", {})
        items = body.get("items", {})
        if not items:
            return []
        item_list = items.get("item", [])
        return item_list if isinstance(item_list, list) else [item_list]
    except Exception as e:
        print(f"    ⚠ API 오류 ({stn_id}, {start_dt}~{end_dt}): {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 전체 수집
# ══════════════════════════════════════════════════════════════════════════════
def collect(api_key: str) -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)

    ranges = month_ranges(START_DATE, END_DATE)
    print(f"  수집 기간: {START_DATE} ~ {END_DATE}  ({len(ranges)}개 월 구간)")
    print(f"  관측소: {list(STATIONS.values())}\n")

    all_rows = []

    for stn_id, stn_nm in STATIONS.items():
        print(f"  [{stn_nm} ({stn_id})] 수집 중...")
        stn_rows = 0
        for start_dt, end_dt in ranges:
            items = fetch_one(api_key, stn_id, start_dt, end_dt)
            all_rows.extend(items)
            stn_rows += len(items)
            time.sleep(0.3)   # API 과부하 방지
        print(f"    → {stn_rows}건")

    if not all_rows:
        print("\n  ❌ 수집된 데이터 없음. API 키와 네트워크를 확인하세요.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 컬럼 정리 및 저장
# ══════════════════════════════════════════════════════════════════════════════
def clean_and_save(df: pd.DataFrame) -> None:
    # API 필드 → 한국어 컬럼명으로 변환
    df = df.rename(columns={k: v for k, v in FIELD_MAP.items() if k in df.columns})

    # 숫자 컬럼 변환 (빈 문자열 → NaN → 0.0)
    num_cols = ["평균기온(°C)", "일강수량(mm)", "평균풍속(m/s)",
                "평균상대습도(%)", "평균전운량(1/10)", "최저기온(°C)", "최고기온(°C)"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 날짜 정렬
    if "일시" in df.columns:
        df["일시"] = pd.to_datetime(df["일시"], errors="coerce")
        df = df.dropna(subset=["일시"])
        df = df.sort_values(["지점명", "일시"] if "지점명" in df.columns else ["일시"])
        df = df.reset_index(drop=True)

    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ 저장 완료: {OUT_FILE}")
    print(f"     총 {len(df):,}행 × {len(df.columns)}컬럼")
    print(f"     컬럼: {list(df.columns)}")
    if "일시" in df.columns:
        print(f"     기간: {df['일시'].min().date()} ~ {df['일시'].max().date()}")
    if "지점명" in df.columns:
        print(f"     관측소별 행 수:\n{df['지점명'].value_counts().to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASOS 일자료 수집")
    parser.add_argument(
        "--api_key",
        default=os.environ.get("ASOS_API_KEY", ""),
        help="공공데이터포털 ASOS 일자료 서비스 인증키 (환경변수 ASOS_API_KEY로도 설정 가능)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("❌ API 키 없음.")
        print("   방법 1: set ASOS_API_KEY=인증키  (Windows)")
        print("   방법 2: python fetch_weather_data.py --api_key 인증키")
        sys.exit(1)

    print("=" * 65)
    print("ASOS 일자료 수집 시작")
    print("=" * 65)

    df = collect(args.api_key)

    print("=" * 65)
    print("저장 처리")
    print("=" * 65)
    clean_and_save(df)

    print("\n" + "=" * 65)
    print("✅ fetch_weather_data.py 완료")
    print("   다음 단계: python kride-project/build_weather_lstm.py")
    print("=" * 65)
