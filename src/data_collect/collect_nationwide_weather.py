"""
collect_nationwide_weather.py
==============================
기상청 ASOS 일자료 API → 전국 관측소 날씨 CSV 수집

기존 fetch_weather_data.py 대비 변경:
  - 관측소 5개 (수도권) → 전국 17개 시도 약 60개 관측소
  - 출력 파일: weather_asos_daily_nationwide.csv (기존 파일 덮어쓰지 않음)
  - --region 옵션으로 특정 시도만 수집 가능
  - --merge 옵션으로 기존 weather_asos_daily.csv 와 합치기 가능

[ 실행 방법 ]
  set ASOS_API_KEY=발급받은_인증키   (Windows)
  export ASOS_API_KEY=발급받은_인증키 (Mac/Linux)

  # 전국 전체 수집
  python src/data_collect/collect_nationwide_weather.py

  # 특정 시도만 수집
  python src/data_collect/collect_nationwide_weather.py --region 강원 경상북도 제주

  # 수집 후 기존 파일과 병합
  python src/data_collect/collect_nationwide_weather.py --merge

  # API 키 직접 전달
  python src/data_collect/collect_nationwide_weather.py --api_key 인증키

[ 출력 파일 ]
  data/dl/kma_weather_raw/weather_asos_daily_nationwide.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, date

import pandas as pd
import requests

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

# ── 경로 설정 (kride-project 루트 기준) ──────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # kride-project/
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()

OUT_DIR  = os.path.join(BASE_DIR, "data", "dl", "kma_weather_raw")
OUT_FILE = os.path.join(OUT_DIR, "weather_asos_daily_nationwide.csv")
EXISTING_FILE = os.path.join(OUT_DIR, "weather_asos_daily.csv")

# ── API 설정 ──────────────────────────────────────────────────────────────────
API_URL    = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
START_DATE = "20230101"
END_DATE   = "20251231"

# API → CSV 컬럼 매핑 (build_weather_lstm.py preprocess() 와 동일 포맷)
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

# ── 전국 ASOS 관측소 (지점번호: (지점명, 시도)) ───────────────────────────────
# KMA 공식 관측소 기준, 전국 17개 시도 약 60개 선별
STATIONS: dict[str, tuple[str, str]] = {
    # 서울
    "108": ("서울",    "서울특별시"),
    # 인천
    "112": ("인천",    "인천광역시"),
    "201": ("강화",    "인천광역시"),
    # 경기
    "098": ("동두천",  "경기도"),
    "099": ("파주",    "경기도"),
    "119": ("수원",    "경기도"),
    "202": ("양평",    "경기도"),
    "203": ("이천",    "경기도"),
    # 강원
    "090": ("속초",    "강원특별자치도"),
    "100": ("대관령",  "강원특별자치도"),
    "101": ("춘천",    "강원특별자치도"),
    "105": ("강릉",    "강원특별자치도"),
    "106": ("동해",    "강원특별자치도"),
    "114": ("원주",    "강원특별자치도"),
    "121": ("영월",    "강원특별자치도"),
    "211": ("인제",    "강원특별자치도"),
    "216": ("태백",    "강원특별자치도"),
    # 대전
    "133": ("대전",    "대전광역시"),
    # 세종
    "239": ("세종",    "세종특별자치시"),
    # 충북
    "131": ("청주",    "충청북도"),
    "226": ("보은",    "충청북도"),
    "221": ("제천",    "충청북도"),
    # 충남
    "129": ("서산",    "충청남도"),
    "232": ("천안",    "충청남도"),
    "235": ("보령",    "충청남도"),
    "236": ("부여",    "충청남도"),
    # 대구
    "143": ("대구",    "대구광역시"),
    # 경북
    "115": ("울릉도",  "경상북도"),
    "130": ("울진",    "경상북도"),
    "135": ("추풍령",  "경상북도"),
    "136": ("안동",    "경상북도"),
    "137": ("상주",    "경상북도"),
    "138": ("포항",    "경상북도"),
    "271": ("봉화",    "경상북도"),
    "272": ("영주",    "경상북도"),
    "273": ("문경",    "경상북도"),
    "279": ("의성",    "경상북도"),
    "281": ("구미",    "경상북도"),
    "283": ("영천",    "경상북도"),
    "284": ("경주",    "경상북도"),
    # 울산
    "152": ("울산",    "울산광역시"),
    # 부산
    "159": ("부산",    "부산광역시"),
    # 경남
    "155": ("창원",    "경상남도"),
    "162": ("통영",    "경상남도"),
    "175": ("진주",    "경상남도"),
    "177": ("거창",    "경상남도"),
    "262": ("합천",    "경상남도"),
    "263": ("밀양",    "경상남도"),
    "264": ("산청",    "경상남도"),
    "266": ("거제",    "경상남도"),
    "268": ("남해",    "경상남도"),
    # 광주
    "156": ("광주",    "광주광역시"),
    # 전북
    "140": ("군산",    "전라북도"),
    "146": ("전주",    "전라북도"),
    "243": ("부안",    "전라북도"),
    "244": ("임실",    "전라북도"),
    "245": ("정읍",    "전라북도"),
    "247": ("남원",    "전라북도"),
    # 전남
    "165": ("목포",    "전라남도"),
    "168": ("여수",    "전라남도"),
    "170": ("완도",    "전라남도"),
    "172": ("고흥",    "전라남도"),
    "174": ("순천",    "전라남도"),
    "192": ("진도",    "전라남도"),
    "252": ("영광",    "전라남도"),
    "254": ("해남",    "전라남도"),
    # 제주
    "184": ("제주",    "제주특별자치도"),
    "189": ("서귀포",  "제주특별자치도"),
}


# ══════════════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════════════
def month_ranges(start: str, end: str) -> list[tuple[str, str]]:
    s = datetime.strptime(start, "%Y%m%d").date()
    e = datetime.strptime(end,   "%Y%m%d").date()
    ranges = []
    cur = s
    while cur <= e:
        if cur.month == 12:
            last = date(cur.year + 1, 1, 1) - timedelta(days=1)
        else:
            last = date(cur.year, cur.month + 1, 1) - timedelta(days=1)
        month_end = min(last, e)
        ranges.append((cur.strftime("%Y%m%d"), month_end.strftime("%Y%m%d")))
        cur = month_end + timedelta(days=1)
    return ranges


def fetch_one(api_key: str, stn_id: str, start_dt: str, end_dt: str,
              retries: int = 3) -> list[dict]:
    """ASOS 일자료 API 1회 호출. 실패 시 최대 retries번 재시도."""
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
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            body = resp.json().get("response", {}).get("body", {})
            items = body.get("items", {})
            if not items:
                return []
            item_list = items.get("item", [])
            return item_list if isinstance(item_list, list) else [item_list]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.0)
            else:
                print(f"    ⚠ API 오류 ({stn_id}, {start_dt}~{end_dt}): {e}")
    return []


# ══════════════════════════════════════════════════════════════════════════════
# 수집 메인
# ══════════════════════════════════════════════════════════════════════════════
def collect(api_key: str, target_sidos: list[str] | None = None) -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)
    ranges = month_ranges(START_DATE, END_DATE)

    # 시도 필터
    if target_sidos:
        stations = {k: v for k, v in STATIONS.items()
                    if any(s in v[1] for s in target_sidos)}
        if not stations:
            print(f"  ❌ 해당 시도 관측소 없음: {target_sidos}")
            print(f"  사용 가능한 시도: {sorted(set(v[1] for v in STATIONS.values()))}")
            sys.exit(1)
    else:
        stations = STATIONS

    # 시도별 그룹 출력
    sido_groups: dict[str, list[str]] = {}
    for stn_id, (stn_nm, sido) in stations.items():
        sido_groups.setdefault(sido, []).append(f"{stn_nm}({stn_id})")

    print(f"  수집 기간    : {START_DATE} ~ {END_DATE}  ({len(ranges)}개 월 구간)")
    print(f"  수집 관측소  : 총 {len(stations)}개 / {len(sido_groups)}개 시도")
    for sido, stns in sorted(sido_groups.items()):
        print(f"    {sido:18s}: {', '.join(stns)}")
    print()

    all_rows: list[dict] = []

    for stn_id, (stn_nm, sido) in stations.items():
        print(f"  [{sido} / {stn_nm} ({stn_id})] 수집 중...", end="", flush=True)
        stn_rows = 0
        for start_dt, end_dt in ranges:
            items = fetch_one(api_key, stn_id, start_dt, end_dt)
            # sido 컬럼 주입 (모델링 시 지역 필터용)
            for item in items:
                item["sido"] = sido
            all_rows.extend(items)
            stn_rows += len(items)
            time.sleep(0.3)
        print(f" {stn_rows}건")

    if not all_rows:
        print("\n  ❌ 수집된 데이터 없음. API 키와 네트워크를 확인하세요.")
        sys.exit(1)

    return pd.DataFrame(all_rows)


# ══════════════════════════════════════════════════════════════════════════════
# 정제 및 저장
# ══════════════════════════════════════════════════════════════════════════════
def clean_and_save(df: pd.DataFrame, merge: bool = False) -> None:
    # API 필드명 → 한국어 컬럼명
    df = df.rename(columns={k: v for k, v in FIELD_MAP.items() if k in df.columns})

    num_cols = ["평균기온(°C)", "일강수량(mm)", "평균풍속(m/s)",
                "평균상대습도(%)", "평균전운량(1/10)", "최저기온(°C)", "최고기온(°C)"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "일시" in df.columns:
        df["일시"] = pd.to_datetime(df["일시"], errors="coerce")
        df = df.dropna(subset=["일시"])
        sort_cols = [c for c in ["sido", "지점명", "일시"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # 기존 파일과 병합
    if merge and os.path.exists(EXISTING_FILE):
        print(f"\n  기존 파일 병합: {EXISTING_FILE}")
        df_old = pd.read_csv(EXISTING_FILE, encoding="utf-8-sig")
        if "일시" in df_old.columns:
            df_old["일시"] = pd.to_datetime(df_old["일시"], errors="coerce")
        df = pd.concat([df_old, df], ignore_index=True)
        # 중복 제거 (지점명+일시 기준)
        dedup_cols = [c for c in ["지점명", "일시"] if c in df.columns]
        if dedup_cols:
            before = len(df)
            df = df.drop_duplicates(subset=dedup_cols, keep="last")
            print(f"  중복 제거: {before - len(df)}행 → 최종 {len(df)}행")
        sort_cols = [c for c in ["sido", "지점명", "일시"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ 저장 완료: {OUT_FILE}")
    print(f"     총 {len(df):,}행 × {len(df.columns)}컬럼")
    if "일시" in df.columns:
        print(f"     기간: {df['일시'].min().date()} ~ {df['일시'].max().date()}")
    if "지점명" in df.columns:
        print(f"     관측소별 행 수 (상위 10):")
        print(df["지점명"].value_counts().head(10).to_string())
    if "sido" in df.columns:
        print(f"\n     시도별 관측소 수:")
        print(df.groupby("sido")["지점명"].nunique().sort_index().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전국 ASOS 일자료 수집")
    parser.add_argument(
        "--api_key",
        default=os.environ.get("ASOS_API_KEY", ""),
        help="공공데이터포털 ASOS 일자료 서비스 인증키 (환경변수 ASOS_API_KEY 권장)",
    )
    parser.add_argument(
        "--region", nargs="+", default=None,
        metavar="시도명",
        help="수집할 시도 목록 (예: --region 강원 경상북도 제주). 생략 시 전국 전체.",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="기존 weather_asos_daily.csv 와 병합하여 저장",
    )
    parser.add_argument(
        "--start", default=START_DATE,
        help=f"수집 시작일 YYYYMMDD (기본: {START_DATE})",
    )
    parser.add_argument(
        "--end", default=END_DATE,
        help=f"수집 종료일 YYYYMMDD (기본: {END_DATE})",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("❌ API 키 없음.")
        print("   방법 1: set ASOS_API_KEY=인증키  (Windows)")
        print("   방법 2: python ... collect_nationwide_weather.py --api_key 인증키")
        print()
        print("   공공데이터포털 (data.go.kr) 에서 발급:")
        print("   '기상청_지상(중관,ASOS) 일자료 조회서비스' 검색 후 활용 신청")
        sys.exit(1)

    # 날짜 오버라이드
    import collect_nationwide_weather as _self
    _self.START_DATE = args.start
    _self.END_DATE   = args.end

    print("=" * 65)
    print("전국 ASOS 일자료 수집 시작")
    print(f"  기존 fetch_weather_data.py 대비: 5개 → {len(STATIONS)}개 관측소")
    print("=" * 65)

    df = collect(args.api_key, target_sidos=args.region)

    print("=" * 65)
    print("정제 및 저장")
    print("=" * 65)
    clean_and_save(df, merge=args.merge)

    print("\n" + "=" * 65)
    print("✅ collect_nationwide_weather.py 완료")
    print("   다음 단계: python src/dl/build_weather_lstm.py")
    print("=" * 65)
