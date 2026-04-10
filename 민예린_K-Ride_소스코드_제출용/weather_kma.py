"""
weather_kma.py
==============
기상청 단기예보 API 연동 모듈

[ 사용법 ]
  from kride-project.weather_kma import get_weather_weight, get_current_weather

[ 전제 ]
  공공데이터포털 (data.go.kr) → "기상청_단기예보 ((구)_동네예보) 조회서비스" API 키 발급
  환경변수 KMA_API_KEY 또는 직접 인자로 전달

[ 의존 패키지 ]
  pip install requests pyproj
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

import requests

# ── 기상청 단기예보 API ────────────────────────────────────────────────────────
KMA_BASE_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"

# 강수형태 (PTY) 코드
PTY_MAP = {
    "0": "없음",
    "1": "비",
    "2": "비/눈",
    "3": "눈",
    "4": "소나기",
}

# 하늘상태 (SKY) 코드
SKY_MAP = {
    "1": "맑음",
    "3": "구름많음",
    "4": "흐림",
}


# ══════════════════════════════════════════════════════════════════════════════
# 좌표 변환 (위경도 → 기상청 격자 nx/ny)
# ══════════════════════════════════════════════════════════════════════════════
def latlon_to_grid(lat: float, lon: float) -> Tuple[int, int]:
    """
    WGS84 위경도 → 기상청 격자 좌표 (nx, ny)
    기상청 공식 변환 공식 (Lambert Conformal Conic)
    """
    RE = 6371.00877   # 지구 반경 (km)
    GRID = 5.0        # 격자 간격 (km)
    SLAT1 = 30.0      # 표준위도 1
    SLAT2 = 60.0      # 표준위도 2
    OLON  = 126.0     # 기준점 경도
    OLAT  = 38.0      # 기준점 위도
    XO    = 43        # 기준점 X 격자
    YO    = 136       # 기준점 Y 격자

    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon  = OLON  * DEGRAD
    olat  = OLAT  * DEGRAD

    sn = math.log(math.cos(slat1) / math.cos(slat2))
    sn /= math.log(math.tan(math.pi * 0.25 + slat2 * 0.5) /
                   math.tan(math.pi * 0.25 + slat1 * 0.5))
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = (sf ** sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro ** sn)

    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / (ra ** sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    nx = int(ra * math.sin(theta) + XO + 0.5)
    ny = int(ro - ra * math.cos(theta) + YO + 0.5)
    return nx, ny


# ══════════════════════════════════════════════════════════════════════════════
# API 호출
# ══════════════════════════════════════════════════════════════════════════════
def _get_base_time(now: datetime) -> Tuple[str, str]:
    """현재 시각 기준 가장 가까운 발표 시각 반환 (base_date, base_time)"""
    # 단기예보 발표: 02, 05, 08, 11, 14, 17, 20, 23시 (10분 후 공개)
    ANNOUNCE_HOURS = [2, 5, 8, 11, 14, 17, 20, 23]
    cur_h = now.hour
    cur_m = now.minute

    base_h = None
    for h in reversed(ANNOUNCE_HOURS):
        if cur_h > h or (cur_h == h and cur_m >= 10):
            base_h = h
            break
    if base_h is None:
        # 자정 이전 마지막 발표 시각 = 전날 23시
        now = now - timedelta(days=1)
        base_h = 23

    base_date = now.strftime("%Y%m%d")
    base_time = f"{base_h:02d}00"
    return base_date, base_time


def fetch_kma_forecast(lat: float, lon: float, api_key: Optional[str] = None) -> dict:
    """
    기상청 단기예보 API 호출 → 최근접 시각의 기상 정보 반환

    반환:
      {
        "pop": int,      강수확률 (0~100)
        "pty": str,      강수형태 ("없음"/"비"/"눈"/"소나기")
        "sky": str,      하늘상태 ("맑음"/"구름많음"/"흐림")
        "tmp": float,    기온 (°C)
        "wsd": float,    풍속 (m/s)
        "weather_label": str,  사람이 읽기 좋은 요약
      }
    """
    api_key = api_key or os.environ.get("KMA_API_KEY", "")
    if not api_key:
        raise ValueError(
            "KMA API 키가 없습니다. "
            "공공데이터포털(data.go.kr)에서 발급 후 "
            "환경변수 KMA_API_KEY에 설정하세요."
        )

    now = datetime.now()
    base_date, base_time = _get_base_time(now)
    nx, ny = latlon_to_grid(lat, lon)

    params = {
        "serviceKey": api_key,
        "pageNo":     "1",
        "numOfRows":  "100",
        "dataType":   "JSON",
        "base_date":  base_date,
        "base_time":  base_time,
        "nx":         str(nx),
        "ny":         str(ny),
    }

    resp = requests.get(KMA_BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    body = resp.json().get("response", {}).get("body", {})
    items = body.get("items", {}).get("item", [])

    # 가장 가까운 예보 시각 파싱
    result = {"pop": 0, "pty": "없음", "sky": "맑음", "tmp": 0.0, "wsd": 0.0}
    fcst_target = (now + timedelta(hours=1)).strftime("%Y%m%d%H00")

    for item in items:
        if item.get("fcstDate") + item.get("fcstTime") == fcst_target:
            cat = item.get("category", "")
            val = item.get("fcstValue", "0")
            if cat == "POP":
                result["pop"] = int(val)
            elif cat == "PTY":
                result["pty"] = PTY_MAP.get(str(val), "없음")
            elif cat == "SKY":
                result["sky"] = SKY_MAP.get(str(val), "맑음")
            elif cat == "TMP":
                try:
                    result["tmp"] = float(val)
                except ValueError:
                    pass
            elif cat == "WSD":
                try:
                    result["wsd"] = float(val)
                except ValueError:
                    pass

    # 요약 레이블
    if result["pty"] != "없음":
        result["weather_label"] = result["pty"]
    elif result["sky"] == "흐림":
        result["weather_label"] = "흐림"
    elif result["sky"] == "구름많음":
        result["weather_label"] = "구름많음"
    else:
        result["weather_label"] = "맑음"

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 안전 가중치 보정
# ══════════════════════════════════════════════════════════════════════════════
def get_weather_weight(
    lat: float,
    lon: float,
    api_key: Optional[str] = None,
    base_w_safety: float = 0.6,
) -> Tuple[float, float, dict]:
    """
    날씨 기반 안전/관광 가중치 자동 보정

    반환:
      (w_safety_adj, w_tourism_adj, weather_info)

    보정 규칙:
      - 강수확률 0%   → w_safety = base (0.60)
      - 강수확률 50%  → w_safety = base + 0.10
      - 강수확률 100% → w_safety = base + 0.20 (최대 0.80)
      - 강수형태 비/눈 → 추가 +0.05
      - 풍속 10m/s 이상 → 추가 +0.05
    """
    weather = fetch_kma_forecast(lat, lon, api_key)

    pop_adj = (weather["pop"] / 100.0) * 0.20      # 최대 +0.20
    pty_adj = 0.05 if weather["pty"] != "없음" else 0.0
    wsd_adj = 0.05 if weather["wsd"] >= 10.0 else 0.0

    w_safety  = min(base_w_safety + pop_adj + pty_adj + wsd_adj, 0.80)
    w_safety  = round(w_safety, 2)
    w_tourism = round(1.0 - w_safety, 2)

    weather["w_safety_adj"]  = w_safety
    weather["w_tourism_adj"] = w_tourism

    return w_safety, w_tourism, weather


def get_current_weather(lat: float, lon: float, api_key: Optional[str] = None) -> dict:
    """날씨 정보만 반환 (가중치 계산 없음)"""
    return fetch_kma_forecast(lat, lon, api_key)


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI 연동용 엔드포인트 헬퍼
# ══════════════════════════════════════════════════════════════════════════════
def weather_to_safety_penalty(weather_label: str) -> float:
    """
    예측 날씨 → safety_score 페널티 반환
    Phase 3-8 WeatherLSTM 출력과 동일한 인터페이스

    맑음  → 0.0
    흐림  → -0.05
    구름  → -0.03
    비/눈 → -0.15 ~ -0.30
    """
    PENALTY = {
        "맑음":   0.0,
        "구름많음": -0.03,
        "흐림":   -0.05,
        "소나기": -0.15,
        "비":     -0.20,
        "비/눈":  -0.25,
        "눈":     -0.30,
    }
    return PENALTY.get(weather_label, 0.0)
