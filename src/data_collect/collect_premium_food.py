"""
collect_premium_food.py
=======================
프리미엄 맛집 POI 수집 — 또간집(나무위키) + 블루리본 + 레드리본 + 지오코딩

[ 지오코딩 전략 — 우선순위 순 ]
  1. Kakao Local API keyword  ← 식당명 검색에 가장 강력 (앱 Local API 활성화 필요)
  2. Vworld 명칭(place) 검색
  3. JUSO 주소 검색 → Vworld 좌표 변환
  4. Vworld 도로명/지번 주소 검색
  5. geopy Nominatim (OpenStreetMap) ← API 키 불필요, 마지막 폴백

[ 실행 방법 ]
  pip install requests beautifulsoup4 pandas python-dotenv geopy
  python kride-project/src/data_collect/collect_premium_food.py

[ .env 설정 ]
  VWORLD_API_KEY=...
  KAKAO_REST_API_KEY=...   ← https://developers.kakao.com 에서 발급 (무료)

[ 출력 ]
  data/raw_ml/premium_food_geocoded.csv
    컬럼: name, search_query, sub_category, address_raw, lat, lon
"""

from __future__ import annotations

import json
import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    _GEOPY_AVAILABLE = True
except ImportError:
    _GEOPY_AVAILABLE = False

_nominatim = Nominatim(user_agent="kride-project-geocoder", timeout=5) if _GEOPY_AVAILABLE else None

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()

RAW_ML_DIR  = os.path.join(BASE_DIR, "data", "raw_ml")
OUTPUT_PATH = os.path.join(RAW_ML_DIR, "premium_food_geocoded.csv")
os.makedirs(RAW_ML_DIR, exist_ok=True)

# ── API 키 로드 ──────────────────────────────────────────────────────────────
def _load_dotenv() -> None:
    for candidate in [os.path.join(BASE_DIR, ".env"), os.path.join(os.getcwd(), ".env")]:
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            break

_load_dotenv()
VWORLD_API_KEY    = os.environ.get("VWORLD_API_KEY", "")
KAKAO_API_KEY     = os.environ.get("KAKAO_REST_API_KEY", "")
JUSO_CONFIRM_KEY  = os.environ.get("JUSO_CONFIRM_KEY", "")
NAVER_CLIENT_ID   = os.environ.get("NAVER_CLIENT_ID", "")       # developers.naver.com (구 방식)
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "")
NCP_CLIENT_ID     = os.environ.get("NCP_CLIENT_ID", "")         # Naver Cloud Platform Maps
NCP_CLIENT_SECRET = os.environ.get("NCP_CLIENT_SECRET", "")

HEADERS_BROWSER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. 나무위키 '또간집' 파싱
# ═════════════════════════════════════════════════════════════════════════════

def get_ttoganjib(region_col: int = 1, name_col: int = 2) -> list[dict]:
    """나무위키 또간집 표에서 식당명 + 지역 추출."""
    print("\n📺 나무위키 '또간집' 파싱 중...")
    url = "https://namu.wiki/w/%EB%98%90%EA%B0%84%EC%A7%91"
    try:
        res = requests.get(url, headers=HEADERS_BROWSER, timeout=15)
        res.raise_for_status()
    except Exception as e:
        print(f"  ⚠️  나무위키 요청 실패: {e}")
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    results: list[dict] = []
    seen: set[str] = set()

    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) <= max(region_col, name_col):
                continue
            name   = re.sub(r"\[.*?\]", "", cols[name_col].get_text(strip=True)).strip()
            region = re.sub(r"\[.*?\]", "", cols[region_col].get_text(strip=True)).strip()
            if not name or len(name) < 2 or name in seen:
                continue
            seen.add(name)
            results.append({
                "name":         name,
                "search_query": f"{region} {name}",
                "sub_category": "또간집",
                "address_raw":  region,
            })

    print(f"  → 추출 완료: {len(results)}건")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 2. 블루리본 / 레드리본 수집
# ═════════════════════════════════════════════════════════════════════════════

def _parse_bluer_next_data(html: str, label: str) -> list[dict]:
    """Next.js __NEXT_DATA__ JSON 블록에서 식당 리스트 추출."""
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.S)
    if not match:
        return []
    try:
        data  = json.loads(match.group(1))
        # Next.js 페이지 props 경로 탐색 (구조에 따라 달라질 수 있음)
        page_props = data.get("props", {}).get("pageProps", {})
        candidates = (
            page_props.get("restaurants")
            or page_props.get("data", {}).get("content")
            or page_props.get("items")
            or []
        )
        results = []
        for item in candidates:
            name    = item.get("name") or item.get("title") or ""
            address = item.get("address") or item.get("addr") or ""
            if name:
                results.append({
                    "name":         name,
                    "search_query": address or name,
                    "sub_category": label,
                    "address_raw":  address,
                })
        return results
    except Exception:
        return []


def get_blueribbon(is_red: bool = False, max_pages: int = 10) -> list[dict]:
    """
    블루리본 / 레드리본 수집.
    우선순위: ① REST API 시도 → ② __NEXT_DATA__ JSON 파싱 → ③ HTML 셀렉터
    사이트가 완전히 SPA(JS 렌더링)면 Selenium 없이 0건. 이 경우 수동 CSV를 사용.
    """
    label    = "레드리본" if is_red else "블루리본"
    base_url = "https://www.bluer.co.kr"
    print(f"\n{'🎀' if is_red else '💙'} {label} 수집 중...")
    results: list[dict] = []

    # ── ① REST API 엔드포인트 시도 ──────────────────────────────────────────
    api_candidates = [
        f"{base_url}/api/v1/restaurants",
        f"{base_url}/api/restaurants",
        f"{base_url}/api/v2/restaurants",
    ]
    for api_url in api_candidates:
        try:
            params = {"page": 0, "size": 50, "ribbon": "red" if is_red else "blue"}
            resp   = requests.get(api_url, params=params,
                                  headers={**HEADERS_BROWSER, "Accept": "application/json"},
                                  timeout=8)
            if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("application/json"):
                raw = resp.json()
                for item in (
                    raw.get("content") or raw.get("data", {}).get("content")
                    or raw.get("_embedded", {}).get("restaurants") or []
                ):
                    name    = item.get("name") or item.get("title") or ""
                    address = item.get("address") or item.get("addr") or ""
                    if name:
                        results.append({
                            "name":         name,
                            "search_query": address or name,
                            "sub_category": label,
                            "address_raw":  address,
                        })
                if results:
                    print(f"  → REST API 성공: {len(results)}건")
                    return results
        except Exception:
            continue

    # ── ② __NEXT_DATA__ + HTML 셀렉터 (페이지네이션) ──────────────────────
    start_url = (
        f"{base_url}/about/redribbon" if is_red
        else f"{base_url}/search?ribbon=true"
    )
    for page in range(1, max_pages + 1):
        url = start_url if page == 1 else f"{start_url}&page={page}"
        try:
            res = requests.get(url, headers=HEADERS_BROWSER, timeout=10)
            if res.status_code != 200:
                break
        except Exception as e:
            print(f"  ⚠️  {label} 요청 실패 (page {page}): {e}")
            break

        # __NEXT_DATA__ 먼저 시도
        next_items = _parse_bluer_next_data(res.text, label)
        if next_items:
            results.extend(next_items)
            print(f"  page {page}: __NEXT_DATA__ {len(next_items)}건 추가")
            time.sleep(0.3)
            continue

        # 정적 HTML 셀렉터 폴백
        soup  = BeautifulSoup(res.text, "html.parser")
        items = (
            soup.select(".restaurant-item")
            or soup.select(".list-item")
            or soup.select("li[class*='restaurant']")
            or soup.select("div[class*='place']")
            or soup.select("article")
        )
        if not items:
            break  # SPA 렌더링 — 정적 HTML에 데이터 없음

        prev = len(results)
        for item in items:
            name_el = item.select_one(".name, [class*='name'], h3, h4, strong")
            addr_el = item.select_one(".address, [class*='addr'], p")
            name    = name_el.get_text(strip=True) if name_el else ""
            address = addr_el.get_text(strip=True) if addr_el else ""
            if name:
                results.append({
                    "name":         name,
                    "search_query": address or name,
                    "sub_category": label,
                    "address_raw":  address,
                })
        if len(results) == prev:
            break
        time.sleep(0.4)

    if not results:
        print(f"  ⚠️  {label}: JS 렌더링으로 0건 — Selenium 또는 수동 CSV 필요")
    print(f"  → {label} 추출: {len(results)}건")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 3. 지오코딩 — Kakao → Vworld 순서
# ═════════════════════════════════════════════════════════════════════════════

_kakao_error_logged = False  # 동일 오류 반복 출력 방지


def _kakao_keyword(query: str) -> tuple[float | None, float | None]:
    """
    Kakao Local API keyword 검색 — 식당명 기반 검색에 가장 강력.
    https://developers.kakao.com/docs/latest/ko/local/dev-guide#search-by-keyword

    앱 설정 주의:
      developers.kakao.com → 내 애플리케이션 → 앱 설정 → 플랫폼
      → Web 플랫폼 추가, 사이트 도메인에 http://localhost 등록 (Python 배치 사용 가능)
    """
    global _kakao_error_logged
    if not KAKAO_API_KEY or not query:
        return None, None
    try:
        resp = requests.get(
            "https://dapi.kakao.com/v2/local/search/keyword.json",
            headers={"Authorization": f"KakaoAK {KAKAO_API_KEY}"},
            params={"query": query, "size": 1},
            timeout=5,
        )
        if resp.status_code != 200:
            if not _kakao_error_logged:
                print(f"  ⚠️  Kakao API 오류 {resp.status_code}: {resp.text[:120]}")
                print("      → developers.kakao.com → 앱 설정 → 플랫폼 → Web 등록 필요")
                _kakao_error_logged = True
            return None, None
        docs = resp.json().get("documents", [])
        if docs:
            return float(docs[0]["y"]), float(docs[0]["x"])
    except Exception as e:
        if not _kakao_error_logged:
            print(f"  ⚠️  Kakao 예외: {e}")
            _kakao_error_logged = True
    return None, None


_ncp_error_logged = False


def _naver_ncp_geocode(query: str) -> tuple[float | None, float | None]:
    """
    Naver Cloud Platform Maps Geocoding API.
    헤더: X-NCP-APIGW-API-KEY-ID / X-NCP-APIGW-API-KEY
    응답: addresses[].x (경도), addresses[].y (위도)
    무료 3,000,000 req/월.
    """
    global _ncp_error_logged
    if not NCP_CLIENT_ID or not NCP_CLIENT_SECRET or not query:
        return None, None
    try:
        resp = requests.get(
            "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode",
            headers={
                "X-NCP-APIGW-API-KEY-ID": NCP_CLIENT_ID,
                "X-NCP-APIGW-API-KEY":    NCP_CLIENT_SECRET,
            },
            params={"query": query},
            timeout=5,
        )
        if resp.status_code != 200:
            if not _ncp_error_logged:
                print(f"  ⚠️  NCP Geocoding 오류 {resp.status_code}: {resp.text[:120]}")
                _ncp_error_logged = True
            return None, None
        addresses = resp.json().get("addresses", [])
        if addresses:
            lon = float(addresses[0]["x"])
            lat = float(addresses[0]["y"])
            if lat and lon:
                return lat, lon
    except Exception as e:
        if not _ncp_error_logged:
            print(f"  ⚠️  NCP Geocoding 예외: {e}")
            _ncp_error_logged = True
    return None, None


def _naver_local(query: str) -> tuple[float | None, float | None]:
    """
    Naver developers.naver.com 검색 API (구 방식) — 폴백용.
    응답: mapx(경도×1e7), mapy(위도×1e7)
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET or not query:
        return None, None
    try:
        resp = requests.get(
            "https://openapi.naver.com/v1/search/local.json",
            headers={
                "X-Naver-Client-Id":     NAVER_CLIENT_ID,
                "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
            },
            params={"query": query, "display": 1},
            timeout=5,
        )
        if resp.status_code != 200:
            return None, None
        items = resp.json().get("items", [])
        if items:
            lon = int(items[0]["mapx"]) / 1e7
            lat = int(items[0]["mapy"]) / 1e7
            if lat and lon:
                return lat, lon
    except Exception:
        pass
    return None, None


def _juso_keyword(query: str) -> tuple[float | None, float | None]:
    """
    JUSO 도로명주소 키워드 검색 API.
    주소 성분이 포함된 쿼리("서울 을지로3가 미진")에 효과적.
    순수 식당명 쿼리에는 효과 낮음 — Kakao 실패 후 보조 폴백으로 사용.
    """
    if not JUSO_CONFIRM_KEY or not query:
        return None, None
    try:
        res = requests.get(
            "https://www.juso.go.kr/addrlink/addrLinkApiJsonp.do",
            params={
                "confmKey":     JUSO_CONFIRM_KEY,
                "currentPage":  1,
                "countPerPage": 1,
                "keyword":      query,
                "resultType":   "json",
            },
            timeout=5,
        )
        data   = res.json()
        juso   = data.get("results", {}).get("juso", [])
        if juso:
            # JUSO는 좌표를 직접 반환하지 않음 → 도로명 주소로 Vworld 재검색
            addr = juso[0].get("roadAddr", "")
            if addr:
                return _vworld_addr(addr, "road")
    except Exception:
        pass
    return None, None


_vworld_error_logged = False  # 동일 오류 반복 출력 방지


def _vworld_check_error(resp_body: dict) -> bool:
    """
    Vworld 공통 오류 응답 처리.
    status == ERROR 이면 오류 코드/메시지 출력 후 True 반환.
    level 2·3 오류(INVALID_KEY, OVER_REQUEST_LIMIT 등)는 로그 1회 출력.
    """
    global _vworld_error_logged
    resp = resp_body.get("response", {})
    if resp.get("status") != "ERROR":
        return False
    err   = resp.get("error", {})
    code  = err.get("code", "UNKNOWN")
    level = err.get("level", 0)
    text  = err.get("text", "")
    if not _vworld_error_logged:
        print(f"  ⚠️  Vworld 오류 [{code}] lv{level}: {text}")
        if code in ("INVALID_KEY", "UNAVAILABLE_KEY"):
            print("      → vworld.kr 개발자 센터에서 키 확인 필요")
        elif code == "OVER_REQUEST_LIMIT":
            print("      → 일일 요청 한도 초과 — 내일 재실행 또는 키 교체 필요")
        if level >= 2:
            _vworld_error_logged = True  # 키/한도 문제면 반복 출력 억제
    return True


def _vworld_place(query: str) -> tuple[float | None, float | None]:
    """
    Vworld 명칭(place) 검색 API.
    응답: response.status = OK | NOT_FOUND | ERROR
    """
    if not VWORLD_API_KEY or not query:
        return None, None
    try:
        body = requests.get(
            "https://api.vworld.kr/req/search",
            params={
                "service":      "search",
                "request":      "search",
                "version":      "2.0",
                "crs":          "epsg:4326",
                "query":        query,
                "type":         "place",
                "format":       "json",
                "errorFormat":  "json",
                "key":          VWORLD_API_KEY,
            },
            timeout=5,
        ).json()
        if _vworld_check_error(body):
            return None, None
        items = body.get("response", {}).get("result", {}).get("items", [])
        if items:
            pt  = items[0].get("point", {})
            lat = float(pt.get("y", 0))
            lon = float(pt.get("x", 0))
            if lat and lon:
                return lat, lon
    except Exception:
        pass
    return None, None


def _vworld_addr(query: str, addr_type: str = "road") -> tuple[float | None, float | None]:
    """
    Vworld Geocoder API 2.0 — 주소 → 좌표 변환.
    addr_type: "road" (도로명) | "parcel" (지번)
    응답: response.status = OK | NOT_FOUND | ERROR
    """
    if not VWORLD_API_KEY or not query:
        return None, None
    try:
        body = requests.get(
            "https://api.vworld.kr/req/address",
            params={
                "service":  "address",
                "request":  "getCoord",        # 대소문자 레퍼런스 기준
                "version":  "2.0",
                "crs":      "epsg:4326",
                "address":  query,
                "refine":   "true",
                "simple":   "false",
                "format":   "json",
                "type":     addr_type.upper(), # ROAD | PARCEL
                "key":      VWORLD_API_KEY,
            },
            timeout=5,
        ).json()
        if _vworld_check_error(body):
            return None, None
        resp = body.get("response", {})
        if resp.get("status") == "OK":
            pt = resp["result"]["point"]
            return float(pt["y"]), float(pt["x"])
    except Exception:
        pass
    return None, None


def _geopy_nominatim(query: str) -> tuple[float | None, float | None]:
    """
    geopy Nominatim (OpenStreetMap) — API 키 불필요, 마지막 폴백.
    한국 식당 커버리지는 낮지만 일부 유명 장소는 찾아줌.
    요청 제한: 1 req/초 (time.sleep은 geocode() 호출부에서 통합 관리)
    """
    if not _GEOPY_AVAILABLE or not query:
        return None, None
    try:
        # "한국 서울 미진" 형식으로 country 힌트 추가
        loc = _nominatim.geocode(f"한국 {query}", language="ko")
        if loc:
            return loc.latitude, loc.longitude
    except (GeocoderTimedOut, GeocoderServiceError):
        pass
    except Exception:
        pass
    return None, None


def geocode(query: str) -> tuple[float | None, float | None]:
    """
    8단계 지오코딩 (성공률 높은 순):
      1. Kakao Local keyword        ← 앱 Local API 활성화 시 최강 (현재 비활성)
      2. NCP Maps Geocoding         ← Naver Cloud Platform, 300만/월 무료 ★현재 주력★
      3. Naver Search API (구 방식) ← developers.naver.com, 25K/일
      4. Vworld place 검색
      5. JUSO keyword → Vworld 재검색
      6. Vworld 도로명 주소
      7. Vworld 지번 주소
      8. geopy Nominatim (API 키 불필요)
    """
    lat, lon = _kakao_keyword(query)
    if lat:
        return lat, lon
    lat, lon = _naver_ncp_geocode(query)
    if lat:
        return lat, lon
    lat, lon = _naver_local(query)
    if lat:
        return lat, lon
    lat, lon = _vworld_place(query)
    if lat:
        return lat, lon
    lat, lon = _juso_keyword(query)
    if lat:
        return lat, lon
    lat, lon = _vworld_addr(query, "road")
    if lat:
        return lat, lon
    lat, lon = _vworld_addr(query, "parcel")
    if lat:
        return lat, lon
    lat, lon = _geopy_nominatim(query)
    return lat, lon


# ═════════════════════════════════════════════════════════════════════════════
# 메인
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("프리미엄 맛집 수집 — 또간집 + 블루리본 + 레드리본")
    print("=" * 60)

    # ── API 키 상태 확인 ──────────────────────────────────────────────────────
    if KAKAO_API_KEY:
        try:
            chk = requests.get(
                "https://dapi.kakao.com/v2/local/search/keyword.json",
                headers={"Authorization": f"KakaoAK {KAKAO_API_KEY}"},
                params={"query": "서울 미진", "size": 1},
                timeout=5,
            )
            if chk.status_code == 200:
                print("✅ Kakao Local API 정상 (1순위 지오코딩 활성)")
            else:
                print(f"⚠️  Kakao API 오류 {chk.status_code}: {chk.text[:100]}")
                print("   → developers.kakao.com → 앱 설정 → 플랫폼 → Web → http://localhost 등록")
        except Exception as e:
            print(f"⚠️  Kakao API 연결 실패: {e}")
    else:
        print("⚠️  KAKAO_REST_API_KEY 미설정 — .env에 추가하면 성공률 크게 향상")
    if VWORLD_API_KEY:
        try:
            chk  = requests.get(
                "https://api.vworld.kr/req/address",
                params={
                    "service": "address", "request": "getCoord", "version": "2.0",
                    "crs": "epsg:4326", "address": "서울특별시 중구 을지로 30",
                    "format": "json", "type": "ROAD", "key": VWORLD_API_KEY,
                },
                timeout=5,
            ).json()
            status = chk.get("response", {}).get("status", "")
            if status == "OK":
                print("✅ Vworld Geocoder API 정상 (2순위 지오코딩 활성)")
            elif status == "ERROR":
                err = chk.get("response", {}).get("error", {})
                print(f"⚠️  Vworld 오류 [{err.get('code')}]: {err.get('text')}")
            else:
                print(f"⚠️  Vworld 응답 이상: status={status}")
        except Exception as e:
            print(f"⚠️  Vworld 연결 실패: {e}")
    else:
        print("⚠️  VWORLD_API_KEY 미설정 — Vworld 지오코딩 비활성화")
    if NCP_CLIENT_ID and NCP_CLIENT_SECRET:
        try:
            chk = requests.get(
                "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode",
                headers={
                    "X-NCP-APIGW-API-KEY-ID": NCP_CLIENT_ID,
                    "X-NCP-APIGW-API-KEY":    NCP_CLIENT_SECRET,
                },
                params={"query": "서울특별시 중구 을지로 30"},
                timeout=5,
            )
            if chk.status_code == 200 and chk.json().get("addresses"):
                print("✅ NCP Maps Geocoding API 정상 (2순위 주력, 300만/월 무료)")
            elif chk.status_code == 200:
                print("✅ NCP Maps Geocoding 연결됨 (테스트 주소 결과 0건)")
            else:
                print(f"⚠️  NCP Geocoding 오류 {chk.status_code}: {chk.text[:80]}")
        except Exception as e:
            print(f"⚠️  NCP Geocoding 연결 실패: {e}")
    else:
        print("⚠️  NCP_CLIENT_ID / NCP_CLIENT_SECRET 미설정")
    if JUSO_CONFIRM_KEY:
        print("✅ JUSO API 활성 (5순위 폴백)")
    if _GEOPY_AVAILABLE:
        print("✅ geopy Nominatim 활성 (6순위 폴백, API 키 불필요)")
    else:
        print("⚠️  geopy 미설치 — pip install geopy")
    print()

    # 1. 수집
    data: list[dict] = []
    data += get_ttoganjib()
    data += get_blueribbon(is_red=False)
    data += get_blueribbon(is_red=True)

    if not data:
        print("\n❌ 수집된 데이터 없음")
        return

    df = pd.DataFrame(data).drop_duplicates(subset=["name"])
    print(f"\n중복 제거 후 총 {len(df)}건")
    print(df["sub_category"].value_counts().to_string())

    # 2. 지오코딩
    print(f"\n🌍 지오코딩 시작 ({len(df)}건) ...")
    lats, lons = [], []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        lat, lon = geocode(row["search_query"])
        lats.append(lat)
        lons.append(lon)
        if i % 20 == 0:
            ok = sum(1 for v in lats if v is not None)
            print(f"  진행 {i}/{len(df)} | 성공 {ok}건 ({ok/i*100:.1f}%)")
        time.sleep(0.08)  # Kakao API 호출 제한 대응

    df["lat"] = lats
    df["lon"] = lons

    # 3. 저장
    df_ok   = df[df["lat"].notna()]
    df_fail = df[df["lat"].isna()]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"✅ 수집 완료")
    print(f"   전체        : {len(df):,}건")
    print(f"   좌표 성공   : {len(df_ok):,}건 ({len(df_ok)/len(df)*100:.1f}%)")
    print(f"   좌표 실패   : {len(df_fail):,}건")
    print(f"   저장 위치   : {OUTPUT_PATH}")
    print(f"{'='*60}")

    if not KAKAO_API_KEY and len(df_fail) > 0:
        print("\n💡 KAKAO_REST_API_KEY 설정 후 재실행하면 성공률이 크게 오릅니다.")


if __name__ == "__main__":
    main()
