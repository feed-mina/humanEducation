"""
K-Ride 2.0 -- 두루누비 둘레길 데이터 수집기
===========================================
Phase 1: 두루누비 공공 API (routeList / courseList) 호출
Phase 2: durunubi.kr 웹사이트에서 이미지 크롤링
Phase 3: 데이터 병합 & JSON/CSV 저장
"""

import os
import sys
import time
import json
import csv
import re
import requests
from pathlib import Path
from urllib.parse import urlencode, quote_plus
from datetime import datetime

# ── Config ──────────────────────────────────────────────
SERVICE_KEY = "1982f962b3451ad1a449051bf6266ac560540e346678c51933ae885bc2b4a95e"
BASE_API = "https://apis.data.go.kr/B551011/Durunubi"
WEBSITE_URL = "https://durunubi.kr"

OUTPUT_DIR = Path(r"e:/krider/kride-project/data/durunubi")
IMAGE_DIR = OUTPUT_DIR / "images"

TIMEOUT = 20
DELAY = 1.0  # API 호출 간 딜레이(초)
PAGE_SIZE = 100  # 한 페이지당 결과 수

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
}


# ═══════════════════════════════════════════════════════
#  Phase 1: 두루누비 API 데이터 수집
# ═══════════════════════════════════════════════════════

def call_api(endpoint: str, extra_params: dict = None) -> list:
    """
    두루누비 API 엔드포인트를 페이지네이션하며 전체 데이터 수집.
    """
    all_items = []
    page = 1

    while True:
        params = {
            "serviceKey": SERVICE_KEY,
            "numOfRows": PAGE_SIZE,
            "pageNo": page,
            "MobileOS": "ETC",
            "MobileApp": "KRide",
            "brdDiv": "DNWW",
            "_type": "json",
        }
        if extra_params:
            params.update(extra_params)

        url = f"{BASE_API}/{endpoint}?{urlencode(params, quote_via=quote_plus)}"
        print(f"  [API] {endpoint} page={page} ...", end=" ")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"ERROR: {e}")
            # Decoding 키로 재시도
            if "serviceKey" not in str(e):
                print("  → Decoding 키로 재시도...")
                params["serviceKey"] = requests.utils.unquote(SERVICE_KEY)
                url = f"{BASE_API}/{endpoint}?{urlencode(params)}"
                try:
                    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
                    resp.raise_for_status()
                except Exception as e2:
                    print(f"  → 재시도 실패: {e2}")
                    break
            else:
                break

        try:
            data = resp.json()
        except json.JSONDecodeError:
            # XML 응답일 수 있음
            print(f"JSON 파싱 실패 (응답: {resp.text[:200]})")
            break

        # 응답 구조 파싱
        body = data.get("response", {}).get("body", {})
        total_count = body.get("totalCount", 0)
        items_wrapper = body.get("items", {})

        if not items_wrapper:
            print(f"데이터 없음 (totalCount={total_count})")
            break

        items = items_wrapper.get("item", [])
        if isinstance(items, dict):
            items = [items]

        all_items.extend(items)
        print(f"OK ({len(items)}건, 누적 {len(all_items)}/{total_count})")

        if len(all_items) >= total_count:
            break

        page += 1
        time.sleep(DELAY)

    return all_items


def fetch_route_list() -> list:
    """길 목록 정보 조회 (/routeList)"""
    print("\n[Phase 1-A] 길 목록(routeList) 수집")
    return call_api("routeList")


def fetch_course_list(route_id: str = None) -> list:
    """코스 목록 정보 조회 (/courseList)"""
    print("\n[Phase 1-B] 코스 목록(courseList) 수집")
    extra = {}
    if route_id:
        extra["routeIdx"] = route_id
    return call_api("courseList", extra)


# ═══════════════════════════════════════════════════════
#  Phase 2: 두루누비 웹사이트 이미지 크롤링
# ═══════════════════════════════════════════════════════

def crawl_trail_pages() -> list:
    """
    durunubi.kr/road-walk.do 에서 걷기 여행길 목록 + 이미지 크롤링.
    JavaScript 기반 페이지네이션이므로 AJAX endpoint를 추적.
    """
    print("\n[Phase 2] 두루누비 웹사이트 이미지 크롤링")
    trail_data = []

    # 메인 페이지에서 총 페이지 수 확인
    for page_no in range(1, 60):  # 최대 60페이지 시도 (540개 / 12개씩)
        url = f"{WEBSITE_URL}/road-walk.do"
        params = {"pageNo": page_no}

        print(f"  [WEB] 페이지 {page_no} ...", end=" ")
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}")
                break
        except requests.RequestException as e:
            print(f"ERROR: {e}")
            break

        html = resp.text

        # 걷기 여행길 카드 파싱 (img + title)
        # 패턴: <div class="img-box"> ... <img src="..."> ... <p class="tit">제목</p>
        card_pattern = re.compile(
            r'<(?:div|a)[^>]*class="[^"]*(?:card|item|img-box|road-item)[^"]*"[^>]*>'
            r'.*?<img[^>]+src=["\']([^"\']+)["\'][^>]*>.*?'
            r'(?:<p[^>]*class="[^"]*tit[^"]*"[^>]*>|<strong[^>]*>|<h[2-6][^>]*>)\s*([^<]+)',
            re.DOTALL | re.IGNORECASE,
        )

        # 더 일반적인 패턴도 시도
        img_pattern = re.compile(
            r'<img[^>]+src=["\'](/upload/[^"\']+)["\'][^>]*>',
            re.IGNORECASE,
        )
        title_pattern = re.compile(
            r'<(?:p|strong|span|h\d)[^>]*class="[^"]*(?:tit|name|title)[^"]*"[^>]*>\s*([^<]+)',
            re.IGNORECASE,
        )

        # roadDetailView 링크에서 ID 추출
        detail_pattern = re.compile(
            r"roadDetailView\(['\"]?(\d+)['\"]?\)",
            re.IGNORECASE,
        )

        # 리스트형 패턴: li 안에 img + 제목
        list_pattern = re.compile(
            r'<li[^>]*>.*?<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
            r'.*?(?:<[^>]*class="[^"]*(?:tit|name)[^"]*"[^>]*>|<strong[^>]*>)\s*([^<]+)',
            re.DOTALL | re.IGNORECASE,
        )

        cards = card_pattern.findall(html)
        if not cards:
            cards = list_pattern.findall(html)

        images = img_pattern.findall(html)
        titles = title_pattern.findall(html)
        detail_ids = detail_pattern.findall(html)

        if cards:
            for img_src, title in cards:
                title = title.strip()
                if not title or len(title) < 2:
                    continue
                if not img_src.startswith("http"):
                    img_src = WEBSITE_URL + img_src
                trail_data.append({
                    "title": title,
                    "image_url": img_src,
                    "source_page": page_no,
                })
        elif images and titles:
            # 이미지와 제목을 짝으로 매칭
            for i, (img, title) in enumerate(zip(images, titles)):
                title = title.strip()
                if not title or len(title) < 2:
                    continue
                img_url = WEBSITE_URL + img if not img.startswith("http") else img
                trail_data.append({
                    "title": title,
                    "image_url": img_url,
                    "source_page": page_no,
                    "detail_id": detail_ids[i] if i < len(detail_ids) else None,
                })

        count = len(cards) if cards else min(len(images), len(titles))
        print(f"발견 {count}건 (누적 {len(trail_data)})")

        if count == 0:
            # 데이터 없으면 마지막 페이지
            break

        time.sleep(DELAY)

    return trail_data


def download_trail_images(trail_data: list):
    """크롤링한 이미지 URL에서 실제 이미지 다운로드."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[Phase 2-B] 이미지 다운로드 ({len(trail_data)}건)")

    downloaded = 0
    for i, trail in enumerate(trail_data):
        img_url = trail.get("image_url", "")
        if not img_url:
            continue

        # 파일명 생성
        safe_name = re.sub(r'[^\w가-힣]', '_', trail.get("title", f"trail_{i}"))
        safe_name = safe_name[:50]  # 길이 제한
        ext = ".jpg"
        if ".png" in img_url.lower():
            ext = ".png"
        elif ".webp" in img_url.lower():
            ext = ".webp"

        filepath = IMAGE_DIR / f"{safe_name}{ext}"
        if filepath.exists():
            trail["local_image"] = str(filepath)
            downloaded += 1
            continue

        try:
            resp = requests.get(img_url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            if resp.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                size_kb = filepath.stat().st_size / 1024
                if size_kb < 1:
                    filepath.unlink()
                    continue
                trail["local_image"] = str(filepath)
                downloaded += 1
                if downloaded % 20 == 0:
                    print(f"  ... {downloaded}건 다운로드 완료")
        except Exception:
            pass

        time.sleep(0.3)

    print(f"  이미지 다운로드 완료: {downloaded}/{len(trail_data)}")


# ═══════════════════════════════════════════════════════
#  Phase 3: 데이터 병합 & 저장
# ═══════════════════════════════════════════════════════

def merge_and_save(routes: list, courses: list, web_trails: list):
    """API 데이터 + 웹 크롤링 데이터를 병합하여 저장."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) 길 목록 저장
    if routes:
        route_path = OUTPUT_DIR / f"routes_{timestamp}.json"
        with open(route_path, "w", encoding="utf-8") as f:
            json.dump(routes, f, ensure_ascii=False, indent=2)
        print(f"\n  [저장] 길 목록: {route_path} ({len(routes)}건)")

        # CSV 변환
        csv_path = OUTPUT_DIR / f"routes_{timestamp}.csv"
        if routes:
            keys = routes[0].keys()
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(routes)
            print(f"  [저장] 길 목록 CSV: {csv_path}")

    # 2) 코스 목록 저장
    if courses:
        course_path = OUTPUT_DIR / f"courses_{timestamp}.json"
        with open(course_path, "w", encoding="utf-8") as f:
            json.dump(courses, f, ensure_ascii=False, indent=2)
        print(f"  [저장] 코스 목록: {course_path} ({len(courses)}건)")

        csv_path = OUTPUT_DIR / f"courses_{timestamp}.csv"
        if courses:
            keys = courses[0].keys()
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(courses)
            print(f"  [저장] 코스 목록 CSV: {csv_path}")

    # 3) 웹 크롤링 이미지 데이터
    if web_trails:
        web_path = OUTPUT_DIR / f"web_trails_{timestamp}.json"
        with open(web_path, "w", encoding="utf-8") as f:
            json.dump(web_trails, f, ensure_ascii=False, indent=2)
        print(f"  [저장] 웹 크롤링 데이터: {web_path} ({len(web_trails)}건)")

    # 4) 통합 요약 저장
    summary = {
        "collected_at": timestamp,
        "api_routes_count": len(routes),
        "api_courses_count": len(courses),
        "web_trails_count": len(web_trails),
        "api_endpoint": BASE_API,
        "website": WEBSITE_URL,
        "routes_sample": routes[:3] if routes else [],
        "courses_sample": courses[:3] if courses else [],
    }
    summary_path = OUTPUT_DIR / f"collection_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  [저장] 수집 요약: {summary_path}")


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  K-Ride 2.0 -- 두루누비 둘레길 데이터 수집기")
    print(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Phase 1: API 데이터
    routes = fetch_route_list()
    print(f"  → 길 목록 총 {len(routes)}건 수집 완료")

    courses = fetch_course_list()
    print(f"  → 코스 목록 총 {len(courses)}건 수집 완료")

    # Phase 2: 웹 크롤링
    web_trails = crawl_trail_pages()
    print(f"  → 웹 크롤링 총 {len(web_trails)}건 수집 완료")

    if web_trails:
        download_trail_images(web_trails)

    # Phase 3: 저장
    print("\n[Phase 3] 데이터 저장")
    merge_and_save(routes, courses, web_trails)

    print(f"\n{'=' * 60}")
    print(f"  수집 완료!")
    print(f"  - API 길 목록: {len(routes)}건")
    print(f"  - API 코스 목록: {len(courses)}건")
    print(f"  - 웹 이미지: {len(web_trails)}건")
    print(f"  - 저장 경로: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
