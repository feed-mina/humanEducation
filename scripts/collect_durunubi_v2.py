"""
K-Ride 2.0 -- 두루누비 둘레길 데이터 수집기 v2
================================================
Phase 1: API - routeList + courseList (재시도 + 딜레이 강화)
Phase 2: Selenium으로 웹사이트 이미지 크롤링
Phase 3: 데이터 병합 & 저장
"""

import os
import sys
import time
import json
import csv
import re
import requests
from pathlib import Path
from urllib.parse import urlencode, quote_plus, unquote
from datetime import datetime

# ── Config ──────────────────────────────────────────────
SERVICE_KEY_ENCODED = "1982f962b3451ad1a449051bf6266ac560540e346678c51933ae885bc2b4a95e"
SERVICE_KEY_DECODED = unquote(SERVICE_KEY_ENCODED)
BASE_API = "https://apis.data.go.kr/B551011/Durunubi"
WEBSITE_URL = "https://durunubi.kr"

OUTPUT_DIR = Path(r"e:/krider/kride-project/data/durunubi")
IMAGE_DIR = OUTPUT_DIR / "images"

TIMEOUT = 30
API_DELAY = 2.0  # API 호출 간 딜레이(초) - 403 방지
PAGE_SIZE = 50    # 페이지당 결과 수 줄임

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


# ═══════════════════════════════════════════════════════
#  Phase 1: 두루누비 API 데이터 수집 (재시도 + 딜레이 강화)
# ═══════════════════════════════════════════════════════

def call_api(endpoint: str, extra_params: dict = None, max_retries: int = 3) -> list:
    """두루누비 API 엔드포인트를 페이지네이션하며 전체 데이터 수집."""
    all_items = []
    page = 1

    while True:
        params = {
            "serviceKey": SERVICE_KEY_ENCODED,
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

        success = False
        for attempt in range(max_retries):
            print(f"  [API] {endpoint} page={page} (attempt {attempt+1}) ...", end=" ")
            try:
                resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
                if resp.status_code == 200:
                    success = True
                    break
                elif resp.status_code == 403:
                    wait = (attempt + 1) * 5
                    print(f"403 → {wait}초 대기 후 재시도")
                    time.sleep(wait)
                else:
                    print(f"HTTP {resp.status_code}")
                    time.sleep(3)
            except requests.RequestException as e:
                print(f"ERROR: {e}")
                time.sleep(3)

        if not success:
            print(f"  [FAIL] {endpoint} page={page} 최종 실패")
            break

        try:
            data = resp.json()
        except json.JSONDecodeError:
            print(f"JSON 파싱 실패")
            break

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
        time.sleep(API_DELAY)

    return all_items


# ═══════════════════════════════════════════════════════
#  Phase 2: Selenium 기반 이미지 크롤링
# ═══════════════════════════════════════════════════════

def crawl_images_with_selenium():
    """
    Selenium으로 durunubi.kr/road-walk.do 의 둘레길 이미지 크롤링.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        print("  [WARN] selenium 미설치 → pip install selenium")
        return []

    print("\n[Phase 2] Selenium 이미지 크롤링")

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"user-agent={HEADERS['User-Agent']}")

    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"  [ERROR] Chrome 드라이버 실행 실패: {e}")
        print("  → pip install selenium webdriver-manager")
        return []

    all_trails = []
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        driver.get(f"{WEBSITE_URL}/road-walk.do")
        time.sleep(5)  # 초기 로딩 대기

        page_no = 1
        max_pages = 55  # 540 / 12 = 45 페이지

        while page_no <= max_pages:
            print(f"  [WEB] 페이지 {page_no} ...", end=" ")

            # 카드 요소 찾기 - 여러 셀렉터 시도
            cards = []
            for selector in [
                "div.road-item", "div.item", "li.item",
                "div.card", "div.walk-item", "a.road-item",
                "div.list-item", "ul.road-list > li",
                "div.content-list > div", "div.search-list > div",
            ]:
                cards = driver.find_elements(By.CSS_SELECTOR, selector)
                if cards:
                    break

            # 이미지 요소 직접 탐색
            if not cards:
                cards = driver.find_elements(By.CSS_SELECTOR, "div.img-box, div.thumb, div.thumbnail")

            if not cards:
                # 전체 이미지 중 컨텐츠 이미지만 필터링
                all_imgs = driver.find_elements(By.TAG_NAME, "img")
                content_imgs = []
                for img in all_imgs:
                    src = img.get_attribute("src") or ""
                    alt = img.get_attribute("alt") or ""
                    if "/upload/" in src or "/data/" in src:
                        content_imgs.append(img)
                    elif alt and len(alt) > 2 and "logo" not in alt.lower():
                        content_imgs.append(img)

                for img in content_imgs:
                    src = img.get_attribute("src") or ""
                    alt = img.get_attribute("alt") or ""
                    if src and alt:
                        all_trails.append({
                            "title": alt.strip(),
                            "image_url": src,
                            "source_page": page_no,
                        })
            else:
                for card in cards:
                    try:
                        img_el = card.find_element(By.TAG_NAME, "img")
                        src = img_el.get_attribute("src") or ""
                        alt = img_el.get_attribute("alt") or ""

                        # 제목 추출
                        title = alt
                        for tag in ["p", "strong", "span", "h3", "h4"]:
                            try:
                                title_el = card.find_element(By.TAG_NAME, tag)
                                t = title_el.text.strip()
                                if t and len(t) > 1:
                                    title = t
                                    break
                            except Exception:
                                continue

                        if src and title:
                            all_trails.append({
                                "title": title,
                                "image_url": src,
                                "source_page": page_no,
                            })
                    except Exception:
                        continue

            count = len(all_trails) - sum(1 for t in all_trails if t["source_page"] < page_no)
            print(f"발견 {count}건 (누적 {len(all_trails)})")

            if count == 0 and page_no > 1:
                break

            # 다음 페이지 이동
            page_no += 1
            try:
                # 페이지 번호 클릭
                page_links = driver.find_elements(By.CSS_SELECTOR, "a.page-link, a.paging, div.paging a, ul.pagination a")
                clicked = False
                for link in page_links:
                    text = link.text.strip()
                    onclick = link.get_attribute("onclick") or ""
                    if text == str(page_no) or f"({page_no})" in onclick:
                        driver.execute_script("arguments[0].click();", link)
                        clicked = True
                        time.sleep(3)
                        break

                if not clicked:
                    # "다음" 버튼
                    for link in page_links:
                        text = link.text.strip()
                        if "다음" in text or "next" in text.lower() or ">" == text:
                            driver.execute_script("arguments[0].click();", link)
                            clicked = True
                            time.sleep(3)
                            break

                if not clicked:
                    # JavaScript 직접 호출
                    driver.execute_script(f"if(typeof goPage === 'function') goPage({page_no});")
                    time.sleep(3)

            except Exception as e:
                print(f"  [페이지 이동 실패] {e}")
                break

            time.sleep(1)

    except Exception as e:
        print(f"  [ERROR] {e}")
    finally:
        driver.quit()

    return all_trails


def crawl_images_fallback():
    """
    Selenium 없이 requests로 상세 페이지에서 이미지 크롤링 시도.
    API에서 받은 routeIdx를 활용하여 상세 페이지 접근.
    """
    print("\n[Phase 2-B] Requests 기반 이미지 크롤링 (Fallback)")

    # 기존 수집된 routes 로드
    routes_files = sorted(OUTPUT_DIR.glob("routes_*.json"), reverse=True)
    if not routes_files:
        print("  [WARN] routes 데이터 없음, 스킵")
        return []

    routes = json.loads(routes_files[0].read_text(encoding="utf-8"))
    trail_images = []

    for route in routes:
        route_idx = route.get("routeIdx", "")
        theme_nm = route.get("themeNm", "")
        if not route_idx:
            continue

        # 상세 페이지 접근
        detail_url = f"{WEBSITE_URL}/4-1-1-walk-Road-view.do?theme_mng={route_idx}"
        print(f"  [{theme_nm}] {detail_url} ...", end=" ")

        try:
            resp = requests.get(detail_url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200:
                # editImgUp.do 패턴 찾기
                img_patterns = re.findall(
                    r'(?:src|data-src|background-image)[=:]\s*["\']?'
                    r'((?:https?://)?[^\s"\'<>]+(?:\.jpg|\.jpeg|\.png|\.webp|editImgUp[^\s"\'<>]+))',
                    resp.text, re.I
                )
                for img_url in img_patterns:
                    if not img_url.startswith("http"):
                        img_url = WEBSITE_URL + "/" + img_url.lstrip("/")
                    trail_images.append({
                        "title": theme_nm,
                        "image_url": img_url,
                        "route_idx": route_idx,
                    })
                print(f"이미지 {len(img_patterns)}개")
            else:
                print(f"HTTP {resp.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(1)

    return trail_images


def download_images(trail_data: list):
    """이미지 다운로드."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[이미지 다운로드] {len(trail_data)}건")

    downloaded = 0
    seen_urls = set()

    for i, trail in enumerate(trail_data):
        img_url = trail.get("image_url", "")
        if not img_url or img_url in seen_urls:
            continue
        seen_urls.add(img_url)

        # 파일명 생성
        title = trail.get("title", f"trail_{i}")
        safe_name = re.sub(r'[^\w가-힣]', '_', title)
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')[:50]
        if not safe_name:
            safe_name = f"trail_{i}"

        ext = ".jpg"
        for e in [".png", ".webp", ".gif"]:
            if e in img_url.lower():
                ext = e
                break

        filepath = IMAGE_DIR / f"{safe_name}{ext}"

        # 중복 방지 - 이미 같은 이름이 있으면 넘버링
        counter = 1
        while filepath.exists():
            filepath = IMAGE_DIR / f"{safe_name}_{counter}{ext}"
            counter += 1

        try:
            resp = requests.get(img_url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            if resp.status_code == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "image" not in content_type and "octet" not in content_type:
                    continue
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
                    print(f"  ... {downloaded}건 완료")
        except Exception:
            pass

        time.sleep(0.2)

    print(f"  이미지 다운로드 완료: {downloaded}건")
    return downloaded


# ═══════════════════════════════════════════════════════
#  Phase 3: 데이터 저장
# ═══════════════════════════════════════════════════════

def save_data(routes, courses, web_trails):
    """모든 데이터를 JSON/CSV로 저장."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_files = []

    # 길 목록
    if routes:
        p = OUTPUT_DIR / f"routes_{ts}.json"
        p.write_text(json.dumps(routes, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_files.append((p, len(routes), "길 목록"))

        cp = OUTPUT_DIR / f"routes_{ts}.csv"
        with open(cp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=routes[0].keys())
            w.writeheader()
            w.writerows(routes)
        saved_files.append((cp, len(routes), "길 목록 CSV"))

    # 코스 목록
    if courses:
        p = OUTPUT_DIR / f"courses_{ts}.json"
        p.write_text(json.dumps(courses, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_files.append((p, len(courses), "코스 목록"))

        cp = OUTPUT_DIR / f"courses_{ts}.csv"
        with open(cp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=courses[0].keys())
            w.writeheader()
            w.writerows(courses)
        saved_files.append((cp, len(courses), "코스 목록 CSV"))

    # 웹 크롤링 이미지
    if web_trails:
        p = OUTPUT_DIR / f"web_trails_{ts}.json"
        p.write_text(json.dumps(web_trails, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_files.append((p, len(web_trails), "웹 크롤링"))

    # 요약
    summary = {
        "collected_at": ts,
        "api_routes": len(routes),
        "api_courses": len(courses),
        "web_images": len(web_trails),
        "files": [{"path": str(p), "count": c, "type": t} for p, c, t in saved_files],
    }
    sp = OUTPUT_DIR / f"summary_{ts}.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[저장 완료]")
    for p, c, t in saved_files:
        print(f"  {t}: {p.name} ({c}건)")
    print(f"  요약: {sp.name}")


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  K-Ride 2.0 -- 두루누비 둘레길 데이터 수집기 v2")
    print(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Phase 1: API
    print("\n[Phase 1-A] 길 목록(routeList) 수집")
    routes = call_api("routeList")
    print(f"  → 길 목록: {len(routes)}건")

    print("\n[Phase 1-B] 코스 목록(courseList) 수집")
    courses = call_api("courseList")
    print(f"  → 코스 목록: {len(courses)}건")

    # Phase 2: 이미지 크롤링
    web_trails = []

    # Selenium 시도
    web_trails = crawl_images_with_selenium()

    # Selenium 실패 시 fallback
    if not web_trails:
        web_trails = crawl_images_fallback()

    # 이미지 다운로드
    if web_trails:
        download_images(web_trails)

    # Phase 3: 저장
    save_data(routes, courses, web_trails)

    print(f"\n{'=' * 60}")
    print(f"  수집 완료!")
    print(f"  - API 길 목록: {len(routes)}건")
    print(f"  - API 코스 목록: {len(courses)}건")
    print(f"  - 웹 이미지: {len(web_trails)}건")
    print(f"  - 저장 경로: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
