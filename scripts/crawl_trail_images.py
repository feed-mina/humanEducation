"""
두루누비 road-walk.do 이미지 크롤러 (Selenium)
- 540개 걷기 여행길의 이름 + 이미지 + 상세 ID 추출
- 이미지 다운로드
"""
import json
import time
import re
import requests
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

OUTPUT_DIR = Path(r"e:/krider/kride-project/data/durunubi")
IMAGE_DIR = OUTPUT_DIR / "images" / "trails"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
}


def setup_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    return webdriver.Chrome(options=options)


def extract_trails_from_page(driver):
    """현재 페이지에서 트레일 카드 데이터 추출."""
    trails = []

    # 여러 셀렉터 시도
    # 1) 모든 img 태그에서 /upload/ 경로 이미지 찾기
    all_imgs = driver.find_elements(By.TAG_NAME, "img")
    upload_imgs = []
    for img in all_imgs:
        src = img.get_attribute("src") or ""
        data_src = img.get_attribute("data-src") or ""
        actual_src = src or data_src
        if "/upload/" in actual_src or "editImgUp" in actual_src:
            upload_imgs.append(img)

    # 2) onclick에서 상세 ID 추출  
    all_links = driver.find_elements(By.CSS_SELECTOR, "a[onclick], div[onclick], li[onclick]")
    detail_ids = []
    for link in all_links:
        onclick = link.get_attribute("onclick") or ""
        match = re.search(r"(?:roadDetailView|fnView|goView)\(['\"]?([A-Z_0-9]+)['\"]?\)", onclick)
        if match:
            detail_ids.append(match.group(1))

    # 3) JavaScript로 직접 DOM 탐색
    js_result = driver.execute_script("""
        var results = [];
        // 방법 1: .road-list 관련 셀렉터
        var items = document.querySelectorAll('.list-cont li, .road-list li, .search-list li, .content-list > div, ul > li');
        items.forEach(function(item) {
            var img = item.querySelector('img');
            var titleEl = item.querySelector('.tit, .name, .title, strong, h3, h4, p');
            var link = item.querySelector('a');
            if (img && titleEl) {
                var src = img.getAttribute('src') || img.getAttribute('data-src') || '';
                var title = titleEl.textContent.trim();
                var onclick = '';
                if (link) onclick = link.getAttribute('onclick') || '';
                if (!onclick) onclick = item.getAttribute('onclick') || '';
                if (title && title.length > 1 && title.length < 50) {
                    results.push({
                        title: title,
                        image_url: src,
                        onclick: onclick,
                        alt: img.getAttribute('alt') || ''
                    });
                }
            }
        });
        
        // 방법 2: 이미지 기반 역탐색
        if (results.length === 0) {
            var imgs = document.querySelectorAll('img');
            imgs.forEach(function(img) {
                var src = img.getAttribute('src') || '';
                if (src.indexOf('/upload/') > -1 || src.indexOf('editImgUp') > -1) {
                    var parent = img.closest('li') || img.closest('div') || img.parentElement;
                    var titleEl = parent ? (parent.querySelector('.tit, .name, strong, p, h3, h4') || parent) : null;
                    var title = titleEl ? titleEl.textContent.trim().split('\\n')[0].trim() : img.getAttribute('alt') || '';
                    if (title && title.length > 1) {
                        results.push({
                            title: title,
                            image_url: src,
                            onclick: parent ? (parent.getAttribute('onclick') || '') : '',
                            alt: img.getAttribute('alt') || ''
                        });
                    }
                }
            });
        }
        
        return results;
    """)

    if js_result:
        for item in js_result:
            # 상세 ID 추출
            detail_id = ""
            onclick = item.get("onclick", "")
            match = re.search(r"['\"]([A-Z_]+MNG\d+)['\"]", onclick)
            if match:
                detail_id = match.group(1)

            img_url = item.get("image_url", "")
            if img_url and not img_url.startswith("http"):
                img_url = "https://durunubi.kr" + img_url

            trails.append({
                "title": item.get("title", ""),
                "image_url": img_url,
                "detail_id": detail_id,
                "alt": item.get("alt", ""),
            })

    # JS 결과가 없으면 페이지 소스에서 추출
    if not trails:
        page_source = driver.page_source

        # editImgUp 패턴 이미지 찾기
        img_matches = re.findall(
            r'<img[^>]+src=["\']([^"\']*(?:upload|editImgUp)[^"\']*)["\'][^>]*>',
            page_source, re.I
        )
        title_matches = re.findall(
            r'class="[^"]*(?:tit|name)[^"]*"[^>]*>\s*([^<]{2,50})',
            page_source, re.I
        )
        id_matches = re.findall(
            r'["\']([A-Z_]+MNG\d+)["\']',
            page_source
        )

        for i, img_url in enumerate(img_matches):
            if not img_url.startswith("http"):
                img_url = "https://durunubi.kr" + img_url
            trails.append({
                "title": title_matches[i] if i < len(title_matches) else f"trail_{i}",
                "image_url": img_url,
                "detail_id": id_matches[i] if i < len(id_matches) else "",
            })

    return trails


def navigate_to_page(driver, page_no):
    """자바스크립트로 페이지 이동."""
    try:
        # 여러 가능한 JS 함수 시도
        for fn_name in ["goPage", "fnGoPage", "changePage", "goList"]:
            try:
                driver.execute_script(f"if(typeof {fn_name} === 'function') {fn_name}({page_no});")
                time.sleep(3)
                return True
            except Exception:
                continue

        # 페이지 번호 클릭
        page_links = driver.find_elements(By.CSS_SELECTOR, "a")
        for link in page_links:
            text = link.text.strip()
            onclick = link.get_attribute("onclick") or ""
            if text == str(page_no) or f"({page_no})" in onclick:
                driver.execute_script("arguments[0].click();", link)
                time.sleep(3)
                return True

        return False
    except Exception as e:
        print(f"    [페이지 이동 실패] {e}")
        return False


def download_image(url, filepath):
    """이미지 다운로드."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20, stream=True)
        if resp.status_code == 200:
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            if filepath.stat().st_size > 500:
                return True
            filepath.unlink()
    except Exception:
        pass
    return False


def main():
    print("=" * 60)
    print("  두루누비 걷기여행길 이미지 크롤러")
    print("=" * 60)

    driver = setup_driver()
    all_trails = []
    seen_titles = set()

    try:
        print("\n[1] 페이지 로딩...")
        driver.get("https://durunubi.kr/road-walk.do")
        time.sleep(8)  # JS 렌더링 대기

        # 총 페이지 확인
        total_text = driver.execute_script("""
            var el = document.querySelector('.total, .count, .result-count');
            return el ? el.textContent : '';
        """)
        print(f"  총 건수 텍스트: {total_text}")

        # 페이지 소스에서 총 건수 확인
        src = driver.page_source
        total_match = re.search(r'(\d+)\s*(?:건|개)', src)
        total_count = int(total_match.group(1)) if total_match else 540
        total_pages = (total_count + 11) // 12
        print(f"  총 {total_count}건, {total_pages} 페이지")

        # 페이지 순회
        for page_no in range(1, total_pages + 1):
            print(f"\n[Page {page_no}/{total_pages}]", end=" ")

            if page_no > 1:
                if not navigate_to_page(driver, page_no):
                    print("페이지 이동 실패, 중단")
                    break
                time.sleep(2)

            trails = extract_trails_from_page(driver)

            # 중복 제거
            new_trails = []
            for t in trails:
                key = t.get("title", "")
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    t["source_page"] = page_no
                    new_trails.append(t)

            all_trails.extend(new_trails)
            print(f"→ {len(new_trails)}건 추출 (누적 {len(all_trails)})")

            if not new_trails and page_no > 1:
                print("  데이터 없음, 종료")
                break

    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        driver.quit()

    print(f"\n\n[2] 총 {len(all_trails)}건 추출 완료")

    # 이미지 다운로드
    if all_trails:
        print(f"\n[3] 이미지 다운로드 중...")
        downloaded = 0
        for i, trail in enumerate(all_trails):
            img_url = trail.get("image_url", "")
            if not img_url:
                continue

            safe_name = re.sub(r'[^\w가-힣]', '_', trail.get("title", f"trail_{i}"))
            safe_name = re.sub(r'_+', '_', safe_name).strip('_')[:50]

            ext = ".jpg"
            for e in [".png", ".webp"]:
                if e in img_url.lower():
                    ext = e
                    break

            filepath = IMAGE_DIR / f"{safe_name}{ext}"
            if filepath.exists():
                trail["local_image"] = str(filepath)
                downloaded += 1
                continue

            if download_image(img_url, filepath):
                trail["local_image"] = str(filepath)
                downloaded += 1

            if downloaded % 50 == 0 and downloaded > 0:
                print(f"  ... {downloaded}건 완료")
            time.sleep(0.2)

        print(f"  이미지 다운로드: {downloaded}건")

    # 저장
    output_path = OUTPUT_DIR / "trail_images.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_trails, f, ensure_ascii=False, indent=2)
    print(f"\n[4] 저장 완료: {output_path} ({len(all_trails)}건)")

    print(f"\n{'=' * 60}")
    print(f"  완료! {len(all_trails)}건 / 이미지: {IMAGE_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
