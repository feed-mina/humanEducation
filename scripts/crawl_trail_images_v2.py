"""
두루누비 걷기여행길 이미지 크롤러 v2 (Selenium - 정확한 DOM 기반)
- 이미지: /editImgUp.do?filePath=... 패턴
- 카드: card-favorites 클래스
- 페이지네이션: pageNum hidden input 기반
- 상세: detailRoutePageCall(theme_Mng, brd_div)
"""
import json
import time
import re
import requests
from pathlib import Path
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

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


def extract_trails_js(driver):
    """JavaScript로 현재 페이지의 트레일 카드 데이터 추출."""
    result = driver.execute_script("""
        var trails = [];
        
        // card-favorites 요소 찾기
        var cards = document.querySelectorAll('.card-favorites, .card-item, [class*="card"]');
        
        // 카드가 없으면 li 요소 시도
        if (cards.length === 0) {
            cards = document.querySelectorAll('.row > div, .list-wrap > div');
        }
        
        cards.forEach(function(card) {
            var img = card.querySelector('img');
            var titleEl = card.querySelector('.tit, .name, strong, h3, h4, p.tit, span.tit');
            var linkEl = card.querySelector('a[onclick]') || card.querySelector('[onclick]');
            
            if (!img) return;
            
            var src = img.getAttribute('src') || img.getAttribute('data-src') || '';
            var alt = img.getAttribute('alt') || '';
            
            // 제목 추출
            var title = '';
            if (titleEl) title = titleEl.textContent.trim();
            if (!title) title = alt;
            
            // 제목이 너무 길거나 없으면 스킵
            if (!title || title.length < 2 || title.length > 100) return;
            // UI 요소 제외
            if (title.indexOf('QR') > -1 || title.indexOf('로그인') > -1) return;
            
            // onclick에서 theme_mng ID 추출
            var themeId = '';
            if (linkEl) {
                var onclick = linkEl.getAttribute('onclick') || '';
                var match = onclick.match(/['"]([A-Z_]+(?:MNG|MNG_)\\d+)['"]/);
                if (match) themeId = match[1];
            }
            
            trails.push({
                title: title,
                image_url: src,
                theme_id: themeId,
                alt: alt
            });
        });
        
        // 카드 기반으로 못 찾으면, editImgUp 이미지 기반 탐색
        if (trails.length === 0) {
            var imgs = document.querySelectorAll('img[src*="editImgUp"], img[src*="upload"]');
            imgs.forEach(function(img) {
                var src = img.getAttribute('src') || '';
                var alt = img.getAttribute('alt') || '';
                var parent = img.closest('a, div, li');
                var title = alt;
                if (parent) {
                    var tEl = parent.querySelector('.tit, strong, p, h3, h4');
                    if (tEl) title = tEl.textContent.trim();
                    if (!title) title = alt;
                }
                
                var themeId = '';
                if (parent) {
                    var onclick = parent.getAttribute('onclick') || '';
                    var match = onclick.match(/['"]([A-Z_]+(?:MNG|MNG_)\\d+)['"]/);
                    if (match) themeId = match[1];
                }
                
                if (title && title.length > 1 && title.length < 100) {
                    trails.push({
                        title: title,
                        image_url: src,
                        theme_id: themeId,
                        alt: alt
                    });
                }
            });
        }
        
        return trails;
    """)
    return result or []


def main():
    print("=" * 60)
    print("  두루누비 걷기여행길 크롤러 v2")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    driver = setup_driver()
    all_trails = []
    seen = set()

    try:
        # 첫 페이지 로드
        print("\n[1] 첫 페이지 로드...")
        driver.get("https://durunubi.kr/road-walk.do")
        time.sleep(8)

        # 총 건수 확인
        total_info = driver.execute_script("""
            var body = document.body.innerHTML;
            var match = body.match(/(\\d{2,4})\\s*개/);
            return match ? match[1] : '0';
        """)
        total_count = int(total_info) if total_info else 540
        per_page = 12
        total_pages = (total_count + per_page - 1) // per_page
        print(f"  총 {total_count}건, {total_pages}페이지")

        # 첫 페이지에서 페이지 소스 일부 확인
        debug_html = driver.execute_script("""
            var row = document.querySelector('.row');
            if (row) return row.innerHTML.substring(0, 3000);
            
            // 대안: body에서 card 관련 부분 찾기
            var body = document.body.innerHTML;
            var idx = body.indexOf('card-favorites');
            if (idx > -1) return body.substring(idx - 200, idx + 2000);
            
            idx = body.indexOf('editImgUp');
            if (idx > -1) return body.substring(idx - 200, idx + 2000);
            
            return 'DEBUG: card-favorites and editImgUp not found in body';
        """)
        print(f"\n  [DEBUG] HTML snippet:\n  {debug_html[:500] if debug_html else 'EMPTY'}")

        # 페이지 순회
        for page_no in range(1, total_pages + 1):
            if page_no > 1:
                # 페이지 이동: URL 파라미터로 직접 접근
                url = f"https://durunubi.kr/road-walk.do?pageNo={page_no}"
                driver.get(url)
                time.sleep(4)

            trails = extract_trails_js(driver)

            new_count = 0
            for t in trails:
                key = t.get("title", "")
                if key and key not in seen:
                    seen.add(key)
                    t["source_page"] = page_no
                    # 이미지 URL 정규화
                    img_url = t.get("image_url", "")
                    if img_url and not img_url.startswith("http"):
                        t["image_url"] = "https://durunubi.kr" + img_url
                    all_trails.append(t)
                    new_count += 1

            if page_no <= 5 or page_no % 10 == 0:
                print(f"  [Page {page_no}/{total_pages}] {new_count}건 (누적 {len(all_trails)})")

            if new_count == 0 and page_no > 2:
                # pageNo 파라미터가 안 되면 다른 방식 시도
                if page_no == 3:
                    print("  → pageNo 파라미터 실패, pageNum 시도...")
                    url2 = f"https://durunubi.kr/road-walk.do?pageNum={page_no}"
                    driver.get(url2)
                    time.sleep(4)
                    trails2 = extract_trails_js(driver)
                    if trails2:
                        for t in trails2:
                            key = t.get("title", "")
                            if key and key not in seen:
                                seen.add(key)
                                t["source_page"] = page_no
                                img_url = t.get("image_url", "")
                                if img_url and not img_url.startswith("http"):
                                    t["image_url"] = "https://durunubi.kr" + img_url
                                all_trails.append(t)
                        print(f"  → pageNum 성공! {len(trails2)}건")
                        continue
                    else:
                        # JS 기반 페이지 이동 시도
                        print("  → JS 기반 페이지 이동 시도...")
                        driver.get("https://durunubi.kr/road-walk.do")
                        time.sleep(5)
                        driver.execute_script(f"""
                            var pageInput = document.querySelector('input[name="pageNum"]');
                            if (pageInput) {{ pageInput.value = {page_no}; }}
                            var form = document.querySelector('form');
                            if (form) form.submit();
                        """)
                        time.sleep(5)
                        trails3 = extract_trails_js(driver)
                        if trails3:
                            for t in trails3:
                                key = t.get("title", "")
                                if key and key not in seen:
                                    seen.add(key)
                                    t["source_page"] = page_no
                                    img_url = t.get("image_url", "")
                                    if img_url and not img_url.startswith("http"):
                                        t["image_url"] = "https://durunubi.kr" + img_url
                                    all_trails.append(t)
                            print(f"  → form submit 성공! {len(trails3)}건")
                            continue
                        else:
                            print("  → 모든 방법 실패, 종료")
                            break
                elif page_no > 3:
                    break

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.quit()

    print(f"\n\n[2] 총 {len(all_trails)}건 추출")

    # 이미지 다운로드
    if all_trails:
        print(f"\n[3] 이미지 다운로드...")
        downloaded = 0
        for i, trail in enumerate(all_trails):
            img_url = trail.get("image_url", "")
            if not img_url or "android_qr" in img_url or "ios_qr" in img_url:
                continue

            safe_name = re.sub(r'[^\w가-힣]', '_', trail.get("title", f"trail_{i}"))
            safe_name = re.sub(r'_+', '_', safe_name).strip('_')[:50]
            if not safe_name:
                safe_name = f"trail_{i}"

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

            try:
                resp = requests.get(img_url, headers=HEADERS, timeout=20, stream=True)
                if resp.status_code == 200:
                    with open(filepath, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            f.write(chunk)
                    if filepath.stat().st_size > 500:
                        trail["local_image"] = str(filepath)
                        downloaded += 1
                    else:
                        filepath.unlink()
            except Exception:
                pass
            time.sleep(0.15)

        print(f"  완료: {downloaded}건")

    # 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"trail_images_{ts}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_trails, f, ensure_ascii=False, indent=2)
    print(f"\n[4] 저장: {output_path} ({len(all_trails)}건)")

    # 요약
    print(f"\n{'=' * 60}")
    print(f"  완료!")
    print(f"  - 트레일 데이터: {len(all_trails)}건")
    print(f"  - 이미지 경로: {IMAGE_DIR}")
    print(f"  - JSON: {output_path}")
    if all_trails:
        print(f"  - 샘플: {all_trails[0].get('title','')} → {all_trails[0].get('image_url','')[:60]}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
