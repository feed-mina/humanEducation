"""
K-Ride 2.0 — 아이돌/유명인 이미지 크롤링 스크립트
온보딩 화면에 사용할 프로필/포스터 이미지를 DuckDuckGo에서 검색 후 다운로드
"""

import os
import sys
import time
import re
import requests
from pathlib import Path

# ─── 설정 ───────────────────────────────────────────────
BASE_DIR = Path(r"e:/krider/kride-project/images")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
TIMEOUT = 15
DELAY = 1.5  # 요청 간 딜레이 (초)

# ─── 데이터 정의 ─────────────────────────────────────────
CATEGORIES = {
    "kpop_groups": [
        ("BTS", "BTS 방탄소년단 kpop group photo"),
        ("BLACKPINK", "BLACKPINK 블랙핑크 kpop group photo"),
        ("Stray_Kids", "Stray Kids 스트레이키즈 kpop group photo"),
        ("SEVENTEEN", "SEVENTEEN 세븐틴 kpop group photo"),
        ("aespa", "aespa 에스파 kpop group photo"),
        ("NewJeans", "NewJeans 뉴진스 kpop group photo"),
        ("G_I_DLE", "(G)I-DLE 여자아이들 kpop group photo"),
        ("TWICE", "TWICE 트와이스 kpop group photo"),
        ("EXO", "EXO 엑소 kpop group photo"),
        ("TXT", "TXT 투모로우바이투게더 kpop group photo"),
        ("ENHYPEN", "ENHYPEN 엔하이픈 kpop group photo"),
        ("LE_SSERAFIM", "LE SSERAFIM 르세라핌 kpop group photo"),
        ("ATEEZ", "ATEEZ 에이티즈 kpop group photo"),
        ("NCT", "NCT 엔시티 kpop group photo"),
        ("ITZY", "ITZY 잇지 kpop group photo"),
        ("IVE", "IVE 아이브 kpop group photo"),
        ("NMIXX", "NMIXX 엔믹스 kpop group photo"),
        ("Red_Velvet", "Red Velvet 레드벨벳 kpop group photo"),
        ("TREASURE", "TREASURE 트레저 kpop group photo"),
        ("MONSTA_X", "MONSTA X 몬스타엑스 kpop group photo"),
        ("GOT7", "GOT7 갓세븐 kpop group photo"),
        ("SHINee", "SHINee 샤이니 kpop group photo"),
        ("2NE1", "2NE1 투애니원 kpop group photo"),
        ("BIGBANG", "BIGBANG 빅뱅 kpop group photo"),
    ],
    "kpop_solo": [
        ("IU", "IU 아이유 kpop singer photo"),
        ("Taeyang", "태양 TAEYANG kpop singer photo"),
        ("Zico", "지코 ZICO kpop singer photo"),
        ("DEAN", "DEAN 딘 kpop singer photo"),
        ("Hwasa", "화사 HWASA mamamoo kpop photo"),
        ("Sunmi", "선미 SUNMI kpop singer photo"),
        ("Chungha", "청하 CHUNGHA kpop singer photo"),
        ("BIBI", "BIBI 비비 kpop singer photo"),
        ("Rose", "로제 ROSE BLACKPINK kpop photo"),
        ("Jennie", "제니 JENNIE BLACKPINK kpop photo"),
        ("Jisoo", "지수 JISOO BLACKPINK kpop photo"),
        ("Lisa", "리사 LISA BLACKPINK kpop photo"),
    ],
    "actors": [
        ("Lee_Minho", "이민호 Lee Min Ho actor photo"),
        ("Song_Hyekyo", "송혜교 Song Hye Kyo actress photo"),
        ("Park_Seojun", "박서준 Park Seo Joon actor photo"),
        ("Jun_Jihyun", "전지현 Jun Ji Hyun actress photo"),
        ("Kim_Soohyun", "김수현 Kim Soo Hyun actor photo"),
        ("Byeon_Wooseok", "변우석 Byeon Woo Seok actor photo"),
        ("Han_Sohee", "한소희 Han So Hee actress photo"),
        ("Song_Kang", "송강 Song Kang actor photo"),
    ],
    "content": [
        ("Squid_Game", "오징어게임 Squid Game netflix poster"),
        ("The_Glory", "더글로리 The Glory drama poster"),
        ("Extraordinary_Attorney_Woo", "이상한변호사우영우 Extraordinary Attorney Woo poster"),
        ("Queen_of_Tears", "눈물의여왕 Queen of Tears drama poster"),
        ("Goblin", "도깨비 Goblin drama poster"),
    ],
}


def search_and_download_ddg(query: str, save_path: Path) -> bool:
    """DuckDuckGo 이미지 검색 후 첫 번째 이미지 다운로드"""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=5))

        if not results:
            print(f"    ⚠️  검색 결과 없음: {query}")
            return False

        # 첫 번째 유효한 이미지 다운로드 시도
        for result in results:
            img_url = result.get("image", "")
            if not img_url:
                continue

            try:
                resp = requests.get(img_url, headers=HEADERS, timeout=TIMEOUT, stream=True)
                if resp.status_code == 200:
                    content_type = resp.headers.get("Content-Type", "")
                    if "image" in content_type or img_url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        # 확장자 결정
                        if "png" in content_type or img_url.lower().endswith('.png'):
                            ext = ".png"
                        elif "webp" in content_type or img_url.lower().endswith('.webp'):
                            ext = ".webp"
                        else:
                            ext = ".jpg"

                        final_path = save_path.with_suffix(ext)
                        with open(final_path, "wb") as f:
                            for chunk in resp.iter_content(8192):
                                f.write(chunk)

                        size_kb = final_path.stat().st_size / 1024
                        if size_kb < 2:  # 2KB 미만은 유효하지 않은 이미지
                            final_path.unlink()
                            continue

                        print(f"    ✅ 저장: {final_path.name} ({size_kb:.0f}KB)")
                        return True
            except Exception:
                continue

        print(f"    ⚠️  다운로드 실패: {query}")
        return False

    except Exception as e:
        print(f"    ❌ 오류: {e}")
        return False


def main():
    print("=" * 60)
    print("  K-Ride 2.0 — 아이돌/유명인 이미지 크롤링")
    print("=" * 60)

    total = sum(len(items) for items in CATEGORIES.values())
    success = 0
    fail = 0
    idx = 0

    for category, items in CATEGORIES.items():
        cat_dir = BASE_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 [{category}] — {len(items)}개")

        for filename, query in items:
            idx += 1
            save_path = cat_dir / filename  # 확장자는 다운로드 시 결정

            # 이미 다운로드된 경우 스킵
            existing = list(cat_dir.glob(f"{filename}.*"))
            if existing:
                print(f"  [{idx}/{total}] ⏭️  이미 존재: {existing[0].name}")
                success += 1
                continue

            print(f"  [{idx}/{total}] 🔍 검색: {query}")
            if search_and_download_ddg(query, save_path):
                success += 1
            else:
                fail += 1

            time.sleep(DELAY)

    print(f"\n{'=' * 60}")
    print(f"  완료! 성공: {success}/{total}  |  실패: {fail}/{total}")
    print(f"  저장 위치: {BASE_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
