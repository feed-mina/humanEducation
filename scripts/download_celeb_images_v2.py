"""
K-Ride 2.0 -- Celeb Image Downloader v2
Bing Image Search scraping (no API key needed)
"""

import os
import sys
import time
import re
import json
import requests
from pathlib import Path
from urllib.parse import quote_plus

# --- Config ---
BASE_DIR = Path(r"e:/krider/kride-project/images")
TIMEOUT = 15
DELAY = 3.0  # seconds between requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
    "Referer": "https://www.bing.com/",
}

# --- Data ---
CATEGORIES = {
    "kpop_groups": [
        ("BTS", "BTS kpop group official photo"),
        ("BLACKPINK", "BLACKPINK kpop group official photo"),
        ("Stray_Kids", "Stray Kids kpop group photo"),
        ("SEVENTEEN", "SEVENTEEN kpop group photo"),
        ("aespa", "aespa kpop group photo"),
        ("NewJeans", "NewJeans kpop group photo"),
        ("G_I_DLE", "G I-DLE kpop group photo"),
        ("TWICE", "TWICE kpop group photo"),
        ("EXO", "EXO kpop group photo"),
        ("TXT", "TXT Tomorrow by Together kpop group photo"),
        ("ENHYPEN", "ENHYPEN kpop group photo"),
        ("LE_SSERAFIM", "LE SSERAFIM kpop group photo"),
        ("ATEEZ", "ATEEZ kpop group photo"),
        ("NCT", "NCT kpop group photo"),
        ("ITZY", "ITZY kpop group photo"),
        ("IVE", "IVE kpop group photo"),
        ("NMIXX", "NMIXX kpop group photo"),
        ("Red_Velvet", "Red Velvet kpop group photo"),
        ("TREASURE", "TREASURE kpop group photo"),
        ("MONSTA_X", "MONSTA X kpop group photo"),
        ("GOT7", "GOT7 kpop group photo"),
        ("SHINee", "SHINee kpop group photo"),
        ("2NE1", "2NE1 kpop group photo"),
        ("BIGBANG", "BIGBANG kpop group photo"),
    ],
    "kpop_solo": [
        ("IU", "IU kpop singer official photo"),
        ("Taeyang", "Taeyang BIGBANG singer photo"),
        ("Zico", "Zico kpop singer photo"),
        ("DEAN", "DEAN korean singer photo"),
        ("Hwasa", "Hwasa MAMAMOO singer photo"),
        ("Sunmi", "Sunmi kpop singer photo"),
        ("Chungha", "Chungha kpop singer photo"),
        ("BIBI", "BIBI korean singer photo"),
        ("Rose", "Rose BLACKPINK singer photo"),
        ("Jennie", "Jennie BLACKPINK singer photo"),
        ("Jisoo", "Jisoo BLACKPINK singer photo"),
        ("Lisa", "Lisa BLACKPINK singer photo"),
    ],
    "actors": [
        ("Lee_Minho", "Lee Min Ho korean actor photo"),
        ("Song_Hyekyo", "Song Hye Kyo korean actress photo"),
        ("Park_Seojun", "Park Seo Joon korean actor photo"),
        ("Jun_Jihyun", "Jun Ji Hyun korean actress photo"),
        ("Kim_Soohyun", "Kim Soo Hyun korean actor photo"),
        ("Byeon_Wooseok", "Byeon Woo Seok korean actor photo"),
        ("Han_Sohee", "Han So Hee korean actress photo"),
        ("Song_Kang", "Song Kang korean actor photo"),
    ],
    "content": [
        ("Squid_Game", "Squid Game netflix poster official"),
        ("The_Glory", "The Glory korean drama poster"),
        ("Extraordinary_Attorney_Woo", "Extraordinary Attorney Woo drama poster"),
        ("Queen_of_Tears", "Queen of Tears korean drama poster"),
        ("Goblin", "Goblin Dokkaebi korean drama poster"),
    ],
}


def download_image(url, save_path):
    """Download a single image from URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        if resp.status_code != 200:
            return False

        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            return False

        # Determine extension
        if "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"
        else:
            ext = ".jpg"

        final_path = save_path.with_suffix(ext)
        with open(final_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        size_kb = final_path.stat().st_size / 1024
        if size_kb < 3:  # Too small = invalid
            final_path.unlink()
            return False

        print(f"    -> Saved: {final_path.name} ({size_kb:.0f}KB)")
        return True
    except Exception as e:
        return False


def search_bing_images(query, max_results=10):
    """Scrape image URLs from Bing Image Search."""
    url = f"https://www.bing.com/images/search?q={quote_plus(query)}&first=1&count={max_results}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            print(f"    [Bing] HTTP {resp.status_code}")
            return []

        # Extract image URLs from murl parameter
        pattern = r'murl&quot;:&quot;(https?://[^&]+?)&quot;'
        matches = re.findall(pattern, resp.text)

        if not matches:
            # Try alternate pattern
            pattern2 = r'"murl":"(https?://[^"]+?)"'
            matches = re.findall(pattern2, resp.text)

        if not matches:
            # Try yet another pattern for image src
            pattern3 = r'src2="(https?://[^"]+?)"'
            matches = re.findall(pattern3, resp.text)

        return matches[:max_results]
    except Exception as e:
        print(f"    [Bing] Error: {e}")
        return []


def search_google_images_fallback(query, max_results=5):
    """Fallback: scrape from Google Images."""
    url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch&ijn=0"

    try:
        google_headers = HEADERS.copy()
        google_headers["Referer"] = "https://www.google.com/"

        resp = requests.get(url, headers=google_headers, timeout=TIMEOUT)
        if resp.status_code != 200:
            return []

        # Extract data: URLs from JSON-like structures
        pattern = r'\["(https?://[^"]+?\.(?:jpg|jpeg|png|webp))"'
        matches = re.findall(pattern, resp.text)

        # Filter out Google's own URLs
        filtered = [m for m in matches if "gstatic.com" not in m and "google.com" not in m]
        return filtered[:max_results]
    except Exception:
        return []


def process_item(filename, query, cat_dir):
    """Search and download image for a single item."""
    save_path = cat_dir / filename

    # Skip if already exists
    existing = list(cat_dir.glob(f"{filename}.*"))
    if existing:
        print(f"  [SKIP] Already exists: {existing[0].name}")
        return True

    print(f"  [SEARCH] {query}")

    # Try Bing first
    urls = search_bing_images(query)

    # Fallback to Google
    if not urls:
        print(f"    Trying Google fallback...")
        urls = search_google_images_fallback(query)

    if not urls:
        print(f"    [FAIL] No image URLs found")
        return False

    # Try downloading each URL until one succeeds
    for i, img_url in enumerate(urls):
        if download_image(img_url, save_path):
            return True

    print(f"    [FAIL] All {len(urls)} URLs failed to download")
    return False


def main():
    print("=" * 60)
    print("  K-Ride 2.0 -- Celeb Image Downloader v2")
    print("  Using Bing Image Search + Google fallback")
    print("=" * 60)

    total = sum(len(items) for items in CATEGORIES.values())
    success = 0
    fail = 0
    idx = 0

    for category, items in CATEGORIES.items():
        cat_dir = BASE_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{category}] -- {len(items)} items")

        for filename, query in items:
            idx += 1
            print(f"\n  ({idx}/{total})", end=" ")

            if process_item(filename, query, cat_dir):
                success += 1
            else:
                fail += 1

            time.sleep(DELAY)

    print(f"\n\n{'=' * 60}")
    print(f"  Done! Success: {success}/{total}  |  Failed: {fail}/{total}")
    print(f"  Saved to: {BASE_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
