"""
PostgreSQL → CSV 내보내기 (Colab 임베딩용)
실행: python scripts/export_poi_for_colab.py
출력: data/poi_for_embedding.csv (~150MB)
"""

import os
import csv
import psycopg2
from dotenv import load_dotenv

load_dotenv(".env")

OUTPUT_PATH = "data/poi_for_embedding.csv"
BATCH_SIZE  = 10_000

print("=" * 55)
print("POI 데이터 내보내기 (PostgreSQL → CSV)")
print("=" * 55)

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur  = conn.cursor()

# 전체 수 확인
cur.execute("SELECT COUNT(*) FROM poi")
total = cur.fetchone()[0]
print(f"총 POI: {total:,}개")

# 스트리밍 커서로 대용량 처리
cur_stream = conn.cursor(name="poi_export_cursor")
cur_stream.execute("""
    SELECT
        id,
        COALESCE(name,         '')  AS name,
        COALESCE(name_en,      '')  AS name_en,
        COALESCE(category,     '')  AS category,
        COALESCE(sub_category, '')  AS sub_category,
        COALESCE(sido,         '')  AS sido,
        COALESCE(sigungu,      '')  AS sigungu,
        COALESCE(address,      '')  AS address,
        COALESCE(description,  '')  AS description,
        CASE WHEN geom IS NOT NULL THEN ST_Y(geom) ELSE NULL END AS lat,
        CASE WHEN geom IS NOT NULL THEN ST_X(geom) ELSE NULL END AS lon,
        COALESCE(image_url,    '')  AS image_url
    FROM poi
    ORDER BY id
""")

os.makedirs("data", exist_ok=True)

COLS = ["id", "name", "name_en", "category", "sub_category",
        "sido", "sigungu", "address", "description", "lat", "lon", "image_url"]

exported = 0
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(COLS)

    while True:
        rows = cur_stream.fetchmany(BATCH_SIZE)
        if not rows:
            break
        writer.writerows(rows)
        exported += len(rows)
        print(f"\r  진행: {exported:,} / {total:,} ({exported/total*100:.1f}%)", end="", flush=True)

print(f"\n✅ 완료! {OUTPUT_PATH} ({exported:,}개)")
print(f"   파일 크기: {os.path.getsize(OUTPUT_PATH)/1024/1024:.1f} MB")
print(f"\n다음 단계: {OUTPUT_PATH} 를 Google Drive에 업로드 후 Colab 실행")

cur_stream.close()
cur.close()
conn.close()
