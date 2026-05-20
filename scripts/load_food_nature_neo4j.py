"""
load_food_nature_neo4j.py — Food/Nature POI를 Neo4j에 적재
=============================================================
실행: cd D:/kride-project && python scripts/load_food_nature_neo4j.py

기능:
  1. tourapi_food_nationwide.csv → Neo4j POI 노드 (category='food')
  2. durunubi courses CSV        → Neo4j POI 노드 (category='nature')
  3. POI-Region 연결 (IN_REGION 엣지)

적재 후: python scripts/build_poi_collections.py  (ChromaDB 재인덱싱)
"""
import os
import sys
import re

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "data", "raw_ml")
DURUNUBI_DIR = os.path.join(BASE_DIR, "dataset", "data", "durunubi")

BATCH_SIZE = 200


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def extract_sido(address: str) -> str:
    """주소에서 시도 추출"""
    if not address:
        return ""
    sido_list = [
        "서울특별시", "부산광역시", "대구광역시", "인천광역시",
        "광주광역시", "대전광역시", "울산광역시", "세종특별자치시",
        "경기도", "강원특별자치도", "충청북도", "충청남도",
        "전북특별자치도", "전라남도", "경상북도", "경상남도",
        "제주특별자치도",
        # 과거 표기
        "강원도", "전라북도",
    ]
    for s in sido_list:
        if address.startswith(s):
            return s
    return ""


# ── 1. Food POI 적재 ─────────────────────────────────────────

def load_food(driver):
    path = os.path.join(RAW_DIR, "tourapi_food_nationwide.csv")
    if not os.path.exists(path):
        print(f"  ❌ 파일 없음: {path}")
        return 0

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  tourapi_food_nationwide.csv: {len(df):,}행 로드")

    records = []
    for _, r in df.iterrows():
        name = str(r.get("name", ""))[:200]
        if not name:
            continue
        lat = r.get("lat")
        lon = r.get("lon")
        try:
            lat, lon = float(lat), float(lon)
            if not (33 < lat < 39 and 124 < lon < 132):
                lat, lon = 0.0, 0.0
        except (TypeError, ValueError):
            lat, lon = 0.0, 0.0

        address = str(r.get("address", ""))[:300]
        sido = str(r.get("sido", ""))[:20] or extract_sido(address)

        records.append({
            "id": f"food_{r.get('contentid', '')}",
            "name": name,
            "category": "food",
            "sub_category": str(r.get("sub_category", ""))[:50] if pd.notna(r.get("sub_category")) else "",
            "address": address,
            "sido": sido,
            "lat": lat,
            "lon": lon,
            "image_url": str(r.get("image_url", ""))[:500] if pd.notna(r.get("image_url")) else "",
        })

    # Neo4j 적재
    total = 0
    with driver.session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            session.run("""
                UNWIND $batch AS row
                MERGE (p:POI {id: row.id})
                SET p.name         = row.name,
                    p.category     = row.category,
                    p.sub_category = row.sub_category,
                    p.address      = row.address,
                    p.sido         = row.sido,
                    p.lat          = toFloat(row.lat),
                    p.lon          = toFloat(row.lon),
                    p.image_url    = row.image_url
            """, {"batch": batch})
            total += len(batch)
            print(f"    food upsert {total}/{len(records)}")

    print(f"  ✅ Food POI {total}건 적재 완료")
    return total


# ── 2. Nature POI 적재 (두루누비 코스) ────────────────────────

def load_nature(driver):
    # 최신 courses CSV 사용
    csv_files = sorted(
        [f for f in os.listdir(DURUNUBI_DIR) if f.startswith("courses_") and f.endswith(".csv")],
        reverse=True,
    )
    if not csv_files:
        print("  ❌ durunubi courses CSV 없음")
        return 0

    path = os.path.join(DURUNUBI_DIR, csv_files[0])
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  {csv_files[0]}: {len(df):,}행 로드")

    records = []
    for _, r in df.iterrows():
        name = str(r.get("crsKorNm", ""))[:200]
        if not name:
            continue

        crs_idx = str(r.get("crsIdx", ""))
        sigun = str(r.get("sigun", ""))[:30] if pd.notna(r.get("sigun")) else ""
        sido = extract_sido(sigun) if sigun else ""

        # 코스 설명을 address 대용으로 사용
        summary = str(r.get("crsSummary", ""))[:300] if pd.notna(r.get("crsSummary")) else ""
        contents = str(r.get("crsContents", ""))[:500] if pd.notna(r.get("crsContents")) else ""
        # HTML 태그 제거
        summary = re.sub(r"<[^>]+>", " ", summary).strip()
        contents = re.sub(r"<[^>]+>", " ", contents).strip()

        distance = str(r.get("crsDstnc", ""))
        level = str(r.get("crsLevel", ""))
        level_map = {"1": "쉬움", "2": "보통", "3": "어려움"}

        records.append({
            "id": f"nature_{crs_idx}",
            "name": name,
            "category": "nature",
            "sub_category": "둘레길",
            "address": sigun or summary[:100],
            "sido": sido,
            "lat": 0.0,   # 둘레길은 좌표 대신 코스 정보 활용
            "lon": 0.0,
            "image_url": "",
            "distance_km": distance,
            "level": level_map.get(str(level), str(level)),
            "description": contents[:300],
        })

    total = 0
    with driver.session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            session.run("""
                UNWIND $batch AS row
                MERGE (p:POI {id: row.id})
                SET p.name         = row.name,
                    p.category     = row.category,
                    p.sub_category = row.sub_category,
                    p.address      = row.address,
                    p.sido         = row.sido,
                    p.lat          = toFloat(row.lat),
                    p.lon          = toFloat(row.lon),
                    p.image_url    = row.image_url,
                    p.distance_km  = row.distance_km,
                    p.level        = row.level,
                    p.description  = row.description
            """, {"batch": batch})
            total += len(batch)
            print(f"    nature upsert {total}/{len(records)}")

    print(f"  ✅ Nature POI {total}건 적재 완료")
    return total


# ── 3. POI-Region 연결 ────────────────────────────────────────

def link_poi_region(driver):
    """새로 추가된 food/nature POI를 기존 Region 노드에 연결"""
    with driver.session() as session:
        regions = session.run("MATCH (r:Region) RETURN r.name AS name").data()
        region_names = [r["name"] for r in regions]
        print(f"  Region 노드: {len(region_names)}개")

        linked = 0
        for rname in region_names:
            result = session.run("""
                MATCH (p:POI)
                WHERE p.category IN ['food', 'nature']
                  AND p.address CONTAINS $rname
                  AND NOT (p)-[:IN_REGION]->(:Region)
                MATCH (r:Region {name: $rname})
                MERGE (p)-[:IN_REGION]->(r)
                RETURN count(p) AS cnt
            """, {"rname": rname}).data()
            c = result[0]["cnt"] if result else 0
            linked += c

        print(f"  ✅ POI-Region 연결: {linked}개 엣지 추가")
        return linked


# ── 메인 ──────────────────────────────────────────────────────

def main():
    driver = get_driver()

    # 적재 전 현황
    with driver.session() as session:
        before = session.run("""
            MATCH (p:POI)
            RETURN p.category AS cat, count(*) AS cnt
            ORDER BY cnt DESC
        """).data()
    print("적재 전 현황:")
    for r in before:
        print(f"  {r['cat']}: {r['cnt']}")

    print(f"\n{'='*50}")
    print("[1] Food POI 적재")
    print(f"{'='*50}")
    food_cnt = load_food(driver)

    print(f"\n{'='*50}")
    print("[2] Nature POI 적재")
    print(f"{'='*50}")
    nature_cnt = load_nature(driver)

    print(f"\n{'='*50}")
    print("[3] POI-Region 연결")
    print(f"{'='*50}")
    link_poi_region(driver)

    # 적재 후 현황
    with driver.session() as session:
        after = session.run("""
            MATCH (p:POI)
            RETURN p.category AS cat, count(*) AS cnt
            ORDER BY cnt DESC
        """).data()
    print(f"\n{'='*50}")
    print("적재 후 현황:")
    for r in after:
        print(f"  {r['cat']}: {r['cnt']}")

    driver.close()
    print(f"\n✅ 완료! Food {food_cnt}건 + Nature {nature_cnt}건 적재")
    print("다음 단계: python scripts/build_poi_collections.py")


if __name__ == "__main__":
    main()
