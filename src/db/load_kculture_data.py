"""
load_kculture_data.py
=====================
K-Drama 촬영지 데이터를 DB에 적재 (Geocoding 실패해도 링크 유지)
"""
import os
import re
import time
import pandas as pd
import requests
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")
# Kakao API가 안되므로 Vworld 또는 다른 방식을 고려하거나, 
# 일단 DB에 있는 기존 POI와 매칭을 시도함.
VWORLD_API_KEY = os.environ.get("VWORLD_API_KEY")

EXCEL_PATH = r"D:\kride-project\data\crawling\K_Drama_Unique_Spots.xlsx"

def get_coords_vworld(query):
    """Vworld API를 이용한 장소 검색"""
    if not query or len(str(query)) < 2: return None
    url = f"http://api.vworld.kr/req/search?service=search&request=search&type=place&format=json&query={query}&key={VWORLD_API_KEY}"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get('response', {}).get('status') == 'OK':
            item = res['response']['result']['items'][0]
            return {
                'lat': float(item['point']['y']),
                'lon': float(item['point']['x']),
                'address': item['address']['road'] if item.get('address') else query
            }
    except: pass
    return None

def load_data():
    print(f"Reading excel: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH, sheet_name='Spots_By_Drama')
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # 1. Artist (Drama) 적재
    dramas = df['drama'].unique()
    print(f"Processing {len(dramas)} dramas...")
    artist_map = {}
    for drama in dramas:
        cur.execute("SELECT id FROM artist WHERE name = %s", (drama,))
        res = cur.fetchone()
        if not res:
            cur.execute("INSERT INTO artist (name, category) VALUES (%s, %s) RETURNING id", (drama, 'drama'))
            artist_id = cur.fetchone()[0]
        else:
            artist_id = res[0]
        artist_map[drama] = artist_id
            
    # 2. POI (K-Culture) 적재
    unique_spots = df['final_clean_spot'].unique()
    print(f"Processing {len(unique_spots)} unique spots...")
    
    spot_to_poi_id = {}
    
    for spot in tqdm(unique_spots):
        # 2-1. 기존 POI에 있는지 확인 (이름 매칭)
        cur.execute("SELECT id FROM poi WHERE name = %s LIMIT 1", (spot,))
        res = cur.fetchone()
        if res:
            spot_to_poi_id[spot] = res[0]
            continue
            
        # 2-2. 없으면 지오코딩 시도
        coords = get_coords_vworld(spot)
        
        # 2-3. 삽입
        if coords:
            cur.execute("""
                INSERT INTO poi (name, category, address, geom, source)
                VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s)
                RETURNING id
            """, (spot, 'kculture', coords['address'], coords['lon'], coords['lat'], 'vworld'))
        else:
            # 좌표 없어도 일단 삽입 (GraphRAG 관계 구성을 위해)
            cur.execute("""
                INSERT INTO poi (name, category, source)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (spot, 'kculture', 'excel_only'))
            
        spot_to_poi_id[spot] = cur.fetchone()[0]
        time.sleep(0.05)
    
    # 3. Artist-POI 연결
    print("Linking artists and POIs...")
    for idx, row in df.iterrows():
        drama = row['drama']
        spot = row['final_clean_spot']
        
        if drama in artist_map and spot in spot_to_poi_id:
            artist_id = artist_map[drama]
            poi_id = spot_to_poi_id[spot]
            
            cur.execute("""
                INSERT INTO artist_poi (artist_id, poi_id, relationship_type)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (artist_id, poi_id, 'FILMING_AT'))
            
    conn.commit()
    cur.close()
    conn.close()
    print("K-Culture data sync complete (without Unicode emojis).")

if __name__ == "__main__":
    load_data()
