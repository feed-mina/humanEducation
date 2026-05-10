"""
update_schema_kculture.py
========================
K-Culture 데이터를 위한 artist 및 artist_poi 테이블 추가
"""
import os
import psycopg2
from dotenv import load_dotenv

# .env 로드
load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")

def update_schema():
    if not DATABASE_URL:
        print("❌ DATABASE_URL이 설정되지 않았습니다.")
        return

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    try:
        print("[1] artist 테이블 생성 중...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS artist (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                name_en VARCHAR(100),
                category VARCHAR(50), -- drama, actor, singer
                image_url VARCHAR(500),
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        print("[2] artist_poi 테이블 생성 중...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS artist_poi (
                artist_id INTEGER REFERENCES artist(id),
                poi_id INTEGER REFERENCES poi(id),
                relationship_type VARCHAR(50) DEFAULT 'FILMING_AT',
                PRIMARY KEY (artist_id, poi_id)
            );
        """)

        conn.commit()
        print("✅ 스키마 업데이트 완료!")
    except Exception as e:
        conn.rollback()
        print(f"❌ 오류 발생: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    update_schema()
