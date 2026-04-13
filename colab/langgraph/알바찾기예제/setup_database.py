import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# .env 파일에서 환경 변수 로드
load_dotenv()

# .env에서 DB 접속 정보 읽기
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 3306)
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME", "test")

# 1. SQLAlchemy 연결 문자열
server_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"

# 2. 테이블 생성 쿼리 (동일)
create_table_sql = """
CREATE TABLE IF NOT EXISTS job_postings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    location VARCHAR(255) NOT NULL,
    company VARCHAR(100),
    hours_desc VARCHAR(255),
    pay_rate INT,
    full_content TEXT,
    INDEX(pay_rate),
    FULLTEXT(full_content)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
"""

# 3. 데이터 삽입 쿼리 (플레이스홀더를 '%s'에서 ':key' 형태로 변경)
insert_data_sql = """
INSERT INTO job_postings (title, location, company, hours_desc, pay_rate, full_content) 
VALUES (:title, :location, :company, :hours_desc, :pay_rate, :full_content);
"""

# 샘플 데이터를 튜플 리스트에서 '딕셔너리 리스트'로 변경
sample_data = [
    {'title': '바리스타', 'location': '경기도 수원시 팔달구', 'company': '스타벅스', 'hours_desc': '주 2일, 4시간', 'pay_rate': 12000, 'full_content': '경기도 수원시 팔달구 스타벅스 바리스타, 주 2일, 4시간, 시급 12000원'},
    {'title': '바리스타', 'location': '경기도 성남시 분당구', 'company': '카페', 'hours_desc': '월화수 오후', 'pay_rate': 12500, 'full_content': '경기도 성남시 분당구 카페 바리스타, 월화수 오후, 시급 12500원'},
    {'title': '편의점 야간 알바', 'location': '서울 강남구', 'company': 'GS25', 'hours_desc': '야간', 'pay_rate': 11000, 'full_content': '서울 강남구 GS25 편의점 야간 알바, 시급 11000원'},
    {'title': '주말 바리스타', 'location': '서울 마포구 홍대입구역', 'company': '스타벅스', 'hours_desc': '주말', 'pay_rate': 10500, 'full_content': '서울 마포구 홍대입구역 스타벅스 주말 바리스타, 시급 10500원'},
    {'title': '주중 마감 알바', 'location': '부산 해운대구', 'company': '파리바게트', 'hours_desc': '주중 마감', 'pay_rate': 10000, 'full_content': '부산 해운대구 파리바게트 주중 마감 알바, 시급 10000원'},
    {'title': '사무보조', 'location': '서울 강남구 삼성동', 'company': 'IT 회사', 'hours_desc': '주 3일', 'pay_rate': 12000, 'full_content': '서울 강남구 삼성동 IT 회사 사무보조, 주 3일, 시급 12000원'},
    {'title': '홀서빙', 'location': '인천 부평구', 'company': '식당', 'hours_desc': '주말 풀타임', 'pay_rate': 11500, 'full_content': '인천 부평구 식당 홀서빙, 주말 풀타임, 시급 11500원'},
    {'title': '카페 바리스타', 'location': '경기도 수원시', 'company': '개인카페', 'hours_desc': '월수금', 'pay_rate': 11500, 'full_content': '경기도 수원시 카페 바리스타, 월수금, 시급 11500원'},
    {'title': '단기 서빙', 'location': '서울 종로구', 'company': '한정식집', 'hours_desc': '주말 11-15시', 'pay_rate': 13000, 'full_content': '서울 종로구 한정식집 단기 서빙, 주말 11-15시, 시급 13000원'},
    {'title': '재고관리', 'location': '경기도 이천시', 'company': '물류센터', 'hours_desc': '주 5일(주간)', 'pay_rate': 12000, 'full_content': '경기도 이천시 물류센터 재고관리, 주 5일(주간), 시급 12000원'}
]

def initialize_database():
    try:
        # 1. 서버에 연결 (DB 생성용)
        print(f"{DB_HOST}:{DB_PORT} ({DB_USER}) 서버 연결 시도...")
        engine = create_engine(server_url)
        
        with engine.connect() as conn:
            # 2. 데이터베이스 생성 (IF NOT EXISTS)
            print(f"데이터베이스 '{DB_NAME}' 생성 시도...")
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
            conn.execute(text(f"USE {DB_NAME}"))
            
            # 3. 테이블 생성
            print("테이블 'job_postings' 생성 시도...")
            conn.execute(text("DROP TABLE IF EXISTS job_postings"))
            conn.execute(text(create_table_sql))
            print("테이블 'job_postings' 생성 완료.")
            
            # 4. 데이터 삽입
            print("샘플 데이터 삽입 중...")
            
            # 딕셔너리 리스트(sample_data)를 파라미터로 전달
            conn.execute(text(insert_data_sql), sample_data) 
            conn.commit() 
            
            print(f"샘플 데이터 {len(sample_data)}개 삽입 완료.")

    except Exception as e:
        print(f"데이터베이스 초기화 중 오류 발생: {e}")
        print("  -> .env 파일의 DB 접속 정보를 확인하거나 MariaDB 서버가 실행 중인지 확인하세요.")
            
    finally:
        if 'engine' in locals():
            engine.dispose()
            print("데이터베이스 연결 종료.")

if __name__ == "__main__":
    initialize_database()