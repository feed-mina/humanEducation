import os
from dotenv import load_dotenv
from langchain.tools import tool
from sqlalchemy import create_engine, text

# .env 파일에서 API 키 및 DB 정보 로드
load_dotenv()

# --- DB 설정 ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 3306)
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME", "test")

# 1. SQLAlchemy 연결 문자열
try:
    DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # 2. SQLAlchemy 엔진 생성
    engine = create_engine(DATABASE_URL)
    print("✓ DB 엔진(커넥션 풀)이 성공적으로 생성되었습니다.")
    
except Exception as e:
    print(f"DB 엔진 생성 실패: {e}")
    print("  -> .env 파일의 DB 접속 정보를 확인하세요.")
    engine = None


def simple_search(query: str, results_count: int = 5) -> list: # [수정] 3개 -> 5개로 변경
    """
    데이터베이스의 'full_content' 컬럼을 기반으로 검색합니다.
    모든 키워드가 AND 조건으로 일치하는 항목만 검색합니다.
    """
    if engine is None:
        print("[simple_search] DB 엔진이 초기화되지 않았습니다.")
        return []
        
    query_lower = query.lower()
    query_words = query_lower.split()
    
    where_clauses = [] # WHERE 절 (AND로 묶일)
    scores = []        # SELECT 절 (점수 계산용)
    params = {}        # SQL 파라미터
    
    for i, word in enumerate(query_words):
        if len(word) > 1: # 1글자 단어 무시
            key = f"word_{i}" # 예: :word_0, :word_1
            
            # 1. WHERE 절에 추가 (AND 조건)
            where_clauses.append(f"full_content LIKE :{key}")
            
            # 2. SCORE 계산에 추가
            scores.append(f"CASE WHEN full_content LIKE :{key} THEN 1 ELSE 0 END")
            
            # 3. 파라미터 추가
            params[key] = f"%{word}%" 
    
    if not where_clauses:
        print("[simple_search] 유효한 검색어가 없습니다.")
        return []

    where_sql = " AND ".join(where_clauses)
    # ---------------------
    
    score_sql = " + ".join(scores)
    
    # 쿼리 생성
    sql_query = text(f"""
    SELECT id, full_content, ({score_sql}) AS match_score
    FROM job_postings
    WHERE {where_sql} 
    ORDER BY match_score DESC
    LIMIT :limit
    """)
    
    # LIMIT 파라미터 추가
    params["limit"] = results_count
    
    results = []
    
    try:
        with engine.connect() as conn:
            print(f"[simple_search] SQL: {sql_query}")
            print(f"[simple_search] PARAMS: {params}")
            
            result_proxy = conn.execute(sql_query, params)
            rows = result_proxy.mappings().fetchall() 
            
            print(f"[simple_search] 쿼리: '{query}' -> {len(rows)}개 항목 발견")
            
            for i, row in enumerate(rows):
                job_dict = {
                    "id": row["id"],
                    "content": row["full_content"]
                }
                results.append(job_dict)
                print(f"  [{i+1}] (점수: {row['match_score']}) {row['full_content']}")
            
    except Exception as e:
        print(f"[simple_search] DB 오류: {e}")
        
    return results


@tool
def search_jobs_tool(query: str) -> str:
    """
    사용자의 희망 조건(키워드)을 바탕으로 알바 공고를 데이터베이스에서 검색합니다.
    (이 함수는 수정할 필요가 없습니다. simple_search만 호출합니다.)
    
    Args:
        query: 검색 키워드 (예: "경기도 수원시 바리스타 12000원")
    
    Returns:
        검색된 알바 공고 목록 또는 안내 메시지
    """
    try:
        print(f"\n[search_jobs_tool] DB 검색 시작: {query}")
        
        if not query or not query.strip():
            msg = "검색 조건이 비어있습니다. 지역과 직종을 입력해주세요."
            print(f"[search_jobs_tool] {msg}")
            return msg
        
        # DB 검색 실행 (상위 5개)
        results = simple_search(query.lower(), results_count=5)
        
        if not results:
            msg = "죄송합니다. 현재 조건에 맞는 알바 공고를 찾지 못했습니다. 다른 조건으로 다시 시도해주세요."
            print(f"[search_jobs_tool] 검색 결과 없음")
            return msg
        
        # 검색 결과 포맷팅
        formatted_results = "\n".join([
            f"• {job['content']}"
            for job in results
        ])
        
        result_msg = f"검색 결과:\n{formatted_results}"
        print(f"[search_jobs_tool] 검색 완료: {len(results)}개 항목 발견")
        
        return result_msg
        
    except Exception as e:
        error_msg = f"검색 중 오류가 발생했습니다: {str(e)}\n다시 시도해주세요."
        print(f"[search_jobs_tool] 에러: {str(e)}")
        import traceback
        traceback.print_exc()
        return error_msg