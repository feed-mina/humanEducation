#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import Tuple, List, Optional
from sqlalchemy import text, inspect
from config import Config


# In[2]:


logger = Config.setup_logging()


# In[3]:


class DatabaseInterface:
    """데이터베이스 작업을 처리하는 클래스"""
    
    def __init__(self):  # engine 파라미터 제거
        """SQLAlchemy 엔진을 초기화합니다."""
        self.engine = Config.get_engine()  # Config에서 직접 engine 생성
        self.inspector = inspect(self.engine)

    def get_schema_info(self) -> str:
        """데이터베이스 스키마 정보를 반환합니다."""
        schema_info = []
        try:
            tables = self.inspector.get_table_names()
            for table in tables:
                columns = self.inspector.get_columns(table)
                column_info = [f"{col['name']} ({col['type'].__class__.__name__})" for col in columns]
                schema_info.append(f"테이블: {table}\n컬럼: {', '.join(column_info)}")
            return "\n\n".join(schema_info)
        except Exception as e:
            logger.error(f"스키마 정보 조회 중 오류 발생: {e}")
            return f"스키마 정보 조회 실패: {str(e)}"

    def execute_query(self, query: str) -> Tuple[Optional[List[str]], List[Tuple]]:
        """SQL 쿼리를 실행하고 결과를 반환합니다."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                columns = list(result.keys()) if result.keys() else []
                rows = result.fetchall()
                return columns, rows
        except Exception as e:
            logger.error(f"쿼리 실행 오류: {e}")
            return None, [(f"쿼리 실행 중 오류가 발생했습니다: {str(e)}",)]