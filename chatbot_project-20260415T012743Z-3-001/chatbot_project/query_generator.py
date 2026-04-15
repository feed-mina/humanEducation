#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama


# In[2]:


class QueryGenerator:
    """SQL 쿼리 생성을 위한 클래스"""

    def __init__(self):
        self.template = """
        당신은 한국어를 잘하고 MySQL 데이터베이스의 쿼리를 생성하는 전문가입니다.
        데이터베이스 스키마 정보:
        {schema_info}
        
        이전 피드백 정보:
        {feedback_info}
        
        위 정보를 바탕으로 다음 질문에 대한 MySQL 쿼리를 생성해주세요.
        질문: {question}
        
        규칙:
        1. 순수한 SQL 쿼리만 작성하세요
        2. 컬럼의 실제 값을 기준으로 쿼리를 작성하세요
        3. 설명이나 주석을 포함하지 마세요
        4. 쿼리는 SELECT 문으로 시작하고 세미콜론(;)으로 끝나야 합니다
        5. WHERE 절에서는 정확한 값 매칭을 위해 = 연산자를 사용하세요
        6. 유사 검색이 필요한 경우 LIKE '%키워드%' 를 사용하세요
        7. 관련된 모든 결과를 찾기 위해 적절히 OR 조건을 활용하세요
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.llm = Ollama(model="gemma2", temperature=0)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate_query(self, question: str, schema_info: str, feedback_info: str = "") -> str:
        """질문에 대한 SQL 쿼리를 생성합니다."""
        response = self.chain.run(
            question=question,
            schema_info=schema_info,
            feedback_info=feedback_info
        )
        return self.extract_sql_query(response)

    @staticmethod
    def extract_sql_query(response: str) -> str:
        """응답에서 SQL 쿼리를 추출합니다."""
        response = response.replace('```sql', '').replace('```', '').strip()
        match = re.search(r'SELECT.*?;', response, re.DOTALL | re.IGNORECASE)
        return match.group(0).strip() if match else response.strip()




