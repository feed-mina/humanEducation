#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import Tuple, List
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from config import Config


# In[2]:


logger = Config.setup_logging()


# In[3]:


class NaturalLanguageGenerator:
    """쿼리 결과를 자연어로 설명하는 클래스"""

    def __init__(self):
        self.template = """
        다음 데이터베이스 쿼리 결과를 자연스러운 한국어로 설명해주세요:
        컬럼: {columns}
        결과: {results}
        
        규칙:
        1. 친근하고 대화체로 설명해주세요
        2. 중요한 정보를 강조해주세요
        3. 결과의 의미를 해석해서 설명해주세요
        4. 필요한 경우 추가 컨텍스트를 제공해주세요
        """
        self.llm = Ollama(model="gemma2", temperature=0.3)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_template(self.template)
        )

    def generate_summary(self, columns: Tuple[str], rows: List[Tuple]) -> str:
        """쿼리 결과를 자연어로 설명합니다."""
        if not rows:
            return "검색된 결과가 없습니다. 다른 검색어로 시도해보시겠어요?"
        
        try:
            summary = self.chain.run(
                columns=", ".join(columns),
                results=str(rows)
            )
            return summary
        except Exception as e:
            logger.error(f"자연어 생성 중 오류 발생: {e}")
            return self._generate_fallback_summary(columns, rows)

    def _generate_fallback_summary(self, columns: Tuple[str], rows: List[Tuple]) -> str:
        """오류 발생 시 기본적인 설명을 생성합니다."""
        summary = []
        for row in rows:
            description = "검색 결과: "
            for col, val in zip(columns, row):
                formatted_val = str(val).strip('()[]{}\\\'\"')
                description += f"{col}는 {formatted_val}, "
            summary.append(description.rstrip(", ") + "입니다.")
        return "\n".join(summary)




