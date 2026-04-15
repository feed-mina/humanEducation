#!/usr/bin/env python
# coding: utf-8

# In[11]:


import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import importlib
import config
importlib.reload(config)
from config import Config


# In[12]:


logger = Config.setup_logging()


# In[13]:


class QueryFeedbackManager:
    """피드백 관리를 위한 클래스"""
    
    def __init__(self, feedback_file: str):
        self.feedback_file = feedback_file
        self.last_question: str = ""
        
    def load_feedback(self) -> List[Dict[str, Any]]:
        """피드백 데이터를 로드하고 유효성을 검증합니다."""
        if not os.path.exists(self.feedback_file):
            return []
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.warning("잘못된 피드백 데이터 형식입니다. 빈 리스트로 초기화합니다.")
                    return []
                return data
        except json.JSONDecodeError as e:
            logger.error(f"피드백 파일 파싱 오류: {e}")
            return []

    def save_feedback(self, feedback_data: List[Dict[str, Any]]) -> None:
        """피드백 데이터를 저장하고 오류를 처리합니다."""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"피드백 저장 중 오류 발생: {e}")

    def find_similar_feedback(self, question: str) -> Optional[Dict[str, Any]]:
        """유사한 질문에 대한 피드백을 찾습니다."""
        feedback_data = self.load_feedback()
        # 정확한 일치 먼저 확인
        exact_match = next((fb for fb in feedback_data if fb['question'].lower() == question.lower()), None)
        if exact_match:
            return exact_match
        
        # 부분 일치 확인 (키워드 기반)
        keywords = question.lower().split()
        for feedback in feedback_data:
            feedback_keywords = feedback['question'].lower().split()
            if any(keyword in feedback_keywords for keyword in keywords):
                return feedback
        return None