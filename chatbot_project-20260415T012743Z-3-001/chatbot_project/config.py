#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import logging
from sqlalchemy import create_engine

class Config:
    # 기본 경로 설정
    BASE_DIR = os.getcwd()
    TEMP_FOLDER = os.path.join(BASE_DIR, "temp")
    MAIN_DOC_PATH = os.path.join(BASE_DIR, "dataset", "company.txt")
    IMG_PATH = os.path.join(BASE_DIR, "imgs", "wordcloud.png")
    FONT_PATH = os.path.join(BASE_DIR, "fonts", "NanumGothic.ttf")

    # LLM 설정
    TEMPERATURE = 0.1
    GEMMA_MODEL = "gemma2"
    LLAVA_MODEL = "llava:7b"

    # 데이터베이스 연결 설정
    DB_URL = "mysql+pymysql://root:비밀번호@localhost:3306/스키마명"
    FEEDBACK_FILE = "query_feedback.json"

    @staticmethod
    def get_engine():
        """SQLAlchemy 엔진 객체 생성"""
        return create_engine(Config.DB_URL)

    @staticmethod
    def setup_logging():
        """로깅 설정"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    # 프롬프트 템플릿
    PROMPT_TEMPLATE = """
    당신은 휴먼(주) 정보를 제공하는 AI 어시스턴트입니다. 모든 답변은 한국어로 답변해 주세요.

    아래는 이전 대화 내용입니다:
    {chat_history}

    관련 문서 내용:
    {context}

    사용자 질문: {question}

    지침:
    1. 문서에서 찾은 정보가 있다면 그 정보를 바탕으로 답변해주세요.
    2. 문서에서 관련 정보를 찾지 못했다면 "죄송합니다만, 해당 질문에 대한 정보를 문서에서 찾을 수 없습니다."라고 답변한 후, 일반적인 대화를 이어갈 수 있습니다.
    3. 모든 답변은 친절하고 전문적으로 제공해주세요.

    답변:
    """