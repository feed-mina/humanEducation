"""
multi_query.py — Groq LLM으로 쿼리 변형 생성
=============================================
원본 쿼리 + N개 변형 = 총 (N+1)개 쿼리로 멀티소스 검색
"""
from __future__ import annotations

from groq import Groq

from chatbot.config import GROQ_MODEL, GROQ_API_KEY, MULTI_QUERY_COUNT


_groq: Groq | None = None


def _get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


def generate_query_variants(query: str) -> list[str]:
    """
    원본 쿼리를 기반으로 N개 변형 쿼리 생성.
    반환: [원본, 변형1, 변형2, ...]
    """
    if not GROQ_API_KEY:
        return [query]

    prompt = f"""아래 사용자 질문에 대해 {MULTI_QUERY_COUNT}개의 다른 표현으로 바꿔주세요.
검색 엔진에서 더 다양한 결과를 얻기 위한 목적입니다.
각 변형을 한 줄씩, 번호 없이 출력하세요.

원본 질문: {query}

변형:"""

    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "당신은 한국어 검색 쿼리 변환 전문가입니다. 변형 쿼리만 출력하세요."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        variants = [line.strip().lstrip("0123456789.-) ") for line in raw.split("\n") if line.strip()]
        variants = [v for v in variants if v and v != query][:MULTI_QUERY_COUNT]
    except Exception as e:
        print(f"[multi_query] 변형 생성 실패: {e}")
        variants = []

    return [query] + variants
