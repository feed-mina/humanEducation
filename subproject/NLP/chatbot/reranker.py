"""
reranker.py — CrossEncoder 기반 리랭커 래퍼
==========================================
기본: BAAI/bge-reranker-v2-m3 (다국어, 한국어 지원)
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from chatbot.config import RERANKER_MODEL


class Reranker:
    """CrossEncoder 리랭커 래퍼"""

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        passages: list[dict],
        text_key: str = "text",
        top_k: int = 10,
    ) -> list[dict]:
        """
        passages를 query 관련도로 리랭킹하여 상위 top_k 반환.

        Parameters
        ----------
        query : str
        passages : list[dict]  — 각 dict에 text_key 필드 필요
        text_key : str         — 패시지 텍스트 필드명
        top_k : int

        Returns
        -------
        list[dict] — 리랭킹된 상위 passages (rerank_score 추가)
        """
        if not passages:
            return []

        pairs = [(query, p.get(text_key, "")) for p in passages]
        scores = self.model.predict(pairs)

        for passage, score in zip(passages, scores):
            passage["rerank_score"] = float(score)

        ranked = sorted(passages, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
