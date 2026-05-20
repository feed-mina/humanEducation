"""
torchserve_client.py — TorchServe HTTP 추론 클라이언트
=====================================================
TorchServe에 배포된 모델에 HTTP로 추론 요청을 보내는 래퍼.
동기(httpx) + 비동기(httpx.AsyncClient) 양쪽 인터페이스 제공.

환경변수:
  TORCHSERVE_URL  — TorchServe 추론 주소 (기본: http://localhost:8085)
"""
from __future__ import annotations

import os

import httpx

TORCHSERVE_URL = os.environ.get("TORCHSERVE_URL", "http://localhost:8085")

# ── 동기 API (Celery 워커 / 일반 함수 호출용) ──────────────────────────────────

def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """SentenceTransformer 임베딩 (동기)"""
    resp = httpx.post(
        f"{TORCHSERVE_URL}/predictions/embedder",
        json={"text": texts},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()


def rerank_sync(query: str, documents: list[str]) -> list[float]:
    """Cross-encoder 리랭킹 (동기)"""
    resp = httpx.post(
        f"{TORCHSERVE_URL}/predictions/reranker",
        json={"query": query, "documents": documents},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()


def predict_weather_sync(sequence: list) -> dict:
    """WeatherLSTM 예측 (동기)"""
    resp = httpx.post(
        f"{TORCHSERVE_URL}/predictions/weather_lstm",
        json={"sequence": sequence},
        timeout=5.0,
    )
    resp.raise_for_status()
    return resp.json()


def classify_event_sync(text: str) -> dict:
    """이벤트 분류 (동기)"""
    resp = httpx.post(
        f"{TORCHSERVE_URL}/predictions/event_ner",
        json={"text": text},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()


# ── 비동기 API (FastAPI 엔드포인트 호출용) ─────────────────────────────────────

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """SentenceTransformer 임베딩 (비동기)"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{TORCHSERVE_URL}/predictions/embedder",
            json={"text": texts},
        )
        resp.raise_for_status()
        return resp.json()


async def rerank(query: str, documents: list[str]) -> list[float]:
    """Cross-encoder 리랭킹 (비동기)"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{TORCHSERVE_URL}/predictions/reranker",
            json={"query": query, "documents": documents},
        )
        resp.raise_for_status()
        return resp.json()


async def predict_weather(sequence: list) -> dict:
    """WeatherLSTM 예측 (비동기)"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(
            f"{TORCHSERVE_URL}/predictions/weather_lstm",
            json={"sequence": sequence},
        )
        resp.raise_for_status()
        return resp.json()


async def classify_event(text: str) -> dict:
    """이벤트 분류 (비동기)"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{TORCHSERVE_URL}/predictions/event_ner",
            json={"text": text},
        )
        resp.raise_for_status()
        return resp.json()
