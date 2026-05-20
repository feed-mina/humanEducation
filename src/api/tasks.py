"""
tasks.py — Celery 비동기 태스크 정의
====================================
Phase 2: 기존 모델 (TorchServe 경유)
Phase 3: 미디어 파이프라인 (GPT-SoVITS, CogVideoX) — 미래 구현
"""
from __future__ import annotations

import os

import httpx

from src.api.celery_app import celery

TORCHSERVE_URL = os.environ.get("TORCHSERVE_URL", "http://localhost:8085")


@celery.task(bind=True, max_retries=3, default_retry_delay=5)
def task_embed_texts(self, texts: list[str]) -> list:
    """배치 임베딩 (TorchServe 경유, 비동기)"""
    try:
        resp = httpx.post(
            f"{TORCHSERVE_URL}/predictions/embedder",
            json={"text": texts},
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise self.retry(exc=exc)


@celery.task(bind=True, max_retries=3, default_retry_delay=5)
def task_rerank(self, query: str, documents: list[str]) -> list:
    """리랭킹 (TorchServe 경유, 비동기)"""
    try:
        resp = httpx.post(
            f"{TORCHSERVE_URL}/predictions/reranker",
            json={"query": query, "documents": documents},
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise self.retry(exc=exc)


@celery.task(bind=True, max_retries=3, default_retry_delay=5)
def task_predict_weather(self, sequence: list) -> dict:
    """날씨 예측 (TorchServe 경유, 비동기)"""
    try:
        resp = httpx.post(
            f"{TORCHSERVE_URL}/predictions/weather_lstm",
            json={"sequence": sequence},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise self.retry(exc=exc)


@celery.task(bind=True, max_retries=3, default_retry_delay=5)
def task_classify_event(self, text: str) -> dict:
    """이벤트 분류 (TorchServe 경유, 비동기)"""
    try:
        resp = httpx.post(
            f"{TORCHSERVE_URL}/predictions/event_ner",
            json={"text": text},
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise self.retry(exc=exc)


# ── Phase 3 태스크 (미래 구현) ──────────────────────────────────────────────

@celery.task(bind=True, max_retries=3, default_retry_delay=10)
def task_generate_tts(self, text: str, voice_id: str) -> str:
    """GPT-SoVITS TTS → GCS URL 반환 (Phase 3)"""
    raise NotImplementedError("Phase 3: GPT-SoVITS TTS 미구현")


@celery.task(bind=True, max_retries=3, default_retry_delay=30)
def task_generate_video(self, image_url: str, track: str) -> str:
    """CogVideoX/AnimatedDrawings → GCS URL 반환 (Phase 3)"""
    raise NotImplementedError("Phase 3: Video generation 미구현")
