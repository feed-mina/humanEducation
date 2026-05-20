"""
celery_app.py — Celery 비동기 작업 큐 설정
==========================================
브로커/백엔드: Redis

환경변수:
  CELERY_BROKER_URL — Redis URL (기본: redis://localhost:6379/1)
"""
from __future__ import annotations

import os

from celery import Celery

REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/1")

celery = Celery("kride", broker=REDIS_URL, backend=REDIS_URL)
celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,          # Spot VM 대비: 작업 완료 후 ACK
    worker_prefetch_multiplier=1, # 한 번에 1개만 프리페치
    task_routes={
        "src.api.tasks.task_embed_texts":     {"queue": "ml"},
        "src.api.tasks.task_predict_weather":  {"queue": "ml"},
        "src.api.tasks.task_generate_tts":     {"queue": "media"},
        "src.api.tasks.task_generate_video":   {"queue": "media"},
    },
)

# 태스크 모듈 자동 탐색
celery.autodiscover_tasks(["src.api"])
