"""
event_ner_handler.py — Zero-shot Event Classification TorchServe Handler
=========================================================================
모델: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (zero-shot classification)
입력: {"text": "잠실종합운동장에서 BTS 콘서트가 열립니다"}
출력: {"event_type": "콘서트/공연", "score": 0.87}
"""
from __future__ import annotations

import json
import logging
import os

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

EVENT_LABELS = [
    "콘서트/공연",
    "스포츠 경기",
    "축제/행사",
    "시위/집회",
    "교통 통제",
]


class EventNERHandler(BaseHandler):

    def initialize(self, context):
        from transformers import pipeline as hf_pipeline

        properties = context.system_properties
        model_dir = properties.get("model_dir", ".")
        gpu_id = properties.get("gpu_id", -1)
        device = gpu_id if gpu_id >= 0 else -1

        # 로컬 저장된 모델이 있으면 사용, 아니면 HuggingFace에서 다운로드
        local_model = os.path.join(model_dir, "event_classifier")
        if os.path.exists(local_model):
            model_name = local_model
        else:
            model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

        self.classifier = hf_pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
        )
        logger.info("EventNERHandler initialized with %s on device=%d", model_name, device)

    def preprocess(self, data):
        texts = []
        for item in data:
            body = item.get("body") or item.get("data")
            if isinstance(body, (bytes, bytearray)):
                body = json.loads(body)
            if isinstance(body, dict):
                texts.append(body.get("text", ""))
            elif isinstance(body, str):
                texts.append(body)
        return texts

    def inference(self, texts):
        results = []
        for text in texts:
            if not text:
                results.append({"event_type": "알수없음", "score": 0.0})
                continue
            out = self.classifier(text, candidate_labels=EVENT_LABELS)
            results.append({
                "event_type": out["labels"][0],
                "score": round(float(out["scores"][0]), 4),
            })
        return results

    def postprocess(self, inference_output):
        return inference_output
