"""
reranker_handler.py — CrossEncoder TorchServe Handler
======================================================
모델: cross-encoder/ms-marco-MiniLM-L-6-v2
입력: {"query": "서울 맛집", "documents": ["서울 맛집 리스트", "부산 해변"]}
출력: [0.95, 0.12]
"""
from __future__ import annotations

import json
import logging

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class RerankerHandler(BaseHandler):

    def initialize(self, context):
        from sentence_transformers import CrossEncoder

        properties = context.system_properties
        self.device = properties.get("gpu_id", -1)
        device_str = f"cuda:{self.device}" if self.device >= 0 else "cpu"

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device_str,
        )
        logger.info("RerankerHandler initialized on %s", device_str)

    def preprocess(self, data):
        """요청 파싱 → (query, document) 페어 리스트"""
        pairs = []
        for item in data:
            body = item.get("body") or item.get("data")
            if isinstance(body, (bytes, bytearray)):
                body = json.loads(body)
            query = body.get("query", "")
            documents = body.get("documents", [])
            pairs.extend([(query, doc) for doc in documents])
        return pairs

    def inference(self, pairs):
        if not pairs:
            return []
        scores = self.model.predict(pairs)
        return scores.tolist()

    def postprocess(self, inference_output):
        return [inference_output]
