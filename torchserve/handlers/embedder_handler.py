"""
embedder_handler.py — SentenceTransformer TorchServe Handler
=============================================================
모델: intfloat/multilingual-e5-small (384-dim)
입력: {"text": ["서울 여행", "부산 맛집"]}
출력: [[0.12, -0.34, ...], [0.56, 0.78, ...]]
"""
from __future__ import annotations

import json
import logging

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class EmbedderHandler(BaseHandler):

    def initialize(self, context):
        """모델 로드 (워커 시작 시 1회)"""
        from sentence_transformers import SentenceTransformer

        properties = context.system_properties
        self.device = properties.get("gpu_id", -1)
        device_str = f"cuda:{self.device}" if self.device >= 0 else "cpu"

        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-small",
            device=device_str,
        )
        logger.info("EmbedderHandler initialized on %s", device_str)

    def preprocess(self, data):
        """요청 파싱 → 텍스트 리스트"""
        texts = []
        for item in data:
            body = item.get("body") or item.get("data")
            if isinstance(body, (bytes, bytearray)):
                body = json.loads(body)
            if isinstance(body, dict):
                t = body.get("text", [])
                if isinstance(t, str):
                    t = [t]
                texts.extend(t)
            elif isinstance(body, str):
                texts.append(body)
        return texts

    def inference(self, texts):
        """SentenceTransformer.encode → 정규화 벡터"""
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def postprocess(self, inference_output):
        return [inference_output]
