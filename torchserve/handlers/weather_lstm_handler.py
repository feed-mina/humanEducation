"""
weather_lstm_handler.py — WeatherLSTM TorchServe Handler
=========================================================
모델: weather_lstm.pt (PyTorch LSTM, 3-class 날씨 분류)
입력: {"sequence": [[월,일,요일,기온,강수,풍속,습도,sgg_idx], ...]}  (14일치)
출력: {"class": 0, "label": "맑음", "proba": [0.8, 0.15, 0.05], "safety_penalty": 0.0}
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

WEATHER_LABELS = {0: "맑음", 1: "흐림", 2: "비·눈"}
SAFETY_PENALTY = {0: 0.0, 1: 0.05, 2: 0.15}


class WeatherLSTMHandler(BaseHandler):

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir", ".")

        # 모델 가중치 로드
        model_path = os.path.join(model_dir, "weather_lstm.pt")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # 메타에서 하이퍼파라미터 복원
        meta_path = os.path.join(model_dir, "weather_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        input_size = meta.get("input_size", 8)
        hidden_size = meta.get("hidden_size", 64)
        num_classes = meta.get("num_classes", 3)

        from weather_lstm_model import WeatherLSTM
        self.model = WeatherLSTM(input_size, hidden_size, num_classes)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        # 스케일러 (선택)
        scaler_path = os.path.join(model_dir, "weather_scaler.pkl")
        self.scaler = None
        if os.path.exists(scaler_path):
            import joblib
            self.scaler = joblib.load(scaler_path)

        logger.info("WeatherLSTMHandler initialized (input=%d, hidden=%d)", input_size, hidden_size)

    def preprocess(self, data):
        sequences = []
        for item in data:
            body = item.get("body") or item.get("data")
            if isinstance(body, (bytes, bytearray)):
                body = json.loads(body)
            seq = np.array(body.get("sequence", []), dtype=np.float32)
            if self.scaler is not None and seq.ndim == 2:
                seq = self.scaler.transform(seq)
            sequences.append(seq)
        return sequences

    def inference(self, sequences):
        results = []
        for seq in sequences:
            tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(tensor)
                proba = torch.softmax(logits, dim=1).squeeze().tolist()
                cls = int(np.argmax(proba))
            results.append({
                "class": cls,
                "label": WEATHER_LABELS.get(cls, "알수없음"),
                "proba": proba,
                "safety_penalty": SAFETY_PENALTY.get(cls, 0.0),
            })
        return results

    def postprocess(self, inference_output):
        return inference_output
