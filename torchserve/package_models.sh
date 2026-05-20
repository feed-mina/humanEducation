#!/usr/bin/env bash
# package_models.sh — TorchServe .mar 파일 패키징
# 사용법: cd torchserve && bash package_models.sh
set -euo pipefail

HANDLER_DIR="handlers"
OUT_DIR="."
MODEL_STORE="../models/dl"

echo "=== Packaging TorchServe .mar files ==="

# 1) embedder.mar — SentenceTransformer (모델 파일 없이 핸들러만 패키징, 런타임 다운로드)
torch-model-archiver \
  --model-name embedder \
  --version 1.0 \
  --handler "${HANDLER_DIR}/embedder_handler.py" \
  --extra-files "${HANDLER_DIR}/embedder_handler.py" \
  --export-path "${OUT_DIR}" \
  --force
echo "  [OK] embedder.mar"

# 2) reranker.mar — CrossEncoder
torch-model-archiver \
  --model-name reranker \
  --version 1.0 \
  --handler "${HANDLER_DIR}/reranker_handler.py" \
  --extra-files "${HANDLER_DIR}/reranker_handler.py" \
  --export-path "${OUT_DIR}" \
  --force
echo "  [OK] reranker.mar"

# 3) weather_lstm.mar — WeatherLSTM (모델 파일 포함)
if [ -f "${MODEL_STORE}/weather_lstm.pt" ]; then
  EXTRA_FILES="${MODEL_STORE}/weather_lstm.pt"
  [ -f "${MODEL_STORE}/weather_scaler.pkl" ] && EXTRA_FILES="${EXTRA_FILES},${MODEL_STORE}/weather_scaler.pkl"
  [ -f "${MODEL_STORE}/weather_meta.json" ] && EXTRA_FILES="${EXTRA_FILES},${MODEL_STORE}/weather_meta.json"

  torch-model-archiver \
    --model-name weather_lstm \
    --version 1.0 \
    --handler "${HANDLER_DIR}/weather_lstm_handler.py" \
    --extra-files "${EXTRA_FILES}" \
    --export-path "${OUT_DIR}" \
    --force
  echo "  [OK] weather_lstm.mar"
else
  echo "  [SKIP] weather_lstm.mar — ${MODEL_STORE}/weather_lstm.pt 없음"
fi

# 4) event_ner.mar — zero-shot classification
if [ -d "${MODEL_STORE}/event_classifier" ]; then
  torch-model-archiver \
    --model-name event_ner \
    --version 1.0 \
    --handler "${HANDLER_DIR}/event_ner_handler.py" \
    --extra-files "${MODEL_STORE}/event_classifier" \
    --export-path "${OUT_DIR}" \
    --force
  echo "  [OK] event_ner.mar"
else
  # 로컬 모델 없으면 핸들러만 패키징 (런타임 HuggingFace 다운로드)
  torch-model-archiver \
    --model-name event_ner \
    --version 1.0 \
    --handler "${HANDLER_DIR}/event_ner_handler.py" \
    --extra-files "${HANDLER_DIR}/event_ner_handler.py" \
    --export-path "${OUT_DIR}" \
    --force
  echo "  [OK] event_ner.mar (runtime download mode)"
fi

echo "=== Done! .mar files: ==="
ls -lh "${OUT_DIR}"/*.mar 2>/dev/null || echo "  (no .mar files found)"
