"""
build_event_ner.py
==================
이벤트 감지 파이프라인 (Phase 3-7)

# [메모] Zero-shot이 무엇인지 알려주세요 
#  
[ 방식 1 ] Zero-shot 분류 (빠른 대안 — 데이터 불필요)
# [메모] snunlp/KR-FinBert-SC 또는 MoritzLaurer/mDeBERTa-v3-base-mnli-xnli 이 무엇을 가르키는지 알려주세요 

  snunlp/KR-FinBert-SC 또는 MoritzLaurer/mDeBERTa-v3-base-mnli-xnli 사용
  → 뉴스 텍스트를 5개 이벤트 유형으로 분류
# [메모] KLUE-BERT NER 파인튜닝이 무엇인지 알려주세요 그리고 각 라벨이 무엇인지 알려주세요 
[ 방식 2 ] KLUE-BERT NER 파인튜닝 (AI Hub 데이터 활용)
  데이터: AI Hub 한국어 다중 이벤트 추출 (dataSetSn=71729)
  라벨: O / B-LOC / I-LOC / B-EVT / I-EVT
  → 장소명(LOC) + 이벤트명(EVT) 추출

[ 실행 ]
  python kride-project/build_event_ner.py --mode zero_shot   # 즉시 사용 가능
  python kride-project/build_event_ner.py --mode finetune    # AI Hub 데이터 필요

[ 출력 ]
  models/dl/event_classifier/   ← zero-shot pipeline 저장
  models/dl/event_ner/          ← NER 파인튜닝 모델 저장 (방식 2)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()

DL_MODELS_DIR = os.path.join(BASE_DIR, "models", "dl")
os.makedirs(DL_MODELS_DIR, exist_ok=True)

EVENT_TYPES = ["스포츠_경기", "공연_행사", "재난_사고", "도로_공사", "기타"]

# 이벤트 → 경로 영향 매핑
EVENT_IMPACT = {
    "스포츠_경기": {"w_tourism_delta": -0.2,  "congestion_alert": True,  "danger_alert": False},
    "공연_행사":   {"w_tourism_delta": -0.2,  "congestion_alert": True,  "danger_alert": False},
    "재난_사고":   {"safety_delta":    -0.3,  "congestion_alert": False, "danger_alert": True},
    "도로_공사":   {"safety_delta":    -0.15, "congestion_alert": False, "danger_alert": False},
    "기타":        {"safety_delta":     0.0,  "congestion_alert": False, "danger_alert": False},
}


# ══════════════════════════════════════════════════════════════════════════════
# 방식 1: Zero-shot 분류
# ══════════════════════════════════════════════════════════════════════════════
def build_zero_shot_classifier():
    """zero-shot-classification 파이프라인 구성 및 저장"""
    try:
        from transformers import pipeline
    except ImportError:
        print("  ❌ transformers 없음: pip install transformers torch")
        sys.exit(1)

    print("=" * 65)
    print("Zero-shot 이벤트 분류 파이프라인 구성")
    print("=" * 65)

    # 한국어 지원 zero-shot 모델
    # Option A: 다국어 DeBERTa (추천 — 한국어 성능 우수)
    # Option B: KR-FinBert-SC (한국어 전용이나 zero-shot 공식 미지원)
    MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    print(f"  모델 다운로드: {MODEL_ID}")
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_ID,
        device=-1,   # CPU (GPU 있으면 device=0)
    )

    # 테스트
    test_texts = [
        "잠실야구장에서 한화 이글스와 LG 트윈스의 경기가 오늘 18시에 열립니다.",
        "서울 강남구 일대 집중호우로 자전거 통행이 위험합니다.",
        "올림픽공원에서 아이유 콘서트가 이번 주말 개최됩니다.",
        "올림픽대로 여의도 구간 도로공사로 우회가 필요합니다.",
    ]
    print("\n  === 테스트 결과 ===")
    for text in test_texts:
        result = classifier(text, candidate_labels=EVENT_TYPES, multi_label=False)
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        print(f"  [{top_label} {top_score:.2f}] {text[:40]}...")

    # 저장
    save_path = os.path.join(DL_MODELS_DIR, "event_classifier")
    classifier.save_pretrained(save_path)
    print(f"\n  ✅ 저장: {save_path}")

    # 설정 메타 저장
    meta = {
        "model_id": MODEL_ID,
        "event_types": EVENT_TYPES,
        "event_impact": EVENT_IMPACT,
        "mode": "zero_shot",
    }
    with open(os.path.join(DL_MODELS_DIR, "event_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("  ✅ event_meta.json 저장")

    return classifier


# ══════════════════════════════════════════════════════════════════════════════
# 방식 2: KLUE-BERT NER 파인튜닝
# ══════════════════════════════════════════════════════════════════════════════
def convert_aihub_to_ner(data_dir: str) -> list:
    """
    AI Hub 이벤트 추출 데이터 → NER 학습 포맷 변환

    원본 형식 (예상):
      { "sentence": "...", "events": [ {"entity_value": "잠실", "event_type": "스포츠_경기"} ] }

    출력:
      [ {"tokens": [...], "labels": ["O", "B-LOC", "I-EVT", ...]} ]
    """
    import glob

    ner_data = []
    json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
    print(f"  JSON 파일 수: {len(json_files)}")

    for fpath in json_files[:100]:   # 테스트용 100개만
        try:
            with open(fpath, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            continue

        # AI Hub 포맷 파싱 (실제 포맷에 따라 조정 필요)
        sentences = raw if isinstance(raw, list) else raw.get("data", [])
        for item in sentences:
            text = item.get("sentence", item.get("text", ""))
            events = item.get("events", item.get("event_list", []))
            if not text:
                continue

            tokens = list(text)   # 음절 단위 토큰화 (BIO 태깅용)
            labels = ["O"] * len(tokens)

            for evt in events:
                entity = str(evt.get("entity_value", ""))
                etype  = str(evt.get("event_type", "기타"))

                # 장소명(LOC) 태깅
                idx = text.find(entity)
                if idx >= 0:
                    labels[idx] = "B-LOC"
                    for i in range(idx + 1, idx + len(entity)):
                        if i < len(labels):
                            labels[i] = "I-LOC"

            ner_data.append({"tokens": tokens, "labels": labels})

    return ner_data


def build_ner_finetune(data_dir: str):
    """KLUE-BERT NER 파인튜닝"""
    # [메모] try 밑에서부터 에러가 나는것 같습니다. 
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            AdamW,
            get_linear_schedule_with_warmup,
        )
    except ImportError:
        print("  ❌ torch/transformers 없음: pip install torch transformers")
        sys.exit(1)

    print("=" * 65)
    print("KLUE-BERT NER 파인튜닝")
    print("=" * 65)

    LABEL2ID = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-EVT": 3, "I-EVT": 4}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}
    MODEL_ID = "klue/bert-base"
    MAX_LEN  = 128
    EPOCHS   = 3
    LR       = 2e-5

    # 데이터 변환
    ner_data = convert_aihub_to_ner(data_dir)
    print(f"  NER 학습 샘플: {len(ner_data):,}개\n")

    if len(ner_data) == 0:
        print("  ❌ 학습 데이터 없음. AI Hub 데이터 경로를 확인하세요.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    class NERDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            tokens = item["tokens"][:MAX_LEN]
            labels = item["labels"][:MAX_LEN]
# [메모] encoding이 무엇인지 알려주세요 
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            label_ids = [LABEL2ID.get(l, 0) for l in labels]
            # 패딩 구간은 -100 (loss 무시)
            label_ids += [-100] * (MAX_LEN - len(label_ids))
            return {
                "input_ids":      encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels":         torch.tensor(label_ids[:MAX_LEN]),
            }

    dataset = NERDataset(ner_data)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(loader),
        num_training_steps=len(loader) * EPOCHS,
    )

    print(f"  device: {device}  |  epochs: {EPOCHS}  |  lr: {LR}\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()
        avg = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}")

    # 저장
    save_path = os.path.join(DL_MODELS_DIR, "event_ner")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n  ✅ 저장: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 추론 함수 (FastAPI에서 import해서 사용)
# ══════════════════════════════════════════════════════════════════════════════
def classify_event(text: str, classifier=None) -> dict:
    """
    뉴스/이벤트 텍스트 → 이벤트 유형 분류

    classifier: transformers pipeline (None이면 내부에서 로드)
    반환: { "event_type": str, "score": float, "impact": dict }
    """
    if classifier is None:
        from transformers import pipeline as hf_pipeline
        meta_path = os.path.join(DL_MODELS_DIR, "event_meta.json")
        if os.path.exists(os.path.join(DL_MODELS_DIR, "event_classifier")):
            classifier = hf_pipeline(
                "zero-shot-classification",
                model=os.path.join(DL_MODELS_DIR, "event_classifier"),
                device=-1,
            )
        else:
            classifier = hf_pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                device=-1,
            )

    result = classifier(text, candidate_labels=EVENT_TYPES, multi_label=False)
    top_type  = result["labels"][0]
    top_score = result["scores"][0]

    return {
        "event_type": top_type,
        "score":      round(top_score, 4),
        "impact":     EVENT_IMPACT.get(top_type, {}),
    }


def geocode_venue(venue_name: str) -> tuple | None:
    """
    장소명 → (lat, lon) 변환 (geopy Nominatim)
    반환: (lat, lon) 또는 None
    """
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="kride-event")
        location   = geolocator.geocode(venue_name + " 대한민국", timeout=5)
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CLI 진입점
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "finetune"],
        default="zero_shot",
        help="zero_shot: 즉시 사용 가능 | finetune: AI Hub 데이터 필요",
    )
    parser.add_argument(
        "--data_dir",
        default="",
        help="AI Hub 이벤트 추출 데이터 디렉토리 (finetune 모드 필수)",
    )
    args = parser.parse_args()

    if args.mode == "zero_shot":
        build_zero_shot_classifier()
    else:
        if not args.data_dir:
            print("❌ --data_dir 인자가 필요합니다.")
            sys.exit(1)
        build_ner_finetune(args.data_dir)

    print("\n✅ build_event_ner.py 완료")
