"""
build_event_ner.py
==================
이벤트 감지 파이프라인 (Phase 3-7)

# [메모] Zero-shot이 무엇인지 알려주세요
# → Zero-shot(제로샷) 이란 모델을 별도로 학습(파인튜닝)하지 않고 바로 사용하는 방식입니다.
#   일반적으로 ML 모델은 "스포츠 경기" 라는 카테고리를 분류하려면 해당 라벨이 붙은
#   학습 데이터가 필요합니다. 반면 Zero-shot 은 사전학습된 언어 모델이 이미 언어의
#   의미를 이해하고 있기 때문에, 카테고리 이름(후보 라벨)만 텍스트로 넘겨주면
#   "이 문장이 어느 카테고리에 가장 가까운가?" 를 바로 판단합니다.
#   → 데이터 없이 즉시 사용 가능하지만, 정확도는 파인튜닝 모델보다 낮을 수 있습니다.
#
[ 방식 1 ] Zero-shot 분류 (빠른 대안 — 데이터 불필요)
# [메모] snunlp/KR-FinBert-SC 또는 MoritzLaurer/mDeBERTa-v3-base-mnli-xnli 이 무엇을 가르키는지 알려주세요
# → 두 모델 모두 HuggingFace에 올라온 사전학습된 언어 모델입니다.
#
#   ■ snunlp/KR-FinBert-SC
#     · 서울대(SNU)가 한국어 금융 뉴스로 학습한 BERT 모델입니다.
#     · "SC" = Sentence Classification (문장 감성 분류)에 특화되어 있습니다.
#     · 한국어 전용이라 한국어 이해 능력은 뛰어나지만,
#       zero-shot 분류를 공식 지원하지 않아 파인튜닝 없이는 사용이 제한됩니다.
#
#   ■ MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
#     · Microsoft의 DeBERTa 구조를 다국어(100개 언어)로 확장한 모델입니다.
#     · MNLI(영어) + XNLI(다국어) 데이터로 학습되어 zero-shot 분류를 공식 지원합니다.
#     · "텍스트가 이 카테고리에 해당하는가?" 를 직접 판단하므로 데이터 없이 바로 쓸 수 있습니다.
#     → 이 프로젝트에서는 mDeBERTa를 사용합니다 (한국어 + zero-shot 동시 지원).

  snunlp/KR-FinBert-SC 또는 MoritzLaurer/mDeBERTa-v3-base-mnli-xnli 사용
  → 뉴스 텍스트를 5개 이벤트 유형으로 분류
# [메모] KLUE-BERT NER 파인튜닝이 무엇인지 알려주세요 그리고 각 라벨이 무엇인지 알려주세요
# → KLUE-BERT NER 파인튜닝이란:
#     KLUE-BERT = 카카오가 한국어 대규모 말뭉치로 학습한 BERT 모델입니다.
#     NER(Named Entity Recognition) = "개체명 인식" 으로, 문장 속에서
#     사람 이름, 장소명, 날짜, 이벤트명 등을 자동으로 찾아내는 작업입니다.
#     파인튜닝(Fine-tuning) = 이미 학습된 KLUE-BERT에 우리 프로젝트 라벨이 붙은
#     데이터를 추가로 학습시켜서 KRide 전용 NER 모델로 만드는 과정입니다.
#
#   라벨 설명 (BIO 태깅 방식):
#     O       = Outside  → 아무 의미 없는 일반 글자 (이벤트/장소 아님)
#     B-LOC   = Begin Location  → 장소명의 첫 번째 글자
#               예) "잠실야구장" → B-LOC(잠), I-LOC(실), I-LOC(야), ...
#     I-LOC   = Inside Location → 장소명의 두 번째 이후 글자
#     B-EVT   = Begin Event     → 이벤트명의 첫 번째 글자
#               예) "콘서트" → B-EVT(콘), I-EVT(서), I-EVT(트)
#     I-EVT   = Inside Event    → 이벤트명의 두 번째 이후 글자
#
#   B/I 구분이 필요한 이유: 연속된 글자 중 어디서 새 단어가 시작하는지 표시하기 위함.
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
    # → 원인: transformers 최신 버전(4.30+)에서 AdamW가 transformers에서 제거되었습니다.
    #   해결: AdamW를 torch.optim 에서 가져오도록 변경합니다.
    try:
        import torch
        from torch.optim import AdamW                          # ← 수정: torch.optim에서 import
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
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
            # → encoding이란 "텍스트를 모델이 이해할 수 있는 숫자 배열로 변환한 결과물" 입니다.
            #   tokenizer()를 호출하면 아래 3가지 정보가 딕셔너리 형태로 반환됩니다:
            #
            #   ① input_ids      : 각 글자/단어를 모델 어휘사전의 숫자 ID로 변환한 배열
            #                      예) ["잠","실"] → [12543, 7821, ...]
            #   ② attention_mask : 실제 토큰(1)과 패딩(0)을 구분하는 마스크
            #                      예) [1, 1, 1, 0, 0, ...] (뒤쪽 0이 패딩)
            #   ③ token_type_ids : 문장 A/B 구분용 (단일 문장이면 전부 0)
            #
            #   인자 설명:
            #     is_split_into_words=True → 이미 글자 단위로 나눠진 리스트를 입력받음
            #     max_length=MAX_LEN       → 최대 128 토큰으로 자르기
            #     padding="max_length"     → 128 미만이면 [PAD] 토큰으로 채우기
            #     truncation=True          → 128 초과면 뒤를 잘라내기
            #     return_tensors="pt"      → PyTorch 텐서 형태로 반환 (pt = pytorch)
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
