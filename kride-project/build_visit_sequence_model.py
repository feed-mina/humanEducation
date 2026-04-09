"""
build_visit_sequence_model.py
=============================================================
AI Hub 여행로그 방문지 시퀀스 → GRU 다음 방문지 예측 모델

[ 데이터 ]
  tn_visit_area_info_방문지정보_E.csv
  - TRAVEL_ID : 여행 고유 ID (2,560개)
  - VISIT_ORDER: 방문 순서 (1~38)
  - VISIT_AREA_NM: 방문지명 (결측 없음, 9,881 고유값)

[ 모델 ]
  Embedding(vocab, 32) → GRU(64, layers=2, dropout=0.3) → Linear → Softmax

[ 분할 방식 ]
  TRAVEL_ID 기준 70 / 20 / 10 (동일 여행이 분할 간 섞이지 않도록)

[ 슬라이딩 윈도우 ]
  여행 시퀀스 [A,B,C,D,E] → ([A,B,C,D], E), ([B,C,D,E], F), ...
  window_size=5 (입력 4, 타겟 1)

[ 출력 파일 ]
  models/dl/visit_seq_gru.pt      ← GRU 가중치
  models/dl/visit_seq_meta.json   ← 하이퍼파라미터·결과
  models/dl/poi_encoder.pkl       ← 방문지명 → 인덱스 인코더
  data/dl/visit_sequences.csv     ← 전처리된 시퀀스 데이터
"""

import os
import sys
import json
import pickle
import warnings
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── ArgumentParser ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",   type=int,   default=50)
parser.add_argument("--batch",    type=int,   default=64)
parser.add_argument("--lr",       type=float, default=1e-3)
parser.add_argument("--embed_d",  type=int,   default=32)
parser.add_argument("--gru_h",    type=int,   default=64)
parser.add_argument("--gru_l",    type=int,   default=2)
parser.add_argument("--dropout",  type=float, default=0.3)
parser.add_argument("--window",   type=int,   default=5,
                    help="슬라이딩 윈도우 크기 (입력 window-1개 → 다음 1개 예측)")
parser.add_argument("--patience", type=int,   default=7)
args = parser.parse_args()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))

AIHUB_DIR  = os.path.join(BASE_DIR, "data", "ai-hub",
                           "국내 여행로그 수도권_2023", "02.라벨링데이터")
DL_DATA    = os.path.join(BASE_DIR, "data", "dl")
MODELS_DL  = os.path.join(BASE_DIR, "models", "dl")

VISIT_CSV  = os.path.join(AIHUB_DIR, "tn_visit_area_info_방문지정보_E.csv")
SEQ_CSV    = os.path.join(DL_DATA,   "visit_sequences.csv")
MODEL_PT   = os.path.join(MODELS_DL, "visit_seq_gru.pt")
META_JSON  = os.path.join(MODELS_DL, "visit_seq_meta.json")
ENC_PKL    = os.path.join(MODELS_DL, "poi_encoder.pkl")

os.makedirs(DL_DATA,   exist_ok=True)
os.makedirs(MODELS_DL, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드 및 전처리
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 1: 데이터 로드 및 시퀀스 생성")
print("=" * 65)

if not os.path.exists(VISIT_CSV):
    print(f"  ❌ {VISIT_CSV} 없음.")
    sys.exit(1)

df = pd.read_csv(VISIT_CSV, encoding="utf-8-sig")
print(f"  원본 행수: {len(df):,}")

# 필수 컬럼만 사용
df = df[["TRAVEL_ID", "VISIT_ORDER", "VISIT_AREA_NM"]].copy()
df = df.dropna(subset=["VISIT_AREA_NM"])
df["VISIT_AREA_NM"] = df["VISIT_AREA_NM"].astype(str).str.strip()
df = df[df["VISIT_AREA_NM"] != ""]

# VISIT_ORDER 기준 정렬
df = df.sort_values(["TRAVEL_ID", "VISIT_ORDER"]).reset_index(drop=True)
print(f"  유효 행수: {len(df):,}")
print(f"  여행 수:   {df['TRAVEL_ID'].nunique():,}")
print(f"  장소 고유값: {df['VISIT_AREA_NM'].nunique():,}")

# ── LabelEncoder: 방문지명 → 정수 인덱스 ──────────────────────────────────────
le = LabelEncoder()
df["poi_idx"] = le.fit_transform(df["VISIT_AREA_NM"])
VOCAB = len(le.classes_)
print(f"  vocab 크기: {VOCAB:,}")

# ── 시퀀스 저장 ───────────────────────────────────────────────────────────────
df.to_csv(SEQ_CSV, index=False, encoding="utf-8-sig")
print(f"  시퀀스 저장: {SEQ_CSV}")

# ── 슬라이딩 윈도우 생성 ──────────────────────────────────────────────────────
WINDOW = args.window   # 입력 (window-1)개 → 타겟 1개
SEQ_IN = WINDOW - 1

samples = []  # (X_sequence, y_target, travel_id)
for tid, grp in df.groupby("TRAVEL_ID"):
    seq = grp["poi_idx"].tolist()
    if len(seq) < WINDOW:
        continue
    for i in range(len(seq) - SEQ_IN):
        x = seq[i : i + SEQ_IN]
        y = seq[i + SEQ_IN]
        samples.append((x, y, tid))

print(f"  슬라이딩 윈도우 샘플 수: {len(samples):,} "
      f"(window={WINDOW}, 입력={SEQ_IN}개 → 다음 1개)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: TRAVEL_ID 기준 70 / 20 / 10 분할
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: TRAVEL_ID 기준 train / val / test 분할")
print("=" * 65)

np.random.seed(42)
all_tids = df["TRAVEL_ID"].unique()
np.random.shuffle(all_tids)

n = len(all_tids)
n_train = int(n * 0.70)
n_val   = int(n * 0.20)

train_tids = set(all_tids[:n_train])
val_tids   = set(all_tids[n_train : n_train + n_val])
test_tids  = set(all_tids[n_train + n_val :])

def split_samples(samples, id_set):
    return [(x, y) for x, y, tid in samples if tid in id_set]

train_data = split_samples(samples, train_tids)
val_data   = split_samples(samples, val_tids)
test_data  = split_samples(samples, test_tids)

print(f"  TRAVEL_ID — train: {len(train_tids):,} / val: {len(val_tids):,} / test: {len(test_tids):,}")
print(f"  샘플 수   — train: {len(train_data):,} / val: {len(val_data):,} / test: {len(test_data):,}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Dataset / DataLoader
# ══════════════════════════════════════════════════════════════════════════════
class VisitSeqDataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor([x for x, _ in data], dtype=torch.long)
        self.y = torch.tensor([y for _, y in data], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(VisitSeqDataset(train_data), batch_size=args.batch, shuffle=True)
val_loader   = DataLoader(VisitSeqDataset(val_data),   batch_size=args.batch)
test_loader  = DataLoader(VisitSeqDataset(test_data),  batch_size=args.batch)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: GRU 모델 정의
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: GRU 모델 구성")
print("=" * 65)

class VisitGRU(nn.Module):
    def __init__(self, vocab, embed_d, gru_h, gru_l, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_d, padding_idx=0)
        self.gru   = nn.GRU(
            input_size=embed_d,
            hidden_size=gru_h,
            num_layers=gru_l,
            batch_first=True,
            dropout=dropout if gru_l > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(gru_h, vocab)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embed(x)           # (batch, seq_len, embed_d)
        out, _ = self.gru(emb)        # (batch, seq_len, gru_h)
        last = out[:, -1, :]          # 마지막 타임스텝 (batch, gru_h)
        return self.fc(self.drop(last))  # (batch, vocab)


model = VisitGRU(
    vocab=VOCAB,
    embed_d=args.embed_d,
    gru_h=args.gru_h,
    gru_l=args.gru_l,
    dropout=args.dropout,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"  vocab={VOCAB:,} / embed={args.embed_d} / GRU hidden={args.gru_h} / layers={args.gru_l}")
print(f"  파라미터 수: {total_params:,}")

# ── 클래스 불균형 처리: 방문 빈도 역수 가중치 ──────────────────────────────────
class_counts = np.bincount(
    [y for _, y in train_data], minlength=VOCAB
).astype(np.float32)
class_counts = np.where(class_counts == 0, 1, class_counts)  # 0 방지
weights = len(train_data) / (VOCAB * class_counts)
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: 학습
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5: 학습 (train / val)")
print("=" * 65)
print(f"  epochs={args.epochs} / batch={args.batch} / lr={args.lr} / patience={args.patience}")

def evaluate(loader):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            preds = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc


best_val_loss = float("inf")
best_epoch    = 0
patience_cnt  = 0
history       = []

for epoch in range(1, args.epochs + 1):
    # ── train ──
    model.train()
    train_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * len(y)
    train_loss /= len(train_loader.dataset)

    # ── val ──
    val_loss, val_acc = evaluate(val_loader)
    scheduler.step(val_loss)

    history.append({"epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "val_acc": val_acc})

    if epoch % 5 == 0 or epoch == 1:
        print(f"  epoch {epoch:3d} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    # ── early stopping ──
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch
        patience_cnt  = 0
        torch.save(model.state_dict(), MODEL_PT)
    else:
        patience_cnt += 1
        if patience_cnt >= args.patience:
            print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: 테스트 평가
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6: 테스트 평가 (test set — 학습 중 미사용)")
print("=" * 65)

model.load_state_dict(torch.load(MODEL_PT, map_location=DEVICE))
test_loss, test_acc = evaluate(test_loader)

# Top-5 accuracy
def top5_acc(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            top5   = logits.topk(5, dim=1).indices  # (batch, 5)
            correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total   += len(y)
    return correct / total

test_top5 = top5_acc(test_loader)
val_loss_best, val_acc_best = evaluate(val_loader)

print(f"  best_epoch : {best_epoch}")
print(f"  val_loss   : {val_loss_best:.4f}")
print(f"  val_acc    : {val_acc_best:.4f}")
print(f"  test_loss  : {test_loss:.4f}")
print(f"  test_acc   : {test_acc:.4f}  (Top-1)")
print(f"  test_top5  : {test_top5:.4f}  (Top-5)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: 저장
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 7: 모델 및 인코더 저장")
print("=" * 65)

# LabelEncoder 저장
with open(ENC_PKL, "wb") as f:
    pickle.dump(le, f)
print(f"  ✅ poi_encoder.pkl → {ENC_PKL}")

# 메타 저장
meta = {
    "vocab":        VOCAB,
    "embed_d":      args.embed_d,
    "gru_h":        args.gru_h,
    "gru_l":        args.gru_l,
    "dropout":      args.dropout,
    "window":       WINDOW,
    "seq_in":       SEQ_IN,
    "n_trips_train": len(train_tids),
    "n_trips_val":   len(val_tids),
    "n_trips_test":  len(test_tids),
    "n_samples_train": len(train_data),
    "n_samples_val":   len(val_data),
    "n_samples_test":  len(test_data),
    "best_epoch":   best_epoch,
    "val_acc":      round(val_acc_best, 4),
    "test_acc":     round(test_acc, 4),
    "test_top5":    round(test_top5, 4),
}
with open(META_JSON, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"  ✅ visit_seq_meta.json → {META_JSON}")
print(f"  ✅ visit_seq_gru.pt   → {MODEL_PT}")

print("\n" + "=" * 65)
print("✅ build_visit_sequence_model.py 완료")
print(f"   test_acc={test_acc:.4f}  test_top5={test_top5:.4f}")
print("=" * 65)
