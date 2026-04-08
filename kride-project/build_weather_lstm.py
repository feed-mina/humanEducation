"""
build_weather_lstm.py
=====================
기상청 과거 날씨 CSV → WeatherLSTM 학습 → models/dl/weather_lstm.pt

[ 데이터 준비 ]
  기상자료개방포털 (data.kma.go.kr)
  → 지상관측자료 → 일별 자료 → 서울/경기 관측소 → CSV 다운
  저장 경로: kride-project/data/dl/kma_weather_raw/

[ 실행 ]
  python kride-project/build_weather_lstm.py
  python kride-project/build_weather_lstm.py --data_dir data/dl/kma_weather_raw

[ 출력 ]
  models/dl/weather_lstm.pt        ← 학습된 모델 가중치
  models/dl/weather_scaler.pkl     ← 입력 피처 StandardScaler
  models/dl/weather_meta.json      ← 학습 설정 및 성능 메타

[ WeatherLSTM 입출력 ]
  입력 (시퀀스 길이=14):
    [월, 일, 요일, 기온_평균, 강수량, 풍속, 습도, sgg_idx]
  출력:
    3분류 — 0:맑음 / 1:흐림 / 2:비·눈
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "kride-project"))
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.getcwd()

DL_DATA_DIR   = os.path.join(BASE_DIR, "data", "dl")
DL_MODELS_DIR = os.path.join(BASE_DIR, "models", "dl")
os.makedirs(DL_DATA_DIR,   exist_ok=True)
os.makedirs(DL_MODELS_DIR, exist_ok=True)

# 날씨 3분류 기준
# 강수량 >= 1mm → 비·눈(2) / 운량 >= 8 → 흐림(1) / 나머지 → 맑음(0)
SEQ_LEN    = 14    # 과거 14일치 시퀀스
INPUT_SIZE = 8     # 피처 수
NUM_CLASS  = 3
HIDDEN     = 64
LAYERS     = 2
DROPOUT    = 0.2
EPOCHS     = 30
BATCH      = 64
LR         = 1e-3

WEATHER_PENALTY = {0: 0.0, 1: -0.05, 2: -0.20}   # safety_score 보정값
# [메모] safety_score 보정을 어떻게 진행하는지와 왜 하는지 궁금합니다 
WEATHER_LABEL   = {0: "맑음", 1: "흐림", 2: "비·눈"}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드 및 전처리
# ══════════════════════════════════════════════════════════════════════════════
def load_kma_csvs(data_dir: str) -> pd.DataFrame:
    """기상청 일별 관측 CSV 병합"""
    csv_files = (
        glob.glob(os.path.join(data_dir, "*.csv"))
        + glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    )
    if not csv_files:
        return pd.DataFrame()

    dfs = []
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, encoding="utf-8-sig") # [메모] utf-8-sig와 cp949의 차이는 무엇인가요?
        except UnicodeDecodeError:
            df = pd.read_csv(fpath, encoding="cp949")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"  원본 CSV {len(csv_files)}개 병합 → {merged.shape}")
    return merged


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼 정규화 + 날씨 라벨 생성"""
    # 컬럼명 후보 매핑 (기상청 CSV 포맷 다양)
    col_map = {}
    candidates = {
        "date":   ["일시", "날짜", "date", "Date"],
        "stn":    ["지점", "지점명", "stn", "station"],
        "tavg":   ["평균기온(°C)", "평균기온", "tavg", "TavgC"],
        "precip": ["일강수량(mm)", "강수량(mm)", "precip", "RnDayC"],
        "wspd":   ["평균풍속(m/s)", "풍속(m/s)", "wspd", "WavgC"],
        "humid":  ["평균상대습도(%)", "상대습도(%)", "humid", "HmAvgP"],
        "cloud":  ["평균전운량(1/10)", "전운량", "cloud", "CAvgO"],
    }
    for key, names in candidates.items():
        for n in names:
            if n in df.columns:
                col_map[n] = key
                break

    df = df.rename(columns=col_map)

    for col in ["tavg", "precip", "wspd", "humid", "cloud"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["month"]      = df["date"].dt.month
        df["day"]        = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.dayofweek
    else:
        df["month"] = df["day"] = df["day_of_week"] = 0

    # 관측소 → 정수 인덱스
    if "stn" in df.columns:
        le = LabelEncoder()
        df["sgg_idx"] = le.fit_transform(df["stn"].astype(str))
    else:
        df["sgg_idx"] = 0

    # 날씨 3분류 라벨
    def label_weather(row):
        if row["precip"] >= 1.0:
            return 2   # 비·눈
        if row.get("cloud", 0) >= 8:
            return 1   # 흐림
        return 0       # 맑음

    df["weather_label"] = df.apply(label_weather, axis=1)

    df = df.sort_values(["sgg_idx", "date"] if "date" in df.columns else ["sgg_idx"])
    df = df.reset_index(drop=True)
    return df


def make_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    시계열 데이터 → (X, y) 시퀀스 생성
    X: (N, seq_len, INPUT_SIZE)
    y: (N,) — 다음 날 날씨 라벨
    """
    # [메모] X. y값에 대해 설명해주세요 
    FEAT_COLS = ["month", "day", "day_of_week", "tavg", "precip", "wspd", "humid", "sgg_idx"]
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0

    X_list, y_list = [], []
    for sgg in df["sgg_idx"].unique():
        sub = df[df["sgg_idx"] == sgg].reset_index(drop=True)
        if len(sub) < seq_len + 1:
            continue
        feats  = sub[FEAT_COLS].values.astype(float)
        labels = sub["weather_label"].values
        for i in range(len(sub) - seq_len):
            X_list.append(feats[i:i + seq_len])
            y_list.append(labels[i + seq_len])

    if not X_list:
        return np.array([]), np.array([])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: WeatherLSTM 모델 정의
# ══════════════════════════════════════════════════════════════════════════════
def build_model():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  ❌ PyTorch 없음: pip install torch")
        sys.exit(1)

    class WeatherLSTM(nn.Module):
        def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN,
                     num_layers=LAYERS, num_classes=NUM_CLASS, dropout=DROPOUT):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            out    = self.dropout(out[:, -1, :])   # 마지막 타임스텝
            return self.fc(out)

    return WeatherLSTM


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: 학습
# ══════════════════════════════════════════════════════════════════════════════
def train(data_dir: str):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print("=" * 65)
    print("STEP 1: 데이터 로드")
    print("=" * 65)

    df_raw = load_kma_csvs(data_dir)
    if df_raw.empty:
        print(f"  ❌ CSV 없음: {data_dir}")
        print("  기상자료개방포털(data.kma.go.kr) → 지상관측자료 → 일별 자료 다운")
        sys.exit(1)

    df = preprocess(df_raw)
    print(f"  전처리 후: {df.shape}  날씨 분포: {df['weather_label'].value_counts().to_dict()}\n")

    print("=" * 65)
    print("STEP 2: 시퀀스 생성 + 스케일링")
    print("=" * 65)

    X, y = make_sequences(df, SEQ_LEN)
    if len(X) == 0:
        print("  ❌ 시퀀스 생성 실패 (데이터 부족 — 관측소당 최소 15일 이상 필요)")
        sys.exit(1)
# [메모] 여기서 15일 이상 필요하다는 점은 어떻게 알 수 있나요 

    print(f"  X: {X.shape}  y: {y.shape}")

    # 피처 스케일링 (seq 차원 펼쳐서 fit)
    N, T, F = X.shape
    scaler  = StandardScaler()
    X_flat  = X.reshape(-1, F)
    X_flat  = scaler.fit_transform(X_flat)
    X       = X_flat.reshape(N, T, F).astype(np.float32)

# [메모]  X_flat 이 구해지는 방법이 잘 이해가 가지 않습니다


    scaler_path = os.path.join(DL_MODELS_DIR, "weather_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ weather_scaler.pkl 저장\n")
    # [메모] joblib.dump는 머신러닝에서 사용하는걸로 알고 있씁니다. joblib.save로 해야 되지 않나요? 


    # Train / Val 분할 (80:20)
    split  = int(N * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}\n")

    tr_ds  = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH)

    print("=" * 65)
    print("STEP 3: WeatherLSTM 학습")
    print("=" * 65)

    WeatherLSTM = build_model()
    model       = WeatherLSTM().to(device)
    criterion   = nn.CrossEntropyLoss()
    optimizer   = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
# [메모] optimizer.zero_grad()는 왜 해야 하나요?

        # 검증
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds   = model(xb.to(device)).argmax(dim=1).cpu()
                correct += (preds == yb).sum().item()
                total   += len(yb)
        val_acc = correct / total if total else 0.0
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={tr_loss/len(tr_dl):.4f}  val_acc={val_acc:.4f}")

    # 최적 가중치 저장
    model.load_state_dict(best_state)
    model_path = os.path.join(DL_MODELS_DIR, "weather_lstm.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n  ✅ weather_lstm.pt 저장 → {model_path}")
    print(f"  최고 val_acc: {best_val_acc:.4f}")

    # 메타 저장
    meta = {
        "input_size":   INPUT_SIZE,
        "hidden_size":  HIDDEN,
        "num_layers":   LAYERS,
        "num_classes":  NUM_CLASS,
        "seq_len":      SEQ_LEN,
        "best_val_acc": round(best_val_acc, 4),
        "weather_label": WEATHER_LABEL,
        "weather_penalty": WEATHER_PENALTY,
    }
    meta_path = os.path.join(DL_MODELS_DIR, "weather_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  ✅ weather_meta.json 저장")


# ══════════════════════════════════════════════════════════════════════════════
# 추론 함수 (FastAPI에서 import해서 사용)
# ══════════════════════════════════════════════════════════════════════════════
def predict_weather(
    seq: np.ndarray,   # shape: (SEQ_LEN, INPUT_SIZE)
    model_path: str | None = None,
    scaler_path: str | None = None,
    meta_path: str | None = None,
) -> dict:
    """
    과거 14일 시퀀스 → 다음 날 날씨 예측

    반환:
      { "class": int, "label": str, "proba": [float, float, float],
        "safety_penalty": float }
    """
    import torch

    model_path  = model_path  or os.path.join(DL_MODELS_DIR, "weather_lstm.pt")
    scaler_path = scaler_path or os.path.join(DL_MODELS_DIR, "weather_scaler.pkl")
    meta_path   = meta_path   or os.path.join(DL_MODELS_DIR, "weather_meta.json")

    if not os.path.exists(model_path):
        return {"class": 0, "label": "맑음", "proba": [1.0, 0.0, 0.0], "safety_penalty": 0.0,
                "note": "모델 없음 (build_weather_lstm.py 실행 필요)"}

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    scaler = joblib.load(scaler_path)
    flat   = scaler.transform(seq.reshape(-1, INPUT_SIZE))
    x      = torch.from_numpy(flat.reshape(1, SEQ_LEN, INPUT_SIZE).astype(np.float32))

    WeatherLSTM = build_model()
    model       = WeatherLSTM(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        num_classes=meta["num_classes"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(x)
        proba  = torch.softmax(logits, dim=1)[0].tolist()
        cls    = int(torch.argmax(logits, dim=1).item())

    return {
        "class":          cls,
        "label":          WEATHER_LABEL[cls],
        "proba":          [round(p, 4) for p in proba],
        "safety_penalty": WEATHER_PENALTY[cls],
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI 진입점
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=os.path.join(DL_DATA_DIR, "kma_weather_raw"),
        help="기상청 일별 관측 CSV 디렉토리",
    )  # [메모] 여기서 ArgumentParser함수는 어떤건가요 

    # [메모] parser.parse_args는 어떤의미인가요 
    args = parser.parse_args()
    train(args.data_dir)

    print("\n" + "=" * 65)
    print("✅ build_weather_lstm.py 완료")
    print("=" * 65)
