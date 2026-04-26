"""
visualize_weather_lstm.py
=========================
WeatherLSTM 모델 성능 시각화 및 보고서 저장

[ 실행 전 필요 ]
  python src/dl/build_weather_lstm.py  ← 먼저 실행해야 모델/스케일러 존재

[ 실행 ]
  python src/dl/visualize_weather_lstm.py

[ 출력 파일 ]
  report/figures/weather_lstm_confusion_matrix.png   ← Confusion Matrix
  report/figures/weather_lstm_class_metrics.png      ← 클래스별 Precision/Recall/F1 바 차트
  report/figures/weather_lstm_class_distribution.png ← 클래스 분포 파이 차트
  report/figures/weather_lstm_learning_curve.png     ← 학습 곡선 (weather_history.json 있을 때만)
  report/tables/weather_lstm_classification_report.csv
  report/tables/weather_lstm_performance_summary.json
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # src/dl → src → kride-project
except NameError:
    BASE_DIR = os.getcwd()

DL_DATA_DIR   = os.path.join(BASE_DIR, "data", "dl", "kma_weather_raw")
DL_MODELS_DIR = os.path.join(BASE_DIR, "models", "dl")
FIG_DIR       = os.path.join(BASE_DIR, "report", "figures")
TBL_DIR       = os.path.join(BASE_DIR, "report", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

LABEL_NAMES = ["맑음(0)", "흐림(1)", "비·눈(2)"]
LABEL_COLORS = ["#FFD700", "#A0A0A0", "#4A90D9"]

SEQ_LEN    = 14
INPUT_SIZE = 8


# ── 필수 파일 체크 ─────────────────────────────────────────────────────────────
def check_required_files():
    required = {
        "모델 가중치": os.path.join(DL_MODELS_DIR, "weather_lstm.pt"),
        "스케일러":    os.path.join(DL_MODELS_DIR, "weather_scaler.pkl"),
        "메타 정보":   os.path.join(DL_MODELS_DIR, "weather_meta.json"),
    }
    missing = [name for name, path in required.items() if not os.path.exists(path)]
    if missing:
        print("❌ 필수 파일 없음:")
        for m in missing:
            print(f"   - {m}")
        print("   → python src/dl/build_weather_lstm.py 먼저 실행하세요.")
        sys.exit(1)


# ── 데이터 재로드 및 test split 재생성 ──────────────────────────────────────────
def rebuild_test_set():
    """
    build_weather_lstm.py와 동일한 파이프라인으로 test set 재생성.
    seasonal_split은 시간 순서 기반이므로 재현 가능.
    """
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))
    from dl.build_weather_lstm import load_kma_csvs, preprocess, make_sequences, seasonal_split

    print("  데이터 로드 중...")
    df_raw = load_kma_csvs(DL_DATA_DIR)
    if df_raw.empty:
        print(f"  ❌ CSV 없음: {DL_DATA_DIR}")
        sys.exit(1)

    df = preprocess(df_raw)
    X, y, dates = make_sequences(df, SEQ_LEN)
    if X is None:
        print("  ❌ 시퀀스 생성 실패")
        sys.exit(1)

    scaler = joblib.load(os.path.join(DL_MODELS_DIR, "weather_scaler.pkl"))
    N, T, F = X.shape
    X_flat = X.reshape(-1, F)
    X_flat = scaler.transform(X_flat)
    X = X_flat.reshape(N, T, F).astype(np.float32)

    _, _, X_test, _, _, y_test = seasonal_split(X, y, dates, 0.7, 0.2, 0.1)
    print(f"  Test set: {len(X_test)}개 시퀀스")
    return X_test, y_test


# ── 모델 추론 ──────────────────────────────────────────────────────────────────
def run_inference(X_test, y_test):
    import torch
    from dl.build_weather_lstm import build_model

    with open(os.path.join(DL_MODELS_DIR, "weather_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    WeatherLSTM = build_model()
    model = WeatherLSTM(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        num_classes=meta["num_classes"],
    )
    model.load_state_dict(
        torch.load(os.path.join(DL_MODELS_DIR, "weather_lstm.pt"), map_location="cpu")
    )
    model.eval()

    x_tensor = torch.from_numpy(X_test)
    with torch.no_grad():
        logits = model(x_tensor)
        y_pred = logits.argmax(dim=1).numpy()

    return y_pred, np.array(y_test), meta


# ── 1. Confusion Matrix ────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("WeatherLSTM — Confusion Matrix", fontsize=14, fontweight="bold", y=1.01)

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Count", "Normalized (Recall)"],
        [".0f", ".2f"],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(LABEL_NAMES, fontsize=9)
        ax.set_yticklabels(LABEL_NAMES, fontsize=9)
        thresh = data.max() / 2.0
        for i in range(3):
            for j in range(3):
                ax.text(
                    j, i,
                    format(data[i, j], fmt),
                    ha="center", va="center",
                    color="white" if data[i, j] > thresh else "black",
                    fontsize=11,
                )

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "weather_lstm_confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


# ── 2. 클래스별 Precision / Recall / F1 바 차트 ────────────────────────────────
def plot_class_metrics(y_true, y_pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2])

    x = np.arange(3)
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, prec, width, label="Precision", color="#4A90D9", alpha=0.85)
    ax.bar(x,         rec,  width, label="Recall",    color="#E67E22", alpha=0.85)
    ax.bar(x + width, f1,   width, label="F1-Score",  color="#27AE60", alpha=0.85)

    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        ax.text(i - width, p + 0.01, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i,         r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width, f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("WeatherLSTM — Per-Class Precision / Recall / F1", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "weather_lstm_class_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


# ── 3. 클래스 분포 파이 차트 ──────────────────────────────────────────────────
def plot_class_distribution(y_true):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.bincount(y_true, minlength=3)
    labels = [f"{n}\n{c:,}개 ({c/counts.sum()*100:.1f}%)"
              for n, c in zip(LABEL_NAMES, counts)]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts = ax.pie(
        counts, labels=labels, colors=LABEL_COLORS,
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 11},
    )
    ax.set_title("WeatherLSTM — Test Set Class Distribution", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "weather_lstm_class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


# ── 4. 학습 곡선 (weather_history.json 있을 때만) ─────────────────────────────
def plot_learning_curve():
    history_path = os.path.join(DL_MODELS_DIR, "weather_history.json")
    if not os.path.exists(history_path):
        print(f"  ⚠️  학습 곡선 스킵 — weather_history.json 없음")
        print(f"     (다음 학습부터 자동 저장됩니다)")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    epochs     = range(1, len(history["train_loss"]) + 1)
    train_loss = history["train_loss"]
    val_acc    = history["val_acc"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("WeatherLSTM — Learning Curve", fontsize=14, fontweight="bold")

    ax1.plot(epochs, train_loss, color="#E74C3C", linewidth=2, marker="o", markersize=3)
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_acc, color="#3498DB", linewidth=2, marker="o", markersize=3)
    best_epoch = int(np.argmax(val_acc)) + 1
    best_acc   = max(val_acc)
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7,
                label=f"Best epoch={best_epoch} ({best_acc:.4f})")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "weather_lstm_learning_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


# ── 5. Classification Report CSV ──────────────────────────────────────────────
def save_classification_report(y_true, y_pred):
    from sklearn.metrics import classification_report

    report_dict = classification_report(
        y_true, y_pred,
        target_names=["맑음", "흐림", "비·눈"],
        output_dict=True,
    )
    df_report = pd.DataFrame(report_dict).T.round(4)
    out = os.path.join(TBL_DIR, "weather_lstm_classification_report.csv")
    df_report.to_csv(out, encoding="utf-8-sig")
    print(f"  ✅ {out}")
    print()
    print(classification_report(y_true, y_pred, target_names=["맑음", "흐림", "비·눈"]))


# ── 6. 성능 요약 JSON ──────────────────────────────────────────────────────────
def save_performance_summary(y_true, y_pred, meta):
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average="macro")
    prec, rec, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2])

    summary = {
        "model":          "WeatherLSTM",
        "test_accuracy":  round(float(acc), 4),
        "f1_macro":       round(float(f1), 4),
        "best_val_acc":   meta.get("best_val_acc"),
        "architecture": {
            "hidden_size": meta["hidden_size"],
            "num_layers":  meta["num_layers"],
            "seq_len":     meta["seq_len"],
            "input_size":  meta["input_size"],
        },
        "per_class": {
            name: {
                "precision": round(float(p), 4),
                "recall":    round(float(r), 4),
                "f1":        round(float(f), 4),
            }
            for name, p, r, f in zip(["맑음", "흐림", "비·눈"], prec, rec, f1_cls)
        },
        "class_distribution": {
            name: int(c)
            for name, c in zip(["맑음", "흐림", "비·눈"], np.bincount(y_true, minlength=3))
        },
    }

    out = os.path.join(TBL_DIR, "weather_lstm_performance_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {out}")
    return summary


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("WeatherLSTM 시각화 시작")
    print("=" * 60)

    check_required_files()

    print("\n[1/3] Test set 재생성 중...")
    X_test, y_test = rebuild_test_set()

    print("\n[2/3] 모델 추론 중...")
    y_pred, y_true, meta = run_inference(X_test, y_test)

    print("\n[3/3] 시각화 및 저장 중...")
    plot_confusion_matrix(y_true, y_pred)
    plot_class_metrics(y_true, y_pred)
    plot_class_distribution(y_true)
    plot_learning_curve()
    save_classification_report(y_true, y_pred)
    summary = save_performance_summary(y_true, y_pred, meta)

    print("\n" + "=" * 60)
    print("✅ 시각화 완료")
    print(f"   Test Acc  : {summary['test_accuracy']:.4f}")
    print(f"   F1-Macro  : {summary['f1_macro']:.4f}")
    print(f"   저장 위치 : report/figures/  |  report/tables/")
    print("=" * 60)


if __name__ == "__main__":
    main()
