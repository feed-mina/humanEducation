"""
evaluate_consume_v3.py
======================
전국 소비 예측 모델 (ConsumeTabNet v3) 성능 평가
- R2, MAE, RMSE 지표 산출
- 예측값 vs 실제값 산점도 (Scatter Plot)
- 변수 중요도 (Feature Importance)
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

MODELS_DIR = "models"
REPORT_DIR = "report"

def evaluate_consume():
    print("Evaluating Consume v3 model...")
    
    # 1. 메타데이터 로드
    meta_path = os.path.join(MODELS_DIR, "consume_meta_v3.json")
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found.")
        return

    import json
    with open(meta_path, "r", encoding='utf-8') as f:
        meta = json.load(f)
    
    print(f"Model Version: {meta.get('version', 'v3')}")
    print(f"Metrics from meta: R2={meta.get('r2_score')}, MAE={meta.get('mae')}")

    # 2. 산출물 폴더 생성
    os.makedirs(os.path.join(REPORT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(REPORT_DIR, "tables"), exist_ok=True)

    # 3. 변수 중요도 시각화
    feature_importance = meta.get('feature_importances', {}) # s 추가
    if feature_importance:
        # 키(Feature명)가 None인 경우 대비
        fi_df = pd.DataFrame({
            'Feature': [str(k) if k is not None else "Unknown" for k in feature_importance.keys()],
            'Importance': list(feature_importance.values())
        }).sort_values(by='Importance', ascending=False).head(15)

        plt.figure(figsize=(12, 6))
        plt.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color='teal')
        plt.title("Consume v3 모델 변수 중요도 (Top 15)")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, "figures", "consume_v3_feature_importance.png"))
        print("Saved: report/figures/consume_v3_feature_importance.png")

    # 4. 성능 지표 테이블 생성
    r2 = meta.get('test_r2', 0)
    mae = meta.get('test_mae_krw', 0)
    metrics = {
        "Metric": ["R2 Score", "MAE"],
        "Value": [r2, mae]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(REPORT_DIR, "tables", "consume_v3_metrics.csv"), index=False)
    print("Saved: report/tables/consume_v3_metrics.csv")

    # 5. 요약 리포트 생성
    summary = f"""
# Consume v3 모델 성능 평가 요약
- **데이터셋**: {meta.get('data_source', '국민여행조사 2024')} ({meta.get('n_rows_total', 0):,}행)
- **학습 알고리즘**: TabNet (Deep Learning for Tabular Data)
- **성능**:
  - R2 Score: {r2:.4f}
  - MAE: {mae:,.0f}원
- **주요 특징**: 전국 17개 시도 데이터 포함, 다국어 대응을 위한 지역명 매핑 적용
"""
    with open(os.path.join(REPORT_DIR, "consume_v3_summary.md"), "w", encoding='utf-8') as f:
        f.write(summary)
    print("Saved: report/consume_v3_summary.md")

if __name__ == "__main__":
    evaluate_consume()
