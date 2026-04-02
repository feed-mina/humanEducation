import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent

# 1. 파일 로드 함수 (캐싱 적용으로 성능 최적화)
@st.cache_resource
def load_assets():
    model = joblib.load(BASE_DIR / 'model' / 'attrition_model.pkl')
    scaler = joblib.load(BASE_DIR / 'model' / 'scaler.pkl')
    return model, scaler

def main():
    st.title("퇴사 여부 예측 시스템")
    
    try:
        model, scaler = load_assets()
    except FileNotFoundError:
        st.error("모델 파일(.pkl)을 찾을 수 없습니다. 먼저 학습을 진행해 주세요.")
        return

    # 2. UI 구성
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        sat_level = st.slider("직원 만족도 (0.0 ~ 1.0)", 0.0, 1.0, 0.5)
        num_projects = st.number_input("참여 프로젝트 수 (1 ~ 10)", 1, 10, 3)
        time_spend = st.number_input("회사 근무 기간 (1 ~ 20년)", 1, 20, 3)

    # 3. 예측 로직
    if st.button("퇴사 여부 예측하기"):
        # 입력 데이터 배열 생성
        input_data = np.array([[sat_level, num_projects, time_spend]])
        
        # 저장된 스케일러로 변환
        input_scaled = scaler.transform(input_data)
        
        # 예측
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)

        st.write("---")
        if prediction[0] == 1:
            st.error(f"결과: 퇴사 위험군 (확률: {prob[0][1]:.2%})")
        else:
            st.success(f"결과: 잔류 예상 (확률: {prob[0][0]:.2%})")

if __name__ == "__main__":
    main()