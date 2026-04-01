import pandas as pd
import numpy as np
import joblib
import streamlit as st

# 3. Streamlit 앱
st.title('당뇨병 예측 시스템')
st.write('Glucose, BMI, Age 값을 입력하여 당뇨병 예측을 해보세요.')

# 사용자 입력받기
# Glucose (혈당 수치) 슬라이더
glucose = st.slider('Glucose (혈당 수치)', min_value=0, max_value=200, value=100)
# BMI (체질량지수) 슬라이더
bmi = st.slider('BMI (체질량지수)', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
# Age (나이) 슬라이더
age = st.slider('Age (나이)', min_value=0, max_value=100, value=30)

# 예측하기 버튼
if st.button('예측하기'):
    try:
        # joblib을 사용하여 저장된 모델 로드
        model = joblib.load('diabetes_model.pkl')
        # 입력값을 모델에 전달할 수 있는 형태로 변환
        input_data = np.array([[glucose, bmi, age]])
        # 모델 예측 수행
        prediction = model.predict(input_data)[0]

        # 예측 결과 출력
        if prediction == 1:
            st.error('예측 결과: 당뇨병 가능성이 높습니다.')
        else:
            st.success('예측 결과: 당뇨병 가능성이 낮습니다.')
    except FileNotFoundError:
        st.error("오류: 'diabetes_model.pkl' 파일을 찾을 수 없습니다. 모델 파일이 앱과 같은 디렉터리에 있는지 확인해주세요.")
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")

