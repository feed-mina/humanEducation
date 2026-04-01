import streamlit as st 


st.title('테스트')
st.write("이곳에 텍스트나 모델 결과를 출력합니다.")

data = {"모델": ["Model A", "Model B"], "정확도(%)": [88.5, 92.1]}
st.table(data)