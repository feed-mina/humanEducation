from fastapi import APIRouter, HTTPException
from app.services.waterPredict02 import predict_water_electricity, get_forecast_data
import pandas as pd

router = APIRouter()

@router.get("/")
async def predict():
    """
    예측 데이터를 조회하고 반환합니다.
    """
    try:
        # 예측 수행
        predict_water_electricity()

        # 예측 데이터 조회
        forecast_data = get_forecast_data()
        # 1. NaN 값을 0이나 적절한 값으로 채우기 (JSON은 NaN을 처리 못함)
        forecast_data = forecast_data.fillna(0)

        # 2. 모든 데이터를 파이썬 표준 타입으로 강제 변환
        # .astype(float) 등을 쓰거나, 아래처럼 dict 변환 시 안전하게 처리
        result = forecast_data.to_dict(orient='records')
        
        # 만약 데이터에 날짜나 특수 타입이 많다면, 
        # 리턴하기 전에 데이터를 한번 더 정제하는 로직이 필요할 수 있어.
        return result
    except Exception as e:
        # 에러 내용을 터미널에 더 자세히 찍어보자
        print(f"Error detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))
