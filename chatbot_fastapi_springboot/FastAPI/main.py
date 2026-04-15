# 사전설치 : pip install fastapi uvicorn pandas numpy scikit-learn tensorflow 
# 사전설치 : pip install sqlalchemy pymysql python-dotenv pydantic requests
import uvicorn
from fastapi import FastAPI
from app.routers import chat_router

app = FastAPI(title="Chatbot API", description="Ollama Gemma2 기반 챗봇 서비스")

app.include_router(chat_router.router)

@app.get("/")
async def root():
    return {"message": "Chatbot API Running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    