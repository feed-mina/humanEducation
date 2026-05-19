"""
chatbot_server.py — K-Ride 챗봇 FastAPI 서버 (포트 8001)
========================================================
실행:
    cd subproject/NLP && python -m chatbot.chatbot_server
    또는
    cd subproject/NLP && uvicorn chatbot.chatbot_server:app --port 8001 --reload

엔드포인트:
    POST /chat        — 챗봇 대화
    POST /chat/reset  — 세션 초기화
    GET  /health      — 서버 상태
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot.chatbot_chain import chat, reset_session

app = FastAPI(
    title="K-Ride Chatbot API",
    description="멀티쿼리 + 리랭커 기반 한국 여행 챗봇",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 스키마 ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    context: dict | None = None


class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = Field(default_factory=list)
    pois: list[dict] = Field(default_factory=list)


class ResetRequest(BaseModel):
    session_id: str = "default"


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "kride-chatbot"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """챗봇 대화"""
    result = chat(
        message=req.message,
        session_id=req.session_id,
        context=req.context,
    )
    return ChatResponse(**result)


@app.post("/chat/reset")
def reset_endpoint(req: ResetRequest):
    """세션 초기화"""
    reset_session(req.session_id)
    return {"status": "ok", "session_id": req.session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
