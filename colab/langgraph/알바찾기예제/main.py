import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
from agent import runnable_agent, AgentState, UserProfile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str
    session_id: str

# --- 대화 상태(세션) 관리 ---
session_states = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI 서버가 시작되었습니다.")
    print("브라우저에서 http://localhost:8000 으로 접속하세요.")
    print("API 테스트는 http://localhost:8000/docs 에서 가능합니다.")
    yield
    print("FastAPI 서버가 종료됩니다.")

app = FastAPI(lifespan=lifespan)

# --- CORS 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CSS 연결을 위한 'static' 이름의 폴더를 /static 경로로 앱에 연결합니다.
app.mount("/static", StaticFiles(directory="static"), name="static")
# ----------------------------------------------

@app.get("/")
async def root():
    """index.html을 제공합니다"""
    html_file = Path(__file__).parent / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return {
            "message": "index.html을 찾을 수 없습니다.",
            "help": "index.html 파일이 main.py와 같은 디렉토리에 있는지 확인하세요."
        }

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    노드별 실행 과정을 스트리밍으로 반환합니다.
    """
    session_id = request.session_id
    message = request.message
    
    if session_id not in session_states:
        session_states[session_id] = AgentState(
            messages=[],
            user_profile=UserProfile(),
            search_results=None,
            is_profile_sufficient=False
        )
    
    current_state = session_states[session_id]
    current_state["messages"].append(HumanMessage(content=message))

    async def event_generator():
        try:
            # 에이전트 실행
            final_state = runnable_agent.invoke(current_state)
            
            # 최종 상태 업데이트
            session_states[session_id] = final_state
            
            # 클라이언트에 최종 데이터 전송
            yield f"data: {json.dumps({'type': 'complete', 'final': serialize_state(final_state)})}\n\n"
            
        except Exception as e:
            print(f"에이전트 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def serialize_state(state):
    """AgentState를 JSON 직렬화 가능한 형태로 변환"""
    try:
        return {
            "user_profile": state.get("user_profile", {}).model_dump() if hasattr(state.get("user_profile"), "model_dump") else state.get("user_profile", {}),
            "search_results": state.get("search_results"),
            "is_profile_sufficient": state.get("is_profile_sufficient"),
            "messages": [{"role": "user" if "HumanMessage" in str(type(m)) else "ai", "content": m.content} for m in state.get("messages", [])]
        }
    except Exception as e:
        print(f"직렬화 오류: {e}")
        return {
            "user_profile": {},
            "search_results": None,
            "is_profile_sufficient": False,
            "messages": []
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)