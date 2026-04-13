from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agents.graph import create_graph
import os
import json
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI(title="LangGraph Agent Chatbot")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str
    max_revisions: int = 3

class ChatResponse(BaseModel):
    response: str
    plan: str
    revision_count: int

# Initialize the graph
graph = create_graph()

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        initial_state = {
            "messages": [request.message],
            "max_revisions": request.max_revisions,
            "revision_count": 0,
            "revisions": []
        }
        
        result = graph.invoke(initial_state)
        
        return {
            "response": result.get("draft", "No response generated."),
            "plan": result.get("plan", "No plan generated."),
            "revision_count": result.get("revision_count", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            user_message = request_data.get("message")
            max_revisions = request_data.get("max_revisions", 3)

            initial_state = {
                "messages": [user_message],
                "max_revisions": max_revisions,
                "revision_count": 0,
                "revisions": []
            }

            # Stream the graph execution
            async for event in graph.astream(initial_state):
                for key, value in event.items():
                    # Prepare the message to send to the client
                    response_message = {
                        "type": "update",
                        "node": key,
                        "data": value
                    }
                    await websocket.send_text(json.dumps(response_message))
            
            # Send completion message
            await websocket.send_text(json.dumps({"type": "complete"}))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
