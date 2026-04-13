from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    """
    클라이언트가 보내는 채팅 요청 모델
    """
    message: str
    session_id: str  # 대화의 연속성을 위한 세션 ID

class ChatResponse(BaseModel):
    """
    서버가 반환하는 채팅 응답 모델 (확장됨)
    """
    reply: str
    user_profile: Dict[str, Any]  # 현재 사용자 프로필 상태
    search_results: Optional[str] # 현재 검색 결과