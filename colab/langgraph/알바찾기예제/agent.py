import os
import json
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# tools.py에서 search_jobs_tool 임포트
from tools import search_jobs_tool 

# LLM 모델 초기화 (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # 또는 "gemini-pro"
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# --- 1. Agent State 정의 ---

class UserProfile(BaseModel):
    """사용자의 알바 희망 조건 프로필"""
    location: str | None = Field(default=None, description="희망 근무 지역")
    job_type: str | None = Field(default=None, description="희망 직종")
    hours: str | None = Field(default=None, description="희망 근무 시간")
    pay: str | None = Field(default=None, description="희망 시급")

class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_profile: UserProfile
    search_query: Optional[str] 
    search_results: Optional[str]
    is_profile_sufficient: bool

# --- 2. Nodes 정의 ---

def classify_intent_node(state: AgentState):
    """사용자의 최근 메시지를 분석하여 의도를 분류합니다."""
    last_message = state["messages"][-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "당신은 사용자의 의도를 분류하는 AI입니다. "
         "사용자의 최근 메시지와 현재 프로필 정보를 분석해서 의도를 분류하세요. "
         "현재 프로필: 지역={location}, 직종={job_type}, 시간={hours}, 시급={pay}\n"
         "사용자 메시지에서 새로운 프로필 정보가 있으면 'profile_update'\n"
         "프로필 정보가 충분(지역+직종)하면 'request_search'\n"
         "그 외의 경우는 'greeting_or_advice'\n"
         "반드시 다음 중 하나만 답변하세요: profile_update, request_search, greeting_or_advice"),
        ("human", "{last_message}")
    ])
    
    chain = prompt | llm
    profile = state["user_profile"]
    
    # 'user_profile'이 없는 초기 상태일 경우를 대비
    if not profile:
        profile = UserProfile()
        
    intent = chain.invoke({
        "last_message": last_message,
        "location": profile.location or "없음",
        "job_type": profile.job_type or "없음",
        "hours": profile.hours or "없음",
        "pay": profile.pay or "없음"
    }).content.strip().lower()
    
    print(f"[classify_intent] 메시지: {last_message}")
    print(f"[classify_intent] 현재 프로필: 지역={profile.location}, 직종={profile.job_type}, 시간={profile.hours}, 시급={profile.pay}")
    print(f"[classify_intent] 의도: {intent}")
    
    if "profile_update" in intent or (not (profile.location and profile.job_type)):
        return {"next_node": "update_profile"}
    elif "request_search" in intent:
        return {"next_node": "check_sufficiency"}
    else:
        # 단순 인사말 등의 응답
        return {"next_node": "generate_response"}


def update_profile_node(state: AgentState):
    """사용자 메시지에서 프로필 정보를 추출합니다."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "당신은 대화에서 사용자 프로필 정보를 추출하는 AI입니다. "
         "현재 프로필 상태와 최근 대화를 바탕으로 'user_profile'을 업데이트하세요. "
         "사용자가 정보를 제공하지 않은 필드는 'None'으로 두세요."
         "\n<current_profile>\n{profile}\n</current_profile>"),
        ("user", "{last_message}")
    ])
    
    structured_llm = llm.with_structured_output(UserProfile)
    chain = prompt | structured_llm
    
    last_message = state["messages"][-1].content
    
    # 'user_profile'이 None일 경우를 대비하여 기본값 사용
    current_profile = state.get("user_profile") or UserProfile()
    current_profile_dict = current_profile.model_dump()
    
    updated_profile_struct = chain.invoke({
        "profile": json.dumps(current_profile_dict, ensure_ascii=False),
        "last_message": last_message
    })
    
    # 기존 프로필에 업데이트된 내용을 병합
    update_data = {k: v for k, v in updated_profile_struct.model_dump().items() if v is not None}
    merged_profile = current_profile.model_copy(update=update_data)
    
    print(f"[update_profile] 업데이트: {merged_profile.model_dump()}")
    
    return {"user_profile": merged_profile}

def check_sufficiency_node(state: AgentState):
    """알바 검색에 필요한 최소 정보가 있는지 확인합니다."""
    profile = state.get("user_profile") or UserProfile()
    is_sufficient = bool(profile.location and profile.job_type)
    
    status = "충분" if is_sufficient else "부족"
    print(f"[✓ check_sufficiency] 정보 상태: {status}")
    print(f"  - 지역: {profile.location or '없음'}")
    print(f"  - 직종: {profile.job_type or '없음'}")
    
    return {"is_profile_sufficient": is_sufficient}

def format_search_query_node(state: AgentState):
    """검색 쿼리를 생성합니다."""
    profile = state.get("user_profile") or UserProfile()
    query_parts = []
    if profile.location: 
        query_parts.append(profile.location)
    if profile.job_type: 
        query_parts.append(profile.job_type)
    if profile.hours: 
        query_parts.append(profile.hours)
    if profile.pay: 
        pay_str = str(profile.pay)
        if pay_str.isdigit():
             query_parts.append(f"시급 {pay_str}원")
        else:
            query_parts.append(pay_str)
            
    query = " ".join(query_parts)
    print(f"[format_search_query] 쿼리: {query}")
    
    return {"search_query": query}

def call_search_tool_node(state: AgentState):
    """search_jobs_tool을 호출합니다."""
    query = state["search_query"]
    if not query:
        return {"search_results": "검색 쿼리가 생성되지 않았습니다."}
        
    result = search_jobs_tool.invoke(query)
    print(f"[call_search_tool] 결과:\n{result}")
    return {"search_results": result}

def generate_response_node(state: AgentState):
    """최종 응답을 생성합니다."""
    profile = state.get("user_profile") or UserProfile()
    is_sufficient = state.get("is_profile_sufficient") 
    
    if is_sufficient:
        # 검색을 수행한 경우의 프롬프트
        system_msg = (
            "당신은 친절한 AI 알바 추천 에이전트 '알바고(AlbaGo)'입니다. "
            "사용자의 조건에 맞는 알바 정보를 검색 결과를 바탕으로 요약하여 제시하세요. "
            "검색 결과를 그대로 보여주지 말고, 자연스러운 문장으로 만들어서 추천해주세요. "
            "답변 마지막에는 추가로 도움이 필요한지 묻거나 다른 조건으로 검색할지 물어보세요."
            "짧고 친절하게 응답하세요.\n"
            f"사용자 조건: {profile.location or ''} {profile.job_type or ''} "
            f"{profile.hours or ''} {profile.pay or ''}\n"
            f"검색 결과:\n{state.get('search_results') or '검색 결과 없음'}"
        )
    elif is_sufficient is False: 
        # 정보가 부족하여 추가 질문을 해야 하는 경우의 프롬프트
        missing = []
        if not profile.location:
            missing.append("희망 근무 지역")
        if not profile.job_type:
            missing.append("희망 직종")
        
        system_msg = (
            "당신은 친절한 AI 알바 추천 에이전트 '알바고(AlbaGo)'입니다. "
            f"알바를 추천해 드리기 위해 다음 정보가 더 필요합니다: {', '.join(missing)}\n"
            "사용자에게 이 정보들을 친절하게 질문하세요. "
            "같은 질문을 반복하지 말고, 이미 수집한 정보는 칭찬하고 부족한 것만 물어보세요.\n"
            f"현재까지 파악된 정보: 지역={profile.location or '미입력'}, 직종={profile.job_type or '미입력'}"
        )
    else: 
        system_msg = (
            "당신은 친절한 AI 알바 추천 에이전트 '알바고(AlbaGo)'입니다. "
            "사용자의 일반적인 질문이나 인사에 응답하세요. "
            "만약 알바 추천과 관련 없는 대화라면, 알바 추천이 필요하면 말해달라고 안내하세요."
        )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        *state["messages"],
        ("human", "위 대화 내용을 바탕으로 답변해주세요.")
    ])
    
    chain = prompt_template | llm
    
    response = chain.invoke({}) 
    
    print(f"[generate_response] 응답: {response.content}")
    
    new_messages = state["messages"] + [AIMessage(content=response.content)]
    
    return {
        "messages": new_messages,
        "user_profile": profile,
        "search_results": state.get("search_results") 
    }
    # ---------------------

# --- 3. Graph 정의 ---
def build_agent_graph():
    graph = StateGraph(AgentState)
    
    # 노드 추가
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("update_profile", update_profile_node)
    graph.add_node("check_sufficiency", check_sufficiency_node)
    graph.add_node("format_search_query", format_search_query_node)
    graph.add_node("call_search_tool", call_search_tool_node)
    graph.add_node("generate_response", generate_response_node)

    # 진입점
    graph.set_entry_point("classify_intent")

    # 조건부 엣지
    graph.add_conditional_edges(
        "classify_intent",
        lambda x: x.get("next_node"), 
        {
            "update_profile": "update_profile",
            "check_sufficiency": "check_sufficiency",
            "generate_response": "generate_response",
            None: END
        }
    )
    
    graph.add_edge("update_profile", "check_sufficiency")
    
    graph.add_conditional_edges(
        "check_sufficiency",
        lambda state: "sufficient" if state.get("is_profile_sufficient") else "insufficient",
        {
            "sufficient": "format_search_query",
            "insufficient": "generate_response"
        }
    )
    
    graph.add_edge("format_search_query", "call_search_tool")
    graph.add_edge("call_search_tool", "generate_response")
    graph.add_edge("generate_response", END)

    # 컴파일
    return graph.compile()

# 실행 가능한 에이전트 그래프
runnable_agent = build_agent_graph()