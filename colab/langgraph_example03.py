import streamlit as st
import os
import operator
from typing import Annotated, List, Union

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# -----------------------------
# 1. 설정 및 스키마 정의
# -----------------------------
# LLM이 출력할 데이터의 구조를 정의합니다
class AnalysisResult(BaseModel):
    is_specific: bool = Field(description="사용자의 질문이 사내 규정을 검색하기에 충분히 구체적인지 여부")
    search_keywords: List[str] = Field(description="문서 검색을 위해 추출된 핵심 키워드 리스트")
    clarify_question: str = Field(description="정보가 부족하여 사용자에게 다시 물어볼 내용 (구체적이면 빈 문자열)")

# -----------------------------
# 2. 모델 및 DB 로드
# -----------------------------
GROQ_API_KEY = "GROQ API KEY를 넣으세요"
EMBEDDING_MODEL = "BAAI/bge-m3"

llm = ChatGroq(
    model='llama-3.3-70b-versatile', 
    temperature=0.1, 
    api_key=GROQ_API_KEY
)

embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

DB_PATH = './chroma_company'
DOC_PATH = './dataset/company.txt'

def load_db():
    if os.path.exists(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embedding)
    if not os.path.exists(DOC_PATH):
        return None
    with open(DOC_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = text_splitter.create_documents([text])
    return Chroma.from_documents(docs, embedding=embedding, persist_directory=DB_PATH)

db = load_db()
retriever = db.as_retriever(search_kwargs={'k': 4}) if db else None

# -----------------------------
# 3. 상태(State) 정의
# -----------------------------
class AgentState(BaseModel):
    # Annotated를 사용하여 메시지가 리스트에 순차적으로 추가되도록 설정
    messages: Annotated[List[BaseMessage], operator.add] = []
    tasks: List[str] = []
    docs: List[str] = []
    answer: str = ""
    need_clarify: bool = False
    logs: List[str] = []

# -----------------------------
# 4. 노드 정의 (Nodes)
# -----------------------------

def analyze_node(state: AgentState):
    # 마지막 사용자 메시지 가져오기
    user_input = state.messages[-1].content
    logs = [f"질문 분석 시작: '{user_input}'"]
    
    analysis_prompt = f"""
    당신은 인사업무 분석가입니다. 사용자의 질문이 사내 규정(연차, 급여, 복지 등)을 검색하기에 충분한 정보가 있는지 판단하세요.
    
    질문: {user_input}
    
    정보가 부족하다면(예: '연차에 대해 알려줘' -> 무엇이 궁금한지 부족함), 
    사용자에게 어떤 정보를 더 제공해야 할지 친절하게 질문을 만드세요.
    """
    
    # Pydantic 모델을 사용하여 구조화된 출력 유도
    structured_llm = llm.with_structured_output(AnalysisResult)
    analysis = structured_llm.invoke(analysis_prompt)
    
    if not analysis.is_specific:
        logs.append("분석 결과: 정보 부족. 역질문 생성.")
        return {
            "need_clarify": True, 
            "answer": analysis.clarify_question, 
            "logs": logs
        }
    
    logs.append(f"분석 결과: 검색 키워드 {analysis.search_keywords} 추출 완료.")
    return {
        "tasks": analysis.search_keywords, 
        "need_clarify": False, 
        "logs": logs
    }

def retrieve_node(state: AgentState):
    docs = []
    logs = state.logs
    for task in state.tasks:
        logs.append(f"'{task}' 주제로 사내 문서 검색 중...")
        results = retriever.invoke(task)
        docs.extend([d.page_content for d in results])
    
    return {"docs": list(set(docs)), "logs": logs}

def answer_node(state: AgentState):
    logs = state.logs
    logs.append("답변 생성 중...")
    
    context = "\n".join(state.docs)
    # 메시지 히스토리를 포함하여 맥락 파악
    history_text = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in state.messages])
    
    prompt = f"""
    당신은 회사의 인사 규정 안내 도우미입니다. 아래의 대화 히스토리와 문서를 바탕으로 답변하세요.
    
    [대화 히스토리]
    {history_text}
    
    [참고 문서]
    {context}
    
    문서에 명확한 답변이 없다면 억지로 답변하지 말고 규정이 없다고 정직하게 말하세요.
    """
    
    response = llm.invoke(prompt)
    return {
        "answer": response.content, 
        "logs": logs,
        "messages": [AIMessage(content=response.content)]
    }

def clarify_node(state: AgentState):
    # 역질문을 AI 메시지로 기록
    return {"messages": [AIMessage(content=state.answer)]}

# -----------------------------
# 5. 그래프 구성
# -----------------------------
workflow = StateGraph(AgentState)

workflow.add_node("analyze", analyze_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("answer", answer_node)
workflow.add_node("clarify", clarify_node)

workflow.set_entry_point("analyze")

# 분석 후 분기 처리
workflow.add_conditional_edges(
    "analyze", 
    lambda x: "clarify" if x.need_clarify else "retrieve"
)
workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", END)
workflow.add_edge("clarify", END)

app = workflow.compile()

# -----------------------------
# 6. Streamlit UI
# -----------------------------
st.set_page_config(page_title="HR Smart Agent", layout="wide")
st.title("🏢 사내 규정 AI 에이전트")
st.caption("질문의 맥락을 이해하고 부족한 정보를 역으로 질문합니다.")

# 세션 상태에 채팅 내역 저장
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# 기존 채팅 표시
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 받기
if prompt := st.chat_input("질문을 입력하세요..."):
    # 1. 사용자 질문 표시
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 에이전트 프로세스 시작
    with st.chat_message("assistant"):
        # 실시간 처리 로그용 상태바
        status_bar = st.status("에이전트가 처리 중입니다...", expanded=True)
        answer_placeholder = st.empty()
        
        # LangGraph 입력 구성
        # 이전 히스토리를 모두 Human/AI 메시지 객체로 변환하여 전달
        initial_messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.chat_messages
        ]
        
        inputs = {"messages": initial_messages, "logs": []}
        
        final_answer = ""
        
        # app.stream을 통한 실시간 로그 출력
        for output in app.stream(inputs):
            for node_name, chunk_data in output.items():
                if "logs" in chunk_data:
                    for log in chunk_data["logs"]:
                        status_bar.write(f"✅ {log}")
                
                # 최종 답변 업데이트
                if "answer" in chunk_data:
                    final_answer = chunk_data["answer"]
        
        status_bar.update(label="처리 완료", state="complete", expanded=False)
        answer_placeholder.markdown(final_answer)
        
        # 3. 세션 상태에 답변 저장
        st.session_state.chat_messages.append({"role": "assistant", "content": final_answer})