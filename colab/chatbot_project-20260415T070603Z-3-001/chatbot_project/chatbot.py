#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from config import Config


# In[2]:


class Chatbot:
    def __init__(self, document_handler, image_handler):
        self.document_handler = document_handler
        self.image_handler = image_handler
        self.llm = ChatOllama(model=Config.GEMMA_MODEL, temperature=Config.TEMPERATURE)
        self.llm_image = ChatOllama(model=Config.LLAVA_MODEL, temperature=Config.TEMPERATURE)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        self.prompt = PromptTemplate(
            template=Config.PROMPT_TEMPLATE,
            input_variables=["chat_history", "context", "question"]
        )
        self.qa_chain = self.initialize_qa_chain()

    def initialize_qa_chain(self, custom_db=None):
        db = custom_db if custom_db else self.document_handler.initialize_db()
        return ConversationalRetrievalChain.from_llm(
            self.llm,
            db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            memory=self.memory
        )

    def chat(self, message, history, image=None):
        try:
            chat_history = [(human, ai) for human, ai in history]
            answer = ""
            
            if image is not None:
                processed_image = self.image_handler.process_image(image)
                if processed_image:
                    messages = [
                        HumanMessage(
                            content=[
                                {"type": "text", "text": message},
                                {"type": "image_url", "image_url": processed_image}
                            ]
                        )
                    ]
                    
                    try:
                        response = self.llm_image.invoke(messages)
                        image_answer = response.content
                        answer = f"이미지 분석 결과: {image_answer}"
                    except Exception as img_error:
                        print(f"이미지 분석 중 오류 발생: {str(img_error)}")
                        answer = f"이미지 분석 중 오류가 발생했습니다: {str(img_error)}"
            
            if not image:
                text_response = self.qa_chain({"question": message, "chat_history": chat_history})
                text_answer = text_response['answer']
                
                if 'source_documents' in text_response:
                    sources = set([os.path.basename(doc.metadata.get('source', 'Unknown')) 
                                 for doc in text_response['source_documents']])
                    source_info = f"\n\n참고 파일: {', '.join(sources)}" if sources else ""
                    text_answer += source_info
                
                answer = text_answer if not answer else f"{answer}\n\n텍스트 응답: {text_answer}"
                
            chat_history.append((message, answer))
            return "", chat_history, None
        
        except Exception as e:
            print(f"채팅 함수 실행 중 오류 발생: {str(e)}")
            error_message = f"죄송합니다. 오류가 발생했습니다: {str(e)}"
            chat_history.append((message, error_message))
            return "", chat_history, None