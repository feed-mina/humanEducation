import os
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import shutil
from config import Config
import time

class DocumentHandler:
    def __init__(self):
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )

    def initialize_db(self, file_path=None, use_main_only=True):
        try:
            doc_path = Config.MAIN_DOC_PATH if use_main_only else file_path
            if not doc_path:
                raise ValueError("파일 경로가 지정되지 않았습니다.")

            if not os.path.exists(doc_path):
                if doc_path == Config.MAIN_DOC_PATH:
                    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
                    with open(doc_path, 'w', encoding='utf-8') as f:
                        f.write("휴먼(주) 정보:\n")
                else:
                    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {doc_path}")

            loader = TextLoader(doc_path, encoding='utf-8')
            docs = loader.load()
            db = Chroma.from_documents(docs, self.embeddings_model)
            
            return db
            
        except Exception as e:
            print(f"DB 초기화 중 오류 발생: {str(e)}")
            raise

    def add_new_file(self, file):
        if file is None:
            return "파일을 선택해주세요.", None
    
        try:
            # TEMP 폴더 생성
            os.makedirs(Config.TEMP_FOLDER, exist_ok=True)
    
            # 파일명에 타임스탬프 추가
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{timestamp}_{os.path.basename(file.name)}"
            file_path = os.path.join(Config.TEMP_FOLDER, filename)
    
            # 파일 저장(리눅스)
            # with open(file_path, 'wb') as f_out:
            #     f_out.write(file.read())
            
            # 파일 저장(윈도우)
            with open(file_path, 'wb') as f_out:
                with open(file.name, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out)
    
            # 파일이 비어 있는지 확인
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                return "파일이 비어 있습니다. 다시 시도해 주세요.", None
    
            return f"파일이 성공적으로 추가되었습니다: {filename}", None
    
        except Exception as e:
            return f"파일 추가 중 오류가 발생했습니다: {str(e)}", None

    def add_new_information(self, new_info):
        if new_info.strip():
            try:
                with open(Config.MAIN_DOC_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"\n{new_info.strip()}")
                return "새로운 정보가 성공적으로 추가되었습니다."
            except Exception as e:
                return f"정보 추가 중 오류가 발생했습니다: {str(e)}"
        return "추가할 정보를 입력해주세요."