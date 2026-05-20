"""
test_pdf_ingest.py — PDF 인제스트 파이프라인 테스트
===================================================
실행:
    cd D:/kride-project
    pytest tests/test_pdf_ingest.py -v

외부 의존성(PyPDFLoader, ChromaDB, SentenceTransformer)은 mock 처리.
"""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 외부 패키지 stub (pdf_ingest.py의 top-level import 대응)
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_stub(name: str):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return sys.modules[name]


# sentence_transformers
_st = _ensure_stub("sentence_transformers")
_st.SentenceTransformer = MagicMock
_st.CrossEncoder = MagicMock

# chromadb — PersistentClient 포함
_chroma = _ensure_stub("chromadb")
_chroma.PersistentClient = MagicMock

# groq
_groq = _ensure_stub("groq")
_groq.Groq = MagicMock

# langchain_community.document_loaders.PyPDFLoader
_lc = _ensure_stub("langchain_community")
_lc_loaders = _ensure_stub("langchain_community.document_loaders")
_lc_loaders.PyPDFLoader = MagicMock
_lc.document_loaders = _lc_loaders

# langchain_text_splitters.RecursiveCharacterTextSplitter
_lc_text = _ensure_stub("langchain_text_splitters")
_lc_text.RecursiveCharacterTextSplitter = MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# chatbot.config stub
# ─────────────────────────────────────────────────────────────────────────────
_chatbot_pkg = _ensure_stub("chatbot")
_chatbot_pkg.__path__ = []

_config = _ensure_stub("chatbot.config")
_config.CHROMA_PATH = "/tmp/chroma_test"
_config.PDF_DIR = "/tmp/pdf_test"
_config.MODELS_DIR = "/tmp/models_test"
_config.PDF_COLLECTION = "test_pdf_collection"
_config.POI_COLLECTIONS = []
_config.EMBED_MODEL = "test-model"
_config.GROQ_MODEL = "test-groq"
_config.RERANKER_MODEL = "test-reranker"
_config.CHUNK_SIZE = 100
_config.CHUNK_OVERLAP = 20
_config.MAX_HISTORY_TURNS = 3
_config.MULTI_QUERY_COUNT = 2
_config.RETRIEVE_TOP_K_PDF = 3
_config.RETRIEVE_TOP_K_POI = 2
_config.RERANK_TOP_K = 5
_config.GROQ_API_KEY = ""


# ─────────────────────────────────────────────────────────────────────────────
# pdf_ingest 모듈 로드
# ─────────────────────────────────────────────────────────────────────────────
import importlib.util

_NLP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "subproject", "NLP")
_path = os.path.join(_NLP_DIR, "chatbot", "pdf_ingest.py")
_spec = importlib.util.spec_from_file_location("chatbot.pdf_ingest", _path)
pdf_mod = importlib.util.module_from_spec(_spec)
sys.modules["chatbot.pdf_ingest"] = pdf_mod
_spec.loader.exec_module(pdf_mod)


# ═════════════════════════════════════════════════════════════════════════════
# 1. build_chunk_id 테스트
# ═════════════════════════════════════════════════════════════════════════════
class TestBuildChunkId:
    def test_basic(self):
        cid = pdf_mod.build_chunk_id("guide.pdf", 3, 2)
        assert cid == "guide_p3_c2"

    def test_special_characters(self):
        """파일명 특수문자 제거"""
        cid = pdf_mod.build_chunk_id("한국 여행(2023).pdf", 1, 1)
        assert "(" not in cid
        assert ")" not in cid
        assert " " not in cid

    def test_preserves_alphanumeric(self):
        cid = pdf_mod.build_chunk_id("ABC_123.pdf", 0, 0)
        assert cid == "ABC_123_p0_c0"

    def test_page_zero(self):
        cid = pdf_mod.build_chunk_id("test.pdf", 0, 1)
        assert "_p0_c1" in cid

    def test_unique_ids(self):
        """같은 PDF 다른 페이지/청크 → 다른 ID"""
        id1 = pdf_mod.build_chunk_id("doc.pdf", 1, 1)
        id2 = pdf_mod.build_chunk_id("doc.pdf", 1, 2)
        id3 = pdf_mod.build_chunk_id("doc.pdf", 2, 1)
        assert id1 != id2 != id3


# ═════════════════════════════════════════════════════════════════════════════
# 2. load_pdfs 테스트
# ═════════════════════════════════════════════════════════════════════════════
class TestLoadPdfs:
    def test_empty_directory(self, tmp_path):
        """PDF 없는 디렉토리 → 빈 리스트"""
        result = pdf_mod.load_pdfs(str(tmp_path))
        assert result == []

    def test_loads_pdf_files(self, tmp_path):
        """PDF 파일이 있으면 로드 시도"""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_doc.page_content = "test content"

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [mock_doc]

        # pdf_ingest 모듈에서 참조하는 PyPDFLoader를 직접 교체
        original = pdf_mod.PyPDFLoader
        pdf_mod.PyPDFLoader = MagicMock(return_value=mock_loader_instance)
        try:
            result = pdf_mod.load_pdfs(str(tmp_path))
            assert len(result) == 1
            assert result[0].metadata["source_pdf"] == "test.pdf"
        finally:
            pdf_mod.PyPDFLoader = original

    def test_skips_failed_pdf(self, tmp_path):
        """로드 실패한 PDF는 skip"""
        pdf_file = tmp_path / "bad.pdf"
        pdf_file.write_bytes(b"not a pdf")

        original = pdf_mod.PyPDFLoader
        pdf_mod.PyPDFLoader = MagicMock(side_effect=Exception("파싱 실패"))
        try:
            result = pdf_mod.load_pdfs(str(tmp_path))
            assert result == []
        finally:
            pdf_mod.PyPDFLoader = original


# ═════════════════════════════════════════════════════════════════════════════
# 3. chunk_documents 테스트
# ═════════════════════════════════════════════════════════════════════════════
class TestChunkDocuments:
    def test_splits_documents(self):
        """문서를 청크로 분할"""
        mock_doc = MagicMock()
        mock_doc.page_content = "A" * 300
        mock_doc.metadata = {"source_pdf": "test.pdf", "page": 0}

        mock_chunks = [MagicMock(), MagicMock()]
        for c in mock_chunks:
            c.metadata = {"source_pdf": "test.pdf", "page": 0}
            c.page_content = "chunk content"

        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = mock_chunks

        original = pdf_mod.RecursiveCharacterTextSplitter
        pdf_mod.RecursiveCharacterTextSplitter = MagicMock(return_value=mock_splitter)
        try:
            result = pdf_mod.chunk_documents([mock_doc])
            assert len(result) == 2
        finally:
            pdf_mod.RecursiveCharacterTextSplitter = original

    def test_empty_input(self):
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = []

        original = pdf_mod.RecursiveCharacterTextSplitter
        pdf_mod.RecursiveCharacterTextSplitter = MagicMock(return_value=mock_splitter)
        try:
            result = pdf_mod.chunk_documents([])
            assert result == []
        finally:
            pdf_mod.RecursiveCharacterTextSplitter = original


# ═════════════════════════════════════════════════════════════════════════════
# 4. ingest 통합 테스트 (모든 외부 의존성 mock)
# ═════════════════════════════════════════════════════════════════════════════
class TestIngest:
    def test_no_pdfs_early_return(self):
        """PDF 없으면 즉시 반환"""
        with patch.object(pdf_mod, "load_pdfs", return_value=[]):
            pdf_mod.ingest()  # 예외 없이 통과

    def test_no_chunks_early_return(self):
        """청크 0개면 즉시 반환"""
        mock_doc = MagicMock()
        mock_doc.page_content = "test"
        mock_doc.metadata = {}
        with patch.object(pdf_mod, "load_pdfs", return_value=[mock_doc]), \
             patch.object(pdf_mod, "chunk_documents", return_value=[]):
            pdf_mod.ingest()  # 예외 없이 통과

    def test_full_pipeline(self):
        """전체 인제스트 파이프라인 mock 테스트"""
        # mock chunks
        mock_chunks = []
        for i in range(3):
            c = MagicMock()
            c.metadata = {"source_pdf": "test.pdf", "page": 0}
            c.page_content = f"chunk content number {i} with enough text"
            mock_chunks.append(c)

        # mock embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(
            tolist=lambda: [0.1] * 384
        )

        # mock chromadb client + collection
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_collection

        with patch.object(pdf_mod, "load_pdfs", return_value=[MagicMock()]), \
             patch.object(pdf_mod, "chunk_documents", return_value=mock_chunks):
            # pdf_ingest 내부의 SentenceTransformer / chromadb 교체
            orig_st = pdf_mod.SentenceTransformer
            orig_chroma = pdf_mod.chromadb
            pdf_mod.SentenceTransformer = MagicMock(return_value=mock_embedder)
            pdf_mod.chromadb = MagicMock()
            pdf_mod.chromadb.PersistentClient.return_value = mock_client
            try:
                pdf_mod.ingest(batch_size=10)
            finally:
                pdf_mod.SentenceTransformer = orig_st
                pdf_mod.chromadb = orig_chroma

        # upsert 호출 확인
        assert mock_collection.upsert.called

    def test_skips_short_chunks(self):
        """10자 미만 청크 skip"""
        short_chunk = MagicMock()
        short_chunk.metadata = {"source_pdf": "t.pdf", "page": 0}
        short_chunk.page_content = "short"  # < 10 chars

        long_chunk = MagicMock()
        long_chunk.metadata = {"source_pdf": "t.pdf", "page": 0}
        long_chunk.page_content = "this is a long enough chunk content for testing"

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_collection

        with patch.object(pdf_mod, "load_pdfs", return_value=[MagicMock()]), \
             patch.object(pdf_mod, "chunk_documents", return_value=[short_chunk, long_chunk]):
            orig_st = pdf_mod.SentenceTransformer
            orig_chroma = pdf_mod.chromadb
            pdf_mod.SentenceTransformer = MagicMock(return_value=mock_embedder)
            pdf_mod.chromadb = MagicMock()
            pdf_mod.chromadb.PersistentClient.return_value = mock_client
            try:
                pdf_mod.ingest(batch_size=100)
            finally:
                pdf_mod.SentenceTransformer = orig_st
                pdf_mod.chromadb = orig_chroma

        # upsert에 1개만 포함 (short는 skip)
        if mock_collection.upsert.called:
            args, kwargs = mock_collection.upsert.call_args
            ids = kwargs.get("ids", args[0] if args else [])
            assert len(ids) == 1
