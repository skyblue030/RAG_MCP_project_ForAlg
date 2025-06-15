# -*- Python Standard Library -*-
import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime

# -*- Third-Party Libraries -*-
# 我們把所有外部專家的聯絡電話都放在這裡一次打完
try:
    # Web & API
    import requests
    import uvicorn
    from contextlib import asynccontextmanager
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    # Database
    from sqlalchemy import (JSON as SQL_JSON, Column, DateTime, ForeignKey, String,
                            Text, create_engine)
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

    # AI, NLP & Embeddings
    import torch
    import chromadb
    import google.generativeai as genai_llm
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import CrossEncoder, SentenceTransformer

    # Utilities
    import fitz  # PyMuPDF
    from dotenv import load_dotenv

except ImportError as e:
    # 只要上面有任何一個專家聯絡不上，就執行這裡的備用計畫
    print("錯誤：您的環境缺少必要的函式庫。")
    print("請在您的 backend 資料夾中，啟用虛擬環境後，執行以下指令來安裝所有依賴：")
    print("uv pip install -r requirements.txt")
    print(f"\n詳細錯誤訊息: {e}")
    sys.exit(1) # 直接退出程式，不往下執行

# 如果程式能執行到這裡，代表所有專家都已就位，可以安心使用

# 一個全域的字典，用來存放共享的模型和資料
shared_resources = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 伺服器啟動時執行的程式碼 ---
    logger.info("伺服器啟動中，開始載入所有資源...")
    
    try:
        # 1. 載入父區塊
        parent_store = {}
        if os.path.exists(PARENT_CHUNKS_FILE_PATH):
            with open(PARENT_CHUNKS_FILE_PATH, 'r', encoding='utf-8') as f:
                parent_store = json.load(f)
            logger.info(f"成功載入 {len(parent_store)} 個父區塊。")
        shared_resources["parent_store"] = parent_store

        # 2. 連接 ChromaDB
        collection = get_or_create_chroma_collection(VECTOR_DB_PATH, COLLECTION_NAME)
        shared_resources["chroma_collection"] = collection
        logger.info(f"向量資料庫集合 '{COLLECTION_NAME}' 載入成功。")

        # 3. 載入 AI 模型 (請確保這幾段都在)
        device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 載入 Embedding 模型
        shared_resources["embedding_model"] = SentenceTransformer(BGE_EMBEDDING_MODEL_NAME, device=device_to_use)
        logger.info(f"Embedding 模型 ({BGE_EMBEDDING_MODEL_NAME}) 載入成功，使用設備: {device_to_use}")

        # 載入 Cross-Encoder 模型
        shared_resources["cross_encoder"] = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device_to_use)
        logger.info(f"Cross-Encoder 模型 ({CROSS_ENCODER_MODEL_NAME}) 載入成功。")

        # 載入 Generative LLM
        api_key = os.getenv(API_KEY_LLM_ENV_VAR)
        if not api_key:
            raise ValueError(f"未設定環境變數: {API_KEY_LLM_ENV_VAR}")
        genai_llm.configure(api_key=api_key)
        shared_resources["llm_model"] = genai_llm.GenerativeModel(LLM_MODEL_NAME)
        logger.info(f"Generative LLM ({LLM_MODEL_NAME}) 載入成功。")

    except Exception as e:
        logger.error(f"資源載入過程中發生致命錯誤: {e}", exc_info=True)
    
    logger.info("應用程式啟動準備完成。")
    yield

    # --- 伺服器關閉時執行的程式碼 ---
    logger.info("伺服器關閉中，釋放資源...")
    shared_resources.clear()


# 在所有其他程式碼之前載入 .env 檔案中的環境變數
# 這會查找與此 main.py 同目錄下的 .env 檔案
load_dotenv(override=True) # 強制使用 .env 檔案中的值覆蓋已存在的環境變數

app = FastAPI(lifespan=lifespan)
# --- CORS 設定 ---
origins = [
    "http://localhost:5173", # 允許您的 React 前端來源
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # 允許所有方法 (GET, POST 等)
    allow_headers=["*"], # 允許所有標頭
)

# --- 新增 /ask 端點 ---
class AskRequest(BaseModel):
    question: str

# 解決requests.post死結問題
# =================================================================
# 1. 這是我們乾淨、可重用的「檢索食譜」
# =================================================================
# =================================================================
# FINAL API LOGIC - 請將這整塊貼到您的 main.py 中
# =================================================================

def _perform_retrieval(query_for_retrieval: str, original_user_question: str) -> list[str]:
    """
    這是一個內部輔助函式，專門負責檢索與重排的邏輯。
    它接收查詢字串，並返回一個包含上下文文字的列表。
    """
    # 從共享資源中獲取所有需要的工具
    embedding_model = shared_resources.get("embedding_model")
    cross_encoder = shared_resources.get("cross_encoder")
    collection = shared_resources.get("chroma_collection")
    parent_store = shared_resources.get("parent_store")

    # 檢查工具是否齊全
    if not all([embedding_model, cross_encoder, collection, parent_store is not None]):
        logger.error("檢索所需的一個或多個資源未初始化。")
        return []  # 如果工具不齊全，直接返回一個空列表

    try:
        # 步驟 1: 將查詢問題轉換為向量
        logger.info("正在生成查詢向量...")
        query_embedding = embedding_model.encode(query_for_retrieval, normalize_embeddings=True).tolist()

        # 步驟 2: 從向量資料庫中初步檢索相關的子區塊
        logger.info("正在查詢向量資料庫...")
        child_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            include=['metadatas']
        )

        # 步驟 3: 根據子區塊找到對應的、不重複的父區塊
        if not (child_results and child_results['metadatas'] and child_results['metadatas'][0]):
            logger.warning("初步檢索未找到任何相關的子區塊。")
            return []

        unique_parent_ids = set(
            child_meta['parent_id'] for child_meta in child_results['metadatas'][0] if child_meta.get('parent_id')
        )
        retrieved_parent_contexts_data = [
            {"text": parent_store.get(p_id)}
            for p_id in list(unique_parent_ids)
            if parent_store.get(p_id)
        ]

        # 步驟 4: 使用 Cross-Encoder 進行精細重排
        if retrieved_parent_contexts_data and cross_encoder:
            logger.info(f"正在對 {len(retrieved_parent_contexts_data)} 個上下文進行重排...")
            sentence_pairs = [(original_user_question, ctx["text"]) for ctx in retrieved_parent_contexts_data]
            scores = cross_encoder.predict(sentence_pairs, show_progress_bar=False) # 在伺服器日誌中關閉進度條
            scored_contexts = sorted(zip(scores, retrieved_parent_contexts_data), key=lambda x: x[0], reverse=True)
            final_contexts_data = [ctx for score, ctx in scored_contexts[:NUM_CONTEXTS_AFTER_RERANK]]
            logger.info(f"成功檢索並重排了 {len(final_contexts_data)} 個上下文。")
            return [item.get("text", "") for item in final_contexts_data]

        logger.warning("未執行重排，返回初步檢索結果。")
        return [ctx.get("text", "") for ctx in retrieved_parent_contexts_data[:NUM_CONTEXTS_AFTER_RERANK]]

    except Exception as e:
        logger.error(f"執行檢索 (_perform_retrieval) 時發生錯誤: {e}", exc_info=True)
        return []  # 發生任何錯誤都返回空列表，確保程式穩定


@app.post("/ask")
async def ask_question(ask_request_data: AskRequest):
    """
    這個 API 端點現在非常乾淨，只負責編排整個 RAG 流程。
    """
    original_user_question = ask_request_data.question
    logger.info(f"收到來自前端 /ask 的問題: '{original_user_question[:50]}...'")

    llm_model = shared_resources.get("llm_model")
    if not llm_model:
        raise HTTPException(status_code=503, detail="LLM 服務未初始化。")

    # HyDE 查詢擴展
    query_for_retrieval = original_user_question
    if USE_HYDE_QUERY_EXPANSION:
        logger.info("正在執行 HyDE 查詢擴展...")
        hyde_prompt = EVAL_HYDE_PROMPT_TEMPLATE.format(user_question_for_processing=original_user_question)
        try:
            hyde_response = llm_model.generate_content(hyde_prompt)
            query_for_retrieval = hyde_response.text.strip()
            logger.info("HyDE 查詢擴展完成。")
        except Exception as e:
            logger.error(f"HyDE 失敗，將使用原始問題進行檢索: {e}", exc_info=True)

    # 內部函式呼叫，以獲取上下文
    retrieved_contexts_text = _perform_retrieval(
        query_for_retrieval=query_for_retrieval,
        original_user_question=original_user_question
    )

    # 建構提示詞
    context_string = "\n\n---\n\n".join(retrieved_contexts_text)
    final_prompt = EVAL_FINAL_ANSWER_PROMPT_TEMPLATE.format(
        original_user_question=original_user_question,
        decomposed_queries_text=original_user_question,
        context_string=context_string
    )

    # 生成並回傳最終答案
    logger.info("正在請求 LLM 生成最終回答...")
    try:
        llm_response = llm_model.generate_content(final_prompt)
        final_answer = llm_response.text or "抱歉，我無法生成回答。"
        return {"answer": final_answer.strip()}
    except Exception as e:
        logger.error(f"LLM 回答生成失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="生成回答時發生內部錯誤。")


@app.post("/retrieve")
async def retrieve_context(request: Request):
    """
    這個 API 端點現在只是一個簡單的包裝，
    核心邏輯已移至 _perform_retrieval。
    """
    try:
        data = await request.json()
        query = data.get("query_for_retrieval")
        original_question = data.get("original_user_question")

        if not query or not original_question:
            raise HTTPException(status_code=400, detail="請求中缺少必要欄位。")

        retrieved_texts = _perform_retrieval(query, original_question)
        return {"contexts": [{"text": text} for text in retrieved_texts]}
    except Exception as e:
        logger.error(f"處理 /retrieve 請求時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="處理檢索請求時發生內部錯誤。")


@app.post("/ask_rag_async")
async def ask_rag_asynchronously(request: Request, background_tasks: BackgroundTasks):
    # 從共享資源中獲取 LLM 模型
    llm_model = shared_resources.get("llm_model")

    data = await request.json()
    original_question = data.get("question")
    conversation_id = data.get("conversation_id") # 從請求中獲取 conversation_id

    if not original_question:
        return {"error": "請求中缺少 'question' 欄位"}, 400
    if not llm_model: # 檢查 LLM 模型是否已載入
            return {"error": "伺服器端 LLM 模型未初始化，無法處理非同步 RAG 請求。"}, 503
    # 注意：這裡我們假設 conversation_id 是可選的。
    task_id = str(uuid.uuid4())
    logger.info(f"收到非同步 RAG 請求，問題: '{original_question[:50]}...'，對話 ID: {conversation_id}，分配任務 ID: {task_id}")
    
    background_tasks.add_task(perform_full_rag_in_background, task_id, original_question, conversation_id)
    
    return {"message": "請求已接收，正在背景處理中。", "task_id": task_id}

# --- Configuration Constants ---
# 現在這些值會優先從 .env 檔案讀取，如果 .env 中未定義或為空，則使用提供的預設值
PDF_FILE_PATH = os.getenv("PDF_FILE_PATH", "./Introduction_to_algorithms-3rd Edition.pdf")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
BGE_EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", 'BAAI/bge-m3')
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_pdf_collection")
RETRIEVAL_SERVER_URL = os.getenv("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8000/retrieve")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_mcp_chat.db") # 新增資料庫 URL
API_KEY_LLM_ENV_VAR = "GOOGLE_API_KEY_LLM" # 這仍然是環境變數的 *名稱*
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-latest")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", None) # 從 .env 讀取期望的設備，預設為 None (自動)
PARENT_CHUNKS_FILE_PATH = os.getenv("PARENT_CHUNKS_FILE_PATH", "./parent_chunks_store.json")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2") # 交叉編碼器模型
NUM_CONTEXTS_AFTER_RERANK = int(os.getenv("NUM_CONTEXTS_AFTER_RERANK", "4")) # 重排後傳遞給 LLM 的上下文數量
USE_HYDE_QUERY_EXPANSION = os.getenv("USE_HYDE_QUERY_EXPANSION", "false").lower() == "true" # 是否使用 HyDE 進行查詢擴展
PERFORM_QUERY_DECOMPOSITION = os.getenv("PERFORM_QUERY_DECOMPOSITION", "false").lower() == "true"

# NUM_CONTEXTS_FOR_LLM = int(os.getenv("NUM_CONTEXTS_FOR_LLM", "6")) # 移除或註解此行，不再需要從 .env 讀取
# --- Logging Setup ---

# --- Prompt Definitions for Evaluation Mode ---
EVAL_DECOMPOSITION_PROMPT_TEMPLATE = (
    "你是一位邏輯分析專家。你的任務是將一個指定的「使用者問題」分解成一系列清晰、獨立、且符合邏輯的子問題。"
    "這些子問題將作為一份**寫作大綱**，用來指導最終答案的生成，確保最終答案的完整性與結構性。\n\n"
    "**分解準則：**\n"
    "1. **完全覆蓋**：所有子問題合起來必須完全覆蓋原始問題的所有面向。\n"
    "2. **單一職責**：每個子問題應只關注一個核心要點（例如：「是什麼」、「為什麼」、「如何運作」、「優缺點比較」等）。\n"
    "3. **保持中立**：僅進行問題分解，不要添加額外資訊或嘗試回答問題。\n"
    "4. **簡潔問題優先**：如果原始問題本身已經足夠簡單，無法有意義地分解，則直接將原始問題作為唯一的子問題輸出。\n\n"
    "使用者問題：{user_question_for_processing}\n\n"
    "分解後的子問題清單："
)

EVAL_HYDE_PROMPT_TEMPLATE = (
    "你是一位精通技術寫作的專家。你的任務是為以下「問題」生成一個簡潔、專業、且資訊密度極高的「假設性答案」。"
    "這個假設性答案將被用來在專業教科書中進行語意搜尋，因此它必須聽起來像教科書中的一段精闢論述。\n\n"
    "**撰寫準則：**\n"
    "1. **大膽使用專業術語**：直接使用問題領域最核心的術語。\n"
    "2. **包含可信的細節**：如果問題涉及演算法，可以虛構一個合理的複雜度，如 `O(n log n)`；如果涉及比較，可以直接斷言其優缺點。\n"
    "3. **專注事實，而非對話**：使用陳述句，避免使用口語化的詞彙。\n"
    "4. **保持簡潔**：長度控制在 2-4 句話。\n\n"
    "問題：{user_question_for_processing}\n\n"
    "假設性答案："
)

EVAL_FINAL_ANSWER_PROMPT_TEMPLATE = (
    "你是一位知識淵博、教學經驗豐富的 AI 教授。你的任務是利用以下提供的三項資訊，為「原始問題」合成一份全面、精確、且有條理的最終回答：\n"
    "1. **原始問題**：使用者最一開始的提問。\n"
    "2. **回答大綱**：一份由子問題組成的清單，你必須在回答中逐一回應這些子問題。\n"
    "3. **相關上下文**：從教科書中檢索到的、與問題最相關的原始文本片段。\n\n"
    
    "### 高品質回答的準則與指令：\n" # <-- 將兩個標題合併
    "1. **嚴格遵循大綱**：你的回答結構必須清晰地反映「回答大綱」中的順序和內容，確保每個子問題都得到回應。\n"
    "2. **絕對忠於上下文**：你的回答中的每一個論點、例子和細節都**必須**直接來源於「相關上下文」。**絕對禁止使用任何外部知識。**\n" # <-- 合併了重複的指令
    "3. **提取並呈現關鍵細節**：如果上下文中包含定義、數學公式、演算法偽代碼或關鍵參數，你必須將這些精確的細節融入解釋中。\n"
    "4. **深入解釋「為什麼」**：當子問題涉及原因或原理時，主動在上下文中尋找並總結其背後的理論依據或證明思路。\n"
    "5. **智慧處理語言**：你的回答應主要使用「原始問題」中的**主要語言**。然而，如果「原始問題」中包含了特定語言的專有名詞（尤其是英文技術術語），請在你的回答中**優先保留這些專有名詞的原文**，而不是將它們翻譯掉，以確保技術的精確性。例如，如果問題是「請解釋 `B-tree` 的 `fan-out`」，你的解釋就應該圍繞 `B-tree` 和 `fan-out` 這兩個詞展開。\n"
    "6. **處理資訊不足**：如果上下文不足以回答某個子問題，請在回答的相應部分明確指出「根據提供的資料，無法回答關於...的細節」。\n\n"
    
    "---[輸入資訊開始]---\n"
    "**1. 原始問題：**\n{original_user_question}\n\n"
    "**2. 回答大綱 (子問題清單)：**\n{decomposed_queries_text}\n\n"
    "**3. 相關上下文：**\n{context_string}\n\n"
    "---[輸入資訊結束]---\n\n"
    "**教授的最終回答：**"
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # 從 .env 讀取日誌級別，預設為 INFO
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_python_environment():
    logger.debug("--- Python 環境偵錯資訊 ---")
    logger.debug(f"Python 可執行檔: {sys.executable}")
    logger.debug(f"Python 版本: {sys.version}")
    logger.debug(f"目前工作目錄: {os.getcwd()}")
    logger.debug(f"VIRTUAL_ENV 環境變數: {os.getenv('VIRTUAL_ENV')}")
    logger.debug("sys.path 內容:")
    for i, path_item in enumerate(sys.path):
        logger.debug(f"  [{i}] {path_item}")
    logger.debug("------------------------------------")

log_python_environment() # Call it once at the start
# --- SQLAlchemy Setup ---
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    sender_type = Column(String, nullable=False)  # "user" or "ai"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(SQL_JSON, nullable=True) # 將 metadata 更名為 meta_data
    conversation = relationship("Conversation", back_populates="messages")

engine = None
SessionLocal = None

def init_db():
    global engine, SessionLocal
    logger.info(f"嘗試初始化資料庫，DATABASE_URL: {DATABASE_URL}")
    logger.info(f"資料庫初始化時的目前工作目錄: {os.getcwd()}")
    try:
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
        logger.info(f"SQLAlchemy 引擎已為 {DATABASE_URL} 創建。")
        logger.info("正在嘗試創建所有資料表 (Base.metadata.create_all)...")
        Base.metadata.create_all(bind=engine)
        logger.info("Base.metadata.create_all() 已完成。")
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info(f"資料庫已成功初始化於: {DATABASE_URL}")
    except Exception as e:
        logger.error(f"資料庫初始化失敗，DATABASE_URL '{DATABASE_URL}': {e}", exc_info=True)

def get_db():
    if SessionLocal is None:
        logger.error("資料庫會話 (SessionLocal) 未初始化。請先調用 init_db()。")
        return None # 或者拋出異常
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_or_create_chroma_collection(db_path: str, collection_name: str, recreate_if_exists: bool = False):
    """
    初始化 ChromaDB 客戶端並獲取或創建集合。
    """

    client = chromadb.PersistentClient(path=db_path)
    
    if recreate_if_exists:
        try:
            logger.info(f"正在嘗試刪除可能已存在的舊集合 (因 recreate_if_exists=True): {collection_name}...")
            client.delete_collection(name=collection_name)
            logger.info(f"舊集合 '{collection_name}' 已成功刪除。")
        except chromadb.errors.NotFoundError: # 根據追蹤訊息，應該是 NotFoundError
            logger.info(f"舊集合 '{collection_name}' 不存在，無需刪除。")
        except Exception as delete_exc: # 捕獲其他可能的刪除錯誤
            logger.warning(f"嘗試刪除舊集合 '{collection_name}' 時發生未預期錯誤: {delete_exc}")
            pass # 繼續嘗試創建
    
    try:
        # ChromaDB 通常可以自動檢測嵌入向量的維度
        # 對於 SentenceTransformer 模型，維度通常是固定的，ChromaDB 可以自動處理
        if recreate_if_exists: # 只有在需要重建時才明確創建
            collection = client.create_collection(name=collection_name)
            logger.info(f"新集合 '{collection_name}' 已創建。")
        else: # 否則嘗試獲取，如果不存在則創建
            try:
                collection = client.get_collection(name=collection_name)
                logger.info(f"已成功連接到現有集合: {collection_name}")
            except Exception: # 簡化：如果 get 失敗，則嘗試 create
                logger.info(f"集合 '{collection_name}' 不存在，正在創建...")
                collection = client.create_collection(name=collection_name)
                logger.info(f"新集合 '{collection_name}' 已創建。")
    except Exception as create_e: # 捕獲創建或獲取過程中的最終錯誤
        logger.error(f"處理集合 '{collection_name}' 失敗: {create_e}", exc_info=True)
        # 嘗試在失敗後獲取集合，以防它是由於競爭條件或其他原因創建的
        try:
            collection = client.get_collection(name=collection_name) # type: ignore
            logger.info(f"已連接到先前創建失敗但現在存在的集合: {collection_name}") # type: ignore
        except Exception as get_e_after_create_fail:
            logger.error(f"創建集合 '{collection_name}' 失敗後，嘗試獲取也失敗: {get_e_after_create_fail}", exc_info=True)
            return None # 如果所有嘗試都失敗，則返回 None
    return collection

def run_ingestion_pipeline():
    logger.info("啟動資料導入與索引流程...")

    logger.info(f"正在載入嵌入模型: {BGE_EMBEDDING_MODEL_NAME}...")
    try:
        # 如果 EMBEDDING_DEVICE 有效設定 (例如 "cuda" 或 "cpu")，則使用它
        # 否則 (為 None)，讓 SentenceTransformer 自動檢測
        device_to_use = EMBEDDING_DEVICE if EMBEDDING_DEVICE in ["cuda", "cpu", "mps"] else None
        logger.info(f"嘗試使用設備: {device_to_use if device_to_use else '自動檢測'}")
        embedding_model = SentenceTransformer(BGE_EMBEDDING_MODEL_NAME, device=device_to_use)
    except Exception as e:
        logger.error(f"載入嵌入模型 {BGE_EMBEDDING_MODEL_NAME} 失敗: {e}", exc_info=True)
        logger.error(f"請確保 PyTorch 和 CUDA (如果嘗試使用 GPU) 已正確安裝，或者將 EMBEDDING_DEVICE 設為 'cpu'。")
        return
    logger.info(f"嵌入模型使用的設備: {embedding_model.device}")
    logger.info("嵌入模型載入完成。")


    logger.info(f"正在載入 PDF: {PDF_FILE_PATH}...")
    doc = fitz.open(PDF_FILE_PATH)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text()
    doc.close()
    logger.info(f"PDF 內容載入完成，總字數: {len(full_text)}")

    # 2. 文本切割器
    logger.info("正在進行文本切割...")
    # 父區塊切割器 - 較大的區塊以獲取更完整的上下文
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # 父區塊大小示例值
        chunk_overlap=300, # 父區塊重疊示例值
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",  # 按段落
            "\n",    # 按行
            ". ",    # 按句子
            " ",     # 按詞
            "",
        ],
    )
    # 子區塊切割器 - 較小的區塊用於精確嵌入和檢索
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,   # 子區塊大小示例值
        chunk_overlap=50,  # 子區塊重疊示例值
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""], # 可以與父區塊使用不同的分隔策略
    )

    parent_documents = parent_splitter.create_documents([full_text])
    logger.info(f"已將原始文本切割成 {len(parent_documents)} 個父區塊。")

    all_child_chunks_for_embedding = []
    all_child_metadatas_for_embedding = []
    parent_chunk_store = {} # 用於儲存 {parent_id: parent_text}

    for i, p_doc in enumerate(parent_documents):
        parent_id = f"parent_{uuid.uuid4()}"
        parent_text = p_doc.page_content
        parent_chunk_store[parent_id] = parent_text

        # 從每個父區塊生成子區塊
        child_docs_from_parent = child_splitter.create_documents([parent_text])
        for j, c_doc in enumerate(child_docs_from_parent):
            child_text = c_doc.page_content
            all_child_chunks_for_embedding.append(child_text)
            all_child_metadatas_for_embedding.append({
                "source": PDF_FILE_PATH,
                "parent_id": parent_id, # 關聯到父區塊
                "child_order_in_parent": j # 子區塊在父區塊內的順序 (可選)
            })
    logger.info(f"總共從父區塊生成了 {len(all_child_chunks_for_embedding)} 個子區塊用於嵌入。")

    # 3. 嵌入模型 (使用 Gemini API)
    # print(f"準備使用 Gemini 嵌入模型: {GEMINI_EMBEDDING_MODEL_NAME}") # 已在步驟 0 載入本地模型
    # 4. 向量資料庫 (ChromaDB 範例)
    logger.info(f"正在連接或創建向量資料庫集合 '{COLLECTION_NAME}' 於: {VECTOR_DB_PATH}...")
    collection = get_or_create_chroma_collection(VECTOR_DB_PATH, COLLECTION_NAME, recreate_if_exists=True)
    if collection is None:
        logger.error("無法初始化向量資料庫集合，資料導入中止。")
        return

    logger.info("正在生成嵌入並存入向量資料庫...")
    # Gemini API 嵌入有其批次限制 (例如，models/embedding-001 一次最多100個文件)
    # SentenceTransformer 的 encode 方法可以處理批次，ChromaDB 添加數據也受益於批次處理
    chroma_add_batch_size = 100 # 每次調用 collection.add 的項目數量

    # 使用 SentenceTransformer 的 encode 方法進行批次嵌入
    # 可以調整 batch_size 以優化效能，bge-m3 模型較大，如果記憶體有限，批次大小不宜過大
    # 同時，encode 方法本身可以接受一個文本列表，它會在內部進行批處理
    logger.info(f"正在使用本地模型 '{BGE_EMBEDDING_MODEL_NAME}' 生成所有 {len(all_child_chunks_for_embedding)} 個區塊的嵌入...")
    if not all_child_chunks_for_embedding:
        logger.warning("沒有子區塊可供嵌入，導入中止。")
        return
        
    try:
        # normalize_embeddings=True 通常對 BGE 模型是推薦的
        all_embeddings_from_model = embedding_model.encode(all_child_chunks_for_embedding, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        all_embeddings_list = [emb.tolist() for emb in all_embeddings_from_model] # 轉換為 list
        # all_documents_list 現在是 all_child_chunks_for_embedding
        # all_metadatas_list 現在是 all_child_metadatas_for_embedding
        all_ids_list = [str(uuid.uuid4()) for _ in range(len(all_child_chunks_for_embedding))]
        logger.info("所有嵌入生成完畢。")
    except Exception as e:
        logger.error(f"使用本地模型生成嵌入失敗: {e}", exc_info=True) # exc_info=True 會記錄堆疊追蹤
        # 為了避免輸出過多，只顯示前幾個有問題的區塊的部分內容
        logger.debug(f"  問題批次的內容 (前 2 個區塊的前 50 字元): {[text[:50] + '...' for text in all_child_chunks_for_embedding[:2]]}")
        return

    # all_documents_list 已由 all_child_chunks_for_embedding 替換
    if not all_child_chunks_for_embedding:
        logger.warning("沒有成功生成任何嵌入，導入中止。")
        return

    logger.info(f"總共生成 {len(all_child_chunks_for_embedding)} 個嵌入。正在分批存入向量資料庫...")
    for i in range(0, len(all_child_chunks_for_embedding), chroma_add_batch_size):
        collection.add(
            embeddings=all_embeddings_list[i:i + chroma_add_batch_size],
            documents=all_child_chunks_for_embedding[i:i + chroma_add_batch_size], # 儲存子區塊文本
            ids=all_ids_list[i:i + chroma_add_batch_size],
            metadatas=all_child_metadatas_for_embedding[i:i + chroma_add_batch_size] # 儲存包含 parent_id 的元數據
        )
        logger.info(f"已存入 {min(i + chroma_add_batch_size, len(all_child_chunks_for_embedding))}/{len(all_child_chunks_for_embedding)} 個子區塊到 ChromaDB...")

    logger.info(f"所有 {len(all_child_chunks_for_embedding)} 個子區塊已存入向量資料庫。")

    # 儲存父區塊
    try:
        with open(PARENT_CHUNKS_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(parent_chunk_store, f, ensure_ascii=False, indent=4)
        logger.info(f"所有 {len(parent_chunk_store)} 個父區塊已儲存到 {PARENT_CHUNKS_FILE_PATH}")
    except Exception as e:
        logger.error(f"儲存父區塊到 JSON 檔案失敗: {e}", exc_info=True)

    logger.info("資料導入與索引完成。")
def start_mcp_retrieval_server():
    # 在伺服器啟動時初始化資料庫 (如果尚未初始化)
    init_db() # <--- 新增調用

    # Imports for FastAPI and Pydantic models
    logger.info("啟動 MCP 檢索服務 (RAG Retrieval Server)...")



    # 決定服務端模型使用的設備
    device_to_use_serving = EMBEDDING_DEVICE if EMBEDDING_DEVICE in ["cuda", "cpu", "mps"] else None
    logger.info(f"檢索服務將嘗試在設備上載入模型: {device_to_use_serving if device_to_use_serving else '自動檢測'}")

    # --- 在伺服器啟動時載入 LLM 模型 (用於 RAG 背景任務) ---
    llm_model = None
    api_key_llm = os.getenv(API_KEY_LLM_ENV_VAR)
    if not api_key_llm:
        logger.error(f"錯誤：請設定 {API_KEY_LLM_ENV_VAR} 環境變數以供 Gemini LLM 使用於背景 RAG 任務。")
        # 即使 LLM 載入失敗，檢索服務本身仍可運行，但 /ask_rag_async 會受影響
    else:
        try:
            genai_llm.configure(api_key=api_key_llm)
            llm_model = genai_llm.GenerativeModel(LLM_MODEL_NAME)
            logger.info(f"背景 RAG 任務的 LLM 模型 ({LLM_MODEL_NAME}) 載入成功。")
        except Exception as e:
            logger.error(f"背景 RAG 任務載入 LLM 模型 '{LLM_MODEL_NAME}' 失敗: {e}", exc_info=True)
            # 服務仍可啟動，但 /ask_rag_async 功能將受限

    # 任務結果儲存 (在生產環境中，應使用更持久的儲存，如 Redis 或資料庫)
    task_results_store = {}

    # 在伺服器啟動時載入嵌入模型 (只載入一次)
    logger.info(f"檢索服務正在載入嵌入模型: {BGE_EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(BGE_EMBEDDING_MODEL_NAME, device=device_to_use_serving)
        logger.info(f"檢索服務嵌入模型 ({BGE_EMBEDDING_MODEL_NAME}) 使用的設備: {embedding_model.device}")
        logger.info("檢索服務嵌入模型載入完成。")
    except Exception as e:
        logger.error(f"檢索服務載入嵌入模型 '{BGE_EMBEDDING_MODEL_NAME}' 失敗: {e}", exc_info=True)
        return None # 如果嵌入模型載入失敗，則不啟動服務

    logger.info(f"正在連接向量資料庫集合 '{COLLECTION_NAME}' 於: {VECTOR_DB_PATH}...")
    collection = get_or_create_chroma_collection(VECTOR_DB_PATH, COLLECTION_NAME, recreate_if_exists=False) # 檢索服務不應重建
    if collection is None:
        logger.error("無法連接到向量資料庫，檢索服務無法啟動。")
        return None

    # 載入父區塊儲存
    parent_store = {}
    if os.path.exists(PARENT_CHUNKS_FILE_PATH):
        try:
            with open(PARENT_CHUNKS_FILE_PATH, 'r', encoding='utf-8') as f:
                parent_store = json.load(f)
            logger.info(f"已從 {PARENT_CHUNKS_FILE_PATH} 載入 {len(parent_store)} 個父區塊。")
        except Exception as e:
            logger.error(f"從 {PARENT_CHUNKS_FILE_PATH} 載入父區塊失敗: {e}", exc_info=True)
    else:
        logger.warning(f"父區塊儲存檔案 {PARENT_CHUNKS_FILE_PATH} 未找到。父子檢索可能無法正常工作。")

    # --- 在伺服器啟動時載入交叉編碼器模型 (用於重排) ---
    cross_encoder = None
    if CROSS_ENCODER_MODEL_NAME:
        logger.info(f"檢索服務正在載入交叉編碼器模型: {CROSS_ENCODER_MODEL_NAME}...")
        try:
            cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device_to_use_serving)
            logger.info(f"檢索服務交叉編碼器模型 ({CROSS_ENCODER_MODEL_NAME}) 已配置使用設備: {device_to_use_serving if device_to_use_serving else '自動檢測'}")
        except Exception as e:
            logger.error(f"檢索服務載入交叉編碼器模型 '{CROSS_ENCODER_MODEL_NAME}' 失敗: {e}", exc_info=True)
            # 即使交叉編碼器載入失敗，服務仍可繼續，只是不提供重排功能

    # --- 背景 RAG 處理函數 ---
    def perform_full_rag_in_background(task_id: str, original_question: str, conversation_id: str = None): # 新增 conversation_id
        logger.info(f"背景任務 {task_id}: 開始處理問題 '{original_question[:50]}...'")
        task_results_store[task_id] = {"status": "processing", "original_question": original_question, "answer": None, "retrieved_contexts_summary": None, "error": None}

        if not llm_model:
            error_msg = "LLM 模型未成功載入，無法執行 RAG。"
            logger.error(f"背景任務 {task_id}: {error_msg}")
            task_results_store[task_id].update({"status": "failed", "error": error_msg})
            return

        # 如果提供了 conversation_id，嘗試從資料庫獲取對話歷史作為額外上下文
        history_context = ""
        if conversation_id and SessionLocal: # 確保 SessionLocal 已初始化
            db_for_history_gen = get_db()
            if db_for_history_gen:
                db_for_history: Session = next(db_for_history_gen)
                if db_for_history:
                    try:
                        # 獲取最近 N 條訊息
                        recent_messages = db_for_history.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.desc()).limit(5).all()
                        history_context = "\n".join([f"{msg.sender_type}: {msg.content}" for msg in reversed(recent_messages)]) # 反轉以保持時間順序
                        logger.info(f"背景任務 {task_id}: 為對話 {conversation_id} 獲取到歷史上下文:\n{history_context[:200]}...")
                    except SQLAlchemyError as e:
                        logger.error(f"背景任務 {task_id}: 從資料庫獲取對話 {conversation_id} 的歷史失敗: {e}", exc_info=True)
                    finally:
                        db_for_history.close()

        # 將使用者問題存入資料庫 (如果提供了 conversation_id)
        if conversation_id and SessionLocal:
            db_for_user_msg_gen = get_db()
            if db_for_user_msg_gen:
                db_for_user_msg: Session = next(db_for_user_msg_gen)
                if db_for_user_msg:
                    try:
                        user_msg_db = Message(conversation_id=conversation_id, sender_type="user", content=original_question)
                        db_for_user_msg.add(user_msg_db)
                        db_for_user_msg.commit()
                        logger.info(f"背景任務 {task_id}: 已將使用者問題存入對話 {conversation_id}")
                    except SQLAlchemyError as e:
                        db_for_user_msg.rollback()
                        logger.error(f"背景任務 {task_id}: 儲存使用者問題到對話 {conversation_id} 失敗: {e}", exc_info=True)
                    finally:
                        db_for_user_msg.close()

        current_rag_results = {
            "decomposed_queries_text": None,
            "hypothetical_document": None,
            "query_for_retrieval": original_question,
            "retrieved_contexts_from_server": [],
            "final_prompt_to_llm": None,
            "llm_final_answer": None,
            "error_message_detail": None
        }

        # 1. (可選) 查詢分解
        if PERFORM_QUERY_DECOMPOSITION:
            decomposition_prompt = EVAL_DECOMPOSITION_PROMPT_TEMPLATE.format(user_question_for_processing=original_question)
            try:
                decomposition_response = llm_model.generate_content(decomposition_prompt)
                current_rag_results["decomposed_queries_text"] = decomposition_response.text.strip()
            except Exception as e:
                current_rag_results["error_message_detail"] = f"問題分解失敗: {e}"

        # 2. (可選) HyDE
        query_for_retrieval_actual = original_question
        if USE_HYDE_QUERY_EXPANSION:
            hyde_prompt = EVAL_HYDE_PROMPT_TEMPLATE.format(user_question_for_processing=original_question)
            try:
                hyde_response = llm_model.generate_content(hyde_prompt)
                current_rag_results["hypothetical_document"] = hyde_response.text.strip()
                query_for_retrieval_actual = current_rag_results["hypothetical_document"]
            except Exception as e:
                current_rag_results["error_message_detail"] = (current_rag_results.get("error_message_detail","") + f"; HyDE失敗: {e}").strip(";")
        current_rag_results["query_for_retrieval"] = query_for_retrieval_actual

        # 3. 檢索 (直接調用檢索邏輯，而不是 HTTP call)
        try:
            query_embedding_np = embedding_model.encode(query_for_retrieval_actual, normalize_embeddings=True)
            query_embedding = query_embedding_np.tolist()
            child_results = collection.query(query_embeddings=[query_embedding], n_results=10, include=['metadatas']) # type: ignore
            
            parent_contexts_for_rag = []
            if child_results and child_results['metadatas'] and child_results['metadatas'][0]:
                unique_parent_ids = set(meta.get('parent_id') for meta in child_results['metadatas'][0] if meta.get('parent_id'))
                for p_id in list(unique_parent_ids):
                    parent_text = parent_store.get(p_id)
                    if parent_text:
                        parent_contexts_for_rag.append({"text": parent_text, "metadata": {"source": PDF_FILE_PATH, "retrieved_parent_id": p_id}})
            
            if parent_contexts_for_rag and cross_encoder:
                current_rag_results["retrieved_contexts_from_server"] = rerank_contexts(original_question, parent_contexts_for_rag, cross_encoder, NUM_CONTEXTS_AFTER_RERANK)
            else:
                current_rag_results["retrieved_contexts_from_server"] = parent_contexts_for_rag[:NUM_CONTEXTS_AFTER_RERANK]
        except Exception as e:
            current_rag_results["error_message_detail"] = (current_rag_results.get("error_message_detail","") + f"; 檢索失敗: {e}").strip(";")

        # 4. 生成最終答案
        context_text_list = [item.get("text", "") for item in current_rag_results["retrieved_contexts_from_server"]]
        context_string = "\n\n".join(context_text_list)
        
        # 準備主提示詞內容
        main_prompt_content = EVAL_FINAL_ANSWER_PROMPT_TEMPLATE.format(
            original_user_question=original_question, 
            decomposed_queries_text=current_rag_results["decomposed_queries_text"] or "", 
            context_string=context_string
        )

        # 整合歷史上下文到最終提示詞
        final_llm_prompt_to_use = main_prompt_content
        if history_context:
            # 確保歷史上下文被正確地整合到提示中
            # 這裡我們假設 EVAL_FINAL_ANSWER_PROMPT_TEMPLATE 內部沒有直接處理 history_context 的佔位符
            # 所以我們將其加在前面
            final_llm_prompt_to_use = (
                f"以下是本次對話的先前內容，請在回答時適當參考，以保持對話的連貫性：\n"
                f"先前對話：\n{history_context}\n\n"
                f"---\n\n"
                f"{main_prompt_content}"
            )
        current_rag_results["final_prompt_to_llm"] = final_llm_prompt_to_use

        try:
            llm_response = llm_model.generate_content(final_llm_prompt_to_use) # 這裡不使用 stream=True，因為是背景任務
            current_rag_results["llm_final_answer"] = llm_response.text.strip()
            task_results_store[task_id].update({"status": "completed", "answer": current_rag_results["llm_final_answer"], "retrieved_contexts_summary": [ctx.get('metadata', {}) for ctx in current_rag_results["retrieved_contexts_from_server"]]})
            logger.info(f"背景任務 {task_id}: 完成。")
        except Exception as e:
            error_msg = f"LLM回答生成失敗: {e}"
            current_rag_results["error_message_detail"] = (current_rag_results.get("error_message_detail","") + f"; {error_msg}").strip(";")
            task_results_store[task_id].update({"status": "failed", "error": error_msg, "detail": current_rag_results["error_message_detail"]})
            logger.error(f"背景任務 {task_id}: {error_msg}")
        
        # 將 AI 回應存入資料庫 (如果提供了 conversation_id 且成功生成了答案)
        if conversation_id and SessionLocal and current_rag_results["llm_final_answer"]:
            db_for_ai_msg_gen = get_db()
            if db_for_ai_msg_gen:
                db_for_ai_msg: Session = next(db_for_ai_msg_gen)
                if db_for_ai_msg:
                    try:
                        ai_msg_db = Message(
                            conversation_id=conversation_id,
                            sender_type="ai",
                            content=current_rag_results["llm_final_answer"],
                            meta_data={"retrieved_contexts_summary": [ctx.get('metadata', {}) for ctx in current_rag_results["retrieved_contexts_from_server"]]} # 更新屬性名稱
                        )
                        db_for_ai_msg.add(ai_msg_db)
                        db_for_ai_msg.commit()
                        logger.info(f"背景任務 {task_id}: 已將 AI 回應存入對話 {conversation_id}")
                    except SQLAlchemyError as e:
                        db_for_ai_msg.rollback()
                        logger.error(f"背景任務 {task_id}: 儲存 AI 回應到對話 {conversation_id} 失敗: {e}", exc_info=True)
                    finally:
                        db_for_ai_msg.close()

    

    @app.get("/task_status/{task_id}")
    async def get_task_status(task_id: str):
        result = task_results_store.get(task_id)
        if not result:
            return {"error": "任務 ID 不存在或任務尚未開始處理。"}, 404
        return result


    return app # <--- 新增：返回 FastAPI 應用程式實例

def rerank_contexts(query: str, contexts_with_metadata: list, cross_encoder_model, top_n: int):
    """
    使用交叉編碼器對檢索到的上下文進行重排。
    """
    if not contexts_with_metadata or not cross_encoder_model:
        logger.info("沒有上下文或交叉編碼器模型，跳過重排。")
        return contexts_with_metadata

    # 提取文本內容用於評分
    # 過濾掉那些可能沒有 'text' 鍵或 'text' 為空的上下文
    valid_contexts_for_reranking = [ctx for ctx in contexts_with_metadata if ctx.get("text")]
    if not valid_contexts_for_reranking:
        logger.info("沒有有效的文本內容用於重排。")
        return contexts_with_metadata

    sentence_pairs = [(query, ctx["text"]) for ctx in valid_contexts_for_reranking]

    logger.info(f"正在使用交叉編碼器對 {len(sentence_pairs)} 個上下文-查詢對進行評分...")
    try:
        scores = cross_encoder_model.predict(sentence_pairs, show_progress_bar=False) # 假設 predict API
    except Exception as e:
        logger.error(f"使用交叉編碼器進行預測時發生錯誤: {e}", exc_info=True)
        logger.warning("重排失敗，將返回原始順序的上下文。")
        return contexts_with_metadata # 返回原始上下文以避免中斷流程

    # 將分數與原始上下文（包含元數據）對應起來
    scored_contexts = list(zip(scores, valid_contexts_for_reranking))

    # 按分數降序排序
    scored_contexts.sort(key=lambda x: x[0], reverse=True)

    # 選取前 N 個
    reranked_contexts = [ctx for score, ctx in scored_contexts[:top_n]]
    logger.info(f"上下文已重排，選取前 {len(reranked_contexts)} 個。")
    return reranked_contexts

def start_qa_application_cli():
    logger.info("啟動智慧型文檔問答系統 CLI...")
    

    # --- 配置 Gemini LLM API ---
    # 注意：這裡的 API 金鑰是給 LLM (gemini-1.5-flash-latest) 使用的，不是嵌入模型
    api_key = os.getenv(API_KEY_LLM_ENV_VAR)
    
    if not api_key:
        logger.error(f"錯誤：請設定 {API_KEY_LLM_ENV_VAR} 環境變數以供 Gemini LLM 使用。")
        return
    else:
        # 在 DEBUG 模式下，可以記錄 API 金鑰已設定，或只記錄部分資訊以供識別
        logger.debug(f"環境變數 {API_KEY_LLM_ENV_VAR} 已成功讀取。")
        # 例如，只記錄金鑰的前後幾位，避免完整暴露
        # logger.debug(f"API 金鑰 (部分): '{api_key[:5]}...{api_key[-5:]}'")

    try:
        genai_llm.configure(api_key=api_key)
    except Exception as e:
        logger.error(f"配置 Gemini LLM API 金鑰 ({API_KEY_LLM_ENV_VAR}) 失敗: {e}", exc_info=True)
        return

    try:
        model = genai_llm.GenerativeModel(LLM_MODEL_NAME)
    except Exception as e:
        logger.error(f"載入 Gemini LLM 模型 '{LLM_MODEL_NAME}' 失敗: {e}", exc_info=True)
        return
    # --- 初始化資料庫 ---
    init_db() # 確保在 CLI 啟動時初始化資料庫
    db_session_gen = get_db()
    if db_session_gen is None:
        logger.error("無法獲取資料庫會話，CLI 模式下的對話歷史功能將不可用。")
        # 根據需要決定是否退出

    # 交叉編碼器模型和重排現在由 serve_mcp 處理
    # --- 模擬 MCP 客戶端和 RAG 流程 ---

    # --- 對話管理 ---
    current_conversation_id = None
    db: Session = next(db_session_gen) if db_session_gen else None

    while True:
        if not current_conversation_id:
            action = input("開始新對話 (n) 或載入現有對話 (l) 或退出 (q)？ ").lower()
            if action == 'n':
                if db:
                    new_conversation = Conversation()
                    db.add(new_conversation)
                    try:
                        db.commit()
                        db.refresh(new_conversation)
                        current_conversation_id = new_conversation.id
                        logger.info(f"已創建新對話，ID: {current_conversation_id}")
                    except SQLAlchemyError as e:
                        db.rollback()
                        logger.error(f"創建新對話失敗: {e}", exc_info=True)
                        continue # 重新開始循環
                else:
                    current_conversation_id = str(uuid.uuid4()) # 如果沒有資料庫，則使用臨時 ID
                    logger.info(f"資料庫未連接，使用臨時對話 ID: {current_conversation_id}")
            elif action == 'l':
                logger.info("載入現有對話功能暫未實現。請開始新對話。")
                continue
            elif action == 'q':
                break
            else:
                logger.warning("無效的選擇。")
                continue

        # 1. 接收使用者問題
        original_user_question = input(f"\n[對話 {current_conversation_id[:8]}...] 請輸入您的問題 (或輸入 'exit' 結束目前對話)：")
        if not original_user_question:
            logger.warning("未輸入問題。")
            continue
        if original_user_question.lower() == 'exit':
            logger.info(f"結束對話 {current_conversation_id[:8]}...")
            current_conversation_id = None # 重設以開始新對話或退出
            continue

        # 將使用者訊息存入資料庫
        if db:
            user_message_db = Message(conversation_id=current_conversation_id, sender_type="user", content=original_user_question)
            db.add(user_message_db)
            try:
                db.commit()
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"儲存使用者訊息失敗: {e}", exc_info=True)
                # 即使儲存失敗，也繼續處理問題

    decomposed_queries_text = "" # 初始化 decomposed_queries_text 以避免 NameError

    user_question_for_processing = original_user_question # 用於後續處理的查詢，可能被修改

    # 1.a (可選) 查詢分解
    if PERFORM_QUERY_DECOMPOSITION:
        logger.info("\n正在嘗試分解使用者問題...")
        decomposition_prompt = (
            f"請分析以下使用者問題。如果它是一個複雜的問題，可以分解為更簡單、獨立的子問題，請列出這些子問題，每個子問題佔一行。"
            f"如果問題已經很簡單且無法有意義地分解，只需重複原始問題即可。\n\n"
            f"使用者問題：{user_question_for_processing}\n\n"
            f"分解結果："
        )
        try:
            decomposition_response = model.generate_content(decomposition_prompt)
            decomposed_queries_text = decomposition_response.text
            logger.info(f"問題分解結果：\n{decomposed_queries_text}")
            # 注意：目前僅記錄分解結果。
        except Exception as e:
            logger.error(f"問題分解失敗: {e}", exc_info=True)

    query_for_retrieval = original_user_question # 預設使用原始問題進行檢索

    # 1.a (可選) HyDE 查詢擴展
    if USE_HYDE_QUERY_EXPANSION:
        logger.info(f"\n正在為問題 '{original_user_question}' 生成假設性文件 (HyDE)...")

        hyde_prompt = (
            f"請針對以下問題撰寫一個簡短的、假設性的段落作為答案。這個段落將用於尋找相關文件。"
            f"請專注於提供一個聽起來像事實且簡潔的答案，即使您需要根據問題的主題虛構合理的細節。\n\n"
            f"問題：{user_question_for_processing}\n\n" # 使用 user_question_for_processing 進行 HyDE
            f"假設性答案："
        )
        try:
            hyde_response = model.generate_content(hyde_prompt)
            hypothetical_document = hyde_response.text.strip()
            logger.info(f"生成的 HyDE 文件內容：\n{hypothetical_document}")
            query_for_retrieval = hypothetical_document # 使用 HyDE 文件進行檢索
        except Exception as e:
            logger.error(f"生成 HyDE 文件失敗: {e}", exc_info=True)
            logger.warning(f"HyDE 失敗，將使用 '{user_question_for_processing}' 進行檢索。")
    # 獲取對話歷史 (如果適用)
    history_for_prompt = "" # <--- 確保 history_for_prompt 總是被定義
    if db and current_conversation_id:
        try:
            # 獲取此對話中最近的 N 條訊息 (不包括當前使用者輸入的這條)
            recent_messages_db = db.query(Message)\
                                .filter(Message.conversation_id == current_conversation_id)\
                                .filter(Message.content != original_user_question).order_by(Message.timestamp.desc()).limit(5).all()
            history_for_prompt = "\n".join([f"{msg.sender_type}: {msg.content}" for msg in reversed(recent_messages_db)])
        except SQLAlchemyError as e:
            logger.error(f"從資料庫獲取對話歷史失敗: {e}", exc_info=True)
            
    # 2. 作為 MCP 客戶端，向您搭建的「MCP 檢索服務」發送請求
    logger.info(f"\n正在向檢索服務 ({RETRIEVAL_SERVER_URL}) 請求上下文...")
    try:
        # 直接呼叫我們之前建立的 _perform_retrieval 輔助函式
        # 傳入需要的參數即可
        retrieved_contexts_text = _perform_retrieval(
            query_for_retrieval=query_for_retrieval,
            original_user_question=original_user_question
        )
        # 我們可以假設 retrieved_metadatas 和 retrieved_items_for_llm_and_display
        # 在這個簡化流程中暫時不需要，或者 _perform_retrieval 可以回傳一個更豐富的物件
        
        logger.info(f"成功從內部檢索到 {len(retrieved_contexts_text)} 個上下文片段。")

    except Exception as e:
        # 錯誤處理變得更通用，因為不再有網路請求的特定錯誤
        logger.error(f"在執行內部檢索時發生錯誤: {e}", exc_info=True)
        logger.warning("檢索失敗，將使用空上下文進行回答。")
        retrieved_contexts_text = []

    # 3. 建構最終的提示 (prompt) 給 Gemini
    # 將所有檢索到的上下文都傳遞給 LLM
    context_string = "\n\n".join(retrieved_contexts_text) 
    prompt = (
        f"你是一位知識淵博、教學經驗豐富的 AI 教授。你的任務是利用以下提供的三項資訊，為「原始問題」合成一份全面、精確、且有條理的最終回答：\n" +
        f"1. **原始問題**：使用者最一開始的提問。\n" +
        f"2. **回答大綱**：一份由子問題組成的清單，你必須在回答中逐一回應這些子問題。\n" +
        f"3. **相關上下文**：從教科書中檢索到的、與問題最相關的原始文本片段。\n\n" +
        
        f"### 高品質回答的準則與指令：\n" +
        f"1. **嚴格遵循大綱**：你的回答結構必須清晰地反映「回答大綱」中的順序和內容，確保每個子問題都得到回應。\n" +
        f"2. **絕對忠於上下文**：你的回答中的每一個論點、例子和細節都**必須**直接來源於「相關上下文」。**絕對禁止使用任何外部知識。**\n" +
        f"3. **提取並呈現關鍵細節**：如果上下文中包含定義、數學公式、演算法偽代碼或關鍵參數，你必須將這些精確的細節融入解釋中。\n" +
        f"4. **深入解釋「為什麼」**：當子問題涉及原因或原理時，主動在上下文中尋找並總結其背後的理論依據或證明思路。\n\n" +
        f"5. **智慧處理語言**：你的回答應主要使用「原始問題」中的**主要語言**。然而，如果「原始問題」中包含了特定語言的專有名詞（尤其是英文技術術語），請在你的回答中**優先保留這些專有名詞的原文**，而不是將它們翻譯掉，以確保技術的精確性。例如，如果問題是「請解釋 `B-tree` 的 `fan-out`」，你的解釋就應該圍繞 `B-tree` 和 `fan-out` 這兩個詞展開。\n" +
        f"6. **處理資訊不足**：如果上下文不足以回答某個子問題，請在回答的相應部分明確指出「根據提供的資料，無法回答關於...的細節」。\n\n" +
        
        (f"以下是本次對話的先前內容，請在回答時適當參考，以保持對話的連貫性：\n" +
        f"先前對話：\n{history_for_prompt}\n\n" +
        f"---\n\n" if history_for_prompt else "") + 

        f"---[輸入資訊開始]---\n" +
        f"**1. 原始問題：**\n{original_user_question}\n\n" +
        f"**2. 回答大綱 (子問題清單)：**\n{decomposed_queries_text}\n\n" +
        f"**3. 相關上下文：**\n{context_string}\n\n" +
        f"---[輸入資訊結束]---\n\n" +
        f"**教授的最終回答：**"
    )

    # 4. 調用 Gemini API
    logger.info("\n正在向 Gemini API 發送請求...")
    try:
        # 以串流模式獲取回應
        response_stream = model.generate_content(prompt, stream=True)
    except Exception as e:
        logger.error(f"呼叫 Gemini API (模型: {LLM_MODEL_NAME}) 失敗: {e}", exc_info=True)
        # 您可能希望在這裡印出提示詞以供調試
        # logger.debug(f"\n使用的提示詞：\n{prompt}")
        logger.error("無法生成回答。")
        return

    # 5. 顯示結果
    print("\nGemini 的回答：") # 直接 print 給使用者看
    gemini_answer_text = "" # 用於儲存完整的 Gemini 回答
    chunk_count = 0
    try:
        for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text:
                print(chunk.text, end="", flush=True) # 即時印出每個 token，並刷新輸出
                gemini_answer_text += chunk.text
                chunk_count += 1
        print() # 串流結束後換行
        logger.debug(f"[串流資訊：共收到 {chunk_count} 個回應區塊]")
    except Exception as e:
        logger.error(f"處理 Gemini API 串流回應時發生錯誤: {e}", exc_info=True)
        print("\n[串流回應處理錯誤]")
    # 將 AI 回應存入資料庫
    if db and current_conversation_id and gemini_answer_text:
        ai_message_db = Message(
            conversation_id=current_conversation_id, 
            sender_type="ai", 
            content=gemini_answer_text,
            meta_data={"retrieved_contexts_summary": [item.get("metadata", {}) for item in retrieved_items_for_llm_and_display]} if retrieved_items_for_llm_and_display else None # 更新屬性名稱
        )
        db.add(ai_message_db)
        try:
            db.commit()
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"儲存 AI 回應失敗: {e}", exc_info=True)

    # 顯示引用的來源
    if retrieved_items_for_llm_and_display: # 檢查是否有用於顯示的項目
        print("\n引用來源 (部分元數據)：") # 直接 print 給使用者看
        for i, meta in enumerate(retrieved_metadatas):
            parent_id_cited = meta.get('retrieved_parent_id', '未知')
            print(f"  片段 {i+1}: 來源: {meta.get('source', '未知')}, 父區塊 ID: {parent_id_cited}") # type: ignore

        # 詢問使用者是否要查看引用區塊的內容
        view_chunks_choice = input("\n是否要查看引用父區塊的詳細內容？(y/n): ").lower()
        if view_chunks_choice == 'y':
            print(f"\n----------------------------------------------------")
            print(f"回顧 - 您的問題是：{original_user_question}")
            print(f"回顧 - Gemini 的回答是：\n{gemini_answer_text}")
            # 使用從伺服器檢索到的 retrieved_items_for_llm_and_display 中的文本
            for i, item in enumerate(retrieved_items_for_llm_and_display): # 使用已檢索的項目
                parent_text_cited = item.get("text")
                parent_id_cited = item.get("metadata", {}).get('retrieved_parent_id', '未知')
                if parent_text_cited:
                    print(f"\n--- 片段 {i+1} (父區塊 ID: {parent_id_cited}) 的內容 ---")
                    print(parent_text_cited)
                    print("----------------------------------------------------")
                else:
                    print(f"\n--- 片段 {i+1} (父區塊 ID: {parent_id_cited}) 的內容未提供 ---")

    logger.info("問答系統 CLI 已關閉。")

def process_single_query_for_evaluation(original_user_question: str, llm_model, retrieval_service_url: str):
    """
    處理單個查詢，執行完整的 RAG 流程，並返回詳細的中間結果和最終答案，
    以便進行自動化評估。

    Args:
        original_user_question: 使用者的原始問題。
        llm_model: 已初始化的 Gemini LLM 模型實例。
        retrieval_service_url: 檢索服務的 URL。

    Returns:
        一個字典，包含 RAG 流程各階段的輸出。
    """
    logger.debug(f"評估模式：正在處理問題: {original_user_question}")
    results = {
        "original_question": original_user_question,
        "decomposed_queries_text": None,
        "hypothetical_document": None,
        "query_for_retrieval": original_user_question, # 預設
        "retrieved_contexts_from_server": [], # 從伺服器獲取的原始上下文列表 (包含文本和元數據)
        "final_prompt_to_llm": None,
        "llm_final_answer": None,
        "error_message": None
    }

    user_question_for_processing = original_user_question

    if PERFORM_QUERY_DECOMPOSITION:
        logger.info(f"評估模式：正在為問題 '{original_user_question}' 進行分解...")
        decomposition_prompt = EVAL_DECOMPOSITION_PROMPT_TEMPLATE.format(user_question_for_processing=user_question_for_processing)
        try:
            decomposition_response = llm_model.generate_content(decomposition_prompt)
            results["decomposed_queries_text"] = decomposition_response.text.strip()
        except Exception as e:
            results["error_message"] = f"問題分解失敗: {e}"
            logger.error(f"問題 '{original_user_question}' 分解失敗: {e}")

    query_for_retrieval_actual = original_user_question # 預設使用原始問題

    if USE_HYDE_QUERY_EXPANSION:
        logger.info(f"評估模式：正在為問題 '{original_user_question}' 生成 HyDE...")
        hyde_prompt = EVAL_HYDE_PROMPT_TEMPLATE.format(user_question_for_processing=user_question_for_processing)
        try:
            hyde_response = llm_model.generate_content(hyde_prompt)
            results["hypothetical_document"] = hyde_response.text.strip()
            query_for_retrieval_actual = results["hypothetical_document"]
        except Exception as e:
            error_msg_hyde = f"HyDE失敗: {e}"
            results["error_message"] = (results.get("error_message","") + f";{error_msg_hyde}").strip(";")
            logger.error(f"問題 '{original_user_question}' 的 HyDE 生成失敗: {e}")
    
    results["query_for_retrieval"] = query_for_retrieval_actual

    try:
        logger.info(f"評估模式：正在為問題 '{original_user_question}' 從服務檢索上下文...")
        payload = {
            "query_for_retrieval": query_for_retrieval_actual,
            "original_user_question": original_user_question
        }
        response = requests.post(retrieval_service_url, json=payload)
        response.raise_for_status()
        retrieval_result_json = response.json()
        results["retrieved_contexts_from_server"] = retrieval_result_json.get("contexts", [])
    except Exception as e:
        error_msg_retrieval = f"檢索失敗: {e}"
        results["error_message"] = (results.get("error_message","") + f";{error_msg_retrieval}").strip(";")
        logger.error(f"問題 '{original_user_question}' 的上下文檢索失敗: {e}")

    retrieved_contexts_text_list = [item.get("text", "") for item in results["retrieved_contexts_from_server"]]
    context_string = "\n\n".join(retrieved_contexts_text_list)
    
    final_prompt = EVAL_FINAL_ANSWER_PROMPT_TEMPLATE.format(
        original_user_question=original_user_question,
        decomposed_queries_text=results.get("decomposed_queries_text") or "", # 使用 .get() 並提供預設值
        context_string=context_string
    )
    results["final_prompt_to_llm"] = final_prompt

    try:
        logger.info(f"評估模式：正在為問題 '{original_user_question}' 生成最終 LLM 回答...")
        llm_response = llm_model.generate_content(final_prompt)
        results["llm_final_answer"] = llm_response.text.strip()
    except Exception as e:
        error_msg_llm = f"LLM回答生成失敗: {e}"
        results["error_message"] = (results.get("error_message","") + f";{error_msg_llm}").strip(";")
        logger.error(f"問題 '{original_user_question}' 的 LLM 回答生成失敗: {e}")
        if results["llm_final_answer"] is None: # 確保即使失敗也有預設值
             results["llm_final_answer"] = "無法生成回答因先前錯誤。"
    return results

def run_evaluation_batch(questions_file_path: str, output_file_path: str):
    logger.info(f"啟動 RAG 系統批次評估，問題集: {questions_file_path}, 輸出至: {output_file_path}")
    
    # 1. 載入評估問題集 (假設是一個 JSON 檔案，包含一個問題列表)
    try:
        with open(questions_file_path, 'r', encoding='utf-8') as f:
            # --- Debugging: Log the start of the file ---
            file_content_snippet = f.read(200) # 讀取前 200 個字元
            logger.debug(f"嘗試載入問題集檔案 '{questions_file_path}' 的開頭內容 (前200字元): {file_content_snippet!r}")
            f.seek(0) # 將檔案指標重設回開頭，以便 json.load 正常工作
            # --- End Debugging ---
            evaluation_questions = json.load(f) # 預期格式: [{"question_id": "...", "original_question": "..."}, ...]
        if not isinstance(evaluation_questions, list):
            logger.error(f"問題集檔案 {questions_file_path} 的內容不是一個列表。請檢查格式。")
            return
    except Exception as e:
        logger.error(f"載入評估問題集 '{questions_file_path}' 失敗: {e}", exc_info=True)
        return

    # 2. 初始化 Gemini LLM 模型 (用於 RAG)
    try:
        api_key = os.getenv(API_KEY_LLM_ENV_VAR)
        if not api_key:
            logger.error(f"錯誤：請設定 {API_KEY_LLM_ENV_VAR} 環境變數以供 Gemini LLM 使用。")
            return
        genai_llm.configure(api_key=api_key)
        llm_model_for_rag = genai_llm.GenerativeModel(LLM_MODEL_NAME)
    except Exception as e:
        logger.error(f"初始化 Gemini LLM 模型 ({LLM_MODEL_NAME}) 失敗: {e}", exc_info=True)
        return

    all_evaluation_results = []
    total_questions = len(evaluation_questions)
    logger.info(f"將處理 {total_questions} 個評估問題...")

    for i, question_item in enumerate(evaluation_questions):
        original_question = question_item.get("original_question")
        question_id = question_item.get("question_id", f"eval_q_{i+1}") # 如果沒有 ID，則生成一個

        if not original_question:
            logger.warning(f"問題集中的第 {i+1} 個項目缺少 'original_question' 欄位，跳過。")
            continue
        
        logger.info(f"正在處理問題 {i+1}/{total_questions} (ID: {question_id}): {original_question[:100]}...")
        
        # 調用處理單個查詢的函數
        rag_output = process_single_query_for_evaluation(
            original_user_question=original_question,
            llm_model=llm_model_for_rag,
            retrieval_service_url=RETRIEVAL_SERVER_URL
        )
        
        # 將 question_id 添加到結果中
        rag_output["question_id"] = question_id
        all_evaluation_results.append(rag_output)

    # 5. 儲存所有結果
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_evaluation_results, f, ensure_ascii=False, indent=4)
        logger.info(f"所有 {len(all_evaluation_results)} 個問題的評估結果已儲存到: {output_file_path}")
    except Exception as e:
        logger.error(f"儲存評估結果到 '{output_file_path}' 失敗: {e}", exc_info=True)

    logger.info("批次評估完成。")

def main():
    parser = argparse.ArgumentParser(description="智慧型文檔問答系統 (MCP-Enhanced RAG) - 主控制程式")
    parser.add_argument("action", choices=["ingest", "serve_mcp", "ask_cli", "evaluate"],
                        help="選擇要執行的操作：'ingest', 'serve_mcp', 'ask_cli', 'evaluate' (批次評估)")
    parser.add_argument("--questions_file", type=str, default="./evaluation_questions.json",
                        help="用於 'evaluate' 操作的問題集 JSON 檔案路徑。")
    parser.add_argument("--output_file", type=str, default="./evaluation_results.json",
                        help="用於 'evaluate' 操作的輸出結果 JSON 檔案路徑。")
    args = parser.parse_args()

    if args.action == "ingest":
        run_ingestion_pipeline()
    elif args.action == "serve_mcp":
        app_instance = start_mcp_retrieval_server()
        if app_instance: # 僅當 FastAPI 應用成功創建時才運行
            uvicorn.run(app_instance, host="127.0.0.1", port=8000)
    elif args.action == "ask_cli":
        start_qa_application_cli()
    elif args.action == "evaluate":
        run_evaluation_batch(args.questions_file, args.output_file)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
