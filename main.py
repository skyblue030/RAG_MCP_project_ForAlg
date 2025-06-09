import argparse
import uvicorn # 將 uvicorn 的匯入移到檔案頂部
import os # 用於讀取環境變數
import uuid # 用於生成唯一 ID
import sys # 匯入 sys 模組以便退出程式
import json # 用於處理父區塊的 JSON 儲存
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter # 將 import 移到檔案頂部
from dotenv import load_dotenv

# 在所有其他程式碼之前載入 .env 檔案中的環境變數
# 這會查找與此 main.py 同目錄下的 .env 檔案
load_dotenv(override=True) # 強制使用 .env 檔案中的值覆蓋已存在的環境變數

# --- Configuration Constants ---
# 現在這些值會優先從 .env 檔案讀取，如果 .env 中未定義或為空，則使用提供的預設值
PDF_FILE_PATH = os.getenv("PDF_FILE_PATH", "./Introduction_to_algorithms-3rd Edition.pdf")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
BGE_EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", 'BAAI/bge-m3')
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_pdf_collection")
RETRIEVAL_SERVER_URL = os.getenv("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8000/retrieve")
API_KEY_LLM_ENV_VAR = "GOOGLE_API_KEY_LLM" # 這仍然是環境變數的 *名稱*
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-latest")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", None) # 從 .env 讀取期望的設備，預設為 None (自動)
PARENT_CHUNKS_FILE_PATH = os.getenv("PARENT_CHUNKS_FILE_PATH", "./parent_chunks_store.json")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2") # 交叉編碼器模型
NUM_CONTEXTS_AFTER_RERANK = int(os.getenv("NUM_CONTEXTS_AFTER_RERANK", "4")) # 重排後傳遞給 LLM 的上下文數量
USE_HYDE_QUERY_EXPANSION = os.getenv("USE_HYDE_QUERY_EXPANSION", "false").lower() == "true" # 是否使用 HyDE 進行查詢擴展
PERFORM_QUERY_DECOMPOSITION = os.getenv("PERFORM_QUERY_DECOMPOSITION", "false").lower() == "true" # 是否執行查詢分解
# NUM_CONTEXTS_FOR_LLM = int(os.getenv("NUM_CONTEXTS_FOR_LLM", "6")) # 移除或註解此行，不再需要從 .env 讀取
# --- Logging Setup ---
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


def get_or_create_chroma_collection(db_path: str, collection_name: str, recreate_if_exists: bool = False):
    """
    初始化 ChromaDB 客戶端並獲取或創建集合。
    """
    try:
        import chromadb
    except ImportError:
        logger.error("請先安裝必要的套件: pip install chromadb")
        return None

    client = chromadb.PersistentClient(path=db_path)
    
    if recreate_if_exists:
        try:
            logger.info(f"正在嘗試刪除可能已存在的舊集合 (因 recreate_if_exists=True): {collection_name}...")
            client.delete_collection(name=collection_name)
            logger.info(f"舊集合 '{collection_name}' 已成功刪除。")
        except chromadb.api.errors.CollectionNotFoundError: # 假設 ChromaDB 有這樣的特定錯誤
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

    # 0. 載入嵌入模型 (本地)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("請先安裝必要的套件: pip install sentence-transformers")
        return

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

    # 1. 文件載入 (PDF) 和文本切割器
    try:
        import fitz  # PyMuPDF
    # from langchain_text_splitters import RecursiveCharacterTextSplitter # 從這裡移除
    except ImportError:
        logger.error("請先安裝必要的套件: pip install PyMuPDF") # chromadb is already a general dependency
        return

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
    logger.info("啟動 MCP 檢索服務 (RAG Retrieval Server)...")
    try:
        from fastapi import FastAPI, Request
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError:
        logger.error("請先安裝必要的套件: pip install fastapi uvicorn chromadb sentence-transformers")
        return

    app = FastAPI()

    # 決定服務端模型使用的設備
    device_to_use_serving = EMBEDDING_DEVICE if EMBEDDING_DEVICE in ["cuda", "cpu", "mps"] else None
    logger.info(f"檢索服務將嘗試在設備上載入模型: {device_to_use_serving if device_to_use_serving else '自動檢測'}")

    # 在伺服器啟動時載入嵌入模型 (只載入一次)
    logger.info(f"檢索服務正在載入嵌入模型: {BGE_EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model_retrieval = SentenceTransformer(BGE_EMBEDDING_MODEL_NAME, device=device_to_use_serving)
        logger.info(f"檢索服務嵌入模型 ({BGE_EMBEDDING_MODEL_NAME}) 使用的設備: {embedding_model_retrieval.device}")
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
    cross_encoder_retrieval = None
    if CROSS_ENCODER_MODEL_NAME:
        logger.info(f"檢索服務正在載入交叉編碼器模型: {CROSS_ENCODER_MODEL_NAME}...")
        try:
            from sentence_transformers import CrossEncoder # 確保 CrossEncoder 已匯入
            cross_encoder_retrieval = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device_to_use_serving)
            logger.info(f"檢索服務交叉編碼器模型 ({CROSS_ENCODER_MODEL_NAME}) 已配置使用設備: {device_to_use_serving if device_to_use_serving else '自動檢測'}")
        except Exception as e:
            logger.error(f"檢索服務載入交叉編碼器模型 '{CROSS_ENCODER_MODEL_NAME}' 失敗: {e}", exc_info=True)
            # 即使交叉編碼器載入失敗，服務仍可繼續，只是不提供重排功能

    @app.post("/retrieve")
    async def retrieve_context(request: Request):
        """
        接收查詢，從向量資料庫檢索相關上下文。
        這是一個簡化的 MCP 檢索介面模擬。
        """
        try:
            data = await request.json()
            query_for_initial_retrieval = data.get("query_for_retrieval") # 用於初步檢索的查詢 (可能是 HyDE)
            original_user_question_for_rerank = data.get("original_user_question") # 用於重排的原始問題

            if not query_for_initial_retrieval or not original_user_question_for_rerank:
                return {"error": "Missing 'query' in request body"}, 400

            logger.info(f"收到檢索請求，初步檢索查詢: '{query_for_initial_retrieval}'")
            logger.info(f"  用於重排的原始問題: '{original_user_question_for_rerank}'")

            # 執行檢索
            try:
                # 使用本地模型對查詢進行嵌入，normalize_embeddings=True 通常對 BGE 模型是推薦的
                query_embedding_np = embedding_model_retrieval.encode(query_text, normalize_embeddings=True)
                query_embedding = query_embedding_np.tolist() # 轉換為 list
            except Exception as e:
                logger.error(f"本地模型嵌入查詢 '{query_for_initial_retrieval}' 失敗: {e}", exc_info=True)
                return {"error": f"無法使用本地模型嵌入查詢: {e}"}, 500

            # 1. 檢索相關的子區塊，可以檢索比最終需要的父區塊數量更多的子區塊
            # 以便有更多機會找到不同的父區塊，或為重排提供更多候選
            num_child_candidates = 10 # 增加初步檢索的子區塊候選數量，以便為重排提供更多選擇
            child_results = collection.query( # type: ignore
                query_embeddings=[query_embedding],
                n_results=num_child_candidates,
                include=['metadatas'] # 只需要元數據來獲取 parent_id
            )

            retrieved_parent_contexts = []
            if child_results and child_results['metadatas'] and child_results['metadatas'][0]:
                unique_parent_ids = set()
                for child_meta in child_results['metadatas'][0]:
                    parent_id = child_meta.get('parent_id')
                    if parent_id:
                        unique_parent_ids.add(parent_id)
                
                logger.info(f"從 {len(child_results['metadatas'][0])} 個子區塊中找到 {len(unique_parent_ids)} 個唯一的父區塊 ID。")

                # 獲取父區塊文本，這裡獲取所有找到的唯一父區塊，後續由重排器篩選
                num_final_parent_contexts = len(unique_parent_ids) # 獲取所有唯一父區塊
                for p_id in list(unique_parent_ids)[:num_final_parent_contexts]:
                    parent_text = parent_store.get(p_id)
                    if parent_text:
                        retrieved_parent_contexts.append({
                            "text": parent_text,
                            "metadata": {"source": PDF_FILE_PATH, "retrieved_parent_id": p_id} # 可以添加更多父區塊元數據
                        })
                    else:
                        logger.warning(f"未在父區塊儲存中找到 parent_id '{p_id}' 對應的文本。")
            
            logger.info(f"初步檢索到 {len(retrieved_parent_contexts)} 個父區塊上下文片段。")

            # 在伺服器端進行重排
            if retrieved_parent_contexts and cross_encoder_retrieval:
                logger.info(f"準備對 {len(retrieved_parent_contexts)} 個上下文進行重排 (使用原始問題)...")
                reranked_contexts_for_client = rerank_contexts(
                    original_user_question_for_rerank, # 使用原始問題進行重排
                    retrieved_parent_contexts,
                    cross_encoder_retrieval,
                    NUM_CONTEXTS_AFTER_RERANK # 使用 .env 中定義的數量
                )
                logger.info(f"重排後，將返回 {len(reranked_contexts_for_client)} 個上下文給客戶端。")
                return {"contexts": reranked_contexts_for_client}
            return {"contexts": retrieved_parent_contexts[:NUM_CONTEXTS_AFTER_RERANK]} # 如果不重排或無交叉編碼器，則按數量截斷

        except Exception as e:
            logger.error(f"檢索過程中發生錯誤: {e}", exc_info=True)
            return {"error": str(e)}, 500

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
    try:
        # 仍然需要 google.generativeai 用於 LLM (Gemini)
        import google.generativeai as genai_llm
        import requests # 用於呼叫 MCP 檢索服務
        # from sentence_transformers import CrossEncoder # 不再需要在 CLI 中載入
    except ImportError:
        logger.error("請先安裝必要的套件: pip install google-generativeai requests sentence-transformers")
        return

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

    # 交叉編碼器模型和重排現在由 serve_mcp 處理
    # --- 模擬 MCP 客戶端和 RAG 流程 ---
    # 1. 接收使用者問題
    original_user_question = input("請輸入您的問題：")
    if not original_user_question:
        logger.warning("未輸入問題。")
        return

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

    # 2. 作為 MCP 客戶端，向您搭建的「MCP 檢索服務」發送請求
    logger.info(f"\n正在向檢索服務 ({RETRIEVAL_SERVER_URL}) 請求上下文...")
    try:
        # 向檢索服務發送 query_for_retrieval (可能為 HyDE) 和 original_user_question (用於重排)
        payload = {
            "query_for_retrieval": query_for_retrieval,
            "original_user_question": original_user_question
        }
        response = requests.post(RETRIEVAL_SERVER_URL, json=payload)
        response.raise_for_status() # 如果請求失敗 (非 2xx 狀態碼)，拋出異常
        retrieval_result = response.json()
        # 現在從伺服器接收到的上下文已經是重排和篩選過的
        retrieved_items_for_llm_and_display = retrieval_result.get("contexts", [])
        logger.info(f"從檢索服務獲取到 {len(retrieved_items_for_llm_and_display)} 個（已重排的）上下文片段。")

        # 從返回的項目中提取文本和元數據
        retrieved_contexts_text = [item.get("text", "") for item in retrieved_items_for_llm_and_display]
        retrieved_metadatas = [item.get("metadata", {}) for item in retrieved_items_for_llm_and_display]

    except requests.exceptions.RequestException as e:
        logger.error(f"呼叫檢索服務失敗: {e}", exc_info=True)
        logger.warning("將使用空上下文進行回答。")
        retrieved_contexts_text = []
        retrieved_metadatas = []
        retrieved_items_for_llm_and_display = []

    # 3. 建構最終的提示 (prompt) 給 Gemini
    # 將所有檢索到的上下文都傳遞給 LLM
    context_string = "\n\n".join(retrieved_contexts_text) 
    # prompt = f"請嚴格根據以下提供的「上下文」資訊來回答「問題」。如果「上下文」中沒有明確提及與「問題」直接相關的內容，請回答「根據提供的資料，我無法找到關於此問題的直接資訊」。絕對不要使用任何「上下文」之外的知識。\n\n上下文：\n{context_string}\n\n問題：{original_user_question}\n\n回答："
    prompt = (
        f"作為一位知識淵博且樂於助人的AI老師，請你嚴格根據以下提供的「上下文」資訊來回答「問題」。"
        f"你的回答應該清晰、易懂，並且有條理，就像在向學生解釋概念一樣。"
        f"如果可能，可以適當使用點列、步驟說明或簡短的例子來幫助理解，但所有解釋都必須基於「上下文」。"
        f"如果「上下文」中沒有明確提及與「問題」直接相關的內容，請回答「根據提供的資料，我無法找到關於此問題的直接資訊」。"
        f"請絕對不要使用任何「上下文」之外的知識。\n\n上下文：\n{context_string}\n\n問題：{original_user_question}\n\n回答："
    )

    # 4. 調用 Gemini API
    logger.info("\n正在向 Gemini API 發送請求...")
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        logger.error(f"呼叫 Gemini API (模型: {LLM_MODEL_NAME}) 失敗: {e}", exc_info=True)
        # 您可能希望在這裡印出提示詞以供調試
        # logger.debug(f"\n使用的提示詞：\n{prompt}")
        logger.error("無法生成回答。")
        return
    # 5. 顯示結果
    # 使用 logger.info 或直接 print 都可以，取決於您希望如何呈現給使用者
    print("\nGemini 的回答：") # 直接 print 給使用者看
    gemini_answer_text = response.text # 儲存 Gemini 的回答
    print(gemini_answer_text) # 直接 print 給使用者看

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

def main():
    parser = argparse.ArgumentParser(description="智慧型文檔問答系統 (MCP-Enhanced RAG) - 主控制程式")
    parser.add_argument("action", choices=["ingest", "serve_mcp", "ask_cli"],
                        help="選擇要執行的操作：'ingest' (處理並索引來源文檔), 'serve_mcp' (啟動 MCP 檢索服務器), 'ask_cli' (啟動命令列問答應用)")
    args = parser.parse_args()

    if args.action == "ingest":
        run_ingestion_pipeline()
    elif args.action == "serve_mcp":
        app_instance = start_mcp_retrieval_server()
        if app_instance: # 僅當 FastAPI 應用成功創建時才運行
            uvicorn.run(app_instance, host="127.0.0.1", port=8000)
    elif args.action == "ask_cli":
        start_qa_application_cli()

if __name__ == "__main__":
    main()
