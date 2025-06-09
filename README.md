# RAG MCP Project for Algorithms

這是一個基於檢索增強生成 (RAG) 的智慧型文檔問答系統，專為演算法教科書（例如 "Introduction to Algorithms, 3rd Edition"）設計。它使用本地嵌入模型 (BAAI/bge-m3) 進行文本嵌入與檢索，並結合 Google Gemini API (gemini-1.5-flash-latest) 生成答案。

## 功能特性

*   **資料導入與索引 (`ingest`)**:
    *   從 PDF 文件載入內容。
    *   將文本切割成適當的區塊。
    *   使用本地 `BAAI/bge-m3` 模型生成文本嵌入。
    *   將嵌入向量和文本儲存到本地 ChromaDB 向量資料庫。
*   **MCP 檢索服務 (`serve_mcp`)**:
    *   啟動一個 FastAPI 伺服器。
    *   提供 `/retrieve` API 端點，接收查詢，使用本地嵌入模型生成查詢嵌入，並從 ChromaDB 檢索最相關的上下文片段。
*   **命令列問答應用 (`ask_cli`)**:
    *   允許使用者在命令列輸入問題。
    *   呼叫 MCP 檢索服務獲取相關上下文。
    *   將問題和上下文傳遞給 Gemini API (gemini-1.5-flash-latest) 生成答案。
    *   顯示 LLM 的回答以及引用的來源片段。
*   **配置管理**:
    *   使用 `.env` 檔案管理所有重要配置（檔案路徑、模型名稱、API 金鑰等）。
*   **日誌記錄**:
    *   詳細的日誌記錄，方便追蹤程式執行流程和偵錯。

## 先決條件

*   Python >= 3.12
*   `uv` (用於專案和虛擬環境管理，可選，但推薦)
*   一個有效的 Google Gemini API 金鑰。

## 安裝步驟

1.  **克隆 (Clone) 專案 (如果適用)**:
    ```bash
    git clone <your-repository-url>
    cd rag-mcp-project-foralg
    ```

2.  **建立並啟用虛擬環境**:
    建議使用 `uv` 來管理虛擬環境和依賴：
    ```bash
    # 如果您還沒有安裝 uv，請先安裝：pip install uv
    uv venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    # source .venv/bin/activate
    ```

3.  **安裝依賴**:
    使用 `uv` 安裝 `pyproject.toml` 中定義的依賴：
    ```bash
    uv pip install -r requirements.lock  # 或者 uv sync 如果您有 requirements.lock
    # 如果沒有 lock 檔案，可以直接從 pyproject.toml 安裝
    # uv pip install .
    ```
    或者，如果您不使用 `uv`，可以使用 pip：
    ```bash
    pip install -r requirements.txt # 您可能需要先從 pyproject.toml 生成 requirements.txt
    # 或者 pip install .
    ```

## 配置

在專案根目錄下創建一個名為 `.env` 的檔案，並填入以下內容。請將佔位符替換為您的實際值。

```env
# d:\Program_project\RAG_MCP_project_ForAlg\.env

PDF_FILE_PATH="./Introduction_to_algorithms-3rd Edition.pdf" # 您的 PDF 檔案路徑
VECTOR_DB_PATH="./chroma_db" # ChromaDB 本地儲存路徑
EMBEDDING_MODEL="BAAI/bge-m3" # 本地嵌入模型名稱
COLLECTION_NAME="my_pdf_collection" # ChromaDB 集合名稱
RETRIEVAL_SERVER_URL="http://127.0.0.1:8000/retrieve" # MCP 檢索服務 URL
GOOGLE_API_KEY_LLM="YOUR_ACTUAL_GEMINI_LLM_API_KEY" # 您的 Gemini LLM API 金鑰
LLM_MODEL_NAME="gemini-1.5-flash-latest" # Gemini LLM 模型名稱
LOG_LEVEL="INFO" # 日誌級別 (例如 DEBUG, INFO, WARNING, ERROR)
PARENT_CHUNKS_FILE_PATH="./parent_chunks_store.json" # 父區塊儲存路徑 (用於父子檢索)
```
**重要**: 確保將 `YOUR_ACTUAL_GEMINI_LLM_API_KEY` 替換為您有效的 Google Gemini API 金鑰。

## 使用方式

確保您的虛擬環境已啟用。

1.  **資料導入與索引**:
    首次運行或當 PDF 內容更新時，執行此命令來處理 PDF 並建立向量資料庫。
    ```bash
    uv run python main.py ingest
    ```

2.  **啟動 MCP 檢索服務**:
    在一個終端視窗中啟動 FastAPI 伺服器。
    ```bash
    uv run python main.py serve_mcp
    ```
    伺服器將在 `http://127.0.0.1:8000` 上運行。

3.  **啟動命令列問答應用**:
    在**另一個**終端視窗中啟動 CLI 問答介面。
    ```bash
    uv run python main.py ask_cli
    ```
    然後您可以輸入您的問題。

## 專案結構 (簡要)

*   `main.py`: 主應用程式邏輯，包含資料導入、檢索服務和問答 CLI 的實現。
*   `pyproject.toml`: 專案元數據和依賴定義。
*   `.env`: 環境配置檔案 (應被 `.gitignore` 忽略)。
*   `chroma_db/`: ChromaDB 本地儲存資料庫的目錄 (應被 `.gitignore` 忽略)。
*   `README.md`: 本檔案。

## 貢獻

歡迎提出問題 (Issues) 和拉取請求 (Pull Requests)。

## 授權條款

請在此處添加您的專案授權條款 (例如 MIT, Apache 2.0 等)。如果未指定，則默認為無授權。