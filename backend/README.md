# RAG MCP Project for Algorithms

This is a Retrieval Augmented Generation (RAG) based intelligent document question-answering system, specifically designed for algorithm textbooks (e.g., "Introduction to Algorithms, 3rd Edition"). It utilizes a local embedding model (BAAI/bge-m3) for text embedding and retrieval, combined with a local Cross-Encoder for re-ranking, and the Google Gemini API (e.g., gemini-1.5-flash-latest) for query processing (decomposition, HyDE) and answer generation.

## Features

*   **Data Ingestion and Indexing (`ingest`)**:
    *   Loads content from PDF documents.
    *   Splits text into appropriate chunks (parent and child chunks for better context).
    *   Generates text embeddings using the local `BAAI/bge-m3` model.
    *   Stores embedding vectors and corresponding text (child chunks) into a local ChromaDB vector database.
    *   Saves parent chunks to a JSON file for later retrieval.
*   **MCP Retrieval Service (`serve_mcp`)**:
    *   Launches a FastAPI server.
    *   Provides a `/retrieve` API endpoint that:
        *   Accepts a query (potentially expanded by HyDE) and the original user question.
        *   Generates query embeddings using the local embedding model (configured for GPU if available).
        *   Retrieves relevant child chunks from ChromaDB.
        *   Fetches corresponding parent chunks.
        *   Re-ranks the parent chunks using a local Cross-Encoder model (configured for GPU if available) based on the original user question.
        *   Returns the top re-ranked context passages.
*   **Command-Line Q&A Application (`ask_cli`)**:
    *   Allows users to input questions via the command line.
    *   Optionally performs query decomposition and HyDE query expansion using the LLM (client-side).
    *   Calls the MCP Retrieval Service to get relevant context.
    *   Passes the question, (optional) decomposed outline, and context to the Gemini API to generate an answer (client-side).
    *   Displays the LLM's answer and the cited source passages.
*   **Batch Evaluation Mode (`evaluate`)**:
    *   Processes a list of questions from a specified JSON file.
    *   For each question, performs the full RAG pipeline (optional decomposition, optional HyDE, retrieval via `serve_mcp`, answer generation).
    *   Records all intermediate outputs (original question, decomposed queries, hypothetical document, retrieved contexts, final prompt, LLM answer, errors) into a structured JSON output file.
    *   Facilitates systematic testing and iteration of prompts and system performance.
*   **Configuration Management**:
    *   Uses a `.env` file to manage all important configurations (file paths, model names, API keys, feature flags, device settings, etc.).
*   **Logging**:
    *   Detailed logging for tracing program execution flow and debugging.

## Prerequisites

*   Python >= 3.12
*   `uv` (for project and virtual environment management, optional but recommended)
*   A valid Google Gemini API Key.
*   (Optional) A CUDA-enabled GPU for accelerated embedding and re-ranking.

## Installation

1.  **Clone the Project (if applicable)**:
    ```bash
    git clone <your-repository-url>
    cd RAG_MCP_project_ForAlg
    ```

2.  **Create and Activate a Virtual Environment**:
    Using `uv` is recommended:
    ```bash
    # If you haven't installed uv: pip install uv
    uv venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    # source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    Use `uv` to install dependencies. If your project is configured with a `pyproject.toml` for installation:
    ```bash
    uv pip install .
    ```
    Alternatively, if you have a `requirements.txt` file:
    ```bash
    # uv pip install -r requirements.txt
    ```
    If not using `uv`, you can use pip:
    ```bash
    # pip install .
    # or pip install -r requirements.txt
    ```
    **Note**: Ensure you have the correct PyTorch version installed for your CUDA version if you plan to use GPU acceleration. The `requirements.txt` might specify this (e.g., `torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`).

## Configuration

Create a file named `.env` in the project root directory and fill in the following content. Replace placeholders with your actual values.

```env
# d:\Program_project\RAG_MCP_project_ForAlg\.env

PDF_FILE_PATH="./Introduction_to_algorithms-3rd Edition.pdf" # Path to your PDF file
VECTOR_DB_PATH="./chroma_db" # Path for ChromaDB local storage
EMBEDDING_MODEL="BAAI/bge-m3" # Local embedding model name
COLLECTION_NAME="my_pdf_collection" # ChromaDB collection name
RETRIEVAL_SERVER_URL="http://127.0.0.1:8000/retrieve" # MCP Retrieval Service URL
GOOGLE_API_KEY_LLM="YOUR_ACTUAL_GEMINI_LLM_API_KEY" # Your Gemini LLM API Key
LLM_MODEL_NAME="gemini-1.5-flash-latest" # Gemini LLM model name
LOG_LEVEL="INFO" # Logging level (e.g., DEBUG, INFO, WARNING, ERROR)
PARENT_CHUNKS_FILE_PATH="./parent_chunks_store.json" # Path for parent chunk storage (for parent-child retrieval)
EMBEDDING_DEVICE="cuda" # Device for embedding models (e.g., "cuda", "cpu", "mps", or leave empty for auto-detection)
CROSS_ENCODER_MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2" # Cross-encoder model for re-ranking
NUM_CONTEXTS_AFTER_RERANK="4" # Number of contexts to pass to LLM after re-ranking
USE_HYDE_QUERY_EXPANSION="true" # Enable/disable HyDE query expansion ("true" or "false")
PERFORM_QUERY_DECOMPOSITION="true" # Enable/disable query decomposition ("true" or "false")

Important: Ensure you replace YOUR_ACTUAL_GEMINI_LLM_API_KEY with your valid Google Gemini API key. Set EMBEDDING_DEVICE to "cpu" if you do not have a CUDA-enabled GPU or encounter issues.


Usage
Ensure your virtual environment is activated.

Data Ingestion and Indexing: Run this command the first time or when the PDF content is updated to process the PDF and create the vector database.

bash
uv run python main.py ingest
Start MCP Retrieval Service: Start the FastAPI server in one terminal window.

bash
uv run python main.py serve_mcp
The server will run on http://127.0.0.1:8000.

Start Command-Line Q&A Application: Start the CLI Q&A interface in another terminal window.

bash
uv run python main.py ask_cli
You can then type your questions.

Run Batch Evaluation: To evaluate the RAG system on a set of questions:

bash
uv run python main.py evaluate --questions_file ./evaluation_questions.json --output_file ./evaluation_results.json
--questions_file: Path to your JSON file containing questions (defaults to ./evaluation_questions.json). Each item in the JSON list should be an object with an "original_question" key and an optional "question_id" key.
--output_file: Path where the evaluation results will be saved (defaults to ./evaluation_results.json).
Project Structure (Brief)
main.py: Main application logic, including data ingestion, retrieval service, Q&A CLI, and batch evaluation.
pyproject.toml: Project metadata and dependency definitions.
requirements.txt: (Optional) List of dependencies.
.env: Environment configuration file (should be ignored by .gitignore).
chroma_db/: Directory for ChromaDB local storage (should be ignored by .gitignore).
parent_chunks_store.json: Stores parent text chunks for retrieval (generated by ingest).
evaluation_questions.json: (Example) Input file for batch evaluation.
evaluation_results.json: (Example) Output file from batch evaluation.
README.md: This file.
Contributing
Issues and Pull Requests are welcome.

License
Please add your project's license terms here (e.g., MIT, Apache 2.0, etc.). If not specified, it defaults to no license.