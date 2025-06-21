# **RAG MCP Algorithm Q\&A System**

This is an intelligent document question-answering system based on **Retrieval Augmented Generation (RAG)**, specifically designed for algorithm textbooks (e.g., "Introduction to Algorithms, 3rd Edition"). The system follows the spirit of the **Model Context Protocol (MCP)** by separating retrieval and generation, and integrates a frontend user interface to provide a complete interactive experience.

The backend uses a local embedding model (BAAI/bge-m3) for text vectorization and a local Cross-Encoder model for re-ranking to ensure retrieval quality. The Large Language Model (LLM) connects to the Google Gemini API (gemini-1.5-flash-latest) to process queries (decomposition, HyDE), generate final answers, and enable user conversation.

## **System Architecture**

graph TD  
    A\[User\] \--\> B{Frontend (React App)};  
    B \--\> |HTTP POST /ask| C{Backend API (FastAPI)};  
    C \--\> |Execute HyDE (Optional)| D{Gemini LLM};  
    D \--\> |Generate Hypothetical Answer| C;  
    C \--\> |1. Generate Query Vector| E\[Embedding Model (bge-m3)\];  
    C \--\> |2. Query Child Chunks| F\[Vector Database (ChromaDB)\];  
    F \--\> |Return Relevant Child Chunks| C;  
    C \--\> |3. Fetch Parent Chunks| G\[Parent Chunk Store (JSON)\];  
    G \--\> |Return Full Context| C;  
    C \--\> |4. Re-rank Contexts| H\[Cross-Encoder Model\];  
    H \--\> |Return Re-ranked Contexts| C;  
    C \--\> |5. Construct Final Prompt| C;  
    C \--\> |6. Generate Final Answer| D;  
    D \--\> |Return Answer| C;  
    C \--\> |Send Answer to Frontend| B;  
    B \--\> |Display AI Response| A;

## **Main Features**

* **Data Ingestion and Indexing (ingest)**:  
  * Loads content from PDF documents.  
  * Employs a **Parent-Child Chunks** strategy for text splitting to preserve full context.  
  * Generates text embeddings using the local BAAI/bge-m3 model.  
  * Stores vectors and corresponding child chunks into a local ChromaDB vector database.  
  * Saves parent chunks to a JSON file for subsequent retrieval.  
* **Backend Retrieval Service (main.py with FastAPI)**:  
  * Launches a FastAPI server to handle all core logic centrally.  
  * Provides an /ask API endpoint that receives questions from the frontend.  
  * **Query Expansion**: Optionally uses the **HyDE** (Hypothetical Document Embeddings) strategy, calling the LLM to generate a hypothetical answer to improve retrieval accuracy.  
  * **Efficient Retrieval**: Generates query vectors using the local model, retrieves relevant child chunks from ChromaDB, and then fetches the corresponding parent chunks from the JSON file.  
  * **Precise Re-ranking**: Uses a local Cross-Encoder model to re-rank the retrieved parent chunks, identifying the context most relevant to the original question.  
  * **Answer Generation**: Constructs a final prompt by integrating the original question, re-ranked context, and an optional query outline, then calls the Gemini API to generate an accurate answer.  
* **Frontend Interactive Interface (App.jsx with React)**:  
  * Provides an aesthetic and user-friendly chat interface.  
  * Allows users to input questions and see the AI's response in real-time.  
  * Includes a loading state ("thinking..." animation) to enhance user experience.  
  * Handles API requests and errors to ensure application stability.  
* **Command-Line Q\&A Mode (ask\_cli)**:  
  * Offers a pure command-line interface for developers to quickly test the backend RAG flow.  
  * Supports conversation history, allowing for the creation or loading of previous dialogues.  
* **Batch Evaluation Mode (evaluate)**:  
  * Can read a series of questions from a JSON file.  
  * Executes the full RAG pipeline for each question and records all intermediate outputs (e.g., hypothetical documents, retrieved contexts, final prompts).  
  * Saves the results in a structured JSON file, facilitating systematic evaluation and iteration of the system's performance.

## **Prerequisites**

* Python \>= 3.12  
* Node.js and npm (or yarn)  
* uv (recommended for Python virtual environment management)  
* A valid Google Gemini API Key  
* (Optional) A CUDA-enabled NVIDIA GPU for accelerated model computation

## **Installation and Setup**

### **1\. Clone the Project**

git clone https://github.com/skyblue030/RAG\_MCP\_project\_ForAlg.git  
cd RAG\_MCP\_project\_ForAlg

### **2\. Backend Setup**

a. **Create and Activate Python Virtual Environment**: Using uv is recommended: bash \# If you haven't installed uv: pip install uv uv venv \# Windows .\\.venv\\Scripts\\activate \# macOS/Linux source .venv/bin/activate

b. **Install Python Dependencies**: bash uv pip install \-r requirements.txt \> **Note**: If you wish to use GPU acceleration, ensure you have the PyTorch version that matches your CUDA version installed. The requirements.txt file may specify a particular version.

c. **Set Up Environment Variables**: Create a file named .env in the project root and fill it with the following content: \`\`\`env \# \--- File Paths & Database Settings \--- PDF\_FILE\_PATH="./Introduction\_to\_algorithms-3rd Edition.pdf" \# Path to your PDF file VECTOR\_DB\_PATH="./chroma\_db" \# Path for ChromaDB local storage PARENT\_CHUNKS\_FILE\_PATH="./parent\_chunks\_store.json" \# Path for parent chunk storage DATABASE\_URL="sqlite:///./rag\_mcp\_chat.db" \# Database path for conversation history (for ask\_cli)

\# \--- Model Settings \---  
EMBEDDING\_MODEL="BAAI/bge-m3"               \# Local embedding model  
CROSS\_ENCODER\_MODEL\_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2" \# Local re-ranking model  
LLM\_MODEL\_NAME="gemini-1.5-flash-latest"    \# Gemini LLM model

\# \--- API Key \---  
GOOGLE\_API\_KEY\_LLM="YOUR\_ACTUAL\_GEMINI\_LLM\_API\_KEY" \# Your Gemini API Key

\# \--- RAG Flow Control \---  
NUM\_CONTEXTS\_AFTER\_RERANK="4"               \# Number of contexts to pass to LLM after re-ranking  
USE\_HYDE\_QUERY\_EXPANSION="true"             \# Enable HyDE ("true" or "false")  
PERFORM\_QUERY\_DECOMPOSITION="true"          \# Enable query decomposition ("true" or "false")

\# \--- Execution Environment \---  
EMBEDDING\_DEVICE="cuda"                     \# Device for models ("cuda", "cpu", "mps", or leave empty for auto-detect)  
LOG\_LEVEL="INFO"                            \# Logging level (DEBUG, INFO, WARNING, ERROR)  
\`\`\`  
\*\*Important\*\*: You must replace \`YOUR\_ACTUAL\_GEMINI\_LLM\_API\_KEY\` with your own key. If you don't have a GPU, set \`EMBEDDING\_DEVICE\` to \`cpu\`.

### **3\. Frontend Setup**

a. **Navigate to the Frontend Directory** (if your App.jsx is in a subdirectory like frontend): bash \# Assuming the following structure, adjust the path if necessary \# RAG\_MCP\_project\_ForAlg/ \# |- frontend/ \# | |- package.json \# | |- src/ \# | |- App.jsx \# |- main.py \# cd frontend If your App.jsx is in the root directory, you can skip this step.

b. **Install Node.js Dependencies**: bash npm install

## **Usage**

**Please open two terminal windows.**

### **Terminal 1: Start the Backend Service**

1. **Data Ingestion (Run only the first time or when the document is updated)**: With your virtual environment activated, run the following command in the project root:  
   python main.py ingest

2. **Start the FastAPI Server**:  
   \# Run directly with uvicorn  
   uvicorn main:app \--host 127.0.0.1 \--port 8000 \--reload

   The server will be running at http://127.0.0.1:8000.

### **Terminal 2: Start the Frontend Application**

1. **Navigate to the frontend directory** (if necessary).  
2. **Start the React Development Server**:  
   npm run dev

   The application will typically run at http://localhost:5173.

Now you can open your browser to http://localhost:5173 and start interacting with your algorithm Q\&A system\!

### **Other Execution Modes**

* **Start the Command-Line Q\&A Interface**:  
  python main.py ask\_cli

* **Run Batch Evaluation**:  
  python main.py evaluate \--questions\_file ./evaluation\_questions.json \--output\_file ./evaluation\_results.json

## **Project Structure (Brief)**

/  
|-- .env                   \# Environment variables file (git ignored)  
|-- main.py                \# Main application: FastAPI server, CLI, evaluation logic  
|-- App.jsx                \# React frontend main component  
|-- App.css                \# React frontend styles  
|-- package.json           \# Frontend dependencies  
|-- requirements.txt       \# Python dependencies  
|-- chroma\_db/             \# ChromaDB local storage directory (git ignored)  
|-- parent\_chunks\_store.json \# Parent chunk data (generated by ingest)  
|-- evaluation\_questions.json \# (Example) Input questions for batch evaluation  
\`-- README.md              \# This file

## **Contributing**

Issues and Pull Requests are welcome.

## **License**

Please add your project's license terms here (e.g., MIT, Apache 2.0, etc.). If not specified, it defaults to no license.