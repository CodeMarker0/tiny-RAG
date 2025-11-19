# Quick Start Guide

This guide will help you set up and run the tiny RAG System quickly.

## 1. Configure System

### 1.1 Copy Configuration File

```bash
copy .env.example .env
```

### 1.2 Edit .env File

**Important**: Fill in your actual configurations.

```bash
# Embedding Model (will be downloaded automatically on first run)
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# LLM Model (requires Ollama to be installed first)
LLM_MODEL_NAME=qwen3:4b
OLLAMA_BASE_URL=http://localhost:11434

# Document Processing Configuration
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# RAG Retrieval Configuration
RETRIEVAL_TOP_K=5
TEMPERATURE=0.7
MAX_TOKENS=2000

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/rag_system.log
```

## 2. Install Dependencies

### 2.1 Create Virtual Environment

```bash
python -m venv venv
```

### 2.2 Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 2.3 Install Python Dependencies

```bash
pip install -r requirements.txt
```

**If downloads are slow, you can use a domestic mirror (e.g., for China):**
```bash
pip install -r requirements.txt
```

## 3. Install Ollama (Local LLM)

### 3.1 Download and Install

Visit https://ollama.ai/ to download the version suitable for your system.

### 3.2 Pull a Model

```bash
ollama pull qwen3:4b
```

**Or use other models:**
```bash
# Smaller model (requires fewer resources)
ollama pull qwen2:1.8b

# More powerful model (requires more resources)
ollama pull qwen:14b
```

### 3.3 Verify Installation

```bash
# Test if Ollama is running
curl http://localhost:11434/api/tags
```

## 4. Build Knowledge Base

Place your `.txt` and `.pdf` documents in the `data/raw/` directory or any other directory you prefer. Then, run the build command, specifying the paths to your documents.

### 4.1 First Build

```bash
# Example: python main.py --build data/raw/document1.txt data/raw/document2.pdf
python main.py --build <path/to/your/document1.txt> <path/to/your/document2.pdf> ...
```

**This command will:**
1.  Load all specified `.txt` and `.pdf` documents.
2.  Process and clean the documents.
3.  Split them into appropriately sized chunks.
4.  Vectorize and store them in the FAISS database.

**Estimated time:** Depending on the number and size of documents, this may take several minutes to tens of minutes.

### 4.2 Force Rebuild (Clears Old Data)

```bash
# Example: python main.py --rebuild --build data/raw/document1.txt
python main.py --rebuild --build <path/to/your/document1.txt> ...
```

## 5. Start Using

### 5.1 Interactive Query (Recommended)

```bash
python main.py --interactive
```

Then enter your question:
```
Your question: How to handle a server memory overflow issue?
```

### 5.2 Single Query

```bash
python main.py --query "What is the process for database backup?"
```

### 5.3 View Statistics

```bash
python main.py --stats
```

## 6. Frequently Asked Questions

### Q1: Prompt "Embedding model loading failed"

A: The model (approx. 200MB) will be downloaded automatically on the first run. Please ensure your network connection is stable.

If the download is slow, you can configure a HuggingFace mirror:
```bash
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com
```

### Q2: Ollama connection failed

A: Ensure the Ollama service is running:
```bash
# Start Ollama service
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### Q3: Out of memory

A: Adjust configurations to reduce resource usage:

Edit `.env`:
```bash
# Use a smaller model
LLM_MODEL_NAME=qwen2:1.8b

# Reduce chunk size
CHUNK_SIZE=300

# Reduce retrieval quantity
RETRIEVAL_TOP_K=3
```

### Q4: Inaccurate query results

A: Try the following optimizations:

1.  **Increase retrieval quantity**
    ```bash
    RETRIEVAL_TOP_K=10
    ```

2.  **Adjust chunk size**
    ```bash
    CHUNK_SIZE=800
    CHUNK_OVERLAP=100
    ```

3.  **Rebuild the knowledge base**
    ```bash
    python main.py --rebuild --build <path/to/your/document1.txt> ...
    ```

## 7. Data Update

### Regularly Update Knowledge Base

It is recommended to update the knowledge base regularly, especially when new documents are added or existing ones are modified. You can do this by running the build command again with the relevant file paths.

**Method 1: Manual Update**
```bash
# Example: python main.py --build data/raw/new_document.txt
python main.py --build <path/to/new/document.txt> ...
```

**Method 2: Scheduled Task (Example for Linux using Cron)**
Add to crontab (e.g., to run daily at 2 AM):
```bash
0 2 * * * cd /path/to/tiny-RAG && source venv/bin/activate && python main.py --build <path/to/documents/>*.txt <path/to/documents/>*.pdf
```
(Adjust paths and file patterns as needed for your specific setup.)

## 8. System File Description

```
tiny-RAG/
├── .env                        # Configuration file (needs to be created)
├── main.py                     # Main program entry point
├── requirements.txt            # Python dependencies
├── src/                        # Source code
│   ├── config.py              # Configuration management
│   ├── data_loader.py         # Local file data loading
│   ├── document_processor.py  # Document processing
│   ├── vector_store.py        # FAISS vector storage
│   └── rag_system.py          # RAG core logic
├── data/                       # Data directory
│   └── vectorstore/           # Vector database files
└── logs/                       # Log files
    └── rag_system.log         # System log
```

## 9. Next Steps

-   **Advanced Configuration**: Refer to `README.md` for more configuration options.
-   **API Service**: If you need to provide a Web API, refer to the FastAPI example in `README.md`.
-   **Performance Optimization**: Use GPU acceleration; refer to the optimization section in `README.md`.

## Need Help?

-   View full documentation: `README.md`
-   Check log file: `logs/rag_system.log`
-   Submit an Issue or contact the system administrator.