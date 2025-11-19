# tiny-RAG

A minimal, beginner-friendly Retrieval-Augmented Generation (RAG) system specifically designed for processing and querying local .txt and .pdf documents. Built with LangChain and FAISS, Tiny-RAG proves that powerful AI applications don't need to be complicated. It has packed a fully functional document QA system into just ~700 lines of Python code.

## System Features

- **Local File Support**: Automatically loads and processes `.txt` and `.pdf` documents from local directories.
- **Intelligent Document Processing**: Automatic document cleaning, chunking, and vectorization.
- **Local Deployment**: Utilizes local open-source models to ensure data privacy.
- **Efficient Retrieval**: Semantic retrieval based on the FAISS vector database.
- **Interactive Query**: Provides a user-friendly command-line interface.

## Technology Stack

- **RAG Framework**: LangChain
- **Vector Database**: FAISS (pure Python version, no C++ compilation required)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **LLM Model**: Qwen3/ChatGLM3, etc. (via Ollama)
- **Document Processing**: RecursiveCharacterTextSplitter
- **Logging System**: Loguru

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Local File Source                    │
│                 (.txt and .pdf documents)               │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│             Data Loading Module (LocalFileLoader)       │
│  - Reads .txt and .pdf files                            │
│  - Extracts content and basic metadata                  │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│           Document Processing Module (DocumentProcessor)│
│  - Text cleaning                                        │
│  - Intelligent Chunking                                 │
│  - Metadata extraction                                  │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│           Vector Storage Module (VectorStoreManager)    │
│  - Embedding Vectorization                              │
│  - FAISS Vector Database                                │
│  - Persistent Storage                                   │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│               RAG Query System (RAGSystem)              │
│  - Semantic Retrieval                                   │
│  - LLM Generation                                       │
│  - Answer Synthesis                                     │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
tiny-RAG/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── data_loader.py           # Local file data loading
│   ├── document_processor.py    # Document processing
│   ├── vector_store.py          # Vector storage
│   ├── rag_system.py            # RAG system core
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Logging utility
├── data/
│   ├── raw/                     # Raw data (e.g., your .txt and .pdf files)
│   └── vectorstore/             # Vector database persistence
├── logs/                        # Log files
├── main.py                      # Main program entry point
├── requirements.txt             # Project dependencies
├── .env.example                 # Environment variables example
└── README.md                    # Project documentation
```

## Quick Start

### 1. Environment Setup

#### System Requirements
- Python 3.8+
- 8GB+ RAM
- GPU recommended (optional)

#### Install Ollama (Local LLM Service)

**Windows/Mac/Linux:**
```bash
# Visit https://ollama.ai/ to download and install

# After installation, pull a model
ollama pull qwen3:4b
# Or use other models
# ollama pull llama2
```

### 2. Project Installation

```bash
# Clone the repository
git clone <repository_url>
cd tiny-RAG

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure System

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file to configure your parameters
# Important configurations:
# - EMBEDDING_MODEL_NAME: Name of the embedding model (e.g., sentence-transformers/all-MiniLM-L6-v2)
# - LLM_MODEL_NAME: Name of the LLM model (e.g., qwen3:4b)
```

### 4. Build Knowledge Base

Place your `.txt` and `.pdf` documents in the `data/raw/` directory or any other directory you prefer. Then, run the build command:

```bash
# Build the knowledge base from specified local files
# Load all files in the directory data/raw
python main.py --build

# Load specific files: python main.py --build data/raw/document1.txt data/raw/document2.pdf
python main.py --build <path/to/your/document1.txt> <path/to/your/document2.pdf> ...

# Force rebuild the knowledge base (clears old data)
python main.py --rebuild --build <path/to/your/document1.txt> <path/to/your/document2.pdf> ...
```

### 5. Query Usage

#### Interactive Mode (Recommended)
```bash
python main.py --interactive
```

#### Single Query
```bash
python main.py --query "How to handle a server memory overflow issue?"
```

#### View Statistics
```bash
python main.py --stats
```

## Example Usage

### Example 1: Building the Knowledge Base

```python
from src import LocalFileLoader, DocumentProcessor, VectorStoreManager

# 1. Load data from local files
file_paths = ["data/raw/my_document.txt", "data/raw/another_document.pdf"]
loader = LocalFileLoader(file_paths=file_paths)
documents = loader.load()

# 2. Process documents
processor = DocumentProcessor(chunk_size=500)
processed_docs = processor.process_documents(documents)
chunks = processor.split_documents(processed_docs)

# 3. Create vector store
vector_manager = VectorStoreManager()
vector_manager.create_vectorstore(chunks)
```

### Example 2: RAG Query

```python
from src import RAGSystem, VectorStoreManager

# Initialize
vector_manager = VectorStoreManager()
vector_manager.load_vectorstore()

rag = RAGSystem(vector_store_manager=vector_manager)
rag.setup_qa_chain()

# Query
result = rag.query("What is the SOP for database backup?")
print(result['answer'])

# View source documents
for source in result['sources']:
    print(f"- {source['title']}: {source['content'][:100]}...")
```

## Configuration Details

### Core Configuration Items

| Configuration Item     | Description                                  | Default Value                               |
|------------------------|----------------------------------------------|---------------------------------------------|
| EMBEDDING_MODEL_NAME   | Name of the embedding model                  | `sentence-transformers/all-MiniLM-L6-v2`    |
| LLM_MODEL_NAME         | Name of the LLM model                        | `qwen3:4b`                                  |
| CHUNK_SIZE             | Size of document chunks                      | 500                                         |
| CHUNK_OVERLAP          | Overlap between chunks                       | 50                                          |
| RETRIEVAL_TOP_K        | Number of top-k results to retrieve          | 5                                           |
| TEMPERATURE            | LLM temperature parameter                    | 0.7                                         |

### Recommended Embedding Models

**Multilingual Models:**
- `sentence-transformers/all-MiniLM-L6-v2` (Recommended, lightweight)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### Recommended LLM Models

**Via Ollama:**
- `qwen3:4b` (Lightweight, strong performance)
- `qwen:14b` (More powerful, requires more resources)

## Frequently Asked Questions

### Q1: Ollama connection failed
**A**: Ensure the Ollama service is running.
```bash
# Start Ollama service
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### Q2: Slow embedding model download
**A**: Use a mirror source (e.g., for HuggingFace models).
```bash
# Set HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
```

### Q3: Out of memory
**A**: Adjust configurations.
```python
# Reduce chunk_size and retrieval_top_k
CHUNK_SIZE=300
RETRIEVAL_TOP_K=3

# Use a smaller model
LLM_MODEL_NAME=qwen2:1.8b
```

### Q4: Inaccurate query results
**A**: Optimization strategies:
1. Increase retrieval quantity (RETRIEVAL_TOP_K)
2. Adjust chunk size
3. Improve document quality
4. Use a more powerful model

## Advanced Features

### 1. Add API Service

Create `api.py`:
```python
from fastapi import FastAPI
from src import RAGSystem, VectorStoreManager

app = FastAPI()
vector_manager = VectorStoreManager()
vector_manager.load_vectorstore()
rag = RAGSystem(vector_store_manager=vector_manager)

@app.post("/query")
async def query(question: str):
    result = rag.query(question)
    return result

# Run: uvicorn api:app --reload
```

### 2. Regularly Update Knowledge Base

Create a scheduled task script `update_kb.py`:
```python
import schedule
import time
from main import RAGApplication

def update_job():
    app = RAGApplication()
    app.initialize_components()
    # You would need to provide file_paths here, e.g., by scanning a directory
    # For example: app.build_knowledge_base(file_paths=["data/raw/new_doc.txt"], force_rebuild=False)
    print("Knowledge base update job executed.")

# Update every day at 2 AM
schedule.every().day.at("02:00").do(update_job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 3. Batch Query

```python
questions = [
    "What are the standard procedures for server monitoring?",
    "What is the frequency of database backups?",
    "How to handle network failures?"
]

results = rag.batch_query(questions)
for result in results:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}\n")
```

## Performance Optimization

### 1. Use GPU Acceleration

Modify `src/vector_store.py`:
```python
embeddings = HuggingFaceEmbeddings(
    model_name=self.embedding_model_name,
    model_kwargs={'device': 'cuda'},  # Use GPU
    encode_kwargs={'normalize_embeddings': True}
)
```

### 2. Batch Processing

```python
# Batch add documents
vector_manager.add_documents(large_document_list)
```

### 3. Caching Mechanism

Consider adding Redis caching for common query results.

## Contribution Guide

Contributions, bug reports, and suggestions are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT License

## Contact

For questions or suggestions, please open an Issue or contact the maintainers.

---

**Happy RAG-ing!**