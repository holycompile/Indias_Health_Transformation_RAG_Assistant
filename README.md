# India's Health Transformation RAG Assistant

A Retrieval-Augmented Generation (RAG) system built in Python to ingest, chunk, index, and query information from the official Press Information Bureau (PIB) health transformation backgrounder document.

---

## 📁 Project Directory Structure

```text
├── db/                                  # Local vector database storage
│   └── chroma_db/                       # Chroma DB persistent store files (SQLite + index vectors)
├── internship_document/                 # Folder containing source documents for ingestion
│   └── Indian Health Transformation.pdf # Official Press Information Bureau (PIB) health PDF
├── answer_generating.py                 # Script demonstrating end-to-end RAG question answering using Ollama
├── app.py                               # Streamlit Web UI (Basic Single-Turn & Advanced Conversational modes)
├── history_retrieval_convo_skill.py     # Core conversation engine (history tracking, query rewriting, and LLM prompting)
├── ingestion_pipeline.py                # Document ingestion pipeline (loads, semantically chunks, and embeds the PDF)
├── requirements.txt                     # Project python dependencies
├── retrieval_pipeline.py                # Script demonstrating semantic search retrieval from Chroma DB
├── runtime.txt                          # Target Python version configuration
└── README.md                            # Project documentation (this file)
```

---

## 📊 System Architecture & RAG Workflow

The diagram below outlines the local semantic ingestion pipeline, multi-mode Streamlit UI, vector retrieval process, history-aware query rewriting, and LLM synthesis.

```mermaid
flowchart TD
    %% Styling
    classDef ingestion fill:#e1f5fe,stroke:#039be5,stroke-width:2px,color:#01579b;
    classDef storage fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#1b5e20;
    classDef retrieval fill:#fff3e0,stroke:#fb8c00,stroke-width:2px,color:#e65100;
    classDef llm fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#4a148c;
    classDef app fill:#eceff1,stroke:#546e7a,stroke-width:2px,color:#263238;

    subgraph INGESTION["1. Ingestion & Indexing Pipeline (ingestion_pipeline.py)"]
        PDF["📄 Indian Health Transformation.pdf<br>(internship_document/)"] -->|Load PDF| Loader["⚙️ PyPDFLoader<br>(LangChain)"]
        Loader -->|Document Pages| Chunker["✂️ SemanticChunker<br>(Percentile Threshold)"]
        EmbedModel["🧠 sentence-transformers/<br>all-MiniLM-L6-v2"] <-->|Determine Semantic Boundaries| Chunker
        Chunker -->|Cohesive Text Chunks| Embedding["🧮 HuggingFaceEmbeddings<br>(Generate 384D Vectors)"]
        Embedding -->|Vector Embeddings & Chunks| DBStore["💾 Chroma DB<br>(db/chroma_db/)"]
    end

    subgraph RETRIEVAL["2. Search & Retrieval Pipeline (retrieval_pipeline.py)"]
        DBStore <-->|Cosine Similarity Match| ChromaDB["🗄️ Chroma Vector DB<br>(hnsw:space: cosine)"]
    end

    subgraph APP ["3. Streamlit Interface (app.py)"]
        UserInput ["👤 User Input Query"] --> UI_Mode{Select UI Mode}
        UI_Mode --> |Basic Single-Turn| BasicFlow["⚡ Direct Query<br>(History cleared, K=5)"]
        UI_Mode --> |Advanced Conversational| AdvFlow["💬 Conversational Query<br>(K=3)"]
    end

    subgraph CORE["4. Backend Conversation Engine (history_retrieval_convo_skill.py)"]
        AdvFlow --> HistoryCheck{Chat History<br>Exists?}
        HistoryCheck -->|Yes| Rewrite["📝 Standalone Question Rewrite<br>(gemma3:4b via Ollama)"]
        HistoryCheck -->|No| Standalone["🔍 Standalone Query = User Query"]
        Rewrite --> Standalone
        BasicFlow --> Standalone
        
        Standalone -->|Embed Query| QueryEmbed["🧮 all-MiniLM-L6-v2 Embeddings"]
        QueryEmbed -->|Query Vector| ChromaDB
        ChromaDB -->|Top-K Context Chunks| ContextAssemble["🔗 Combine Context + System Instructions"]
        
        ContextAssemble --> Prompt["📝 Structured RAG Prompt"]
        Prompt --> LLM["🤖 ChatOllama<br>(gemma3:4b Model)"]
        LLM -->|Synthesized Response| OutResponse["✨ Grounded Answer"]
        OutResponse -->|Update History| HistoryList["📚 Chat History Session State"]
        HistoryList -.-> Rewrite
    end

    OutResponse -->|Display| AppDisplay["💻 Render in Streamlit UI"]

    class PDF,Loader,Chunker ingestion;
    class DBStore,ChromaDB storage;
    class UserInput,UI_Mode,BasicFlow,AdvFlow,AppDisplay app;
    class Standalone,QueryEmbed,ContextAssemble,Prompt retrieval;
    class LLM,Rewrite,HistoryCheck,HistoryList,OutResponse llm;
```

---


## 1. Ingested and Chunked the PIB Document
* **Ingestion:** The source PDF document (`Indian Health Transformation.pdf`) is loaded directly into Python memory using LangChain's `PyPDFLoader`. This parses each page of the document and extracts the text in-memory.
* **Semantic Chunking:** Rather than using basic character limits, the document is split using LangChain's experimental `SemanticChunker`. 
  * The chunker uses local HuggingFace embeddings to calculate the distance between consecutive sentences.
  * If the semantic difference between two sentences exceeds a calculated threshold (we configured the `breakpoint_threshold_type="percentile"`), a split is created. This ensures sentences are grouped into semantically cohesive sections with clear meaning.

---

## 2. How and Where Embeddings are Stored
* **Embedding Model:** We use `sentence-transformers/all-MiniLM-L6-v2` via LangChain's `HuggingFaceEmbeddings` class. This model maps chunks into a 384-dimensional dense vector space.
* **Vector Database:** The embeddings and associated document chunks are stored locally on disk using **Chroma DB**.
* **Storage Location:** All database files are persisted in the directory:
  ```
  db/chroma_db/
  ```
* **Distance Metric:** The database is configured to use **Cosine Similarity** (`"hnsw:space": "cosine"`) to find the closest vector matches.

---

## 3. How Semantic Search + RAG Works End-to-End
The pipeline processes user queries in a 3-step sequence:
1. **Query Embedding:** The user's input question is passed to the same `all-MiniLM-L6-v2` model to generate its 384-dimensional search vector.
2. **Semantic Search Lookup:** Chroma DB calculates the cosine similarity between the query vector and all chunk vectors in the database, returning the **top-5 most similar text chunks** as context.
3. **LLM Synthesis (RAG):** 
   * The retrieved context chunks are combined with the user query into a structured system prompt.
   * This prompt is sent to the local `gemma` model running on **Ollama**.
   * The LLM synthesizes a clean, natural response grounded strictly in the retrieved snippets, preventing hallucinations.

---

## 4. Setup and Run Instructions

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) (running locally)

### Setup Steps

1. **Activate Virtual Environment:**
   Inside the project directory:
   ```powershell
   # Windows PowerShell:
   .\venv\Scripts\activate
   ```

2. **Pull LLM model:**
   Ensure Ollama has the gemma model pulled:
   ```bash
   ollama pull gemma:2b
   ```
   *(Note: The code is configured to use `gemma3:4b`. If you have a different local model version installed, update the model parameter in `answer_generating.py` and `history_retrieval_convo_skill.py` to match your model name).*

3. **Run Ingestion Pipeline:**
   Build the semantic vector database from the PDF:
   ```bash
   python ingestion_pipeline.py
   ```

4. **Launch Streamlit Web UI:**
   Start the browser interface:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.
   Deployment might take some time please be patient while it loads. 

---

## 5. Short Implementation Note Summary

### Technical Choices
* **Embedding Model:** Selected `sentence-transformers/all-MiniLM-L6-v2` because it is lightweight (80MB), fast, runs completely locally (offline), and captures strong semantic meaning in short sentence groups.
* **Storage/Index:** **Chroma DB** is used as our vector database because it is an embedded system (runs directly in-process without requiring a separate server database) and easily persists embedding indexes locally in the project folder using Cosine Similarity.
* **LLM & Prompt Design:** Instantiated `gemma` locally via Ollama. The prompt is designed to wrap the retrieved source document segments in context boundaries and strictly instruct the model to respond using *only* this context, raising a clean warning if no matching information is found (avoiding hallucinated claims).

### Learning Journey & Acknowledgements
Honestly, I was familiar with the basic concepts of Python, machine learning, embeddings, and LLMs, but I had never built a complete Retrieval-Augmented Generation (RAG) pipeline from scratch.

Over the past few days, I spent considerable time understanding how each component works rather than simply making the project run. While implementing the project, I learned about concepts such as document ingestion, semantic chunking, vector databases (ChromaDB), embeddings, retrieval pipelines, prompt engineering, conversational history, and integrating a local LLM using Ollama.

Whenever I got stuck on implementation details or unfamiliar concepts, I used LLMs to help me understand the underlying ideas, debug issues, and learn how different components interact. 

Reference Repository:
https://github.com/harishneel1/rag-for-beginners

### Limitations & 2-Day Future Improvements
- Hybrid Retrieval (Vector Search + BM25) to improve retrieval of exact numerical values and keyword-heavy queries.
- Re-ranking retrieved documents using a cross-encoder before passing them to the language model.
- Support for multiple uploaded documents and larger knowledge bases.
- Better conversation memory and context management.
- Deployment using a cloud-hosted inference pipeline instead of relying on a local model.
