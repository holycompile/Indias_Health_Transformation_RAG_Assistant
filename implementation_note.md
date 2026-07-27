# Implementation Note: RAG Q&A Assistant on India's Health Transformation

This note documents the design decisions, architectural choices, and lessons learned during the implementation of the RAG (Retrieval-Augmented Generation) system for the Press Information Bureau (PIB) health backgrounder.

---

## 1. Technical Choices & Rationales

### A. Ingestion and Document Parsing
* **Choice:** In-memory PDF parsing using `PyPDFLoader` (via `langchain-community`)
* **Rationale:**
  * **Simplicity & directness:** The PDF (`Indian Health Transformation.pdf`) is loaded directly into Python memory as page document objects. No intermediate or temporary text files are created on disk. This cleanly satisfies the assignment instruction to convert the PDF content to text in-memory.

### B. Embedding Model
* **Choice:** `sentence-transformers/all-MiniLM-L6-v2` (loaded via LangChain's `HuggingFaceEmbeddings`)
* **Rationale:**
  * **Efficiency & Local Execution:** It is a lightweight, local model (approx. 80MB) that runs entirely offline. This avoids dependency on paid third-party APIs and keeps data private.
  * **Relevance Performance:** It is optimized for mapping short sentences/paragraphs to a 384-dimensional vector space, capturing semantic relationships effectively. This is perfect for searching specific schemes and stats (e.g., matching "AB-PMJAY" with health insurance).

### C. Vector Storage & Index
* **Choice:** `Chroma` (via `langchain-chroma`)
* **Rationale:**
  * **Lightweight & Embedded:** Chroma runs directly inside the Python process, saving vector index files locally inside the `db/` directory. No external server is required, simplifying deployment.
  * **Distance Metric:** Cosine similarity (`{"hnsw:space": "cosine"}`) was configured to measure vector proximity. This normalizes document lengths and provides robust similarity scoring for matching the short user query vectors against paragraph vectors.

### D. Chunking Strategy
* **Choice:** `SemanticChunker` (using the `sentence-transformers/all-MiniLM-L6-v2` embedding model)
* **Rationale:**
  * **Semantic Splitting:** Unlike static character splitters, the `SemanticChunker` breaks text at points where embedding vector similarity drops significantly. We configured the breakpoint threshold using the `percentile` method.
  * **Sentence Integrity:** It groups related sentences together based on semantic meaning, ensuring that sentences about specific health initiatives (such as AB-PMJAY or ABDM) are kept in the same context block.
  * **Alignment with Guidelines:** It chunks the document semantically, resulting in chunks of various sizes that represent complete paragraphs and sections with clean semantic boundaries.

### E. LLM & Prompt Design
* **Choice:** `gemma3:4b` (via Ollama)
* **Rationale:**
  * **Zero-Hallucination Constraints:** The system prompt explicitly instructs the LLM to restrict its answers *strictly* to the provided context. If the answer cannot be found in the context, it must output a fallback statement: *"I don't have enough information to answer that question based on the provided documents."* This prevents the model from hallucinating or using its pre-trained general knowledge.
  * **Prompt Structure:** The prompt wraps retrieved chunks in a clear `Documents: ...` container and attaches the user's question, ensuring the LLM acts solely as a summarizer/reasoner over the provided context.

---

## 2. What I Had to Learn/Research to Complete This Assignment

1. **Local Embeddings Loading:** Researched how to instantiate Hugging Face sentence transformer models locally in LangChain without invoking third-party APIs, keeping the pipeline fully offline.
2. **Streamlit Lifecycle & State Management:** Learned how Streamlit re-runs the entire python file on user input and configured standard `st.session_state` to prevent conversational chat histories from clearing on each rerun.
3. **Windows Stdout Encoding:** Researched resolving standard output character map crashes in Windows command line terminals when printing previews of pages containing Unicode Hindi characters.

---

## 3. Limitations & What I'd Improve with 2 More Days

1. **Hybrid Search:** Adding a lexical/BM25 keyword search index in combination with the vector search (Hybrid Search/RRF) would improve the accuracy of lookup for exact statistics and acronyms.
2. **Metadata Filtering:** Filtering retrieved documents dynamically by specific headers/pillars rather than a flat search would prevent unrelated pillars from bleeding into the LLM context.
3. **Chunk Reranking:** Integrating a cross-encoder reranker (like `cohere-rerank` or a local `bge-reranker`) would sort the top-5 retrieved chunks more accurately before feeding them to the LLM.
