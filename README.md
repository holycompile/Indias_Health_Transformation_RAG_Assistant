# India's Health Transformation RAG Assistant

A Retrieval-Augmented Generation (RAG) system built in Python to ingest, chunk, index, and query information from the official Press Information Bureau (PIB) health transformation backgrounder document.

---

## 1. How we Ingested and Chunked the PIB Document
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
