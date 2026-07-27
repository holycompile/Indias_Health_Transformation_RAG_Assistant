import os
from langchain_community.embeddings import HuggingFaceEmbeddings  # using this for now as its free  
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

persistent_directory = "db/chroma_db"

#a small warning message 
if not os.path.exists(persistent_directory):
    raise FileNotFoundError(
        f"Vector database not found at '{persistent_directory}'. Please run ingestion_pipeline.py first "
    )

#using the HuggingFaceEmbeddings 
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}   # Recall!! Using the Cosine similarity 
)

# Search for relevant documents
query = "Tell about Viksit Bharat 2047"

retriever = db.as_retriever(search_kwargs={"k": 5}) #showing the top-5 chunks with higesh similarity



relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("------------------------")
print("--- Context ---")
print("------------------------")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")




