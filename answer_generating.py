import os
# using ollama model gemma3 4b instead of openai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


load_dotenv()

persistent_directory = "db/chroma_db"

# simple warning message in case db not found
if not os.path.exists(persistent_directory):
    raise FileNotFoundError(
        f"Vector database not found at '{persistent_directory}'. Please run ingestion_pipeline.py first "
    )

# using huggingface embeddings instead of openai
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# search for relevant docs in chroma
query = "What features does the Ayushman App offer to beneficiaries, and in how many languages is it available?"

retriever = db.as_retriever(search_kwargs={"k": 5})


relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# display results now
print("               ")
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# combine query and document context
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# using chatollama model now
model = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

# define messages list
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# invoke ollama model with combined input
result = model.invoke(messages)

# display generated response
print("---------------------------------")
print("\n--- Final Response ---")
print("------------------------------------")
# print("Full result:")
# print(result)
print("Here is the content:")
print(result.content)