import os
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()


#Loading the Documnets 
#----------------------------------------------------------------------------------
def load_documents(docs_path="internship_document"):
    print("------------------------------------------")
    print(f"Loading documents from {docs_path}...")
    print("----------------------------------------")
    
    pdf_file = os.path.join(docs_path, "Indian Health Transformation.pdf")
    
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(
            f"The PDF file '{pdf_file}' does not exist. Please make sure it is in your {docs_path} folder."
        )
    
    # load local pdf file and converting pages to text
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No pages could be loaded from {pdf_file}.")
        
    for i, doc in enumerate(documents): 
        print(f"\nDocument Page {i+1}:")
        print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"  Content length: {len(doc.page_content)} characters")
        preview = doc.page_content[:100].encode('ascii', errors='replace').decode('ascii')
        print(f"  Content preview: {preview}...")
        
    return documents

#-----------------------------------------------------------------------------------------------------------------


def split_documents(documents, embedding_model):
    print("---------------------------------------------")
    print("Splitting documents into chunks...")
    print("---------------------------------------------")

    # using semantic as instructed 
    text_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile"
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # printing first few chunks to see if it split correctly
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            word_count = len(chunk.page_content.split())
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
            print(f"Words: {word_count}")
            preview = chunk.page_content[:200].encode('ascii', errors='replace').decode('ascii')
            print(f"Preview: {preview}...")
            print("-" * 50)
            
    return chunks


#-----------------------------------------------------------------------------------------------------------------

# storing vectors in db... hopefully it works fast
def create_vector_store(chunks, embedding_model, persist_directory="db/chroma_db"):
    
    print("---------------------------------------------")
    print("Creating embeddings and storing in ChromaDB...")
    print("---------------------------------------------")

        
    # creating chroma database to store our chunks
    print("---------------------------------------------")
    print("--- Creating vector store ---")
    print("---------------------------------------------")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"} #Note: Cosine simlarity formula (A.B) / ||A|| x ||B||
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


#-----------------------------------------------------------------------------------------------------------------

     
def main():
    print("Main Function working ")
    
    # loading embedding model once to save memory
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # phase 1: loading the documnets
    documents = load_documents(docs_path="internship_document") # folder name is internship_document
    
    # phase 2: spliting pages into smaller chunks
    chunks = split_documents(documents, embedding_model)
    
    # phase 3: creating chroma vector db for search
    vectorstore = create_vector_store(chunks, embedding_model)
    
if __name__ == "__main__":
    main()



