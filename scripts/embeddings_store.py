from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Load a Free Embedding Model (MiniLM is small & efficient)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to Generate and Store Embeddings
def generate_embeddings():
    # Load Documents (Example: Replace with actual text file loading)
    documents = [
        Document(page_content="This is a test document."),
        Document(page_content="We are generating embeddings for this text.")
    ]

    # Store Embeddings in ChromaDB
    persist_directory = "db"  # Directory to save embeddings
    vector_db = Chroma.from_documents(documents, embedding_model, persist_directory=persist_directory)

    print("✅ Embeddings stored successfully in ChromaDB!")

if __name__ == "__main__":
    generate_embeddings()
