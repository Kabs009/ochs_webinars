from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from preprocess import chunks  # Import from preprocessing step

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create ChromaDB
vector_db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="chroma_db"
)

# Save the database
vector_db.persist()
