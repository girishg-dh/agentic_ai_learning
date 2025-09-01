import os
from helper import check_and_download_file, check_qdrant_status
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# ---- pre-set-up -----
url = "https://arxiv.org/pdf/1706.03762"
file_path = "../data/Attention_is_all_you_need.pdf"
if check_and_download_file(file_path, url):
    print("File exists or has been downloaded successfully.")
else:
    print("File not found and download failed.")
    exit()
if check_qdrant_status():
    print("Qdrant is running.")
else:
    print("Qdrant is not running.")
    exit()

# ---- 1. Load the data from pdf -----
loader = PyPDFLoader(file_path)
documents = loader.load()
print(f"Loaded {len(documents)} page(s) from the PDF.)")

# ---- 2. Split the documents into chunks -----
print("Loading embedding model...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# ---- 3. Create the vector store ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---- 4. Store chunks into the Qdrant  ----
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant = QdrantVectorStore.from_documents(
    documents=docs, 
    embedding=embeddings, 
    url=qdrant_url, 
    collection_name="research_assistant"
)
print("--- Ingestion complete ---")