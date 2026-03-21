from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/"
DB_CHROMA_PATH = "vectorstore/db_chroma"

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)

# Initialize embeddings and Chroma
documents = load_pdf_files(DATA_PATH)
text_chunks = create_chunks(documents)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create and persist the database
db = Chroma.from_documents(
    documents=text_chunks, 
    embedding=embedding_model, 
    persist_directory=DB_CHROMA_PATH
)

print(f"Success: Vector database saved at {DB_CHROMA_PATH}")