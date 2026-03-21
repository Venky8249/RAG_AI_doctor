import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Setup
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
DB_CHROMA_PATH = "vectorstore/db_chroma"

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    api_key=GROQ_API_KEY,
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embedding_model)

# RAG Pipeline
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)