import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

DB_CHROMA_PATH = "vectorstore/db_chroma"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embedding_model)

def main():
    st.title("MedVision AI Doctor")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Ask a medical question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
                
        try: 
            vectorstore = get_vectorstore()
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.5,
                api_key=st.secrets["GROQ_API_KEY"]
            )
            
            combine_docs_chain = create_stuff_documents_chain(llm, hub.pull("langchain-ai/retrieval-qa-chat"))
            rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

            with st.spinner("Analyzing..."):
                response = rag_chain.invoke({'input': prompt})
                result = response["answer"]
            
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()