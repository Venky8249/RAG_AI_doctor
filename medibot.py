import os
os.environ["LANGCHAIN_PYDANTIC_V1"] = "1"   # 🔥 MUST BE FIRST LINE

import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()


DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db


def main():
    st.title("Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    user_prompt = st.chat_input("Pass your prompt here")

    if user_prompt:
        st.chat_message('user').markdown(user_prompt)
        st.session_state.messages.append({'role': 'user', 'content': user_prompt})

        try:
            vectorstore = get_vectorstore()

            if vectorstore is None:
                st.error("Vectorstore not loaded. Check if db_faiss exists.")
                return

            GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

            if not GROQ_API_KEY:
                st.error("Missing GROQ_API_KEY in secrets.")
                return

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.5,
                max_tokens=512,
                api_key=GROQ_API_KEY,
            )

            # ✅ Correct prompt for LC 0.2.x
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are a helpful medical assistant.

                Use the context below to answer the question.

                Context:
                {context}

                Question:
                {question}

                Answer:
                """
            )

            combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)

            rag_chain = create_retrieval_chain(
                vectorstore.as_retriever(search_kwargs={'k': 3}),
                combine_docs_chain
            )

            response = rag_chain.invoke({"question": user_prompt})

            # 🔍 Debug (remove later)
            st.write("DEBUG:", response)

            result = response.get("answer", "")

            if not result or not result.strip():
                result = "No relevant answer found."

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()