# app.py — Streamlit Cloud version (no Chroma, in-memory FAISS)

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import fitz  # PyMuPDF

load_dotenv()

st.set_page_config(page_title="Insurance Filing Chatbot", layout="wide")
st.title("📄 Insurance Filing Chatbot (Demo Mode)")

@st.cache_resource
def load_vectorstore():
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    file_count = 0

    for filename in os.listdir("pdfs"):
        if filename.endswith(".pdf") and file_count < 1:  # Limit to 1 file
            file_count += 1
            path = os.path.join("pdfs", filename)
            with fitz.open(path) as doc:
                full_text = "".join(page.get_text() for page in doc)
                chunks = splitter.create_documents([full_text], metadatas=[{"source": filename}])
                docs.extend(chunks[:50])  # Limit to first 50 chunks

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Stable model
    return FAISS.from_documents(docs, embeddings)

question = st.text_input("Ask a question:", placeholder="e.g. What is the expense constant for Citizens?")

if question:
    with st.spinner("Thinking..."):
        vectordb = load_vectorstore()
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        custom_prompt = PromptTemplate.from_template(
            """You are a helpful assistant trained to answer questions about insurance regulatory filings.
Use ONLY the provided context to answer the question.
If you don't know, say "I don’t know based on the filings provided."

Context:
{summaries}

Question:
{question}

Answer in plain English:"""
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": custom_prompt},
        )

        result = qa_chain(question)

        st.subheader("🧠 Answer")
        st.write(result["answer"])

        st.subheader("📄 Sources")
        for source in result.get("sources", "").split(","):
            if source.strip():
                st.write(f"- {source.strip()}")
