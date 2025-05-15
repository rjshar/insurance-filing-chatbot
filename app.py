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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

    file_count = 0

    for filename in os.listdir("pdfs"):
        if filename.endswith(".pdf") and file_count < 1:  # Limit to 1 file
            file_count += 1
            path = os.path.join("pdfs", filename)
            with fitz.open(path) as doc:
                full_text = "".join(page.get_text() for page in doc)
                chunks = splitter.create_documents([full_text], metadatas=[{"source": filename}])
                
                # Filter out junk like Table of Contents or empty text
                filtered_chunks = [
                    c for c in chunks
                    if len(c.page_content.strip()) > 100
                    and "table of contents" not in c.page_content.lower()
                    and "index" not in c.page_content.lower()
                    and not c.page_content.strip().isdigit()
                ]

                
                docs.extend(chunks[:50])  # Limit to first 50 chunks

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Stable model
    return FAISS.from_documents(docs, embeddings)

question = st.text_input("Ask a question:", placeholder="e.g. What is the expense constant for Citizens?")

if question:
    with st.spinner("Thinking..."):
        vectordb = load_vectorstore()
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

        custom_prompt = PromptTemplate.from_template(
            """You are an expert on U.S. workers' compensation insurance filings.
Answer the question using only the provided context from official filings. Think about comparisons between the company manuals you have in your database.
If the answer is not explicitly stated in the context, say: "I don’t know based on the filings provided."

Context:
{summaries}

Question:
{question}

Answer in plain English, as if you were explaining it to an underwriter:"""
)

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": custom_prompt},
        )
    # Preview what context is being retrieved
        docs = retriever.get_relevant_documents(question)

        st.subheader("🔍 Retrieved Snippets")
        for i, doc in enumerate(docs):
            st.markdown(f"**Snippet {i+1}** — *{doc.metadata.get('source', 'unknown')}*")
            st.markdown(f"> {doc.page_content[:500]}...")
            st.markdown("---")
            
        result = qa_chain(question)  # If using RetrievalQAWithSourcesChain, this may still be `result = qa_chain(question)`

        st.subheader("🧠 Answer")
        st.write(result["answer"])

        st.subheader("📄 Sources")
        for source in result.get("sources", "").split(","):
            if source.strip():
                st.write(f"- {source.strip()}")
