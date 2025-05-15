import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

load_dotenv()
CHROMA_DIR = "chroma_store"

st.set_page_config(page_title="Insurance Filing Q&A", layout="wide")
st.title("📄 Insurance Filing Chatbot")

question = st.text_input("Ask a question about the filings:", placeholder="e.g. What is the expense constant for Travelers?")

if question:
    with st.spinner("Thinking..."):
        vectordb = Chroma(
            collection_name="insurance_chunks",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=CHROMA_DIR
        )

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
