# scripts/check_setup.py

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import chromadb

print("✅ PyMuPDF (fitz) is working")
print("✅ LangChain splitter and embeddings imported")
print("✅ ChromaDB available")

# Optional: confirm Chroma in-memory test
client = chromadb.Client()
collection = client.create_collection(name="test")
collection.add(documents=["Hello, world!"], ids=["1"])
results = collection.query(query_texts=["world"], n_results=1)
print("🔍 Vector search result:", results)

print("🚀 All core components are functional")
