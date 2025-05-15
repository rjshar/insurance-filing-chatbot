# scripts/embed_chunks.py

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from tqdm import tqdm
import fitz  # PyMuPDF

# Load OpenAI API key from .env
load_dotenv()

PDF_DIR = "pdfs"
CHROMA_DIR = "chroma_store"

def extract_chunks_from_pdf(filepath):
    doc = fitz.open(filepath)
    text = "".join(page.get_text() for page in doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text], metadatas=[{"source": os.path.basename(filepath)}])

def main():
    all_chunks = []

    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            print(f"📄 Processing: {filename}")
            path = os.path.join(PDF_DIR, filename)
            chunks = extract_chunks_from_pdf(path)
            all_chunks.extend(chunks)

    print(f"\n📚 Total chunks: {len(all_chunks)}")

    embeddings = OpenAIEmbeddings()
    print("🔄 Generating and storing embeddings in Chroma...")

    # Create empty Chroma DB
    vectordb = Chroma(
        collection_name="insurance_chunks",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    batch_size = 100

    for i in tqdm(range(0, len(all_chunks), batch_size)):
        batch = all_chunks[i:i+batch_size]
        vectordb.add_documents(batch)

    print(f"✅ Chroma DB created at: {CHROMA_DIR}")

if __name__ == "__main__":
    main()
