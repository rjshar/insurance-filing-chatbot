# scripts/parse_and_chunk.py

import os
import fitz  # still use this name
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_DIR = "pdfs"

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, filename):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text], metadatas=[{"source": filename}])
    return chunks

def main():
    all_chunks = []
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_DIR, filename)
            print(f"📄 Parsing: {filename}")
            text = extract_text_from_pdf(path)
            chunks = chunk_text(text, filename)
            print(f"✅ {len(chunks)} chunks created from {filename}")
            all_chunks.extend(chunks)

    # Show a sample
    print("\n🧠 Sample Chunk:")
    print(all_chunks[0].page_content[:500])  # Print first 500 characters of first chunk

if __name__ == "__main__":
    main()
