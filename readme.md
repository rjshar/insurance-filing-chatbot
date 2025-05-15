# 🧠 Insurance Filing Chatbot (RAG AI)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Michigan workers' compensation rate filings using GPT-3.5 and a local vector database.

Built with:
- Python + Streamlit
- LangChain + OpenAI
- ChromaDB for vector storage
- PyMuPDF for PDF parsing

## 🚀 Try It Live

> Coming soon: hosted version on Streamlit Cloud.

## 📦 Project Structure
rag_insurance_ai/
├── app.py               # Streamlit UI
├── scripts/
│   ├── embed_chunks.py  # Parses, chunks, and embeds filings
│   └── chat_with_docs.py# Terminal-based chatbot
├── pdfs/                # Source regulatory PDFs
├── chroma_store/        # Auto-generated vector DB (excluded from repo)
├── .env                 # Your OpenAI key (excluded from repo)
├── requirements.txt     # Dependencies
└── README.md

## 💬 How to Use

### 1. Clone the repo

    ```bash
    git clone https://github.com/YOUR_USERNAME/insurance-filing-chatbot.git
    cd insurance-filing-chatbot

### 2. Create your .env file
    OPENAI_API_KEY=sk-...
    
### 3. Create and activate a virtual environment
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    
### 4. Embed the filings
    Put your source PDFs into the /pdfs folder, then run:
    python scripts/embed_chunks.py

### 5. Launch the app
    streamlit run app.py
    
✨ Features
    •    Ask natural language questions like:
    •    “What is the loss cost multiplier for Travelers?”
    •    “How does Accident Fund justify its expense constant?”
    •    Cited responses with document source info
    •    Built for insurance professionals and filings nerds

📚 License

MIT – feel free to use and adapt.

