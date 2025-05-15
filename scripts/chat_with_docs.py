import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

CHROMA_DIR = "chroma_store"

def main():
    # Load vector store and retriever
    print("🔄 Loading Chroma vector store...")
    vectordb = Chroma(
        collection_name="insurance_chunks",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMA_DIR
    )
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Define custom prompt
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

    # Set up the LLM and QA chain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
    )

    print("🤖 Chatbot is ready. Ask a question or type 'exit' to quit.\n")

    while True:
        query = input("💬 Your question: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = qa_chain(query)
        print("\n🧠 Answer:")
        print(result["answer"])
        print("\n📄 Sources:")
        for source in result.get("sources", "").split(","):
            if source.strip():
                print("-", source.strip())
        print("\n" + "-"*40)

if __name__ == "__main__":
    main()
