# scripts/test_openai.py

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

try:
    response = client.embeddings.create(
        input=["test input for embedding"],
        model="text-embedding-3-small"  # cheapest and current model
    )
    print("✅ API key is valid.")
    print("🧠 Embedding response:", response.data[0].embedding[:5], "...")  # show first 5 numbers
except Exception as e:
    print("❌ Error:", e)
