import os
import json
import numpy as np
import pandas as pd
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import google.api_core.exceptions as google_exceptions
from dotenv import load_dotenv


load_dotenv()
# Access the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

with open(f"parenting_knowledge_base_20_clean_en.json", "r") as f:
    data = json.load(f)
    
persist_dir = "./chroma_parenting_db"

# Make sure the directory exists
os.makedirs(persist_dir, exist_ok=True)

# Set up persistent ChromaDB client
client = chromadb.PersistentClient(path=persist_dir)

# Create or get the collection
collection = client.get_or_create_collection(name="parenting_knowledge_db")

# Add data into the collection
for i, item in enumerate(data):
    collection.add(
        ids=[str(i)],
        documents=[item["content"]],
        metadatas=[{
            "title": item["title"],
            "tags": ", ".join(item["tags"])  # Optional: store as comma-separated string
        }]
    )

# print("📚 Parenting knowledge base has been loaded into ChromaDB.")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    return "positive" if score >= 0.4 else "negative" if score <= -0.4 else "neutral"
    
def search_parenting_knowledge(query: str, top_k: int = 3, age_group: str = None, category: str = None):
    conditions = []

    # Build filter conditions dynamically
    if age_group and age_group != "Not sure":
        conditions.append({"age_group": age_group})
    if category and category != "Not sure":
        conditions.append({"category": category})

    # Format `where` correctly for Chroma
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}
    else:
        where = None

    # Perform query
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where
    )
    return results

def list_available_categories():
    metadata = collection.get(include=["metadatas"])
    all_cats = set([meta["category"] for meta in metadata["metadatas"] if "category" in meta])
    return sorted(all_cats)

bloom_keywords = {
    "Remember": ["define", "list", "identify", "name", "recall"],
    "Understand": ["explain", "describe", "summarize", "clarify", "understand", "why"],
    "Apply": ["use", "help", "practice", "try", "implement"],
    "Analyze": ["compare", "analyze", "differentiate", "distinguish", "how vs"],
    "Evaluate": ["judge", "assess", "evaluate", "recommend", "is it good", "should I"],
    "Create": ["design", "plan", "build", "develop", "create"]
}

def classify_bloom_level(question):
    q = question.lower()
    for level, keywords in bloom_keywords.items():
        if any(kw in q for kw in keywords):
            return level
    return "Understand"  # fallback default

# def format_results(results):
#     formatted = []
#     documents = results["documents"][0]
#     metadatas = results["metadatas"][0]
#     distances = results.get("distances", [[None]])[0]

#     seen = set()
#     for i in range(len(documents)):
#         doc = documents[i]
#         meta = metadatas[i]
#         distance = distances[i]

#         if doc in seen:
#             continue
#         seen.add(doc)

#         formatted.append(
#             f"**{meta['title']}**\n"
#             f"_Tags: {meta['tags']}_\n"
#             f"Relevance Score: {round(1 - distance, 2)}\n\n"
#             f"{doc}"
#         )
#     return "\n\n---\n\n".join(formatted)

# def safe_generate(model, prompt, max_attempts=3):
#     """Retries Gemini API call with shorter prompt and timeout protection."""
#     prompt = prompt[:4000]  # limit length for safety
#     for attempt in range(max_attempts):
#         try:
#             response = model.generate_content(prompt)
#             return response
#         except google_exceptions.DeadlineExceeded:
#             print(f"⚠️ Timeout — retrying ({attempt+1}/{max_attempts})...")
#             time.sleep(2)
#         except Exception as e:
#             print(f"💥 Unexpected error: {e}")
#             break
#     return None

chat_history = []
USE_LOCAL_GEMINI = False
BACKEND_URL = "https://maiproject.onrender.com/ask"
def ask_parenting_assistant(user_question: str, age_group: str = None, category: str = None):
    global chat_history
    context = (
        search_parenting_knowledge(user_question, top_k=2, age_group=age_group, category=category)
        if not chat_history
        else chat_history[-1]["context"]
    )

    sentiment = get_sentiment(user_question)
    bloom = classify_bloom_level(user_question)

    tone = (
        "empathic" if sentiment == "negative"
        else "summary" if sentiment == "positive"
        else "coaching" if bloom == "Apply"
        else "educational"
    )

    instructions = {
        "educational": "Give 1–2 clear parenting tips.",
        "empathic": "Be kind and understanding.",
        "coaching": "Encourage with 2 practical steps.",
        "summary": "Summarize concisely."
    }.get(tone, "Be concise and supportive.")
    
    prompt = f"""
You are a warm, empathetic AI parenting coach.
Mode: {tone}
Context: {context}
Instruction: {instructions}
Question: {user_question}
""".strip()

    if USE_LOCAL_GEMINI:
        # Run directly
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        try:
            response = model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f"⚠️ Gemini error: {e}"
    else:
        # Send to backend
        try:
            res = requests.post(BACKEND_URL, json={"prompt": prompt})
            answer = res.json().get("answer", "⚠️ No backend response.")
        except Exception as e:
            answer = f"⚠️ Error contacting backend: {e}"

    chat_history.append({"question": user_question, "answer": answer, "context": context})
    return answer