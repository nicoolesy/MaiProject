import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
# from langchain_text_splitters import CharacterTextSplitter
# from langchain.document_loaders import TextLoader
from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
import google.generativeai as genai
# from IPython.display import HTML, Markdown, display
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import google.api_core.exceptions as google_exceptions
# from fastapi import FastAPI

# app = FastAPI()


load_dotenv()
# Access the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-8b")

with open(f"parenting_knowledge_base_20_clean_en.json", "r") as f:
    data = json.load(f)
    
persist_dir = "./chroma_parenting_db"

# Make sure the directory exists
os.makedirs(persist_dir, exist_ok=True)

# Set up persistent ChromaDB client
client = chromadb.PersistentClient(path=persist_dir)

# Create or get the collection
collection = client.get_or_create_collection(name="parenting_knowledge_db")

# Sample data format
# data = [{"title": "Tip 1", "content": "Parenting tip content", "tags": ["empathy", "toddlers"]}, ...]

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
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.4:
        return "positive"
    elif compound <= -0.4:
        return "negative"
    else:
        return "neutral"
    
def search_parenting_knowledge(query: str, top_k: int = 3, age_group: str = None, category: str = None):
    conditions = []
    if age_group:
        conditions.append({"age_group": age_group})
    if category:
        conditions.append({"category": category})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}
    else:
        where = None

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

def format_results(results):
    formatted = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[None]])[0]

    seen = set()
    for i in range(len(documents)):
        doc = documents[i]
        meta = metadatas[i]
        distance = distances[i]

        if doc in seen:
            continue
        seen.add(doc)

        formatted.append(
            f"**{meta['title']}**\n"
            f"_Tags: {meta['tags']}_\n"
            f"Relevance Score: {round(1 - distance, 2)}\n\n"
            f"{doc}"
        )
    return "\n\n---\n\n".join(formatted)

def safe_generate(model, prompt, max_attempts=3):
    """Retries Gemini API call with shorter prompt and timeout protection."""
    prompt = prompt[:4000]  # limit length for safety
    for attempt in range(max_attempts):
        try:
            response = model.generate_content(prompt)
            return response
        except google_exceptions.DeadlineExceeded:
            print(f"⚠️ Timeout — retrying ({attempt+1}/{max_attempts})...")
            time.sleep(2)
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            break
    return None

chat_history = []
def ask_parenting_assistant(user_question: str, age_group: str = None, category: str = None):
    global chat_history

    # Retrieve context (cached)
    if not chat_history:
        results = search_parenting_knowledge(user_question, top_k=2, age_group=age_group, category=category)
        context = format_results(results)
        if len(context) > 400:
            context = context[:400] + "..."
    else:
        context = chat_history[-1]["context"]

    sentiment = get_sentiment(user_question)
    bloom_level = classify_bloom_level(user_question)

    tone_instructions = {
        "educational": "Give 1–2 clear parenting tips.",
        "empathic": "Be kind and understanding. Offer 1 piece of advice.",
        "coaching": "Encourage with 2 practical steps.",
        "summary": "Summarize clearly and concisely."
    }
    mode = (
        "empathic" if sentiment == "negative"
        else "summary" if sentiment == "positive"
        else "coaching" if bloom_level == "Apply"
        else "educational"
    )

    instructions = tone_instructions.get(mode, "Give a short, supportive answer.")

    bloom_instructions = {
        "Create": "Offer creative ideas parents can try.",
        "Evaluate": "Give a short, balanced view.",
        "Analyze": "Explain why this might happen.",
        "Apply": "Suggest 1–2 small steps.",
        "Understand": "Explain simply why this occurs.",
        "Remember": "List 1–2 key points."
    }
    instructions += " " + bloom_instructions.get(bloom_level, "")

    previous_rounds = ""
    if chat_history:
        last_turn = chat_history[-1]
        previous_rounds = f"User: {last_turn['question']}\nAI: {last_turn['answer']}"

    prompt = f"""
You are a warm, empathetic AI parenting coach.
Use calm, concise language.
If context is not enough, answer based on general parenting knowledge.

Previous chat:
{previous_rounds}

Context:
{context}

Instruction: {instructions}
Question: {user_question}
""".strip()

    response = None
    try:
        with model.generate_content(prompt, stream=True) as stream:
            partial_text = ""
            for chunk in stream:
                if chunk.text:
                    partial_text += chunk.text
                    st.write(partial_text)  # shows incremental output in Streamlit
        answer = partial_text.strip()
    except Exception as e:
        answer = "⚠️ Gemini took too long. Please try again shortly."

    chat_history.append({"question": user_question, "answer": answer, "context": context})
    return answer