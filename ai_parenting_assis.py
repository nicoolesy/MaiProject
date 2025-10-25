import os
import json
import numpy as np
import pandas as pd
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

with open("kaggle.json") as f:
    kaggle_credentials = json.load(f)

# Set environment variables
os.environ['KAGGLE_USERNAME'] = kaggle_credentials['username']
os.environ['KAGGLE_KEY'] = kaggle_credentials['key']

# Load variables from .env into environment
load_dotenv()

# Access the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-lite")

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
    filters = {}
    if age_group:
        filters["age_group"] = age_group
    if category:
        filters["category"] = category

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=filters if filters else None
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

chat_history = []
def ask_parenting_assistant(user_question: str, age_group: str = None, category: str = None):
    global chat_history

    # if history
    if not chat_history:
        results = search_parenting_knowledge(user_question)
        context = format_results(results)
    else:
        context = chat_history[-1]["context"]

    # using VADER sentiment analysis
    sentiment = get_sentiment(user_question)
    bloom_level = classify_bloom_level(user_question)

    # using VADER sentiment analysis logic
    if sentiment == "negative":
        mode = "empathic"
    elif sentiment == "positive":
        mode = "summary"
    elif bloom_level == "Apply":
        mode = "coaching"
    else:
        mode = "educational"
        
    print(f"🧠 Detected sentiment: {sentiment} → Auto-selected mode: {mode}")
    print(f"🧠 Bloom’s Taxonomy level detected: {bloom_level}")

    # select tone and style according to the prompt
    if mode == "educational":
        instructions = "Provide 2–3 concrete parenting activities parents can do related to this topic."
    elif mode == "empathic":
        instructions = (
            "Respond with empathy and encouragement, affirming the parent's concerns while giving constructive advice."
        )
    elif mode == "coaching":
        instructions = (
            "Respond with a warm and encouraging tone, affirming the parent's efforts "
            "while providing 2–3 specific strategies or steps they can try in everyday life. "
            "Avoid judgmental language. Use examples when possible."
        ) 
    else:
        instructions = "Summarize the relevant parenting knowledge to answer the question briefly and clearly."

    # Bloom’s Taxonomy logic
    if bloom_level == "Create":
        instructions += " Provide actionable ideas the parent can adapt or build upon."
    elif bloom_level == "Evaluate":
        instructions += " Offer a balanced perspective, discuss pros and cons, and help the parent reflect on the best course of action."
    elif bloom_level == "Analyze":
        instructions += " Break down the key factors, compare alternatives, and help the parent understand patterns or root causes."
    elif bloom_level == "Apply" and mode == "coaching":
        instructions += " Focus on helping the parent implement these ideas in real-world parenting situations."
    elif bloom_level == "Understand":
        instructions += " Focus on explaining why this behavior happens and what it means in terms of child development."
    elif bloom_level == "Remember":
        instructions += " Briefly list or define key concepts that address the parent's question."

    previous_rounds = "\n".join([
        f"User: {item['question']}\nAI: {item['answer']}"
        for item in chat_history[-2:]
    ])

    prompt = f"""You are an AI parenting assistant who speaks with the warmth, patience, and empathy of a kind and understanding mother.
Your tone should be calm, supportive, and nurturing—like a parent gently guiding another parent or child through a learning moment.

{previous_rounds}

Use the following parenting knowledge to guide your response.
If the documents are not sufficient, use your own knowledge to help the user in a helpful and thoughtful way. 
This question is at the '{bloom_level}' level of Bloom's Taxonomy. Adjust your response to match the user's cognitive intent.

{context}

{instructions}

Now answer this: {user_question}
"""
    response = model.generate_content(prompt)
    answer = response.text
    print(response.text)

    chat_history.append({
        "question": user_question,
        "answer": answer,
        "context": context
    })

    return answer