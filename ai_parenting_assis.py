import os
import json
import requests
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import chromadb
chromadb.telemetry.posthog.capture = lambda *a, **k: None

load_dotenv()
# Access the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

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
if collection.count() == 0:
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

chat_history = []
USE_LOCAL_GEMINI = False
BACKEND_URL = "https://maiproject.onrender.com/ask"
def ask_parenting_assistant(user_question: str, age_group: str = None, category: str = None):
    global chat_history
    context = (
        search_parenting_knowledge(user_question, top_k=3, age_group=age_group, category=category)
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
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
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