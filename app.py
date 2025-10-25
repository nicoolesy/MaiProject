import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
# Load your AI assistant logic
from ai_parenting_assis import ask_parenting_assistant

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize model
# model = genai.GenerativeModel("models/gemini-pro")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "age_group" not in st.session_state:
    st.session_state.age_group = "Not sure"

if "category" not in st.session_state:
    st.session_state.category = "Not sure"

# --- Page config ---
st.set_page_config(page_title="AI Parenting Assistant", page_icon="👨‍👩‍👧", layout="centered")
st.title("🤖 Gemini AI Parenting Assistant")
st.markdown("Ask me anything about parenting! 💬")

# --- Display history ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Dropdowns ---
st.session_state.age_group = st.selectbox(
    "Child's age group",
    ["0–3", "4–5", "6–12", "13–18", "Not sure"],
    index=4
)
st.session_state.category = st.selectbox(
    "Topic category",
    ["behavior", "emotions", "communication", "learning", "Not sure"],
    index=4
)

# --- Chat input ---
if prompt := st.chat_input("Ask me a parenting question..."):
    # User message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ask_parenting_assistant(
                    user_question=prompt,
                    age_group=st.session_state.age_group,
                    category=st.session_state.category
                )
            except Exception as e:
                response = f"❌ Error: {e}"

        st.markdown(f"📌 Filters — Age: **{st.session_state.age_group}**, Topic: **{st.session_state.category}**")
        st.write(f"💡 {response}")

    # Save assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": response})