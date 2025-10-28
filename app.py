import streamlit as st
import os
from dotenv import load_dotenv
from ai_parenting_assis import ask_parenting_assistant

# --- Setup ---
load_dotenv()
st.set_page_config(page_title="AI Parenting Assistant", page_icon="👨‍👩‍👧", layout="centered")

st.title("🤖 Gemini AI Parenting Assistant")
st.markdown("Ask me anything about parenting! 💬")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "age_group" not in st.session_state:
    st.session_state.age_group = "Not sure"

if "category" not in st.session_state:
    st.session_state.category = "Not sure"

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

# --- Display chat history ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask me a parenting question..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_parenting_assistant(
                user_question=prompt,
                age_group=st.session_state.age_group,
                category=st.session_state.category
            )

        st.markdown(
            f"📌 Filters — Age: **{st.session_state.age_group}**, Topic: **{st.session_state.category}**"
        )
        st.write(f"💡 {response}")

    st.session_state.chat_history.append({"role": "assistant", "content": response})