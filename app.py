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
if "chat" not in st.session_state:
    model = genai.GenerativeModel("models/gemini-pro")
    st.session_state.chat = model.start_chat(history=[])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "age_group" not in st.session_state:
    st.session_state.age_group = "Not sure"
if "category" not in st.session_state:
    st.session_state.category = "Not sure"

# Streamlit UI
st.set_page_config(page_title="AI Parenting Assistant", page_icon="👨‍👩‍👧", layout="centered")
st.title("🤖 Gemini AI Parenting Assistant")
st.markdown("Ask me anything about parenting! 💬")

# Display chat history
for msg in st.session_state.chat.history:
    with st.chat_message("user" if msg.role == "user" else "assistant"):
        st.markdown(msg.parts[0].text)

# Collect user input
user_question = st.text_input("What's on your mind today about parenting?")

# Optional dropdowns
st.session_state.age_group = st.selectbox("Child's age group", ["0–3", "4–5", "6–12", "13–18", "Not sure"], index=4)
st.session_state.category = st.selectbox("Topic category", ["behavior", "emotions", "communication", "learning", "Not sure"], index=4)

if st.button("Send"):
    if user_question:
        with st.spinner("Thinking..."):
            answer = ask_parenting_assistant(
                user_question, 
                age_group=st.session_state.age_group, 
                category=st.session_state.category)
        st.markdown(f"📌 Current filters — Age: **{st.session_state.age_group}**, Topic: **{st.session_state.category}**")
        st.write(answer)
    else:
        st.warning("Please enter a question first!")
        
# User input
if prompt := st.chat_input("Ask me a parenting question..."):
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show Gemini response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # You can inject ChromaDB context or sentiment logic here
                response = ask_parenting_assistant(
                    user_question=prompt,
                    age_group=st.session_state.age_group,
                    category=st.session_state.category
            )
            except Exception as e:
                assistant_reply = "❌ Error: " + str(e)
        st.markdown(f"📌 Current filters — Age: **{st.session_state.age_group}**, Topic: **{st.session_state.category}**")
        st.markdown(f"{prompt}")
        st.write(f"💡 {response}")

    # Save assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": response})