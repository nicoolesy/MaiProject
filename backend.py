from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-8b")

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask(request: PromptRequest):
    prompt = request.prompt
    response = model.generate_content(prompt)
    return {"answer": response.text}
