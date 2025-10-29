from fastapi import FastAPI, Request
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-lite")

app = FastAPI()

@app.post("/ask")
async def ask(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            return {"answer": "⚠️ No prompt provided."}

        response = model.generate_content(prompt)
        return {"answer": response.text or "⚠️ Empty Gemini response."}

    except Exception as e:
        # Return JSON error instead of plain text
        return {"answer": f"⚠️ Backend error: {str(e)}"}
