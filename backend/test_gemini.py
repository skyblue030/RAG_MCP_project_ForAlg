import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY_LLM")

if not API_KEY:
    print("錯誤：請設定 GOOGLE_API_KEY_LLM 環境變數。")
    exit()

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content("What is an algorithm?")
    print(response.text)
except Exception as e:
    print(f"發生錯誤: {e}")

