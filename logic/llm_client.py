"""
llm_client.py
Single, centralised LLM client using the Gemini API.
Loads API key from .env — no key is ever hardcoded.
No Streamlit imports.
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

_MODEL_NAME = "gemini-3.5-flash"
_client_initialised = False


def _init():
    global _client_initialised
    if not _client_initialised:
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file: GEMINI_API_KEY=your_key_here"
            )
        genai.configure(api_key=key)
        _client_initialised = True


def ask(prompt: str, temperature: float = 0.3) -> str:
    """Send a prompt to the LLM and return the raw text response."""
    _init()
    model = genai.GenerativeModel(
        _MODEL_NAME,
        generation_config={"temperature": temperature}
    )
    response = model.generate_content(prompt)
    return response.text.strip()
