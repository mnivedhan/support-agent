"""Gemini through LangChain. Needs GEMINI_API_KEY, from the environment or a git-ignored .env file in the repo root.

Get a key at https://aistudio.google.com/apikey and put `GEMINI_API_KEY=...` in store-support-agents/.env.
Pick the model with SSA_GEMINI_MODEL; list the ones your key can use with `python -m ssa.gemini`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "gemini-2.5-flash"


def chat_model():
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("No Gemini key. Put GEMINI_API_KEY=... in store-support-agents/.env "
                           "(get one at https://aistudio.google.com/apikey).")
    return ChatGoogleGenerativeAI(model=os.environ.get("SSA_GEMINI_MODEL", DEFAULT_MODEL), api_key=key,
                                  max_retries=3)


if __name__ == "__main__":
    from google import genai

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            print(m.name.removeprefix("models/"))
