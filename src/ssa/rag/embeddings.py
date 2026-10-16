"""Text embeddings from a local Ollama model (free, no key). Start the server with `ollama serve`.

nomic-embed-text is trained with task prefixes: documents and queries must be embedded with different prefixes,
or similarity scores get noticeably worse. That's an easy thing to miss when swapping embedding models.
"""

import json
import urllib.request

import numpy as np

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"


def _embed(texts: list[str]) -> np.ndarray:
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    request = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        vectors = np.array(json.load(response)["embeddings"], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # unit length, so dot product = cosine


def embed_documents(texts: list[str]) -> np.ndarray:
    return _embed([f"search_document: {t}" for t in texts])


def embed_query(text: str) -> np.ndarray:
    return _embed([f"search_query: {text}"])[0]
