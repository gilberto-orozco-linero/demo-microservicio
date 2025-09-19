import httpx
from typing import Optional, List
import numpy as np
import os

# ------------------------------------------------------------------------------
# OpenAI / Azure OpenAI: client + helpers
# ------------------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
AZURE_OPENAI_KEY = str(os.getenv("AZURE_OPENAI_KEY"))
AZURE_EMBEDDINGS_ENDPOINT = str(os.getenv("AZURE_EMBEDDINGS_ENDPOINT"))

# Funciones auxiliares para caché semántico
async def get_embedding(text: str) -> Optional[List[float]]:
    """Genera embedding para el texto usando Azure OpenAI."""
    payload = {"input": text}
    headers = {
        "api-key": AZURE_OPENAI_KEY,
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(AZURE_EMBEDDINGS_ENDPOINT, json=payload, headers=headers)
            data = response.json()
            return data["data"][0]["embedding"] if "data" in data else None
        except Exception as e:
            print(f"Error generando embedding: {e}")
            return None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calcula similitud coseno entre dos vectores."""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))