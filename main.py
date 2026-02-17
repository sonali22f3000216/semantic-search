from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import time
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load embedding model (local)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents (replace with your 84 docs if needed)
documents = [
    {"id": 0, "content": "How to authenticate using API keys", "metadata": {"source": "auth.md"}},
    {"id": 1, "content": "Installation guide for SDK", "metadata": {"source": "install.md"}},
    {"id": 2, "content": "How to reset password securely", "metadata": {"source": "security.md"}},
    {"id": 3, "content": "Authentication using OAuth tokens", "metadata": {"source": "oauth.md"}},
    {"id": 4, "content": "Deploying application to production", "metadata": {"source": "deploy.md"}},
]

# Compute embeddings once (cache)
doc_texts = [doc["content"] for doc in documents]
doc_embeddings = model.encode(doc_texts)

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# Better Re-ranking function
def rerank_results(query, candidates):
    reranked = []

    query_words = set(query.lower().split())

    for item in candidates:
        doc_words = set(item["content"].lower().split())

        overlap = len(query_words.intersection(doc_words))
        score = overlap / len(query_words) if len(query_words) > 0 else 0

        reranked.append({
            **item,
            "score": score
        })

    reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)

    return reranked

# Request model
class SearchRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True
    rerankK: int = 3

@app.post("/search")
def search(request: SearchRequest):
    start_time = time.time()

    # Embed query
    query_embedding = model.encode([request.query])[0]

    # Initial Retrieval
    scores = []
    for idx, doc_embedding in enumerate(doc_embeddings):
        #score = cosine_similarity(query_embedding, doc_embedding)
        raw_score = cosine_similarity(query_embedding, doc_embedding)
        score = (raw_score + 1) / 2

        scores.append((idx, score))

    # Sort by similarity
    scores.sort(key=lambda x: x[1], reverse=True)

    top_k = scores[:request.k]

    results = []
    for idx, score in top_k:
        results.append({
            "id": documents[idx]["id"],
            "score": float(score),
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        })

    # Re-ranking (simple re-score boost if query word appears)
    # Re-ranking using improved logic
    if request.rerank:
        results = rerank_results(request.query, results)
        results = results[:request.rerankK]

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": results,
        "reranked": request.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }





