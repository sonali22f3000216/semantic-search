from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import time

app = FastAPI()

# CORS fix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Semantic Search API is running"}

# Sample documents
documents = [
    {"id": 0, "content": "How to authenticate using API keys", "metadata": {"source": "auth.md"}},
    {"id": 1, "content": "Installation guide for SDK", "metadata": {"source": "install.md"}},
    {"id": 2, "content": "How to reset password securely", "metadata": {"source": "security.md"}},
    {"id": 3, "content": "Authentication using OAuth tokens", "metadata": {"source": "oauth.md"}},
    {"id": 4, "content": "Deploying application to production", "metadata": {"source": "deploy.md"}},
]

# Lightweight embedding
def simple_embed(text):
    return np.array([len(text), sum(ord(c) for c in text) % 1000])

doc_embeddings = [simple_embed(doc["content"]) for doc in documents]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True
    rerankK: int = 3

@app.post("/search")
def search(request: SearchRequest):
    start_time = time.time()

    query_embedding = simple_embed(request.query)

    scores = []
    for idx, doc_embedding in enumerate(doc_embeddings):
        raw_score = cosine_similarity(query_embedding, doc_embedding)
        score = (raw_score + 1) / 2
        scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_k = scores[:request.k]

    results = []
    for idx, vector_score in top_k:
        rerank_score = 0
        if request.query.lower() in documents[idx]["content"].lower():
            rerank_score = 1

        final_score = 0.7 * vector_score + 0.3 * rerank_score

        results.append({
            "id": documents[idx]["id"],
            "score": float(final_score),
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:request.rerankK]

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": results,
        "reranked": request.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
