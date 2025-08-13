import asyncio
import time
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer
from cos import cosine_similarity

model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)


async def cython_rag(query, top_k=3):
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    collection = client["anime_db"]["animes"]

    query_emb = model.encode(query).astype(np.float32)
    docs = []
    texts = []
    async for doc in collection.find({}):
        docs.append(doc["embedding"])
        texts.append(doc["text"])

    emb_array = np.array(docs, dtype=np.float32)
    scores = cosine_similarity(emb_array, query_emb)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(texts[i], float(scores[i])) for i in top_indices]

if __name__ == "__main__":
    start = time.time()
    results = asyncio.run(cython_rag("action adventure anime"))
    end = time.time()
    for text, score in results:
        print(f"{score:.4f} -> {text}")
    print(f"Time taken: {end - start:.4f} seconds")

