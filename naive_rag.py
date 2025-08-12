import asyncio
import time
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)

async def naive_rag(query, top_k=3):
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    collection = client["anime_db"]["animes"]
    
    query_emb = model.encode(query)
    docs = collection.find({})
    
    all_docs = []
    
    async for doc in docs:
        all_docs.append(doc)
    
    similarities = []
    for doc in all_docs:
        emb = np.array(doc["embedding"])
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        similarities.append((doc["text"], sim))
        
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

if __name__ == "__main__":
    start = time.time()
    results = asyncio.run(naive_rag("action anime"))
    end = time.time()
    for text, score in results:
        print(f"{text} -> {score:.4f}")
    print(f"Time taken: {end - start:.4f} seconds")