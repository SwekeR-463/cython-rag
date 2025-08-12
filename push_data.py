import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer
import asyncio

model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)

async def push_to_mongo():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    collection = client["anime_db"]["animes"]
    
    df = pd.read_csv("animes.csv")
    
    docs = []
    
    for _, row in df.iterrows():
        text = f"{row['Title']} - {row['Synopsis']}"
        # chunking
        for i in range(0, len(text), 200): # 200 = chunk size
            chunk = text[i:i+200]
            embedding = model.encode(chunk).tolist() 
            docs.append({"text": chunk, "embedding": embedding})
    
    if docs:
        await collection.insert_many(docs)
        print(f"Inserted {len(docs)} documents into MongoDB")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(push_to_mongo())
        
    