import asyncio
import time
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer

cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def cosine_similarity(np.ndarray[np.float32_t, ndim=2] embeddings,
                      np.ndarray[np.float32_t, ndim=1] query):
    cdef Py_ssize_t i, n = embeddings.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] scores = np.zeros(n, dtype=np.float32)
    cdef float qnorm = np.linalg.norm(query)
    for i in range(n):
        scores[i] = np.dot(embeddings[i], query) / (np.linalg.norm(embeddings[i]) * qnorm)
    return scores


model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)


async def cythonrag(query: str, int top_k=3):
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
