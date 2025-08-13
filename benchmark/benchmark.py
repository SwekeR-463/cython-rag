import time
import asyncio
from naive_rag import naive_rag
from cython_rag import cython_rag

queries = [
    "school romance anime",
    "post-apocalyptic survival",
    "mecha battle in space",
    "slice of life high school",
    "revenge story anime",
    "time travel mystery",
    "isekai adventure",
    "sports competition anime",
    "historical samurai drama",
    "psychological thriller anime"
]

print("Benchmarking...")

# Naive Python RAG
start_time = time.time()
for q in queries:
    _ = asyncio.run(naive_rag(q, top_k=3))
end_time = time.time()
print(f"Naive Python total: {end_time - start_time:.4f} sec")
print(f"Naive Python avg per query: {(end_time - start_time)/len(queries):.4f} sec")

# Cython RAG
start_time = time.time()
for q in queries:
    _ = asyncio.run(cython_rag(q, top_k=3))
end_time = time.time()
print(f"Cython total: {end_time - start_time:.4f} sec")
print(f"Cython avg per query: {(end_time - start_time)/len(queries):.4f} sec")
