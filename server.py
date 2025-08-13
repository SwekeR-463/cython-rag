# server.py
from fastapi import FastAPI, Query
import naive_rag
import cython_rag

app = FastAPI()

@app.get("/naive_search")
async def naive_search(query: str = Query(...)):
    results = await naive_rag.naive_rag(query)
    return {"results": results}

@app.get("/cython_search")
async def cython_search(query: str = Query(...)):
    results = await cython_rag.cython_rag(query)
    return {"results": results}
