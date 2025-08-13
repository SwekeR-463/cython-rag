# server.py
from fastapi import FastAPI, Query
import naive_rag
from cython_rag import cythonrag

app = FastAPI()

@app.get("/naive_search")
async def naive_search(query: str = Query(...)):
    results = await naive_rag.naive_rag(query)
    return {"results": results}

@app.get("/cython_search")
async def cython_search(query: str = Query(...)):
    results = await cythonrag(query)
    return {"results": results}
