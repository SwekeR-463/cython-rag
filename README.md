# Cython RAG

### Folder Structure

* `data/` -> push data to mongoDB through motor
* `setup/` -> building the cython function
* `naive_rag.py` -> naive rag implementation -- data loading, chunking, cosine sim, return the output
* `cython_rag.pyx` -> rag with cython 
* `benchmark/benchmark.py` -> to check timing perf against 10 queries == 8s per query for naive & 6s for cython
* `server.py` -> FastAPI server with get to serve both the rags


### How to run?

```bash
# Install dependencies
pip install - requirements.txt

# Build Cython extension
python setup.py build_ext --inplace

# Push dataset to MongoDB
python push_data.py

# Run naive RAG
python naive_rag.py

# Benchmark
python benchmark.py

# To run the server
uvicorn server:app --reload
```