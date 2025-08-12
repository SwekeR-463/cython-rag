How to run?
```bash
# Install dependencies
pip install - requirements.txt

# Build Cython extension
python setup.py build_ext --inplace

# Push dataset to MongoDB
python push_data.py

# Run naive RAG
python naive_rag.py

# Run Cython RAG
python cython_rag.py

# Benchmark
python benchmark.py

# To run the server
uvicorn server:app --reload
```