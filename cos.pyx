import cython
import numpy as np
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
# one optimization can be doing all the ops in c only and then returning a np array