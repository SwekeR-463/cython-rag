from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="cython_rag",
    ext_modules=cythonize(
        "cython_rag.pyx",
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[np.get_include()],
)
