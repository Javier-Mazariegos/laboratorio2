from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("external.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)