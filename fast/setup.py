from setuptools import setup
from Cython.Build import cythonize
import numpy

# python setup.py build_ext --inplace
setup(
    ext_modules=cythonize(
        ['cooc_count.pyx'], 
        annotate=True), 
    include_dirs=[numpy.get_include()]
)