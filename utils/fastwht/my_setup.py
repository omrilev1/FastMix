from setuptools import setup, Extension
import numpy as np
import sys

extra_compile_args = []
if sys.platform.startswith("win"):
    # MSVC flags
    extra_compile_args = ["/O2", "/std:c++14"]
else:
    extra_compile_args = ["-O3", "-std=c++11"]

ext = Extension(
    name="_hadamardKernel",
    sources=["../hadamard.cpp", "hadamardKernel.i"],
    include_dirs=[np.get_include(), "."],
    language="c++",
    swig_opts=["-c++"],               # <-- CRITICAL: generate C++ wrapper
    extra_compile_args=extra_compile_args,
)

setup(
    name="fastwht",
    version="0.0.0",
    ext_modules=[ext],
)