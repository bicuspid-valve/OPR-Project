"""Build script for the _fast_core C extension.

Usage:
    python build_fast_core.py

This compiles _fast_core.c into a shared library that Python can import.
The compiled module is placed in the same directory as this script.
"""
from setuptools import setup, Extension
import os
import sys

ext = Extension(
    "_fast_core",
    sources=[os.path.join(os.path.dirname(__file__) or ".", "_fast_core.c")],
    extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3", "-march=native"],
)

setup(
    name="_fast_core",
    version="1.0",
    ext_modules=[ext],
    script_args=["build_ext", "--inplace"],
)
