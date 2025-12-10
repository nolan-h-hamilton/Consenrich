import sys
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import pysam
import numpy


def get_includes():
    class Includes:
        def __iter__(self):
            return iter(pysam.get_include() + [numpy.get_include()])

        def __getitem__(self, i):
            return list(self)[i]

    return Includes()


base_compile = [
    "-O3",
    "-fno-trapping-math",
    "-fno-math-errno",
    "-mtune=generic",
]

if sys.platform == "darwin":
    omp_compile = ["-Xpreprocessor", "-fopenmp"]
    omp_link = ["-lomp"]
elif sys.platform.startswith("linux"):
    omp_compile = ["-fopenmp"]
    omp_link = ["-fopenmp"]
else:
    omp_compile = []
    omp_link = []


extensions = [
    Extension(
        "consenrich.cconsenrich",
        sources=["src/consenrich/cconsenrich.pyx"],
        include_dirs=get_includes(),
        libraries=pysam.get_libraries(),
        extra_compile_args=base_compile + omp_compile,
        extra_link_args=omp_link,
    )
]


setup(
    name="consenrich",
    version="0.7.11b2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    ext_modules=cythonize(extensions, language_level="3"),
    install_requires=[
        "cython>=3.0",
        "numpy>=2.1",
        "pandas>=2.3",
        "scipy>=1.15",
        "pysam>=0.23.3",
        "pybedtools>=0.11.2",
        "PyYAML>=6.0.2",
        "PyWavelets>=1.9.0",
        "matplotlib",
    ],
    python_requires=">=3.11",
    zip_safe=False,
)
