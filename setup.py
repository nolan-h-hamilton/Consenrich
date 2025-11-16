import os
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import pysam
import numpy


def get_includes():
    class Includes:
        def __iter__(self):
            import pysam
            import numpy

            return iter(pysam.get_include() + [numpy.get_include()])

        def __getitem__(self, i):
            return list(self)[i]

    return Includes()


extensions = [
    Extension(
        "consenrich.cconsenrich",
        sources=["src/consenrich/cconsenrich.pyx"],
        include_dirs=get_includes(),
        libraries=pysam.get_libraries(),
        extra_compile_args=[
            "-O3",
            "-fno-trapping-math",
            "-fno-math-errno",
            "-mtune=generic",
        ],
    )
]

setup(
    name="consenrich",
    version="0.7.4b1",
    packages=find_packages(where="src"),
    include_package_data=True,
    package_dir={"": "src"},
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
    ],
    python_requires=">=3.11",
    zip_safe=False,
)
