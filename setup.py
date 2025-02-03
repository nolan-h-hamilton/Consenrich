from setuptools import setup, find_packages

setup(
    name="consenrich",
    version="0.1.0b1",
    description="Genome-wide extraction of reproducible continuous-valued signals hidden in noisy multisample functional genomics data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Nolan H. Hamilton, Benjamin D. McMichael, Michael I. Love, Terrence S. Furey",
    author_email="nolan.hamilton@unc.edu, bdmcmi@ad.unc.edu, milove@email.unc.edu, tsfurey@email.unc.edu",
    url="https://github.com/nolan-h-hamilton/Consenrich",
    python_requires=">=3.9",
    license="MIT",
    license_files=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=[
        "genomics",
        "functional genomics",
        "epigenomics",
        "epigenetics",
        "signal processing",
        "data fusion",
        "state estimator",
        "filter",
    ],
    packages=find_packages(),
    include_package_data=True,
    package_data={"consenrich": ["refdata/*"]},
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.11",
        "pandas",
        "pysam",
        "pybedtools",
        "deeptools",
        "pyBigWig",
    ],
    extras_require={
        "dev": [
            "pytest",
            "sphinx",
            "twine",
        ],
    },
    entry_points={
        "console_scripts": [
            "consenrich=consenrich.consenrich:main",
        ],
    },
)
