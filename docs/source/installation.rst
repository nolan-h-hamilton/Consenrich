Installation
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Installation
   :name: Installation


From PyPI
~~~~~~~~~~

Multiple binaries are distributed via `PyPI <https://pypi.org/project/consenrich/#files>`_ to accommodate different operating systems, Python versions, and architectures. To install the latest version, run:

.. code-block:: bash

  python -m pip install consenrich --upgrade


Previous Versions
""""""""""""""""""""""""""

To install a specific version of Consenrich from PyPI, e.g., ``0.1.13b1``:

.. code-block:: bash

  python -m pip install consenrich==0.1.13b1


If a binary is not available for your platform, see below to build from source.


From Source
~~~~~~~~~~~~~~

To build from source, you will need a C compiler (e.g., `gcc` or `clang`) to build the Cython extensions.


First, clone the repository:

.. code-block:: console

  git clone https://github.com/nolan-h-hamilton/Consenrich.git


Set the working directory and install:

.. code-block:: console

  cd Consenrich
  python -m pip install .



Conda
~~~~~~~~~~~~~~~~~~~~~~

You can easily create a `conda <https://docs.conda.io/en/latest/>`_, `mamba <https://mamba.readthedocs.io/en/latest/>`_ virtual environment to isolate your Consenrich installation and ensure all dependencies are met.

Save the following to `environment.yaml`

.. code-block:: yaml

  name: consenrichEnv
  channels:
    - conda-forge
    - bioconda
    - defaults
  dependencies:
    - c-compiler
    - python>=3.11
    - pip
    - setuptools
    - wheel
    - cython>=3.0
    - numpy>=2.3.0
    - scipy>=1.16.0
    - pandas>=2.3.0
    - samtools>=1.20
    - bedtools>=2.30.0
    - pysam>=0.23.3
    - pybedtools>=0.11.2
    - ucsc-bedgraphtobigwig
    - PyYAML>=6.0.2
    - PyWavelets>=1.9.0

    - pip:
        - consenrich

Then, run the following to create and activate the environment, named ``consenrichEnv``:

.. code-block:: console

  conda config --set channel_priority strict
  conda create -n consenrichEnv -f environment.yaml
  conda activate consenrichEnv

If using `mamba <https://mamba.readthedocs.io/en/latest/>`_, or `micromamba <https://micromamba.readthedocs.io/en/latest/>`_, replace ``conda`` with ``mamba`` or ``micromamba``.
