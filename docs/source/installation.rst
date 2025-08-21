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

If a binary is not available for your platform, see below to build from source.


From Source
~~~~~~~~~~~~~~

To build from source, you will need a C compiler (e.g., `gcc` or `clang`) to build the Cython extensions.



.. code-block:: console

  git clone https://github.com/nolan-h-hamilton/Consenrich.git
  cd Consenrich
  python -m pip install .


Previous Versions
~~~~~~~~~~~~~~~~~~~~~

To install a specific version of Consenrich from PyPI, you can specify the version number in the pip install command, for example:

.. code-block:: bash

  python -m pip install consenrich==0.1.13b1


Conda
~~~~~~~~~~~~~~~~~~~~~~

You can create a `conda <https://docs.conda.io/en/latest/>`_, `mamba <https://mamba.readthedocs.io/en/latest/>`_ virtual environment to isolate your Consenrich installation and ensure all dependencies are met.

Save the following to `environment.yaml` and run ``conda env create -f environment.yaml`` to create an environment `consenrichEnv`:

.. code-block:: yaml

  name: consenrichEnv
  channels:
    - bioconda
    - conda-forge
    - defaults
  dependencies:
    - c-compiler # defaults to a compatible C compiler for your platform
    - python>=3.11
    - pip
    - setuptools
    - wheel
    - build
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

After activating the environment, install using pip/PyPI or building from source as described above.