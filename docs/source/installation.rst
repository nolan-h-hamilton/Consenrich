Installation
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Installation
   :name: Installation


From PyPI
~~~~~~~~~~

Multiple binaries are distributed via `PyPI <https://pypi.org/project/consenrich/#files>`_ to accommodate different operating systems, Python versions (`3.10 - 3.13`), and architectures.

To install the latest version, run:

.. code-block:: console

  % python -m pip install consenrich --upgrade

If a binary is not available for your platform or you wish to optimize compiler flags for your hardware, consider building from source.

Previous Versions
""""""""""""""""""""""""""

To install a specific version of Consenrich from PyPI, e.g., ``0.1.13b1``:

.. code-block:: console

  % python -m pip install consenrich==0.1.13b1

Conda
~~~~~~~~~~~~~~~~~~~~~~

You can easily create a `conda <https://docs.conda.io/en/latest/>`_, `mamba <https://mamba.readthedocs.io/en/latest/>`_ virtual environment and ensure all dependencies are met.

For instance, save the following contents to a file named ``environment.yaml``:

.. code-block:: yaml

  name: consenrichEnv
  channels:
    - conda-forge
    - bioconda
    - defaults
  dependencies:
    - c-compiler
    - python
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

  % conda config --set channel_priority strict
  % conda create -n consenrichEnv -f environment.yaml
  % conda activate consenrichEnv

If using `mamba <https://mamba.readthedocs.io/en/latest/>`_, or `micromamba <https://micromamba.readthedocs.io/en/latest/>`_, replace ``conda`` with ``mamba`` or ``micromamba``.


From Source
~~~~~~~~~~~~~~

.. admonition:: Guidance: C Compiler
  :class: tip
  :collapsible: closed

  To build from source, you will need a C compiler (e.g., `gcc` or `clang`) to build the Cython extensions.

  *It's likely that a C compiler is already available on your system (run``gcc --version`` or ``clang --version`` in a terminal)*.

  If not, try one of the following:

  - macOS: ``xcode-select --install``
  - Ubuntu/Debian Linux: ``sudo apt install build-essential``
  - For Fedora Linux: ``sudo dnf groupinstall "Development Tools"``


First, clone the repository:

.. code-block:: console

  % git clone https://github.com/nolan-h-hamilton/Consenrich.git


Set the working directory and install:

.. code-block:: console

  % cd Consenrich
  % python -m pip install .


