Installation
-----------------


From PyPI
~~~~~~~~~~

Consenrich supports Python 3.10–3.15. Binaries are distributed via
`PyPI <https://pypi.org/project/consenrich/#files>`_ for different operating
systems and architectures.

To install the latest version, run:

.. code-block:: console

  % python -m pip install --upgrade "consenrich>=0.11.9a0" "ccounts>=0.5.0b0"

If a binary is not available for your platform or you wish to optimize compiler flags for your hardware, consider building from source.

Conda
~~~~~~~~~~~~~~~~~~~~~~

You can use `conda <https://docs.conda.io/en/latest/>`_ or
`mamba <https://mamba.readthedocs.io/en/latest/>`_ to create an environment.
The ``all`` extra includes bigWig output and plotting.

Save to ``consenrichEnv.yaml``:

.. literalinclude:: ../consenrichEnv.yaml
   :language: yaml

Create and activate the environment:

.. code-block:: console

  % conda env create -f consenrichEnv.yaml
  % conda activate consenrichEnv

For Mamba, replace ``conda`` with ``mamba``.


From Source
~~~~~~~~~~~~~~

.. admonition:: Guidance: C Compiler
  :class: tip
  :collapsible: closed

  To build from source, you will need a C compiler such as `gcc` or `clang`.

  A compiler is often already present. Check with ``gcc --version`` or ``clang --version``.

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
