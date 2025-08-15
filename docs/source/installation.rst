Installation
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Installation
   :name: Installation


From PyPI
~~~~~~~~~~

Multiple binary `wheels <https://peps.python.org/pep-0427/>`_ are distributed via `PyPI <https://pypi.org/project/consenrich/#files>`_ to accommodate different operating systems, Python versions, and architectures. To install the latest version, run:

.. code-block:: bash

  python -m pip install consenrich --upgrade

If a wheel for your platform is not available, see below to build from source. Please feel free to open an issue on the `GitHub repository <https://github.com/nolan-h-hamilton/Consenrich/issues>`_ if you encounter any problems.


From Source
~~~~~~~~~~~~~~

- Clone the repository: ``git clone https://github.com/nolan-h-hamilton/Consenrich.git``
- Set working directory to ``/path/to/Consenrich``
- Build the package: ``python -m build``
- Install the package: ``python -m pip install .``



Previous Versions
~~~~~~~~~~~~~~~~~~~~~

To install a specific version of Consenrich from PyPI, you can specify the version number in the pip install command, for example:

.. code-block:: bash

  python -m pip install consenrich==0.1.13b1

