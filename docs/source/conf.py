# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from pathlib import Path


docsRoot = Path(__file__).resolve().parents[1]
repoRoot = docsRoot.parent
os.environ.setdefault("HOME", str(docsRoot.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((docsRoot / ".cache").resolve()))
sys.path.insert(0, str((repoRoot / "src").resolve()))
project = 'consenrich'
copyright = '2025, Nolan H. Hamilton'
author = 'Nolan H. Hamilton'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_design', # https://pypi.org/project/sphinx_design/
]
autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = []

html_theme = "furo"
html_logo = None
html_favicon = None
