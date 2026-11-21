# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "Consenrich"
copyright = "2025, Nolan H. Hamilton"
author = "Nolan H. Hamilton"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_design", "sphinx_copybutton"]
templates_path = ["_templates"]

copybutton_prompt_text = "% "
copybutton_only_copy_prompt_lines = False

html_theme = "furo"
html_title = "Consenrich documentation"
html_short_title = html_title
html_context = {
    "docstitle": '<span class="brandName">Consenrich</span> documentation',
}
html_static_path = ["_static"]
html_css_files = ["typography.css"]
html_theme_options = {
    "light_css_variables": {
        "font-stack": '"CMU Serif"',
        "font-stack--monospace": '"CMU Typewriter Text", monospace',
    },
}
html_logo = None
html_favicon = None
