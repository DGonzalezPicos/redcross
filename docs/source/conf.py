# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'redcross'
copyright = '2022, Dario Gonzalez'
author = 'Dario'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['nbsphinx']

templates_path = ['_templates']


source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build', '**.ipynb_checkpoints']
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'petitRADTRANSdoc'
