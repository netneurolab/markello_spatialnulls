# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'markello_spatialnulls'
copyright = '2020, Network Neuroscience Lab'
author = 'Ross Markello'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx.ext.autosectionlabel"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Prefix document path to section labels, to use `path/to/file:heading`
# instead of just `heading`
autosectionlabel_prefix_document = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_book_theme'
html_title = ""
html_logo = "_static/raw.png"
html_theme_options = {
    "show_prev_next": True,
    "search_bar_text": "Search the walkthrough ..."
}
html_context = {
    "github_user": "netneurolab",
    "github_repo": "markello_spatialnulls",
    "github_version": "master",
    "doc_path": "./walkthrough",
}
