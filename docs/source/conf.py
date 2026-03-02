# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

import sphinx


sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'torch-imagetools'
copyright = '2026, johnny95731'
author = 'johnny95731'
release = '0.1.0'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {'navigation_depth': 2}

# Numpy config
autosummary_generate = True


# Custom plugins
def flag(argument: str) -> bool:  # noqa: PRM002
    if argument and argument.strip():
        raise ValueError(f'No argument is allowed; {argument!r} supplied')
    else:
        return True


def remove_module_docstring(app, what, name, obj, options, lines):
    if what == 'module':
        no_docstring = options.get('no-docstring', False)
        if no_docstring:
            lines.clear()


def setup(app):
    app.connect('autodoc-process-docstring', remove_module_docstring)
    sphinx.ext.autodoc.ModuleDocumenter.option_spec['no-docstring'] = flag

    return {'parallel_read_safe': True}
