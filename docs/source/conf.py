# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EarlyStopping"
copyright = "2025, Eric Ziebell, Ratmir Miftachov, Bernhard Stankewitz, Laura Hucker"
author = "Eric Ziebell, Ratmir Miftachov, Bernhard Stankewitz, Laura Hucker"
release = "0.0.4"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "EarlyStopping": ("https://esfiep.github.io/EarlyStopping/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
}

html_extra_path = ["../../extra_files"]  # added manually for search optimization
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_show_sourcelink = False

html_theme_options = {
    "external_links": [
        {"name": "Github", "url": "https://github.com/EarlyStop/EarlyStopping"},
    ],
}
