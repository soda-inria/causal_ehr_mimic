"""Sphinx configuration."""
import pathlib
import sys
from datetime import datetime
from typing import List

import importlib_metadata
from dotenv import load_dotenv

# Load user-specific env vars (e.g. secrets) from a `.env` file
load_dotenv()


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. Note that we are adding an absolute
# path.
_project_directory = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_directory))


# -- Project information -----------------------------------------------------
PACKAGE_NAME = "caumim"
try:
    project_metadata = importlib_metadata.metadata(PACKAGE_NAME)
except importlib_metadata.PackageNotFoundError as err:
    raise RuntimeError(
        f"The package '{PACKAGE_NAME}' must be installed. "
        "Please install the package in editable mode before building docs."
    ) from err


# pylint: disable=invalid-name

# -- Project information -----------------------------------------------------
project = project_metadata["Name"]
author = project_metadata["Author"]
# pylint: disable=redefined-builtin
copyright = f"{datetime.now().year}, {author}"
version = release = project_metadata["Version"]

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",  # MyST .md parsing (https://myst-parser.readthedocs.io/en/latest/index.html)
    "sphinx.ext.autodoc",  # Include documentation from docstrings (https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
    "sphinx.ext.autosummary",  # Generate autodoc summaries (https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html)
    "sphinx.ext.intersphinx",  # Link to other projects’ documentation (https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html)
    "sphinx.ext.viewcode",  # Add documentation links to/from source code (https://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html)
    "sphinx.ext.autosectionlabel",  # Allow reference sections using its title (https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html)
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings (https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    "sphinx_click",  # Automatic documentation of click based CLI (https://github.com/click-contrib/sphinx-click)
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
# Note: `custom-class-template.rst` & `custom-module-template.rst`
#   for sphinx.ext.autosummary extension `recursive` option
#   see: https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion


# List of patterns, relative to source directory, that match files and
#   directories to ignore when looking for source files.
#   This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[str] = ["_build", "Thumbs.db", ".DS_Store", ".venv"]

# Sphinx configs
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "footer_items": ["sphinx-version"],
    "logo": {
        "image_light": "logo-project.svg",
        "image_dark": "logo-project.svg",
    },
    "icon_links": [
        {
            "name": "SoDa",
            "url": "https://team.inria.fr/soda/",
            "icon": "_static/logo-project.svg",
            "type": "local",
        },
        {
            "name": "GitLab",
            "url": "https://gitlab.com/strayMat/caumim",
            "icon": "fa-brands fa-square-gitlab",
            "type": "fontawesome",
        },
    ],
}
html_show_sourcelink = (
    True  # Remove 'view source code' from top of page (for html, not python)
)

# -- Extension configurations ---------------------------------------------------

# sphinx.ext.autosummary configs
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# sphinx.ext.autodoc configs
# autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
# autodoc_inherit_docstrings = (
#     True  # If no class summary, inherit base class summary
# )
# autodoc_typehints = (
#     "description"  # Show typehints as content of function or method
# )

# myst_parser configs
# Prefix document path to section labels, to use:
# `path/to/file:heading` instead of just `heading`
autosectionlabel_prefix_document = True


# -- External mapping --------------------------------------------------------
python_version = ".".join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    "sphinx": ("http://www.sphinx-doc.org/en/stable", None),
    "python": ("https://docs.python.org/" + python_version, None),
}
