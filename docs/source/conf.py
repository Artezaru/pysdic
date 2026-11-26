# -----
# Import the necessary modules
# -----
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import datetime
import inspect

import pydata_sphinx_theme

import pyvista

pyvista.BUILDING_GALLERY = True
pyvista.OFF_SCREEN = True

# -----
# Path setup
# -----

project_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")
doc_source_directory = os.path.join(project_directory, "docs", "source")

sys.path.insert(0, project_directory)
sys.path.insert(0, doc_source_directory)

# -----
# Project information
# -----


def read_version():
    version_file = os.path.join(project_directory, "pysdic", "__version__.py")
    with open(version_file, "r") as file:
        exec(file.read())
    return locals()["__version__"]


project = "pysdic"
copyright = f"2026-{datetime.datetime.now().year}, Artezaru"
author = "Artezaru"
release = read_version()

# -----
# Extensions and general configuration
# -----

html_theme = "pydata_sphinx_theme"
html_static_path = [
    os.path.join(doc_source_directory, "_static"),
]

extensions = [
    "sphinx_copybutton",  # Add copy button to code blocks
    "sphinx.ext.autodoc",  # Automatically document the code
    "sphinx.ext.viewcode",  # Add links to the code
    "sphinx.ext.napoleon",  # Support for Google and Numpy docstring formats
    "sphinx.ext.githubpages",  # Publish the documentation on GitHub
    "sphinx.ext.autosummary",  # Generate summaries of the modules
    "sphinx.ext.mathjax",  # Render math in the documentation
    "sphinx_gallery.gen_gallery",  # Generate galleries of examples
    "sphinx_design",  # Add design elements to the documentation
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
]


# -----
# Autosummary configuration
# -----
autosummary_generate = True
templates_path = ["_templates"]

# -----
# Autodoc configuration
# -----
autodoc_typehints = "description"
add_module_names = False

# -----
# Napoleon configuration
# -----
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -----
# Intersphinx configuration
# -----
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "py3dframe": ("https://Artezaru.github.io/py3dframe/", None),
    "pysolve-gn": ("https://Artezaru.github.io/pysolve-gn/", None),
}


# Legacy "^^^" / "+++" header and footer separators of the sphinx-design cards
suppress_warnings = ["design.card_legacy"]


# -----
# Sphinx Gallery configuration
# -----
from sphinx_gallery.sorting import ExplicitOrder

gallery_directory = os.path.join(project_directory, "gallery")
gallery_sections = ["geometry", "computer_vision", "imaging", "sdic", "workflows"]

sphinx_gallery_conf = {
    # Paths relative to this conf.py: with absolute paths, sphinx-gallery writes absolute
    # paths in the toctrees of _gallery/index.rst, which Sphinx cannot resolve.
    "examples_dirs": [os.path.relpath(gallery_directory, doc_source_directory)],
    "gallery_dirs": ["_gallery"],
    "backreferences_dir": "_gallery_backreferences",
    "doc_module": ("pysdic",),
    "capture_repr": ["__repr__", "__str__", "_repr_html_"],
    "image_scrapers": ("matplotlib", "pyvista"),
    "filename_pattern": r"\.py$",
    # Order of the sections (sub-folders of gallery/), as paths relative to this conf.py
    "subsection_order": ExplicitOrder(
        [os.path.relpath(os.path.join(gallery_directory, name), doc_source_directory) for name in gallery_sections]
    ),
    # Order of the examples in a section: by file name
    "within_subsection_order": "sphinx_gallery.sorting.FileNameSortKey",
}