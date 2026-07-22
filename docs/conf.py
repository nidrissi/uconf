"""Sphinx configuration for the uconf documentation site."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

project = "uconf"
author = "Najib Idrissi and Victor Roca i Lucio"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_preprocess_types = True

myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3

html_title = "uconf documentation"
html_theme = os.environ.get("SPHINX_HTML_THEME", "alabaster")
html_static_path = []


def _skip_nested_module_aliases(app, what, name, obj, skip, options):
    """Avoid documenting nested Sage classes twice on module pages."""
    if name in {"Component", "Element"}:
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", _skip_nested_module_aliases)
