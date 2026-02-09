"""Sphinx configuration for vibroglass documentation."""

from __future__ import annotations

# Workaround: myst-parser 5.0 tries to remove a Sphinx transform that
# Sphinx >= 9.1 no longer registers by default.  Patch the list.remove
# call so it silently succeeds when the item is absent.
import myst_parser.sphinx_ext.main as _myst_main  # noqa: E402
from sphinx.transforms import UnreferencedFootnotesDetector as _SUFD

_orig_setup = _myst_main.setup_sphinx


def _patched_setup(app, load_parser=False):
    if _SUFD not in app.registry.transforms:
        app.registry.transforms.append(_SUFD)
    return _orig_setup(app, load_parser)


_myst_main.setup_sphinx = _patched_setup

project = "vibroglass"
author = "Alfredo Fiorentino, Paolo Pegolo, Enrico Drigo"
release = "0.1.0"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

# MyST settings
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]

# myst-nb: do NOT re-execute notebooks (they ship with outputs)
nb_execution_mode = "off"

# Suppress warnings that are harmless in our context
suppress_warnings = ["myst.header"]

# Exclude build artefacts from source discovery
exclude_patterns = ["_build", "jupyter_execute"]

# Autosummary
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}

# Theme
html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/logo_hydro_glasses.png"
html_title = "vibroglass"
