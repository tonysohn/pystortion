# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys

__minimum_python_version__ = "3.5"

class UnsupportedPythonError(Exception):
    pass

if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise UnsupportedPythonError("pystortion does not support Python < {}".format(__minimum_python_version__))

if not _ASTROPY_SETUP_:
    # For egg_info test builds to pass, put package imports here.
    from .example_mod import *

# Set the bibtex entry to the article referenced in CITATION.
def _get_bibtex():
    citation_file = os.path.join(os.path.dirname(__file__), 'CITATION')

    with open(citation_file, 'r') as citation:
        refs = citation.read().split('@ARTICLE')[1:]