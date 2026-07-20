"""
Metadata for atomate2siesta dataclass modules.

This module defines author, copyright, and license information used across
all dataclass modules. Centralizing this information allows easy updates
without modifying individual files.
"""

# Author information
__author__ = "Arsalan Akhtar"
__email__ = "arsalan.akhtar@gmail.com"
__maintainer__ = "Arsalan Akhtar"

# Copyright and license
__copyright__ = "Copyright (c) 2024-2025, Arsalan Akhtar"
__license__ = "Modified BSD"
__version__ = "1.1.0"

# Project information
__project__ = "atomate2siesta"
__url__ = "https://github.com/materialsproject/atomate2"
__description__ = "SIESTA integration for atomate2 workflow framework"

# Credits
__credits__ = [
    "Arsalan Akhtar (Lead Developer)",
]

# Full metadata dictionary (for programmatic access)
METADATA = {
    "author": __author__,
    "email": __email__,
    "maintainer": __maintainer__,
    "copyright": __copyright__,
    "license": __license__,
    "version": __version__,
    "project": __project__,
    "url": __url__,
    "description": __description__,
    "credits": __credits__,
}


def get_header_comment() -> str:
    """
    Generate standardized header comment for dataclass modules.

    Returns
    -------
        str: Multi-line header comment with author and copyright info
    """
    return f"""
Author: {__author__} <{__email__}>
Copyright: {__copyright__}
License: {__license__}
Project: {__project__}
URL: {__url__}
""".strip()


def get_module_docstring_header() -> str:
    """
    Generate standardized docstring header for dataclass modules.

    Returns
    -------
        str: Formatted docstring header with metadata
    """
    return f"""
.. moduleauthor:: {__author__} <{__email__}>
.. copyright:: {__copyright__}
.. license:: {__license__}
""".strip()
