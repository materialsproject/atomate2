# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Add your project's directory to sys.path
sys.path.insert(0, os.path.abspath("../../src"))

project = "atomate2siesta"
copyright = "2024, Arsalan Akhtar"
author = "Arsalan Akhtar"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",  # Enable Markdown support
    "sphinx_copybutton",  # Add copy button to code blocks
]

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True  # Remove prompts when copying

# MyST Parser configuration
myst_enable_extensions = [
    "colon_fence",  # ::: instead of ```
    "deflist",  # Definition lists
    "tasklist",  # - [ ] Task lists
    "linkify",  # Auto-convert URLs to links
]
# myst_heading_anchors = 3  # Disabled - causes anchorname bug in Sphinx 8.x

# MyST URL schemes configuration
myst_url_schemes = ("http", "https", "mailto", "ftp")

# Allow Markdown and RST files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Internationalization (i18n) configuration -------------------------------
language = "en"  # Default language
locale_dirs = ["locale/"]  # Path to translation files
gettext_compact = False  # Create separate POT files for each document
gettext_uuid = True  # Add unique IDs to make translations easier to update
gettext_auto_build = True

# Supported languages
languages = ["en", "fa"]  # English and Persian (Farsi)

templates_path = ["_templates"]
exclude_patterns = [
    "tutorials-md/**/job_*",  # Job output directories
    "tutorials-md/**/flow_outputs",  # Flow output directories
]

# Configure Sphinx to include files outside source/
# This allows tutorial README.md files to be included
# (Path imported at top of file)
source_dir = Path(__file__).parent
# Set the root document path to include tutorials
html_extra_path = []

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "show-inheritance": True,
}

# Mock imports for modules that may not be available during doc build
# This prevents import errors for optional dependencies
autodoc_mock_imports = [
    "atomate2.forcefields",
    "atomate2.forcefields.jobs",
    "atomate2.vasp",
    "atomate2.vasp.jobs",
    "atomate2.vasp.jobs.base",
    "atomate2.vasp.jobs.core",
    "atomate2.aims",
    "atomate2.aims.jobs",
    "atomate2.aims.jobs.base",
]

# Type hints settings - avoid resolving forward references that may fail
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Suppress specific warnings that are non-critical
suppress_warnings = [
    "autodoc",  # Suppress forward reference warnings from inherited base classes
    "myst.xref_missing",  # Suppress cross-reference warnings from tutorial markdown files
    "misc.highlighting_failure",  # Suppress lexing warnings from Unicode in code blocks
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# HTML theme settings
html_theme = "sphinx_rtd_theme"

# Disable page TOC to work around Sphinx anchorname bug
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}
html_static_path = ["_static"]

# Custom CSS files
html_css_files = [
    "custom.css",
]

# Note: sphinx_rtd_theme doesn't use html_sidebars the same way
# We'll inject the language switcher via a different method
# For now, users can manually navigate to /fa/ directory

# HTML context for language switcher
html_context = {
    "display_github": True,
    "github_user": "arsalan-akhtar",
    "github_repo": "atomate2siesta",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
    "languages": [
        ("en", "English"),
        ("fa", "فارسی"),
    ],
    "current_language": language,
}
