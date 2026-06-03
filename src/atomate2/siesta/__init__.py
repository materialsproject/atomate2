"""SIESTA interface for atomate2."""
# atomate2siesta/src/atomate2/siesta/__init__.py

# import atomate2.siesta as siesta
from atomate2.siesta.siesta_settings import ExtendedAtomate2Settings

# This Part was for Depending atomate2siesta to atomate2
# _DEFAULT_CONFIG_FILE_PATH = "~/.atomate2siesta.yaml"
# Override the SETTINGS with the extended version
# SETTINGS = ExtendedAtomate2Settings(CONFIG_FILE=_DEFAU
# LT_CONFIG_FILE_PATH)

# This Part is for make atomate2siesta independent of atomate2
SETTINGS = ExtendedAtomate2Settings()

# Version information (managed by versioningit)
try:
    from atomate2.siesta._version import __version__
except ImportError:
    __version__ = "0.0.1+unknown"
