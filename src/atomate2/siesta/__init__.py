"""SIESTA interface for atomate2."""

# When integrated into the full atomate2 package, the SIESTA settings live in
# atomate2.settings.Atomate2Settings, so reuse the single shared instance. When
# running as the standalone atomate2siesta distribution (no top-level
# atomate2.SETTINGS), fall back to the bundled ExtendedAtomate2Settings.
try:
    from atomate2 import SETTINGS
except ImportError:
    from atomate2.siesta.siesta_settings import ExtendedAtomate2Settings

    SETTINGS = ExtendedAtomate2Settings()  # type: ignore[assignment, call-arg]  # env-populated settings; distinct BaseSettings subclass in fallback branch

# Version information (managed by versioningit)
try:
    from atomate2.siesta._version import __version__
except ImportError:
    __version__ = "0.0.1+unknown"
