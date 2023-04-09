"""Utilities for dealing with paths."""

from __future__ import annotations

import contextlib
import os
import socket
from pathlib import Path

__all__ = ["get_uri", "strip_hostname"]


def get_uri(dir_name: str | Path) -> str:
    """
    Return the URI path for a directory.

    This allows files hosted on different file servers to have distinct locations.

    Parameters
    ----------
    dir_name : str or Path
        A directory name.

    Returns
    -------
    str
        Full URI path, e.g., "fileserver.host.com:/full/path/of/dir_name".
    """
    fullpath = Path(dir_name).absolute()
    hostname = socket.gethostname()
    with contextlib.suppress(socket.gaierror, socket.herror):
        hostname = socket.gethostbyaddr(hostname)[0]
    return f"{hostname}:{fullpath}"


def strip_hostname(uri_path: str | Path) -> str:
    """
    Strop the hostname from a URI path.

    For example, "fileserver.host.com:/full/path/of/dir_name" will be transformed to
    "/full/path/of/dir_name".

    Parameters
    ----------
    uri_path : str or Path
        A URI path.

    Returns
    -------
    str
        The path without the hostname information.
    """
    dir_name = str(uri_path)
    if ":" in dir_name:
        dir_name = dir_name.split(":", 1)[1]
    return dir_name

