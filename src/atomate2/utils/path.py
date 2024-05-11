"""Utilities for dealing with paths."""

from __future__ import annotations

import contextlib
import os
import socket
from pathlib import Path


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


def find_recent_logfile(
    dir_name: Path | str, logfile_extensions: str | list[str]
) -> str:
    """
    Find the most recent logfile in a given directory.

    Parameters
    ----------
    dir_name
        The path to the directory to search
    logfile_extensions
        The extension (or list of possible extensions) of the logfile to search for.
        For an exact match only, put in the full file name.

    Returns
    -------
    logfile
        The path to the most recent logfile with the desired extension
    """
    mod_time = 0.0
    logfile = None
    if isinstance(logfile_extensions, str):
        logfile_extensions = [logfile_extensions]
    for f in os.listdir(dir_name):
        f_path = os.path.join(dir_name, f)
        for ext in logfile_extensions:
            if ext in f and os.path.getmtime(f_path) > mod_time:
                mod_time = os.path.getmtime(f_path)
                logfile = os.path.abspath(f_path)
    return logfile
