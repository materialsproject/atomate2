"""Utilities for dealing with paths."""

from __future__ import annotations

import socket
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from typing import Union

__all__ = ["get_uri"]


def get_uri(dir_name: Union[str, Path]) -> str:
    """
    Return the URI path for a directory.

    This allows files hosted on different file servers to have distinct locations.

    Parameters
    ----------
    dir_name
        A directory name.

    Returns
    -------
    str
        Full URI path, e.g., "fileserver.host.com:/full/path/of/dir_name".
    """
    fullpath = Path(dir_name).absolute()
    hostname = socket.gethostname()
    try:
        hostname = socket.gethostbyaddr(hostname)[0]
    except socket.gaierror:
        pass
    return "{}:{}".format(hostname, fullpath)
