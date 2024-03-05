"""Module with common file names and classes used for FHI-aims flows."""

import contextlib
import os
import shutil
from collections.abc import Generator

TMPDIR_NAME = "tmpdir"
OUTPUT_FILE_NAME: str = "aims.out"
CONTROL_FILE_NAME: str = "control.in"
PARAMS_JSON_FILE_NAME: str = "parameters.json"
GEOMETRY_FILE_NAME: str = "geometry.in"


@contextlib.contextmanager
def cwd(path: str, mkdir: bool = False, rmdir: bool = False) -> Generator:
    """Change cwd intermediately.

    Example
    -------
    >>> with cwd(some_path):
    >>>     do so some stuff in some_path
    >>> do so some other stuff in old cwd

    Parameters
    ----------
    path: str or Path
        Path to change working directory to
    mkdir: bool
        If True make path if it does not exist
    rmdir: bool
        If True remove the working directory upon exiting
    """
    cwd = os.getcwd()

    if os.path.exists(path) is False and mkdir:
        os.makedirs(path)

    os.chdir(path)
    yield

    os.chdir(cwd)
    if rmdir:
        shutil.rmtree(path)
