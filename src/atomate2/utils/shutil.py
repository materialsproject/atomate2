"""Utilities for file operations."""

from __future__ import annotations

import glob
import os
import shutil
from gzip import GzipFile
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["gzip_files"]


def gzip_files(pathnames: Sequence[str | Path], compresslevel=6):
    """
    Gzip the files matching the pathnames provided.

    Uses glob to match the files.

    Parameters
    ----------
    pathnames : List[str or Path]
        pathnames that will be gzipped.

    """
    for pathname in pathnames:
        for filepath in glob.glob(str(pathname)):
            if not filepath.lower().endswith("gz") and not os.path.isdir(filepath):
                with open(filepath, "rb") as f_in, GzipFile(
                    f"{filepath}.gz", "wb", compresslevel=compresslevel
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)
                shutil.copystat(filepath, f"{filepath}.gz")
                os.remove(filepath)
