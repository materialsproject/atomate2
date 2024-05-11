"""Drones for parsing VASP calculations and related outputs."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from emmet.core.tasks import TaskDoc
from pymatgen.apps.borg.hive import AbstractDrone

logger = logging.getLogger(__name__)


class VaspDrone(AbstractDrone):
    """
    A VASP drone to parse VASP outputs.

    Parameters
    ----------
    **task_document_kwargs
        Additional keyword args passed to :obj:`.TaskDoc.from_directory`.
    """

    def __init__(self, **task_document_kwargs) -> None:
        self.task_document_kwargs = task_document_kwargs

    def assimilate(self, path: str | Path | None = None) -> TaskDoc:
        """
        Parse VASP output files and return the output document.

        Parameters
        ----------
        path : str or Path or None
            Path to the directory containing vasprun.xml and OUTCAR files.

        Returns
        -------
        TaskDoc
            A VASP task document.
        """
        path = path or Path.cwd()

        try:
            doc = TaskDoc.from_directory(path, **self.task_document_kwargs)
        except Exception:
            import traceback

            logger.exception(
                f"Error in {Path(path).absolute()}\n{traceback.format_exc()}"
            )
            raise
        return doc

    def get_valid_paths(self, path: tuple[str, list[str], list[str]]) -> list[str]:
        """Get valid paths to assimilate.

        There are some restrictions on the valid directory structures:

        1. There can be only one vasprun in each directory. Nested directories are fine.
        2. Directories designated "relax1"..."relax9" are considered to be parts of a
           multiple-optimization run.
        3. Directories containing VASP output with ".relax1"...".relax9" are also
           considered as parts of a multiple-optimization run.

        Parameters
        ----------
        path : tuple of (str, list of str, list of str)
            Input path as a tuple generated from ``os.walk``, i.e., (parent, subdirs,
            files).

        Returns
        -------
        list of str
            A list of paths to assimilate.
        """
        parent, subdirs, _ = path
        task_names = ["precondition"] + [f"relax{i}" for i in range(9)]
        if set(task_names).intersection(subdirs):
            return [parent]
        if (
            not any(parent.endswith(os.sep + r) for r in task_names)
            and len(list(Path(parent).glob("vasprun.xml*"))) > 0
        ):
            return [parent]
        return []
