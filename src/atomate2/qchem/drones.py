"""Drones for parsing VASP calculations and related outputs."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from emmet.core.qc_tasks import TaskDoc
from pymatgen.apps.borg.hive import AbstractDrone

logger = logging.getLogger(__name__)


class QChemDrone(AbstractDrone):
    """
    A QChem drone to parse QChem outputs.

    Parameters
    ----------
    **task_document_kwargs
        Additional keyword args passed to :obj: `.TaskDoc.from_directory`.
    """

    def __init__(self, **task_document_kwargs) -> None:
        self.task_document_kwargs = task_document_kwargs

    def assimilate(self, path: str | Path | None = None) -> TaskDoc:
        """
        Parse QChem output files and return the output document.

        Parameters
        ----------
        path : str pr Path or None
            Path to the directory containing mol.qout and other output files.


        Returns
        -------
        TaskDocument
            A QChem task document
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
        task_names = ["mol.qout.*"]
        combined_paths = [parent + os.sep + sdir for sdir in subdirs]
        valid_paths = []
        for sub_dir in combined_paths:
            file_names = os.listdir(sub_dir)
            if any(name.startswith("mol.qout.") for name in file_names):
                valid_paths.append(parent)

            if (
                not any(parent.endswith(os.sep + r) for r in task_names)
                and len(list(Path(parent).glob("mol.qout*"))) > 0
            ):
                valid_paths.append(parent)
        return valid_paths
