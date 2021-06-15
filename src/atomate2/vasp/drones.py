"""Drones for parsing VASP calculations and related outputs."""

from __future__ import annotations

import logging
import os
import typing
from collections import OrderedDict
from pathlib import Path

from pymatgen.apps.borg.hive import AbstractDrone

from atomate2.vasp.schemas.task import TaskDocument

if typing.TYPE_CHECKING:
    from typing import Any, Dict, List, Tuple, Union


logger = logging.getLogger(__name__)

VOLUMETRIC_FILES = ("CHGCAR", "LOCPOT", "AECCAR0", "AECCAR1", "AECCAR2")

__all__ = ["VaspDrone"]


class VaspDrone(AbstractDrone):
    """
    A VASP drone to parse VASP outputs.

    Parameters
    ----------
    task_names
        Naming scheme for multiple calculations in one folder e.g. ["relax1", "relax2"].
    additional_fields
        Dictionary of additional fields to add to output document.
    **task_document_kwargs
        Additional keyword args passed to :obj:`.TaskDocument.from_task_files`.
    """

    def __init__(
        self,
        task_names: List[str] = None,
        additional_fields: Dict[str, Any] = None,
        **task_document_kwargs,
    ):
        self.additional_fields = {} if additional_fields is None else additional_fields

        self.task_names = task_names
        if self.task_names is None:
            self.task_names = ["precondition"] + [f"relax{i}" for i in range(9)]

        self.task_document_kwargs = task_document_kwargs

    def assimilate(self, path: Union[str, Path] = None) -> TaskDocument:
        """
        Parse VASP output files and return the output document.

        Parameters
        ----------
        path
            Path to the directory containing vasprun.xml and OUTCAR files.

        Returns
        -------
        TaskDocument
            A VASP task document.
        """
        if path is None:
            path = Path.cwd()

        logger.info(f"Getting task doc in: {path}")
        task_files = find_vasp_files(self.task_names, path)

        if len(task_files) > 0:
            try:
                doc = TaskDocument.from_task_files(
                    path, task_files, **self.task_document_kwargs
                )
            except Exception:
                import traceback

                logger.error(
                    f"Error in {Path(path).absolute()}\n{traceback.format_exc()}"
                )
                raise
        else:
            raise ValueError("No VASP files found!")

        doc.copy(update=self.additional_fields)
        return doc

    def get_valid_paths(self, path: Tuple[str, List[str], List[str]]) -> List[str]:
        """
        Get valid paths to assimilate.

        There are some restrictions on the valid directory structures:

        1. There can be only one vasprun in each directory. Nested directories are fine.
        2. Directories designated "relax1"..."relax9" are considered to be parts of a
           multiple-optimization run.
        3. Directories containing VASP output with ".relax1"...".relax9" are also
           considered as parts of a multiple-optimization run.

        Parameters
        ----------
        path
            Input path as a tuple generated from ``os.walk``, i.e., (parent, subdirs,
            files).

        Returns
        -------
        list[str]
            A list of paths to assimilate.
        """
        parent, subdirs, _ = path
        if set(self.task_names).intersection(subdirs):
            return [parent]
        if (
            not any([parent.endswith(os.sep + r) for r in self.task_names])
            and len(list(Path(parent).glob("vasprun.xml*"))) > 0
        ):
            return [parent]
        return []


def find_vasp_files(
    task_names: List[str],
    path: Union[str, Path],
    volumetric_files: Tuple[str, ...] = VOLUMETRIC_FILES,
) -> Dict[str, Any]:
    """
    Find VASP files in a directory.

    Only files in folders with names matching a task name (or alternatively files
    with the task name as an extension, e.g., vasprun.relax1.xml) will be returned.

    VASP files in the current directory will be given the task name "standard".

    Parameters
    ----------
    task_names
        Naming scheme for multiple calculations in one folder e.g. ["relax1", "relax2"].
    path
        Path to a directory to search.
    volumetric_files
        Volumetric files to search for.

    Returns
    -------
    dict[str, Any]
        The filenames of the calculation outputs for each VASP task, given as a ordered
        dictionary of::

            {
                task_name: {
                    "vasprun_file": vasprun_filename,
                    "outcar_file": outcar_filename,
                    "volumetric_files": [CHGCAR, LOCPOT, etc]
                },
                ...
            }

    """
    path = Path(path)
    task_files = OrderedDict()

    def _get_task_files(files, suffix=""):
        vasp_files = {}
        vol_files = []
        for file in files:
            if file.match(f"*vasprun.xml{suffix}*"):
                vasp_files["vasprun_file"] = file
            elif file.match(f"*OUTCAR{suffix}*"):
                vasp_files["outcar_file"] = file
            elif any([file.match(f"*{f}{suffix}*") for f in volumetric_files]):
                vol_files.append(file)

        if len(vol_files) > 0 or len(vasp_files) > 0:
            # only add volumetric files if some were found or other vasp files
            # were found
            vasp_files["volumetric_files"] = vol_files

        return vasp_files

    for task_name in task_names:
        subfolder_match = list(path.glob(f"{task_name}/*"))
        suffix_match = list(path.glob(f"*.{task_name}*"))
        if len(subfolder_match) > 0:
            # subfolder match
            task_files[task_name] = _get_task_files(subfolder_match)
        elif len(suffix_match) > 0:
            # try extension schema
            task_files[task_name] = _get_task_files(
                suffix_match, suffix=f".{task_name}"
            )

    if len(task_files) == 0:
        # get any matching file from the root folder
        standard_files = _get_task_files(list(path.glob("*")))
        if len(standard_files) > 0:
            task_files["standard"] = standard_files

    return task_files
