"""Drones for parsing VASP calculations and related outputs."""

from __future__ import annotations

import logging
import typing

from pymatgen.apps.borg.hive import AbstractDrone

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Dict, List, Tuple

    from atomate2.vasp.schemas.task import TaskDocument


logger = logging.getLogger(__name__)

VOLUMETRIC_FILES = ("CHGCAR", "LOCPOT", "AECCAR0", "AECCAR1", "AECCAR2")


class VaspDrone(AbstractDrone):
    """
    A VASP drone to parse VASP outputs.

    Parameters
    ----------
    runs
        Naming scheme for multiple calculations in one folder e.g. ["relax1", "relax2"].
    additional_fields
        Dictionary of additional fields to add to output document.
    task_document_kwargs
        Additional keyword args passed to :obj:`.TaskDocument.from_task_files`.
    """

    def __init__(
        self,
        runs=None,
        additional_fields=None,
        task_document_kwargs=None,
    ):
        self.additional_fields = {} if additional_fields is None else additional_fields
        self.runs = runs or ["precondition"] + ["relax" + str(i + 1) for i in range(9)]

        self.task_document_kwargs = task_document_kwargs
        if self.task_document_kwargs is None:
            self.task_document_kwargs = {}

    def assimilate(self, path: Path) -> TaskDocument:
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
        logger.info(f"Getting task doc for base dir :{path}")
        task_files = find_vasp_files(self.runs, path)

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
        import os
        from pathlib import Path

        parent, subdirs, _ = path
        if set(self.runs).intersection(subdirs):
            return [parent]
        if (
            not any([parent.endswith(os.sep + r) for r in self.runs])
            and len(list(Path(parent).glob("vasprun.xml*"))) > 0
        ):
            return [parent]
        return []


def find_vasp_files(
    task_names: List[str],
    path: Path,
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
    from collections import OrderedDict

    path = Path(path)
    task_files = OrderedDict()

    def _get_task_files(files, suffix=""):
        vasp_files = {"volumetric_files": []}
        for file in files:
            if file.match(f"*vasprun{suffix}*"):
                vasp_files["vasprun_file"] = file
            elif file.match(f"*OUTCAR{suffix}*"):
                vasp_files["outcar_file"] = file
            elif any([file.match(f"*{f}{suffix}*") for f in volumetric_files]):
                vasp_files["volumetric_files"].append(file)
        return vasp_files

    for task_name in task_names:
        subfiles = list(path.glob(f"{task_name}/*"))
        if len(subfiles) > 0:
            # subfolder match
            task_files[task_name] = _get_task_files(subfiles)
        else:
            # try extension schema
            subfiles = list(path.glob("*"))
            task_files[task_name] = _get_task_files(subfiles, suffix=f".{task_name}")

    if len(task_files) == 0:
        # get any matching file from the root folder
        task_files["standard"] = _get_task_files(list(path.glob("*")))

    return task_files
