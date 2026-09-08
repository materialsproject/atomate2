"""Schema definitions for MRGDV task documents."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from emmet.core.structure import StructureMetadata
from jobflow.utils import ValueEnum
from pydantic import Field

from atomate2.abinit.schemas.outfiles import AbinitStoredFile
from atomate2.abinit.utils.common import get_mrgdv_report
from atomate2.utils.path import get_uri

logger = logging.getLogger(__name__)

__all__ = ["MrgdvdbObject", "MrgdvdbTaskDoc", "TaskState"]


class TaskState(ValueEnum):
    """Mrgdv calculation state."""

    SUCCESS = "successful"
    FAILED = "failed"
    UNCONVERGED = "unconverged"


class MrgdvdbObject(ValueEnum):
    """Types of Mrgdvdb data objects."""

    DVDBFILE = "out_DVDB"  # DVDB file as string


class MrgdvdbTaskDoc(StructureMetadata):
    """Task document for an MRGDV job.

    Attributes
    ----------
    dir_name : str or None
        The directory for this MRGDV task.
    completed_at : str or None
        Timestamp for when this task was completed.
    included_objects : list[MrgdvdbObject] or None
        List of MRGDV objects included with this task document.
    mrgdv_objects : dict[MrgdvdbObject, Any] or None
        MRGDV objects associated with this task.
    task_label : str or None
        A description of the task.
    tags : list[str] or None
        Metadata tags for this task document.
    """

    dir_name: str | None = Field(None, description="The directory for this Abinit task")
    completed_at: str | None = Field(
        None, description="Timestamp for when this task was completed"
    )
    included_objects: list[MrgdvdbObject] | None = Field(
        None, description="List of Mrgdv objects included with this task document"
    )
    mrgdv_objects: dict[MrgdvdbObject, Any] | None = Field(
        None, description="Mrgdv objects associated with this task"
    )
    task_label: str | None = Field(None, description="A description of the task")

    tags: list[str] | None = Field(
        None, description="Metadata tags for this task document"
    )

    @classmethod
    def from_directory(
        cls,
        dir_name: Path | str,
        additional_fields: dict[str, Any] | None = None,
    ) -> MrgdvdbTaskDoc:
        """Create a task document from a directory containing MRGDV files.

        Parameters
        ----------
        dir_name : Path or str
            The path to the folder containing the MRGDV calculation outputs.
        additional_fields : dict[str, Any] or None
            Dictionary of additional fields to add to the output document.
            Default is None.

        Returns
        -------
        MrgdvdbTaskDoc
            A task document for the MRGDV calculation.
        """
        logger.info(f"Getting task doc in: {dir_name}")

        if additional_fields is None:
            additional_fields = {}

        dir_name = Path(dir_name)
        task_files = _find_abinit_files(dir_name)

        if len(task_files) == 0:
            raise FileNotFoundError("No Abinit files found!")
        if len(task_files) > 1:
            raise RuntimeError(
                f"Only one mrgdv calculation expected. Found {len(task_files)}"
            )

        std_task_files = next(iter(task_files.values()))
        abinit_mrgdvdb_file = std_task_files["abinit_outdvdb_file"]

        if not abinit_mrgdvdb_file.exists():
            raise RuntimeError(
                f"The output DVDB file {abinit_mrgdvdb_file} does not exist"
            )

        mrgdv_objects: dict[MrgdvdbObject, Any] = {}
        mrgdv_objects[MrgdvdbObject.DVDBFILE] = AbinitStoredFile.from_file(  # type: ignore[index]
            filepath=abinit_mrgdvdb_file,
            data_type=bytes,
        )

        completed_at = str(
            datetime.fromtimestamp(
                os.stat(abinit_mrgdvdb_file).st_mtime, tz=timezone.utc
            )
        )

        report = get_mrgdv_report(logfile=std_task_files["abinit_mrglog_file"])

        if not report["run_completed"]:
            raise RuntimeError("mrgdv execution was not completed")

        tags = additional_fields.pop("tags", None)

        dir_name = get_uri(dir_name)  # convert to full uri path

        included_objects = None
        if mrgdv_objects:
            included_objects = list(mrgdv_objects)

        data = {
            "dir_name": dir_name,
            "completed_at": completed_at,
            "included_objects": included_objects,
            "mrgdv_objects": mrgdv_objects,
            "tags": tags,
        }

        doc = cls(**data)
        return doc.model_copy(update=additional_fields, deep=True)


def _find_abinit_files(
    path: Path | str,
) -> dict[str, Any]:
    """
    Find MRGDV output files in a directory.

    This function searches for MRGDV output files (out_dv and run.log)
    in the specified directory and its outdata subdirectory.

    Parameters
    ----------
    path : Path or str
        The directory to search for MRGDV output files.

    Returns
    -------
    dict[str, Any]
        A dictionary with task files organized by calculation type.
        Keys are task identifiers (e.g., "standard"), values are dicts
        mapping file types to their relative paths.
    """
    path = Path(path)
    task_files = {}

    def _get_task_files(files: list[Path], suffix: str = "") -> dict:
        abinit_files = {}
        for file in files:
            # Here we make assumptions about the output file naming
            if file.match(f"*outdata/out_dv{suffix}*"):
                abinit_files["abinit_outdvdb_file"] = Path(file).relative_to(path)
            elif file.match(f"*run.log{suffix}*"):
                abinit_files["abinit_mrglog_file"] = Path(file).relative_to(path)

        return abinit_files

    # get any matching file from the root folder
    standard_files = _get_task_files(
        list(path.glob("*")) + list(path.glob("outdata/*"))
    )
    if len(standard_files) > 0:
        task_files["standard"] = standard_files

    return task_files
