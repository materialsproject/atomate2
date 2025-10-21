"""Core definitions of Abinit calculations documents."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abipy.dfpt.ddb import DdbFile
from emmet.core.structure import StructureMetadata
from jobflow.utils import ValueEnum
from pydantic import Field
from pymatgen.core.structure import Structure

from atomate2.abinit.schemas.outfiles import AbinitStoredFile
from atomate2.abinit.utils.common import get_mrgddb_report
from atomate2.utils.path import get_uri

logger = logging.getLogger(__name__)


class TaskState(ValueEnum):
    """Mrgddb calculation state."""

    SUCCESS = "successful"
    FAILED = "failed"
    UNCONVERGED = "unconverged"


class MrgddbObject(ValueEnum):
    """Types of Mrgddb data objects."""

    DDBFILE = "ddb"  # DDB file as string


class MrgddbTaskDoc(StructureMetadata, extra="allow"):  # type: ignore[call-arg]
    """Definition of task document about an Mrgddb Job.

    Parameters
    ----------
    dir_name: str
        The directory for this Abinit task
    completed_at: str
        Timestamp for when this task was completed
    structure: Structure
        Final output structure from the task
    state: .TaskState
        State of this task
    included_objects: List[.MrgddbObject]
        List of Abinit objects included with this task document
    abinit_objects: Dict[.MrgddbObject, Any]
        Abinit objects associated with this task
    tags: List[str]
        Metadata tags for this task document
    """

    dir_name: str | None = Field(None, description="The directory for this Abinit task")
    completed_at: str | None = Field(
        None, description="Timestamp for when this task was completed"
    )
    structure: Structure | None = Field(
        None, description="Final output atoms from the task"
    )
    included_objects: list[MrgddbObject] | None = Field(
        None, description="List of Mrgddb objects included with this task document"
    )
    mrgddb_objects: dict[MrgddbObject, Any] | None = Field(
        None, description="Mrgddb objects associated with this task"
    )
    tags: list[str] | None = Field(
        None, description="Metadata tags for this task document"
    )

    @classmethod
    def from_directory(
        cls,
        dir_name: Path | str,
        additional_fields: dict[str, Any] | None = None,
    ) -> MrgddbTaskDoc:
        """Create a task document from a directory containing Abinit/Mrgddb files.

        Parameters
        ----------
        dir_name: Path or str
            The path to the folder containing the calculation outputs.
        additional_fields: Dict[str, Any]
            Dictionary of additional fields to add to output document.

        Returns
        -------
        .MrgddbTaskDoc
            A task document for the calculation.
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
                f"Only one mrgddb calculation expected. Found {len(task_files)}"
            )

        std_task_files = next(iter(task_files.values()))

        abinit_outddb_file = std_task_files["abinit_outddb_file"]

        if not abinit_outddb_file.exists():
            raise RuntimeError(
                f"The output DDB file {abinit_outddb_file} does not exist"
            )

        mrgddb_objects: dict[MrgddbObject, Any] = {}
        abinit_outddb = DdbFile.from_file(abinit_outddb_file)
        structure = abinit_outddb.structure
        mrgddb_objects[MrgddbObject.DDBFILE] = AbinitStoredFile.from_file(  # type: ignore[index]
            filepath=abinit_outddb_file, data_type=str
        )

        completed_at = str(
            datetime.fromtimestamp(
                os.stat(abinit_outddb_file).st_mtime, tz=timezone.utc
            )
        )

        report = get_mrgddb_report(logfile=std_task_files["abinit_mrglog_file"])

        if not report["run_completed"]:
            raise RuntimeError("mrgddb execution was not completed")

        tags = additional_fields.pop("tags", None)

        dir_name = get_uri(dir_name)  # convert to full uri path

        included_objects = None
        if mrgddb_objects:
            included_objects = list(mrgddb_objects)

        data = {
            "dir_name": dir_name,
            "completed_at": completed_at,
            "included_objects": included_objects,
            "mrgddb_objects": mrgddb_objects,
            "tags": tags,
        }

        return cls.from_structure(
            structure=structure, meta_structure=structure, **data, **additional_fields
        )


def _find_abinit_files(
    path: Path | str,
) -> dict[str, Any]:
    """Find Abinit files."""
    path = Path(path)
    task_files = {}

    def _get_task_files(files: list[Path], suffix: str = "") -> dict:
        abinit_files = {}
        for file in files:
            # Here we make assumptions about the output file naming
            if file.match(f"*outdata/out_DDB{suffix}*"):
                abinit_files["abinit_outddb_file"] = Path(file).relative_to(path)
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
