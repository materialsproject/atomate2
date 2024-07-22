"""Core definitions of Abinit calculations documents."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from abipy.dfpt.ddb import DdbError, DdbFile
from abipy.flowtk import events
from abipy.flowtk.utils import File
from emmet.core.structure import StructureMetadata
from jobflow.utils import ValueEnum
from monty.json import MSONable
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure
from typing_extensions import Self

from atomate2.abinit.utils.common import get_event_report
from atomate2.utils.path import get_uri, strip_hostname

logger = logging.getLogger(__name__)


class TaskState(ValueEnum):
    """Mrgddb calculation state."""

    SUCCESS = "successful"
    FAILED = "failed"
    UNCONVERGED = "unconverged"


# need to inherit from MSONable to be stored in the data store
# I tried to combine it with @dataclass, but didn't work...
class DdbFileStr(MSONable):
    """Object storing the raw string of a DDB file."""

    def __init__(self, ddbfilepath: str | Path, ddb_as_str: str) -> None:
        self.ddbfilepath: str | Path = ddbfilepath
        self.ddb_as_str: str = ddb_as_str

    @classmethod
    def from_ddbfile(cls, ddbfile: DdbFile) -> Self:
        """Create a DdbFileStr object from the native DdbFile abipy object."""
        with open(ddbfile.filepath) as f:
            ddb_as_str = f.read()
        return cls(ddbfilepath=ddbfile.filepath, ddb_as_str=ddb_as_str)


class MrgddbObject(ValueEnum):
    """Types of Mrgddb data objects."""

    DDBFILESTR = "ddb"  # DDB file as string


class CalculationOutput(BaseModel):
    """Document defining Mrgddb calculation outputs.

    Parameters
    ----------
    structure: Structure
        The final pymatgen Structure of the system
    """

    structure: Union[Structure] = Field(
        None, description="The final structure from the calculation"
    )

    @classmethod
    def from_abinit_outddb(
        cls,
        output: DdbFile,
    ) -> CalculationOutput:
        """
        Create an Mrgddb output document from the merged Abinit out_DDB file.

        Parameters
        ----------
        output: .DdbFile
            A DdbFile object.

        Returns
        -------
        The Mrgddb calculation output document.
        """
        structure = output.structure

        return cls(
            structure=structure,
        )


class Calculation(BaseModel):
    """Full Mrgddb calculation (inputs) and outputs.

    Parameters
    ----------
    dir_name: str
        The directory for this Mrgddb calculation
    ddb_version: str
        DDB version
    has_mrgddb_completed: .TaskState
        Whether Mrgddb completed the merge successfully
    output: .CalculationOutput
        The Mrgddb calculation output
    completed_at: str
        Timestamp for when the merge was completed
    output_file_paths: Dict[str, str]
        Paths (relative to dir_name) of the Mrgddb output files
        associated with this calculation
    """

    dir_name: str = Field(None, description="The directory for this Abinit calculation")
    ddb_version: str = Field(
        None, description="Abinit version used to perform the calculation"
    )
    has_mrgddb_completed: TaskState = Field(
        None, description="Whether Abinit completed the calculation successfully"
    )
    output: Optional[CalculationOutput] = Field(
        None, description="The Abinit calculation output"
    )
    completed_at: str = Field(
        None, description="Timestamp for when the calculation was completed"
    )
    event_report: events.EventReport = Field(
        None, description="Event report of this abinit job."
    )
    output_file_paths: Optional[dict[str, str]] = Field(
        None,
        description="Paths (relative to dir_name) of the Abinit output files "
        "associated with this calculation",
    )

    @classmethod
    def from_abinit_files(
        cls,
        dir_name: Path | str,
        task_name: str,
        abinit_outddb_file: Path | str = "out_DDB",
        abinit_mrglog_file: Path | str = "run.log",
    ) -> tuple[Calculation, dict[MrgddbObject, dict]]:
        """
        Create a Mrgddb calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name: Path or str
            The directory containing the calculation outputs.
        task_name: str
            The task name.
        abinit_outddb_file: Path or str
            Path to the merged DDB file, relative to dir_name.
        abinit_mrglog_file: Path or str
            Path to the main log of mrgddb job, relative to dir_name.

        Returns
        -------
        .Calculation
            A Mrgddb calculation document.
        """
        dir_name = Path(dir_name)
        abinit_outddb_file = dir_name / abinit_outddb_file

        output_doc = None
        mrgddb_objects: dict[MrgddbObject, Any] = {}
        if abinit_outddb_file.exists():
            abinit_outddb = DdbFile.from_file(abinit_outddb_file)
            mrgddb_objects[MrgddbObject.DDBFILESTR] = DdbFileStr.from_ddbfile(  # type: ignore[index]
                abinit_outddb
            )
            output_doc = CalculationOutput.from_abinit_outddb(abinit_outddb)

            completed_at = str(
                datetime.fromtimestamp(
                    os.stat(abinit_outddb_file).st_mtime, tz=timezone.utc
                )
            )

        report = None
        has_mrgddb_completed = TaskState.FAILED
        try:
            report = get_event_report(
                ofile=File(abinit_mrglog_file), mpiabort_file=File("whatever")
            )
            if report.run_completed or abinit_outddb_file.exists():
                # VT: abinit_outddb_file should not be necessary but
                # report.run_completed is False even when it completed...
                has_mrgddb_completed = TaskState.SUCCESS

        except (DdbError, Exception) as exc:
            msg = f"{cls} exception while parsing event_report:\n{exc}"
            logger.critical(msg)

        return (
            cls(
                dir_name=str(dir_name),
                task_name=task_name,
                ddb_version=str(abinit_outddb.version),
                has_mrgddb_completed=has_mrgddb_completed,
                completed_at=completed_at,
                output=output_doc,
                event_report=report,
            ),
            mrgddb_objects,
        )


class OutputDoc(BaseModel):
    """Summary of the outputs for a Mrgddb calculation.

    Parameters
    ----------
    structure: Structure
        The final pymatgen Structure of the final system
    """

    structure: Union[Structure] = Field(None, description="The output structure object")

    @classmethod
    def from_abinit_calc_doc(cls, calc_doc: Calculation) -> OutputDoc:
        """Create a summary from an abinit CalculationDocument.

        Parameters
        ----------
        calc_doc: .Calculation
            A Mrgddb calculation document.

        Returns
        -------
        .OutputDoc
            The calculation output summary.
        """
        return cls(
            structure=calc_doc.output.structure,
        )


class MrgddbTaskDoc(StructureMetadata):
    """Definition of task document about an Mrgddb Job.

    Parameters
    ----------
    dir_name: str
        The directory for this Abinit task
    completed_at: str
        Timestamp for when this task was completed
    output: .OutputDoc
        The output of the final calculation
    structure: Structure
        Final output structure from the task
    state: .TaskState
        State of this task
    included_objects: List[.MrgddbObject]
        List of Abinit objects included with this task document
    abinit_objects: Dict[.MrgddbObject, Any]
        Abinit objects associated with this task
    task_label: str
        A description of the task
    tags: List[str]
        Metadata tags for this task document
    """

    dir_name: Optional[str] = Field(
        None, description="The directory for this Abinit task"
    )
    completed_at: Optional[str] = Field(
        None, description="Timestamp for when this task was completed"
    )
    output: Optional[OutputDoc] = Field(
        None, description="The output of the final calculation"
    )
    structure: Union[Structure] = Field(
        None, description="Final output atoms from the task"
    )
    state: Optional[TaskState] = Field(None, description="State of this task")
    event_report: Optional[events.EventReport] = Field(
        None, description="Event report of this abinit job."
    )
    included_objects: Optional[list[MrgddbObject]] = Field(
        None, description="List of Mrgddb objects included with this task document"
    )
    mrgddb_objects: Optional[dict[MrgddbObject, Any]] = Field(
        None, description="Mrgddb objects associated with this task"
    )
    task_label: Optional[str] = Field(None, description="A description of the task")
    tags: Optional[list[str]] = Field(
        None, description="Metadata tags for this task document"
    )

    @classmethod
    def from_directory(
        cls,
        dir_name: Path | str,
        additional_fields: dict[str, Any] = None,
        **abinit_calculation_kwargs,
    ) -> MrgddbTaskDoc:
        """Create a task document from a directory containing Abinit/Mrgddb files.

        Parameters
        ----------
        dir_name: Path or str
            The path to the folder containing the calculation outputs.
        additional_fields: Dict[str, Any]
            Dictionary of additional fields to add to output document.
        **abinit_calculation_kwargs
            Additional parsing options that will be passed to the
            :obj:`.Calculation.from_abinit_files` function.

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

        calcs_reversed = []
        all_mrgddb_objects = []
        for task_name, files in task_files.items():
            calc_doc, mrgddb_objects = Calculation.from_abinit_files(
                dir_name, task_name, **files, **abinit_calculation_kwargs
            )
            calcs_reversed.append(calc_doc)
            all_mrgddb_objects.append(mrgddb_objects)

        tags = additional_fields.get("tags")

        dir_name = get_uri(dir_name)  # convert to full uri path
        dir_name = strip_hostname(
            dir_name
        )  # VT: TODO to put here?necessary with laptop at least...

        # only store objects from last calculation
        # TODO: make this an option
        mrgddb_objects = all_mrgddb_objects[-1]
        included_objects = None
        if mrgddb_objects:
            included_objects = list(mrgddb_objects)

        # rewrite the original structure save!

        if isinstance(calcs_reversed[-1].output.structure, Structure):
            attr = "from_structure"
            dat = {
                "structure": calcs_reversed[-1].output.structure,
                "meta_structure": calcs_reversed[-1].output.structure,
                "include_structure": True,
            }
        doc = getattr(cls, attr)(**dat)
        ddict = doc.dict()

        data = {
            "calcs_reversed": calcs_reversed,
            "completed_at": calcs_reversed[-1].completed_at,
            "dir_name": dir_name,
            "event_report": calcs_reversed[-1].event_report,
            "included_objects": included_objects,
            "mrgddb_objects": mrgddb_objects,
            # "input": InputDoc.from_abinit_calc_doc(calcs_reversed[0]),
            "meta_structure": calcs_reversed[-1].output.structure,
            "output": OutputDoc.from_abinit_calc_doc(calcs_reversed[-1]),
            "state": calcs_reversed[-1].has_mrgddb_completed,
            "structure": calcs_reversed[-1].output.structure,
            "tags": tags,
        }

        doc = cls(**ddict)
        doc = doc.model_copy(update=data)
        return doc.model_copy(update=additional_fields, deep=True)


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
