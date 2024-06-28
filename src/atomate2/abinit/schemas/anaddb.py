"""Core definitions of Abinit calculations documents."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

# from typing import Type, TypeVar, Union, Optional, List
from typing import Any, Optional, Union

import abipy.core.abinit_units as abu
import numpy as np
from abipy.dfpt.anaddbnc import AnaddbNcFile
from abipy.flowtk import events
from abipy.flowtk.utils import File
from emmet.core.structure import StructureMetadata
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure

from atomate2.abinit.schemas.calculation import AbinitObject, TaskState
from atomate2.abinit.utils.common import get_event_report
from atomate2.utils.path import get_uri, strip_hostname

logger = logging.getLogger(__name__)


class CalculationOutput(BaseModel):
    """Document defining Anaddb calculation outputs.

    Parameters
    ----------
    structure: Structure
        The final pymatgen Structure of the system
    dijk: list (3x3x3)
        The conventional static SHG tensor in pm/V (Chi^(2)/2)
    epsinf: list (3x3)
        The electronic contribution to the dielectric tensor
    """

    structure: Union[Structure] = Field(
        None, description="The final structure from the calculation"
    )
    dijk: Optional[list] = Field(
        None, description="Conventional SHG tensor in pm/V (Chi^(2)/2)"
    )
    epsinf: Optional[list] = Field(
        None, description="Electronic contribution to the dielectric tensor"
    )

    @classmethod
    def from_abinit_anaddb(
        cls,
        output: AnaddbNcFile,
    ) -> CalculationOutput:
        """
        Create an Anaddb output document from an AnaddbNcFile.

        Parameters
        ----------
        output: .AnaddbNcFile
            An AnaddbNcFile object.

        Returns
        -------
        The Anaddb calculation output document.
        """
        structure = output.structure
        dijk = list(
            output.dchide * 16 * np.pi**2 * abu.Bohr_Ang**2 * 1e-8 * abu.eps0 / abu.e_Cb
        )  # for pm/V units (SI)
        epsinf = list(output.epsinf)

        return cls(
            structure=structure,
            dijk=dijk,
            epsinf=epsinf,
        )


class Calculation(BaseModel):
    """Full anaddb calculation (inputs) and outputs.

    Parameters
    ----------
    dir_name: str
        The directory for this anaddb calculation
    has_anaddb_completed: .TaskState
        Whether anaddb completed the merge successfully
    output: .CalculationOutput
        The anaddb calculation output
    completed_at: str
        Timestamp for when the merge was completed
    output_file_paths: Dict[str, str]
        Paths (relative to dir_name) of the anaddb output files
        associated with this calculation
    """

    dir_name: str = Field(None, description="The directory for this Abinit calculation")
    has_anaddb_completed: TaskState = Field(
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
        abinit_anaddb_file: Path | str = "out_anaddb.nc",
        abinit_analog_file: Path | str = "run.log",
    ) -> tuple[Calculation, dict[AbinitObject, dict]]:
        """
        Create a anaddb calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name: Path or str
            The directory containing the calculation outputs.
        task_name: str
            The task name.
        abinit_anaddb_file: Path or str
            Path to the merged DDB file, relative to dir_name.
        abinit_analog_file: Path or str
            Path to the main log of anaddb job, relative to dir_name.

        Returns
        -------
        .Calculation
            A anaddb calculation document.
        """
        dir_name = Path(dir_name)
        abinit_anaddb_file = dir_name / abinit_anaddb_file

        output_doc = None
        if abinit_anaddb_file.exists():
            abinit_anaddb = AnaddbNcFile.from_file(abinit_anaddb_file)
            output_doc = CalculationOutput.from_abinit_anaddb(abinit_anaddb)

            completed_at = str(
                datetime.fromtimestamp(os.stat(abinit_anaddb_file).st_mtime)
            )

        report = None
        has_anaddb_completed = TaskState.FAILED
        try:
            report = get_event_report(
                ofile=File(abinit_analog_file), mpiabort_file=File("whatever")
            )
            if report.run_completed or abinit_anaddb_file.exists():
                # VT: abinit_anaddb_file should not be necessary but
                # report.run_completed is False even when it completed...
                has_anaddb_completed = TaskState.SUCCESS

        except Exception as exc:
            msg = f"{cls} exception while parsing event_report:\n{exc}"
            logger.critical(msg)

        return (
            cls(
                dir_name=str(dir_name),
                task_name=task_name,
                has_anaddb_completed=has_anaddb_completed,
                completed_at=completed_at,
                output=output_doc,
                event_report=report,
            ),
            None,  # abinit_objects,
        )


class OutputDoc(BaseModel):
    """Summary of the outputs for a anaddb calculation.

    Parameters
    ----------
    structure: Structure
        The final pymatgen Structure of the final system
    dijk: list (3x3x3)
        The conventional static SHG tensor in pm/V (Chi^(2)/2)
    epsinf: list (3x3)
        The electronic contribution to the dielectric tensor
    """

    structure: Union[Structure] = Field(None, description="The output structure object")
    dijk: Optional[list] = Field(
        None, description="Conventional SHG tensor in pm/V (Chi^(2)/2)"
    )
    epsinf: Optional[list] = Field(
        None, description="Electronic contribution to the dielectric tensor"
    )

    @classmethod
    def from_abinit_calc_doc(cls, calc_doc: Calculation) -> OutputDoc:
        """Create a summary from an abinit CalculationDocument.

        Parameters
        ----------
        calc_doc: .Calculation
            A anaddb calculation document.

        Returns
        -------
        .OutputDoc
            The calculation output summary.
        """
        return cls(
            structure=calc_doc.output.structure,
            dijk=calc_doc.output.dijk,
            epsinf=calc_doc.output.epsinf,
        )


class AnaddbTaskDoc(StructureMetadata):
    """Definition of task document about an anaddb Job.

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
    included_objects: List[.AbinitObject]
        List of Abinit objects included with this task document
    abinit_objects: Dict[.AbinitObject, Any]
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
    included_objects: Optional[list[AbinitObject]] = Field(
        None, description="List of Abinit objects included with this task document"
    )
    abinit_objects: Optional[dict[AbinitObject, Any]] = Field(
        None, description="Abinit objects associated with this task"
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
    ) -> AnaddbTaskDoc:
        """Create a task document from a directory containing Abinit/anaddb files.

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
        .AnaddbTaskDoc
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
        all_abinit_objects = []
        for task_name, files in task_files.items():
            calc_doc, abinit_objects = Calculation.from_abinit_files(
                dir_name, task_name, **files, **abinit_calculation_kwargs
            )
            calcs_reversed.append(calc_doc)
            all_abinit_objects.append(abinit_objects)

        tags = additional_fields.get("tags")

        dir_name = get_uri(dir_name)  # convert to full uri path
        dir_name = strip_hostname(
            dir_name
        )  # VT: TODO to put here?necessary with laptop at least...

        # only store objects from last calculation
        # TODO: make this an option
        abinit_objects = all_abinit_objects[-1]
        included_objects = None
        if abinit_objects:
            included_objects = list(abinit_objects.keys())

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
            "abinit_objects": abinit_objects,
            "calcs_reversed": calcs_reversed,
            "completed_at": calcs_reversed[-1].completed_at,
            "dir_name": dir_name,
            "event_report": calcs_reversed[-1].event_report,
            "included_objects": included_objects,
            # "input": InputDoc.from_abinit_calc_doc(calcs_reversed[0]),
            "meta_structure": calcs_reversed[-1].output.structure,
            "output": OutputDoc.from_abinit_calc_doc(calcs_reversed[-1]),
            "state": calcs_reversed[-1].has_anaddb_completed,
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
            if file.match(f"*outdata/out_anaddb.nc{suffix}*"):
                abinit_files["abinit_anaddb_file"] = Path(file).relative_to(path)
            elif file.match(f"*run.log{suffix}*"):
                abinit_files["abinit_analog_file"] = Path(file).relative_to(path)

        return abinit_files

    # get any matching file from the root folder
    standard_files = _get_task_files(
        list(path.glob("*")) + list(path.glob("outdata/*"))
    )
    if len(standard_files) > 0:
        task_files["standard"] = standard_files

    return task_files
