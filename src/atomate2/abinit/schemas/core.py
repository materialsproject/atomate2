"""Core definitions of Abinit calculations documents."""

import logging
import os
from pathlib import Path
from typing import Type, TypeVar, Union

from abipy.abio.inputs import AbinitInput
from abipy.flowtk import events
from abipy.flowtk.utils import File
from jobflow.utils import ValueEnum
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure

from atomate2.abinit.files import load_abinit_input
from atomate2.abinit.sets.base import AbinitInputSet
from atomate2.abinit.utils.common import (
    LOG_FILE_NAME,
    MPIABORTFILE,
    OUTPUT_FILE_NAME,
    get_event_report,
    get_final_structure,
)
from atomate2.common.schemas.structure import StructureMetadata

_T = TypeVar("_T", bound="AbinitTaskDocument")

logger = logging.getLogger(__name__)


class Status(ValueEnum):
    """Abinit calculation state."""

    # TODO: merge this somewhere with vasp => common calculation schema ?

    SUCCESS = "successful"
    UNCONVERGED = "unconverged"
    FAILED = "failed"


class JobMetadata(BaseModel):
    """Definition of job metadata fields."""

    dir_name: str = Field(None, description="The directory of this job.")
    calc_type: str = Field(None, description="The type of calculation of this job.")


class AbinitJobSummary(JobMetadata):
    """Definition of summary information about an Abinit Job."""

    # restart_info: RestartInfo = Field(
    #     None, description="Restart information for the next job."
    # )
    # history: JobHistory = Field(None, description="Job history.")
    abinit_input_set: AbinitInputSet = Field(
        None, description="AbinitInputSet object used to perform calculation."
    )
    structure: Structure = Field(
        None, description="Final structure of the calculation."
    )
    energy: float = Field(None, description="Final energy of the calculation.")


class AbinitTaskDocument(StructureMetadata):
    """Definition of task document about an Abinit Job."""

    state: Status = Field(None, description="State of this job.")
    run_number: int = Field(None, description="Run number of this job.")
    dir_name: str = Field(None, description="The directory of this job.")
    event_report: events.EventReport = Field(
        None, description="Event report of this abinit job."
    )
    task_label: str = Field(None, description="The label for this job/task.")
    abinit_input: AbinitInput = Field(
        None, description="AbinitInput used to perform calculation."
    )

    @classmethod
    def from_directory(
        cls: Type[_T],
        dir_name: Union[Path, str],
        source: str = "log",
        critical_events=None,
        run_number=1,
        run_status=None,
    ) -> _T:
        """Build AbinitTaskDocument from directory."""
        # Files required for the job analysis.
        # TODO: See if we can put the AbinitInputFile object here from
        #  abipy.abivars.AbinitInputFile (currently not MSONable)
        # input_file = File(os.path.join(dir_name, INPUT_FILE_NAME))
        output_file = File(os.path.join(dir_name, OUTPUT_FILE_NAME))
        log_file = File(os.path.join(dir_name, LOG_FILE_NAME))
        mpiabort_file = File(os.path.join(dir_name, MPIABORTFILE))
        ofile = {"output": output_file, "log": log_file}[source]

        report = None
        # TODO: How to detect which status it has here ?
        #  UNCONVERGED would be for scf/nscf/relax when it's not yet converged
        #  FAILED should be for
        state = Status.UNCONVERGED

        try:
            report = get_event_report(ofile=ofile, mpiabort_file=mpiabort_file)
            critical_events_report = report.filter_types(critical_events)
            if not critical_events_report:
                state = Status.SUCCESS

        except Exception as exc:
            msg = "%s exception while parsing event_report:\n%s" % (cls, exc)
            logger.critical(msg)

        abinit_input = load_abinit_input(dir_name)
        structure = get_final_structure(dir_name=dir_name)

        doc = cls.from_structure(
            structure=structure,
            include_structure=True,
            dir_name=dir_name,
            event_report=report,
            state=state,
            run_number=run_number,
            abinit_input=abinit_input,
        )
        return doc
