"""Core definitions of Abinit calculations documents."""

import logging
import os
from pathlib import Path
from typing import Type, TypeVar, Union

from abipy.abio.inputs import AbinitInput
from abipy.electrons.gsr import GsrFile
from abipy.flowtk import events
from abipy.flowtk.utils import Directory, File
from jobflow.utils import ValueEnum
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure

from atomate2.abinit.files import load_abinit_input
from atomate2.abinit.sets.base import AbinitInputSet
from atomate2.abinit.utils.common import (
    LOG_FILE_NAME,
    MPIABORTFILE,
    OUTDIR_NAME,
    OUTPUT_FILE_NAME,
    PostProcessError,
)
from atomate2.common.schemas.structure import StructureMetadata

_T = TypeVar("_T", bound="AbinitTaskDocument")

logger = logging.getLogger(__name__)


class Status(ValueEnum):
    """Abinit calculation state."""

    # TODO: merge this somewhere with vasp => common calculation schema ?

    SUCCESS = "successful"
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
        structure_fixed=True,
    ):
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
        state = Status.FAILED

        try:
            report = cls.get_event_report(ofile=ofile, mpiabort_file=mpiabort_file)
            critical_events_report = report.filter_types(critical_events)
            if not critical_events_report:
                state = Status.SUCCESS

        except Exception as exc:
            msg = "%s exception while parsing event_report:\n%s" % (cls, exc)
            logger.critical(msg)

        abinit_input = load_abinit_input(dir_name)
        if structure_fixed:
            # Get the structure from the AbinitInput directly
            structure = abinit_input.structure
        else:
            # For relaxations and molecular dynamics, get the final structure from
            # the Gsr file.
            structure = cls.get_final_structure(dir_name)

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

    @staticmethod
    def get_final_structure(dir_name):
        """Get the final structure from the Gsr file."""
        gsr_path = Directory(os.path.join(dir_name, OUTDIR_NAME)).has_abiext("GSR")
        if not gsr_path:
            msg = (
                f"No GSR file available in directory "
                f"{os.path.join(dir_name, OUTDIR_NAME)}."
            )
            logger.critical(msg)
            raise PostProcessError(msg)

        # Open the GSR file.
        try:
            gsr_file = GsrFile(gsr_path)
        except Exception as exc:
            msg = "Exception while reading GSR file at %s:\n%s" % (gsr_path, str(exc))
            logger.critical(msg)
            raise PostProcessError(msg)

        return gsr_file.structure

    @staticmethod
    def get_event_report(ofile, mpiabort_file):
        """Get report from abinit calculation.

        This analyzes the main output file for possible Errors or Warnings.
        It will check the presence of an MPIABORTFILE if not output file is found.

        Parameters
        ----------
        ofile : File
            Output file to be parsed. Should be either the standard abinit
            output or the log file (stdout).
        mpiabort_file : File

        Returns
        -------
        EventReport
            Report of the abinit calculation or None if no output file exists.
        """
        parser = events.EventsParser()

        if not ofile.exists:
            if not mpiabort_file.exists:
                return None
            else:
                # ABINIT abort file without log!
                abort_report = parser.parse(mpiabort_file.path)
                return abort_report

        try:
            report = parser.parse(ofile.path)

            # Add events found in the ABI_MPIABORTFILE.
            if mpiabort_file.exists:
                logger.critical("Found ABI_MPIABORTFILE!")
                abort_report = parser.parse(mpiabort_file.path)
                if len(abort_report) == 0:
                    logger.warning("ABI_MPIABORTFILE but empty")
                else:
                    if len(abort_report) != 1:
                        logger.critical("Found more than one event in ABI_MPIABORTFILE")

                    # Add it to the initial report only if it differs
                    # from the last one found in the main log file.
                    last_abort_event = abort_report[-1]
                    if report and last_abort_event != report[-1]:
                        report.append(last_abort_event)
                    else:
                        report.append(last_abort_event)

            return report

        # except parser.Error as exc:
        except Exception as exc:
            # Return a report with an error entry with info on the exception.
            logger.critical(
                "{}: Exception while parsing ABINIT events:\n {}".format(
                    ofile, str(exc)
                )
            )
            return parser.report_exception(ofile.path, exc)
