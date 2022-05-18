"""Core definitions of Abinit calculations documents."""

import logging
import os
from pathlib import Path
from typing import Type, TypeVar, Union

from abipy.flowtk import events
from abipy.flowtk.utils import File
from jobflow.utils import ValueEnum
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure

from atomate2.abinit.sets.base import AbinitInputSet
from atomate2.abinit.utils.common import LOG_FILE_NAME, MPIABORTFILE, OUTPUT_FILE_NAME
from atomate2.common.schemas.structure import StructureMetadata

# from abipy.flowtk.events import EventReport


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

    @classmethod
    def from_directory(
        cls: Type[_T],
        dir_name: Union[Path, str],
        source: str = "log",
        critical_events=None,
        run_number=1,
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

        return cls(dir_name=dir_name, event_report=report, state=state)

    # def job_analysis(self):
    #     """Perform analysis of abinit job."""
    #     self.report = None
    #     try:
    #         self.report = self.get_event_report()
    #     except Exception as exc:
    #         msg = "%s exception while parsing event_report:\n%s" % (self, exc)
    #         logger.critical(msg)
    #
    #     output = AbinitJobSummary(
    #         calc_type=self.calc_type,
    #         dir_name=os.getcwd(),
    #         abinit_input_set=self.abinit_input_set,
    #         structure=self.get_final_structure(),
    #     )
    #     response = Response(output=output)
    #
    #     if self.report is not None:
    #         # the calculation finished without errors
    #         if self.report.run_completed:
    #             self.history.log_end(workdir=self.workdir)
    #             # Check if the calculation converged.
    #             # TODO: where do we define whether a given critical event
    #             #  allows for a restart ?
    #             #  here we seem to assume that we can always restart because it is
    #             #  something unconverged (be it e.g. scf or relaxation)
    #             not_ok = self.report.filter_types(self.critical_events)
    #             if not_ok:
    #                 self.history.log_unconverged()
    #                 num_restarts = self.history.num_restarts
    #                 # num_restarts = (
    #                 #     self.restart_info.num_restarts if self.restart_info else 0
    #                 # )
    #                 if num_restarts < self.settings.MAX_RESTARTS:
    #                     new_job = self.get_restart_job(output=output)
    #                     response.replace = new_job
    #                 else:
    #                     # TODO: check here if we should stop jobflow or children or
    #                     #  if we should throw an error.
    #                     response.stop_jobflow = True
    #                     # response.stop_children = True
    #                     unconverged_error = UnconvergedError(
    #                         self,
    #                         msg="Unconverged after {} restarts.".format(num_restarts),
    #                         abinit_input=self.abinit_input_set.abinit_input,
    #                         # restart_info=self.restart_info,
    #                         history=self.history,
    #                     )
    #                     response.stored_data = {"error": unconverged_error}
    #                     raise unconverged_error
    #             else:
    #                 # calculation converged
    #                 # everything is ok. conclude the job
    #                 # TODO: add convergence of custom parameters (this is used e.g.
    #                 #  for dilatmx convergence)
    #                 response.output.energy = self.get_final_energy()
    #                 stored_data = self.conclude_task()
    #                 response.stored_data = stored_data
    #     else:
    #         # TODO: add possible fixes here ? (no errors from abinit)
    #         raise NotImplementedError("")
    #
    #     return response

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
