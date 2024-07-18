"""Module with common file names and classes used for Abinit flows."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from abipy.abio.outputs import AbinitOutputFile
from abipy.dfpt.ddb import DdbFile
from abipy.electrons.gsr import GsrFile
from abipy.flowtk import events
from abipy.flowtk.utils import Directory, File
from monty.json import MSONable
from monty.serialization import MontyDecoder

if TYPE_CHECKING:
    from pathlib import Path

    from abipy.abio.inputs import AbinitInput
    from abipy.core.structure import Structure
    from abipy.flowtk.events import EventReport
    from jobflow import Flow, Job
    from typing_extensions import Self

    from atomate2.abinit.utils.history import JobHistory

TMPDIR_NAME = "tmpdata"
OUTDIR_NAME = "outdata"
INDIR_NAME = "indata"
TMPDATAFILE_PREFIX = "tmp"
OUTDATAFILE_PREFIX = "out"
INDATAFILE_PREFIX = "in"
TMPDATA_PREFIX = os.path.join(TMPDIR_NAME, TMPDATAFILE_PREFIX)
OUTDATA_PREFIX = os.path.join(OUTDIR_NAME, OUTDATAFILE_PREFIX)
INDATA_PREFIX = os.path.join(INDIR_NAME, INDATAFILE_PREFIX)
STDERR_FILE_NAME = "run.err"
LOG_FILE_NAME = "run.log"
OUTPUT_FILE_NAME = "run.abo"
OUTNC_FILE_NAME = "out_OUT.nc"
INPUT_FILE_NAME: str = "run.abi"
MPIABORTFILE = "__ABI_MPIABORTFILE__"
DUMMY_FILENAME = "__DUMMY__"
ELPHON_OUTPUT_FILE_NAME = "run.abo_elphon"
DDK_FILES_FILE_NAME = "ddk.files"
HISTORY_JSON = "history.json"


logger = logging.getLogger(__name__)


class ErrorCode:
    """Error code to classify the errors."""

    ERROR = "Error"
    UNRECOVERABLE = "Unrecoverable"
    UNCLASSIFIED = "Unclassified"
    UNCONVERGED = "Unconverged"
    UNCONVERGED_PARAMETERS = "Unconverged_parameters"
    INITIALIZATION = "Initialization"
    RESTART = "Restart"
    POSTPROCESS = "Postprocess"
    WALLTIME = "Walltime"


class AbiAtomateError(Exception):
    """Base class for the abinit errors in atomate."""

    ERROR_CODE = ErrorCode.ERROR

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.msg = msg

    def to_dict(self) -> dict:
        """Create dictionary representation of the error."""
        return {"error_code": self.ERROR_CODE, "msg": self.msg}


class AbinitRuntimeError(AbiAtomateError):
    """Exception raised for errors during Abinit calculation.

    Contains the information about the errors and warning extracted from
    the output files.
    Initialized with a job, uses it to prepare a suitable error message.
    """

    ERROR_CODE = ErrorCode.ERROR

    def __init__(
        self,
        job: Job | Flow | None = None,
        msg: str | None = None,
        num_errors: int | None = None,
        num_warnings: int | None = None,
        errors: list | None = None,
        warnings: list | None = None,
    ) -> None:
        """Construct AbinitRuntimeError object.

        If the job has a report all the information will be extracted from it,
        otherwise the arguments will be used.

        Parameters
        ----------
        job
            the atomate2 job
        msg
            the error message
        num_errors
            number of errors in the abinit execution. Only used if job doesn't
            have a report.
        num_warnings
            number of warning in the abinit execution. Only used if job doesn't
            have a report.
        errors
            list of errors in the abinit execution. Only used if job doesn't
            have a report.
        warnings
            list of warnings in the abinit execution. Only used if job doesn't
            have a report.
        """
        # This can handle both the cases of DECODE_MONTY=True and False
        # (Since it has a from_dict method).
        super().__init__(msg)
        self.job = job
        if (
            self.job is not None
            and hasattr(self.job, "report")
            and self.job.report is not None
        ):
            report = self.job.report
            self.num_errors = report.num_errors
            self.num_warnings = report.num_warnings
            self.errors = report.errors
            self.warnings = report.warnings
        else:
            self.num_errors = num_errors
            self.num_warnings = num_warnings
            self.errors = errors
            self.warnings = warnings
        self.msg = msg

    def to_dict(self) -> dict:
        """Create dictionary representation of the error."""
        dct = {"num_errors": self.num_errors, "num_warnings": self.num_warnings}
        if self.errors:
            errors = [error.as_dict() for error in self.errors]
            dct["errors"] = errors
        if self.warnings:
            warnings = [warning.as_dict() for warning in self.warnings]
            dct["warnings"] = warnings
        if self.msg:
            dct["error_message"] = self.msg

        dct["error_code"] = self.ERROR_CODE
        dct["@module"] = type(self).__module__
        dct["@class"] = type(self).__name__

        return dct

    def as_dict(self) -> dict:
        """Create dictionary representation of the error."""
        return self.to_dict()

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Create instance of the error from its dictionary representation."""
        dec = MontyDecoder()
        warnings = (
            [dec.process_decoded(w) for w in d["warnings"]] if "warnings" in d else []
        )
        errors = [dec.process_decoded(w) for w in d["errors"]] if "errors" in d else []
        msg = d.get("error_message")

        return cls(
            warnings=warnings,
            errors=errors,
            num_errors=d["num_errors"],
            num_warnings=d["num_warnings"],
            msg=msg,
        )


class UnconvergedError(AbinitRuntimeError):
    """Exception raised when a calculation didn't converge after the max restarts."""

    ERROR_CODE = ErrorCode.UNCONVERGED

    def __init__(
        self,
        job: Job | Flow | None = None,
        msg: str | None = None,
        num_errors: int | None = None,
        num_warnings: int | None = None,
        errors: list | None = None,
        warnings: list | None = None,
        abinit_input: AbinitInput | None = None,
        restart_info: RestartInfo | None = None,
        history: JobHistory | None = None,
    ) -> None:
        """Construct UnconvergedError object.

        If the job has a report all the information will be extracted from it,
        otherwise the arguments will be used.
        It contains information that can be used to further restart the job.

        Parameters
        ----------
        job
            the atomate2 job
        msg
            the error message
        num_errors
            number of errors in the abinit execution. Only used if job doesn't
            have a report.
        num_warnings
            number of warning in the abinit execution. Only used if job doesn't
            have a report.
        errors
            list of errors in the abinit execution. Only used if job doesn't
            have a report.
        warnings
            list of warnings in the abinit execution. Only used if job doesn't
            have a report.
        abinit_input
            the last AbinitInput used.
        restart_info
            the RestartInfo required to restart the job.
        history
            The history of the job.
        """
        super().__init__(job, msg, num_errors, num_warnings, errors, warnings)
        self.abinit_input = abinit_input
        self.restart_info = restart_info
        self.history = history

    def to_dict(self) -> dict:
        """Create dictionary representation of the error."""
        dct = super().to_dict()
        dct["abinit_input"] = self.abinit_input.as_dict() if self.abinit_input else None
        dct["restart_info"] = self.restart_info.as_dict() if self.restart_info else None
        dct["history"] = self.history.as_dict() if self.history else None
        dct["@module"] = type(self).__module__
        dct["@class"] = type(self).__name__
        return dct

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Create instance of the error from its dictionary representation."""
        dec = MontyDecoder()
        warnings = (
            [dec.process_decoded(w) for w in d["warnings"]] if "warnings" in d else []
        )
        errors = [dec.process_decoded(w) for w in d["errors"]] if "errors" in d else []
        if "abinit_input" in d and d["abinit_input"] is not None:
            abinit_input = dec.process_decoded(d["abinit_input"])
        else:
            abinit_input = None
        if "restart_info" in d and d["restart_info"] is not None:
            restart_info = dec.process_decoded(d["restart_info"])
        else:
            restart_info = None
        if "history" in d and d["history"] is not None:
            history = dec.process_decoded(d["history"])
        else:
            history = None
        return cls(
            warnings=warnings,
            errors=errors,
            num_errors=d["num_errors"],
            num_warnings=d["num_warnings"],
            msg=d["error_message"],
            abinit_input=abinit_input,
            restart_info=restart_info,
            history=history,
        )


class WalltimeError(AbiAtomateError):
    """Exception raised when the calculation didn't complete in time."""

    ERROR_CODE = ErrorCode.WALLTIME


class InitializationError(AbiAtomateError):
    """Exception raised if errors are present during the initialization of the job."""

    ERROR_CODE = ErrorCode.INITIALIZATION


class RestartError(InitializationError):
    """Exception raised if errors show up during the set up of the restart."""

    ERROR_CODE = ErrorCode.RESTART


class PostProcessError(AbiAtomateError):
    """Exception raised if problems are encountered during the post processing."""

    ERROR_CODE = ErrorCode.POSTPROCESS


class RestartInfo(MSONable):
    """Object that contains the information about the restart of a job."""

    def __init__(self, previous_dir: Path | str, num_restarts: int = 0) -> None:
        self.previous_dir = previous_dir
        # self.reset = reset
        self.num_restarts = num_restarts

    def as_dict(self) -> dict:
        """Create dictionary representation of the error."""
        return {
            "previous_dir": self.previous_dir,
            # "reset": self.reset,
            "num_restarts": self.num_restarts,
            "@module": type(self).__module__,
            "@class": type(self).__name__,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Create instance of the error from its dictionary representation."""
        return cls(
            previous_dir=d["previous_dir"],
            # reset=d["reset"],
            num_restarts=d["num_restarts"],
        )

    @property
    def prev_outdir(self) -> Directory:
        """Get the Directory pointing to the output directory of the previous step."""
        return Directory(os.path.join(self.previous_dir, OUTDIR_NAME))

    @property
    def prev_indir(self) -> Directory:
        """Get the Directory pointing to the input directory of the previous step."""
        return Directory(os.path.join(self.previous_dir, INDIR_NAME))


def get_final_structure(dir_name: Path | str) -> Structure:
    """Get the final/last structure of a calculation in a given directory.

    This functions tries to get the structure:
    1. from the output file of abinit (run.abo).
    2. from the gsr file of abinit (out_GSR.nc).
    """
    gsr_path = Directory(os.path.join(dir_name, OUTDIR_NAME)).has_abiext("GSR")
    if gsr_path:
        # Open the GSR file.
        try:
            gsr_file = GsrFile(gsr_path)
        except Exception:
            logging.exception("Exception occurred")
        else:
            return gsr_file.structure

    ddb_path = Directory(os.path.join(dir_name, OUTDIR_NAME)).has_abiext("DDB")
    if ddb_path:
        # Open the GSR file.
        try:
            ddb_file = DdbFile(ddb_path)
        except Exception:
            logging.exception("Exception occurred")
        else:
            return ddb_file.structure

    out_path = File(os.path.join(dir_name, OUTPUT_FILE_NAME))
    if out_path.exists:
        try:
            ab_out = AbinitOutputFile.from_file(out_path.path)
        except Exception:
            logging.exception("Exception occurred")
        else:
            return ab_out.final_structure

    raise RuntimeError("Could not get final structure.")


def get_event_report(ofile: File, mpiabort_file: File) -> EventReport | None:
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
        # ABINIT abort file without log!

        return parser.parse(mpiabort_file.path)

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
    except (ValueError, RuntimeError, Exception) as exc:
        # Return a report with an error entry with info on the exception.
        logger.critical(f"{ofile}: Exception while parsing ABINIT events:\n {exc!s}")
        return parser.report_exception(ofile.path, exc)
    else:
        return report
