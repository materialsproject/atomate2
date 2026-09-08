"""Module with common file names and classes used for ABINIT flows."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from abipy.abio.outputs import AbinitOutputFile
from abipy.dfpt.ddb import DdbFile
from abipy.electrons.gsr import GsrFile
from abipy.flowtk import events
from abipy.flowtk.utils import Directory, File
from monty.json import MSONable
from monty.serialization import MontyDecoder

from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
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
MRGDDB_INPUT_FILE_NAME: str = "mrgddb.in"
MRGDV_INPUT_FILE_NAME: str = "mrgdv.in"
ANADDB_INPUT_FILE_NAME: str = "anaddb.in"
MPIABORTFILE = "__ABI_MPIABORTFILE__"
DUMMY_FILENAME = "__DUMMY__"
ELPHON_OUTPUT_FILE_NAME = "run.abo_elphon"
DDK_FILES_FILE_NAME = "ddk.files"
HISTORY_JSON = "history.json"


logger = logging.getLogger(__name__)

__all__ = [
    "ANADDB_INPUT_FILE_NAME",
    "DDK_FILES_FILE_NAME",
    "DUMMY_FILENAME",
    "ELPHON_OUTPUT_FILE_NAME",
    "HISTORY_JSON",
    "INDATAFILE_PREFIX",
    "INDATA_PREFIX",
    "INDIR_NAME",
    "INPUT_FILE_NAME",
    "LOG_FILE_NAME",
    "MPIABORTFILE",
    "MRGDDB_INPUT_FILE_NAME",
    "MRGDV_INPUT_FILE_NAME",
    "OUTDATAFILE_PREFIX",
    "OUTDATA_PREFIX",
    "OUTDIR_NAME",
    "OUTNC_FILE_NAME",
    "OUTPUT_FILE_NAME",
    "STDERR_FILE_NAME",
    "TMPDATAFILE_PREFIX",
    "TMPDATA_PREFIX",
    "TMPDIR_NAME",
    "AbiAtomateError",
    "AbinitRuntimeError",
    "ErrorCode",
    "InitializationError",
    "PostProcessError",
    "RestartError",
    "RestartInfo",
    "UnconvergedError",
    "WalltimeError",
    "get_event_report",
    "get_final_structure",
    "get_mrgddb_report",
    "get_mrgdv_report",
]


class ErrorCode:
    """
    Error codes for classifying ABINIT calculation errors.

    Attributes
    ----------
    ERROR : str
        Generic error code.
    UNRECOVERABLE : str
        Error that cannot be recovered from.
    UNCLASSIFIED : str
        Error that has not been classified.
    UNCONVERGED : str
        Calculation did not converge.
    UNCONVERGED_PARAMETERS : str
        Calculation parameters did not converge.
    INITIALIZATION : str
        Error during job initialization.
    RESTART : str
        Error during restart procedure.
    POSTPROCESS : str
        Error during post-processing.
    WALLTIME : str
        Calculation exceeded walltime limit.
    """

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
    """
    Base class for ABINIT errors in atomate2.

    Attributes
    ----------
    ERROR_CODE : str
        The error code classification. Default is ErrorCode.ERROR.
    msg : str
        The error message.
    """

    ERROR_CODE = ErrorCode.ERROR

    def __init__(self, msg: str) -> None:
        """
        Initialize an AbiAtomateError.

        Parameters
        ----------
        msg : str
            The error message.
        """
        super().__init__(msg)
        self.msg = msg

    def to_dict(self) -> dict:
        """
        Create dictionary representation of the error.

        Returns
        -------
        dict
            Dictionary containing error_code and msg.
        """
        return {"error_code": self.ERROR_CODE, "msg": self.msg}


class AbinitRuntimeError(AbiAtomateError):
    """
    Exception raised for errors during ABINIT calculation.

    Contains information about errors and warnings extracted from the output
    files. Can be initialized with a job to automatically extract error
    information from its report.

    Attributes
    ----------
    ERROR_CODE : str
        The error code classification. Default is ErrorCode.ERROR.
    job : Job or Flow or None
        The atomate2 job or flow.
    num_errors : int or None
        Number of errors in the ABINIT execution.
    num_warnings : int or None
        Number of warnings in the ABINIT execution.
    errors : list or None
        List of error events from the ABINIT execution.
    warnings : list or None
        List of warning events from the ABINIT execution.
    msg : str or None
        The error message.
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
        """
        Initialize an AbinitRuntimeError.

        If the job has a report, all the information will be extracted from it,
        otherwise the provided arguments will be used.

        Parameters
        ----------
        job : Job or Flow or None
            The atomate2 job or flow. Default is None.
        msg : str or None
            The error message. Default is None.
        num_errors : int or None
            Number of errors in the ABINIT execution. Only used if job doesn't
            have a report. Default is None.
        num_warnings : int or None
            Number of warnings in the ABINIT execution. Only used if job doesn't
            have a report. Default is None.
        errors : list or None
            List of errors in the ABINIT execution. Only used if job doesn't
            have a report. Default is None.
        warnings : list or None
            List of warnings in the ABINIT execution. Only used if job doesn't
            have a report. Default is None.
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
        """
        Create dictionary representation of the error.

        Returns
        -------
        dict
            Dictionary containing error information including num_errors,
            num_warnings, errors, warnings, error_message, error_code,
            @module, and @class.
        """
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
        """
        Create dictionary representation of the error.

        Returns
        -------
        dict
            Dictionary representation of the error (same as to_dict()).
        """
        return self.to_dict()

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """
        Create instance of the error from its dictionary representation.

        Parameters
        ----------
        d : dict
            Dictionary representation of the error containing num_errors,
            num_warnings, and optionally errors, warnings, and error_message.

        Returns
        -------
        Self
            An instance of AbinitRuntimeError reconstructed from the dictionary.
        """
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
    """
    Exception raised when a calculation didn't converge after maximum restarts.

    Contains additional information about the last input, restart information,
    and job history that can be used to further restart the job.

    Attributes
    ----------
    ERROR_CODE : str
        The error code classification. Set to ErrorCode.UNCONVERGED.
    abinit_input : AbinitInput or None
        The last AbinitInput used.
    restart_info : RestartInfo or None
        The RestartInfo required to restart the job.
    history : JobHistory or None
        The history of the job.
    """

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
        """
        Initialize an UnconvergedError.

        If the job has a report, all the information will be extracted from it,
        otherwise the provided arguments will be used. Contains information that
        can be used to further restart the job.

        Parameters
        ----------
        job : Job or Flow or None
            The atomate2 job or flow. Default is None.
        msg : str or None
            The error message. Default is None.
        num_errors : int or None
            Number of errors in the ABINIT execution. Only used if job doesn't
            have a report. Default is None.
        num_warnings : int or None
            Number of warnings in the ABINIT execution. Only used if job doesn't
            have a report. Default is None.
        errors : list or None
            List of errors in the ABINIT execution. Only used if job doesn't
            have a report. Default is None.
        warnings : list or None
            List of warnings in the ABINIT execution. Only used if job doesn't
            have a report. Default is None.
        abinit_input : AbinitInput or None
            The last AbinitInput used. Default is None.
        restart_info : RestartInfo or None
            The RestartInfo required to restart the job. Default is None.
        history : JobHistory or None
            The history of the job. Default is None.
        """
        super().__init__(job, msg, num_errors, num_warnings, errors, warnings)
        self.abinit_input = abinit_input
        self.restart_info = restart_info
        self.history = history

    def to_dict(self) -> dict:
        """
        Create dictionary representation of the error.

        Returns
        -------
        dict
            Dictionary containing all parent class information plus
            abinit_input, restart_info, history, @module, and @class.
        """
        dct = super().to_dict()
        dct["abinit_input"] = self.abinit_input.as_dict() if self.abinit_input else None
        dct["restart_info"] = self.restart_info.as_dict() if self.restart_info else None
        dct["history"] = self.history.as_dict() if self.history else None
        dct["@module"] = type(self).__module__
        dct["@class"] = type(self).__name__
        return dct

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """
        Create instance of the error from its dictionary representation.

        Parameters
        ----------
        d : dict
            Dictionary representation of the error containing num_errors,
            num_warnings, and optionally errors, warnings, error_message,
            abinit_input, restart_info, and history.

        Returns
        -------
        Self
            An instance of UnconvergedError reconstructed from the dictionary.
        """
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
    """
    Exception raised when the calculation didn't complete in time.

    Attributes
    ----------
    ERROR_CODE : str
        The error code classification. Set to ErrorCode.WALLTIME.
    """

    ERROR_CODE = ErrorCode.WALLTIME


class InitializationError(AbiAtomateError):
    """
    Exception raised if errors are present during the initialization of the job.

    Attributes
    ----------
    ERROR_CODE : str
        The error code classification. Set to ErrorCode.INITIALIZATION.
    """

    ERROR_CODE = ErrorCode.INITIALIZATION


class RestartError(InitializationError):
    """
    Exception raised if errors show up during the setup of the restart.

    Attributes
    ----------
    ERROR_CODE : str
        The error code classification. Set to ErrorCode.RESTART.
    """

    ERROR_CODE = ErrorCode.RESTART


class PostProcessError(AbiAtomateError):
    """
    Exception raised if problems are encountered during the post-processing.

    Attributes
    ----------
    ERROR_CODE : str
        The error code classification. Set to ErrorCode.POSTPROCESS.
    """

    ERROR_CODE = ErrorCode.POSTPROCESS


class RestartInfo(MSONable):
    """
    Object that contains information about the restart of a job.

    Attributes
    ----------
    previous_dir : Path or str
        Directory path of the previous calculation.
    num_restarts : int
        Number of times the job has been restarted. Default is 0.
    """

    def __init__(self, previous_dir: Path | str, num_restarts: int = 0) -> None:
        """
        Initialize a RestartInfo object.

        Parameters
        ----------
        previous_dir : Path or str
            Directory path of the previous calculation.
        num_restarts : int
            Number of times the job has been restarted. Default is 0.
        """
        self.previous_dir = previous_dir
        self.num_restarts = num_restarts

    def as_dict(self) -> dict:
        """
        Create dictionary representation of the restart information.

        Returns
        -------
        dict
            Dictionary containing previous_dir, num_restarts, @module, and @class.
        """
        return {
            "previous_dir": self.previous_dir,
            "num_restarts": self.num_restarts,
            "@module": type(self).__module__,
            "@class": type(self).__name__,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """
        Create instance from its dictionary representation.

        Parameters
        ----------
        d : dict
            Dictionary representation containing previous_dir and num_restarts.

        Returns
        -------
        Self
            A RestartInfo instance reconstructed from the dictionary.
        """
        return cls(
            previous_dir=d["previous_dir"],
            num_restarts=d["num_restarts"],
        )

    @property
    def prev_outdir(self) -> Directory:
        """
        Get the Directory pointing to the output directory of the previous step.

        Returns
        -------
        Directory
            Directory object for the previous output directory.
        """
        return Directory(os.path.join(self.previous_dir, OUTDIR_NAME))

    @property
    def prev_indir(self) -> Directory:
        """
        Get the Directory pointing to the input directory of the previous step.

        Returns
        -------
        Directory
            Directory object for the previous input directory.
        """
        return Directory(os.path.join(self.previous_dir, INDIR_NAME))


def get_final_structure(dir_name: Path | str) -> Structure:
    """
    Get the final structure of a calculation in a given directory.

    This function tries to get the structure from multiple sources in order:
    1. From the GSR file (out_GSR.nc).
    2. From the DDB file (out_DDB).
    3. From the ABINIT output file (run.abo).

    Parameters
    ----------
    dir_name : Path or str
        Directory containing the calculation output files.

    Returns
    -------
    Structure
        The final structure from the calculation.

    Raises
    ------
    RuntimeError
        If the final structure could not be retrieved from any source.
    """
    dir_name = strip_hostname(dir_name)
    gsr_path = Directory(os.path.join(dir_name, OUTDIR_NAME)).has_abiext("GSR")
    if gsr_path:
        try:
            gsr_file = GsrFile(gsr_path)
        except Exception:
            logging.exception("Exception occurred")  # noqa: LOG015
        else:
            return gsr_file.structure

    ddb_path = Directory(os.path.join(dir_name, OUTDIR_NAME)).has_abiext("DDB")
    if ddb_path:
        try:
            ddb_file = DdbFile(ddb_path)
        except Exception:
            logging.exception("Exception occurred")  # noqa: LOG015
        else:
            return ddb_file.structure

    out_path = File(os.path.join(dir_name, OUTPUT_FILE_NAME))
    if out_path.exists:
        try:
            ab_out = AbinitOutputFile.from_file(out_path.path)
        except Exception:
            logging.exception("Exception occurred")  # noqa: LOG015
        else:
            return ab_out.final_structure

    raise RuntimeError("Could not get final structure.")


def get_event_report(
    ofile: File, mpiabort_file: File | None = None
) -> EventReport | None:
    """
    Get report from an ABINIT calculation.

    Analyzes the main output file for possible Errors or Warnings. Will check
    the presence of an MPIABORTFILE if no output file is found.

    Parameters
    ----------
    ofile : File
        Output file to be parsed. Should be either the standard ABINIT
        output (run.abo) or the log file (stdout).
    mpiabort_file : File or None
        The MPI abort file to check for errors. Default is None.

    Returns
    -------
    EventReport or None
        Report of the ABINIT calculation, or None if no output file exists.
    """
    parser = events.EventsParser()

    if not ofile.exists:
        if not mpiabort_file or not mpiabort_file.exists:
            return None
        return parser.parse(mpiabort_file.path)

    try:
        report = parser.parse(ofile.path)

        if mpiabort_file and mpiabort_file.exists:
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
    except (ValueError, RuntimeError, Exception) as exc:  # noqa: BLE001
        logger.critical(f"{ofile}: Exception while parsing ABINIT events:\n {exc!s}")
        return parser.report_exception(ofile.path, exc)
    else:
        return report


def get_mrgddb_report(
    logfile: str | Path,
) -> dict:
    """
    Get report from MRGDDB utility.

    Returns a dict with a "run_completed" key that is True if
    "completed successfully" is present in the log.

    Parameters
    ----------
    logfile : str or Path
        Log file to be parsed (stdout from MRGDDB).

    Returns
    -------
    dict
        Dictionary with boolean "run_completed" key.
    """
    if not Path(logfile).exists:
        return {"run_completed": False}
    with open(str(logfile)) as f:
        last_line = f.readlines()[-1]
    return {"run_completed": "completed successfully" in last_line}


def get_mrgdv_report(
    logfile: str | Path,
) -> dict:
    """
    Get report from MRGDV utility.

    Returns a dict with a "run_completed" key that is True if
    "Done" is present in the log.

    Parameters
    ----------
    logfile : str or Path
        Log file to be parsed (stdout from MRGDV).

    Returns
    -------
    dict
        Dictionary with boolean "run_completed" key.
    """
    if not Path(logfile).exists:
        return {"run_completed": False}
    with open(str(logfile)) as f:
        last_line = f.readlines()[-1]
    return {"run_completed": "Done" in last_line}
