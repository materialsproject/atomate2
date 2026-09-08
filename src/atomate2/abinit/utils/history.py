"""Objects for tracking ABINIT job history and events."""

from __future__ import annotations

import collections
import logging
import os
import traceback
from typing import TYPE_CHECKING, Any

from monty.json import MontyDecoder, MSONable, jsanitize

from atomate2.abinit.utils.common import OUTDIR_NAME

if TYPE_CHECKING:
    from pathlib import Path

    from abipy.abio.inputs import AbinitInput
    from abipy.flowtk.events import AbinitEvent
    from abipy.flowtk.utils import Directory
    from jobflow import Flow, Job
    from typing_extensions import Self

logger = logging.getLogger(__name__)

__all__ = [
    "JobEvent",
    "JobHistory",
]


class JobHistory(collections.deque, MSONable):
    """
    History class for tracking the creation and actions performed during a job.

    This class extends collections.deque to provide a chronological record of
    events that occur during an ABINIT job execution, including initialization,
    corrections, restarts, and finalization. All stored objects should be
    MSONable (dicts, lists, or MSONable objects).

    The history is forwarded during job restarts and resets to maintain a
    complete record of the job's lifecycle.

    Notes
    -----
    Expected items include dictionaries, transformations, corrections, autoparal
    results, restart/reset events, and initializations. The first item typically
    contains information about the starting point of the job.
    """

    def as_dict(self) -> dict:
        """
        Create dictionary representation of the history.

        Returns
        -------
        dict
            Dictionary containing items list, @module, and @class.
        """
        items = [i.as_dict() if hasattr(i, "as_dict") else i for i in self]

        return {
            "items": items,
            "@module": type(self).__module__,
            "@class": type(self).__name__,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """
        Create instance of the history from its dictionary representation.

        Parameters
        ----------
        d : dict
            Dictionary representation containing items list.

        Returns
        -------
        Self
            A JobHistory instance reconstructed from the dictionary.
        """
        dec = MontyDecoder()
        return cls([dec.process_decoded(i) for i in d["items"]])

    def log_initialization(
        self, job: Job | Flow, initialization_info: Any | None = None
    ) -> None:
        """
        Log initialization information about the job.

        Parameters
        ----------
        job : Job or Flow
            The atomate2 job or flow being initialized.
        initialization_info : Any or None
            Additional initialization information. Default is None.
        """
        details = {"job_class": type(job).__name__}
        if initialization_info:
            details["initialization_info"] = initialization_info
        self.append(JobEvent(JobEvent.INITIALIZED, details=details))

    def log_corrections(self, corrections: Any | None) -> None:
        """
        Log corrections applied to the job.

        Parameters
        ----------
        corrections : Any or None
            Correction information to be logged.
        """
        self.append(JobEvent(JobEvent.CORRECTIONS, corrections))

    def log_restart(self) -> None:
        """Log that the job is restarted."""
        self.append(
            JobEvent(
                JobEvent.RESTART,
            )
        )

    def log_start(self, workdir: Path | str | Directory, start_time: Any) -> None:
        """
        Log that the job has started.

        Parameters
        ----------
        workdir : Path or str or Directory
            Working directory of the job.
        start_time : Any
            Start time of the job.
        """
        self.append(
            JobEvent(
                JobEvent.START,
                details={"workdir": workdir, "start_time": start_time},
            )
        )

    def log_end(self, workdir: Path | str | Directory) -> None:
        """
        Log that the job has ended.

        Parameters
        ----------
        workdir : Path or str or Directory
            Working directory of the job.
        """
        self.append(
            JobEvent(
                JobEvent.END,
                details={"workdir": workdir},
            )
        )

    @property
    def num_restarts(self) -> int:
        """
        Get the number of restarts of the job.

        Returns
        -------
        int
            Number of times the job has been restarted.

        Notes
        -----
        This counts only explicit RESTART events. Consider that if a job starts
        but does not end (e.g., killed by walltime), the restart accounting may
        not reflect all execution attempts. The relationship between START and
        RESTART events is that typically there is one initial START, and
        subsequent executions are logged as RESTARTs.
        """
        return len(self.get_events_by_types(JobEvent.RESTART))

    @property
    def run_number(self) -> int:
        """
        Get the number of the run.

        Returns
        -------
        int
            Number of times the job has been started.
        """
        return len(self.get_events_by_types(JobEvent.START))

    @property
    def prev_dir(self) -> str:
        """
        Get the last run directory.

        Returns
        -------
        str
            Path to the working directory of the most recent completed run.
        """
        return os.path.join(
            self.get_events_by_types(JobEvent.END)[-1].details["workdir"]
        )

    @property
    def prev_dirs(self) -> list[str]:
        """
        Get all previous run directories.

        Returns
        -------
        list[str]
            List of paths to all working directories except the current one.
        """
        return [
            os.path.join(ievent.details["workdir"])
            for ievent in self.get_events_by_types(JobEvent.START)[:-1]
        ]

    @property
    def prev_outdir(self) -> str:
        """
        Get the output directory of the last run.

        Returns
        -------
        str
            Path to the output directory of the most recent run.
        """
        return os.path.join(self.prev_dir, OUTDIR_NAME)

    @property
    def is_first_run(self) -> bool:
        """
        Determine if this is the first run of the job from the history.

        Returns
        -------
        bool
            True if this is the first run, False otherwise.

        Raises
        ------
        RuntimeError
            If called when the start of the job has not been logged yet.
        """
        nstart = len(self.get_events_by_types(JobEvent.START))
        if nstart == 0:
            raise RuntimeError(
                "Calling is_first_run when the start of the job has not been logged."
            )
        return nstart == 1

    def log_autoparal(self, optconf: Any) -> None:
        """
        Log autoparal execution.

        Parameters
        ----------
        optconf : Any
            Optimal configuration determined by autoparal.
        """
        self.append(JobEvent(JobEvent.AUTOPARAL, details={"optconf": optconf}))

    def log_unconverged(self) -> None:
        """Log that the job has not converged."""
        self.append(JobEvent(JobEvent.UNCONVERGED))

    def log_finalized(self, final_input: AbinitInput | None = None) -> None:
        """
        Log that the job is finalized.

        Parameters
        ----------
        final_input : AbinitInput or None
            The final ABINIT input used. Default is None.
        """
        details = {"total_run_time": self.get_total_run_time()}
        if final_input:
            details["final_input"] = final_input
        self.append(JobEvent(JobEvent.FINALIZED, details=details))

    def log_converge_params(
        self, unconverged_params: dict, abiinput: AbinitInput
    ) -> None:
        """
        Log the change of user-defined convergence parameters.

        Parameters
        ----------
        unconverged_params : dict
            Dictionary of parameter names and their new values.
        abiinput : AbinitInput
            The ABINIT input containing the old parameter values.

        Notes
        -----
        An example is the convergence with respect to dilatmx when relaxing
        a structure.
        """
        params = {}
        for param, new_value in unconverged_params.items():
            params[param] = {
                "old_value": abiinput.get(param, "Default"),
                "new_value": new_value,
            }
        self.append(JobEvent(JobEvent.UNCONVERGED_PARAMS, details={"params": params}))

    def log_error(self, exc: Any) -> None:
        """
        Log an error in the job.

        Parameters
        ----------
        exc : Any
            The exception that occurred.
        """
        tb = traceback.format_exc()
        event_details = {"stacktrace": tb}
        # If the exception is serializable, save its details
        try:
            exception_details = exc.to_dict()
        except AttributeError:
            exception_details = None
        except BaseException:
            logger.exception("Exception couldn't be serialized")
            exception_details = None
        if exception_details:
            event_details["exception_details"] = exception_details
        self.append(JobEvent(JobEvent.ERROR, details=event_details))

    def log_abinit_stop(self, run_time: Any | None = None) -> None:
        """
        Log that ABINIT has stopped.

        Parameters
        ----------
        run_time : Any or None
            The run time of the ABINIT execution. Default is None.
        """
        self.append(JobEvent(JobEvent.ABINIT_STOP, details={"run_time": run_time}))

    def get_events_by_types(self, types: list | AbinitEvent) -> list:
        """
        Return the events in history of the selected types.

        Parameters
        ----------
        types : list or AbinitEvent
            Single event type or list of event types to filter.

        Returns
        -------
        list
            List of JobEvent objects matching the specified types.
        """
        types = types if isinstance(types, list | tuple) else [types]

        return [e for e in self if e.event_type in types]

    def get_total_run_time(self) -> Any:
        """
        Get the total run time by summing ABINIT stop event run times.

        Returns
        -------
        Any
            Total run time accumulated across all ABINIT executions.
        """
        total_run_time = 0
        for te in self.get_events_by_types(JobEvent.ABINIT_STOP):
            run_time = te.details.get("run_time")
            if run_time:
                total_run_time += run_time

        return total_run_time


class JobEvent(MSONable):
    """
    Object used to categorize events in JobHistory.

    Attributes
    ----------
    INITIALIZED : str
        Event type for job initialization.
    CORRECTIONS : str
        Event type for corrections applied.
    START : str
        Event type for job start.
    END : str
        Event type for job end.
    RESTART : str
        Event type for job restart.
    AUTOPARAL : str
        Event type for autoparal execution.
    UNCONVERGED : str
        Event type for unconverged job.
    FINALIZED : str
        Event type for job finalization.
    UNCONVERGED_PARAMS : str
        Event type for unconverged parameter changes.
    ERROR : str
        Event type for errors.
    ABINIT_STOP : str
        Event type for ABINIT stop.
    event_type : AbinitEvent
        The type of this event.
    details : Any or None
        Additional details about the event.
    """

    INITIALIZED = "initialized"
    CORRECTIONS = "corrections"
    START = "start"
    END = "end"
    RESTART = "restart"
    AUTOPARAL = "autoparal"
    UNCONVERGED = "unconverged"
    FINALIZED = "finalized"
    UNCONVERGED_PARAMS = "unconverged parameters"
    ERROR = "error"
    ABINIT_STOP = "abinit stop"

    def __init__(self, event_type: AbinitEvent, details: Any | None = None) -> None:
        """
        Initialize a JobEvent object.

        Parameters
        ----------
        event_type : AbinitEvent
            The type of event being logged.
        details : Any or None
            Additional details about the event. Default is None.

        Notes
        -----
        Future enhancement could include timestamp and location information
        about when and where the JobEvent occurred.
        """
        self.event_type = event_type
        self.details = details

    def as_dict(self) -> dict:
        """
        Create dictionary representation of the job event.

        Returns
        -------
        dict
            Dictionary containing event_type, details (if present),
            @module, and @class.
        """
        dct = {"event_type": self.event_type}
        if self.details:
            dct["details"] = jsanitize(self.details, strict=True)
        dct["@module"] = type(self).__module__
        dct["@class"] = type(self).__name__
        return dct

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """
        Create instance of the job event from its dictionary representation.

        Parameters
        ----------
        d : dict
            Dictionary representation containing event_type and optionally details.

        Returns
        -------
        Self
            A JobEvent instance reconstructed from the dictionary.
        """
        dec = MontyDecoder()
        details = dec.process_decoded(d["details"]) if "details" in d else None
        return cls(event_type=d["event_type"], details=details)
