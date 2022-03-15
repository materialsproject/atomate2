"""Job history related objects."""

import collections
import logging
import os
import traceback

from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.util.serialization import pmg_serialize

from atomate2.abinit.utils.common import OUTDIR_NAME

logger = logging.getLogger(__name__)


class JobHistory(collections.deque, MSONable):
    """History class for tracking the creation and actions performed during a job.

    The objects provided should be MSONable, thus dicts, lists or MSONable objects.
    The expected items are dictionaries, transformations, corrections, autoparal, restart/reset
    and initializations (when factories will be handled).
    Possibly, the first item should contain information about the starting point of the job.
    This object will be forwarded during job restarts and resets, in order to keep track of the
    full history of the job.
    """

    @pmg_serialize
    def as_dict(self):
        """Create dictionary representation of the history."""
        items = [i.as_dict() if hasattr(i, "as_dict") else i for i in self]
        return dict(items=items)

    @classmethod
    def from_dict(cls, d):
        """Create instance of the history from its dictionary representation."""
        dec = MontyDecoder()
        return cls([dec.process_decoded(i) for i in d["items"]])

    def log_initialization(self, job, initialization_info=None):
        """Log initialization information about the job."""
        details = {"job_class": job.__class__.__name__}
        if initialization_info:
            details["initialization_info"] = initialization_info
        self.append(JobEvent(JobEvent.INITIALIZED, details=details))

    def log_corrections(self, corrections):
        """Log corrections applied to the job."""
        self.append(JobEvent(JobEvent.CORRECTIONS, corrections))

    def log_restart(self):
        """Log that the job is restarted."""
        self.append(
            JobEvent(
                JobEvent.RESTART,
            )
        )

    def log_start(self, workdir):
        """Log that the job has started."""
        self.append(
            JobEvent(
                JobEvent.START,
                details=dict(workdir=workdir),
            )
        )

    def log_end(self, workdir):
        """Log that the job has ended."""
        self.append(
            JobEvent(
                JobEvent.END,
                details=dict(workdir=workdir),
            )
        )

    @property
    def num_restarts(self):
        """Get the number of restarts of the job."""
        # TODO: what happens if a job starts but does not end (e.g. killed by walltime)
        #  how should we could the number of restarts ? do we need both START and RESTART events ? One could assume
        #  there is one "start" and the following are restarts.
        return len(
            self.get_events_by_types(JobEvent.RESTART)
        )  # [event for event in self if event.event_type == JobEvent.RESTART])

    @property
    def prev_dir(self):
        """Get the last run directory."""
        return os.path.join(
            self.get_events_by_types(JobEvent.END)[-1].details["workdir"]
        )

    @property
    def prev_outdir(self):
        """Get the output directory of the last run."""
        return os.path.join(self.prev_dir, OUTDIR_NAME)

    @property
    def is_first_run(self):
        """Determine if it is the first run of the job from the history."""
        nstart = len(self.get_events_by_types(JobEvent.START))
        if nstart == 0:
            raise RuntimeError(
                "Calling is_first_run when the start of the job has not been logged."
            )
        return nstart == 1

    def log_autoparal(self, optconf):
        """Log autoparal execution."""
        self.append(JobEvent(JobEvent.AUTOPARAL, details={"optconf": optconf}))

    def log_unconverged(self):
        """Log that the job is not converged."""
        self.append(JobEvent(JobEvent.UNCONVERGED))

    def log_finalized(self, final_input=None):
        """Log that the job is finalized."""
        details = dict(total_run_time=self.get_total_run_time())
        if final_input:
            details["final_input"] = final_input
        self.append(JobEvent(JobEvent.FINALIZED, details=details))

    def log_converge_params(self, unconverged_params, abiinput):
        """Log the change of user-defined convergence parameters.

        An example is the convergence with respect to dilatmx when relaxing a structure.
        """
        params = {}
        for param, new_value in unconverged_params.items():
            params[param] = dict(
                old_value=abiinput.get(param, "Default"), new_value=new_value
            )
        self.append(JobEvent(JobEvent.UNCONVERGED_PARAMS, details={"params": params}))

    def log_error(self, exc):
        """Log an error in the job."""
        tb = traceback.format_exc()
        event_details = dict(stacktrace=tb)
        # If the exception is serializable, save its details
        try:
            exception_details = exc.to_dict()
        except AttributeError:
            exception_details = None
        except BaseException as e:
            logger.error("Exception couldn't be serialized: {} ".format(e))
            exception_details = None
        if exception_details:
            event_details["exception_details"] = exception_details
        self.append(JobEvent(JobEvent.ERROR, details=event_details))

    def log_abinit_stop(self, run_time=None):
        """Log that abinit has stopped."""
        self.append(JobEvent(JobEvent.ABINIT_STOP, details={"run_time": run_time}))

    def get_events_by_types(self, types):
        """Return the events in history of the selected types.

        Parameters
        ----------
        types
            Single type or list of types.
        """
        types = types if isinstance(types, (list, tuple)) else [types]

        events = [e for e in self if e.event_type in types]

        return events

    def get_total_run_time(self):
        """Get the total run time based summing the run times saved in the abinit stop event."""
        total_run_time = 0
        for te in self.get_events_by_types(JobEvent.ABINIT_STOP):
            run_time = te.details.get("run_time", None)
            if run_time:
                total_run_time += run_time

        return total_run_time


class JobEvent(MSONable):
    """Object used to categorize the events in the JobHistory."""

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

    def __init__(self, event_type, details=None):
        # TODO: add when and where the JobEvent occurred ?
        """Construct JobEvent object."""
        self.event_type = event_type
        self.details = details

    @pmg_serialize
    def as_dict(self):
        """Create dictionary representation of the job event."""
        d = dict(event_type=self.event_type)
        if self.details:
            d["details"] = jsanitize(self.details, strict=True)

        return d

    @classmethod
    def from_dict(cls, d):
        """Create instance of the job event from its dictionary representation."""
        dec = MontyDecoder()
        details = dec.process_decoded(d["details"]) if "details" in d else None
        return cls(event_type=d["event_type"], details=details)
