"""Definition of base ABINIT job maker."""

from __future__ import annotations

import logging
import os
import time
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Union

import jobflow
import pseudo_dojo
from abipy.flowtk.events import AbinitEvent
from jobflow import Maker, Response, job
from monty.inspect import all_subclasses
from monty.json import jsanitize
from monty.serialization import dumpfn
from monty.string import is_string
from pymatgen.core.structure import Structure
from pymatgen.io.abinit.pseudos import PseudoTable

from atomate2.abinit.run import run_abinit
from atomate2.abinit.schemas.core import AbinitTaskDocument, Status
from atomate2.abinit.sets.base import AbinitInputSetGenerator
from atomate2.abinit.utils.common import (
    LOG_FILE_NAME,
    STDERR_FILE_NAME,
    InitializationError,
    UnconvergedError,
)
from atomate2.abinit.utils.history import JobHistory
from atomate2.abinit.utils.settings import AbinitAtomateSettings, get_abipy_manager

logger = logging.getLogger(__name__)

__all__ = ["BaseAbinitMaker"]


def as_event_class(event_string):
    """Convert event string into a subclass of AbinitEvent.

    The string can be the class name or the YAML tag.
    """
    if is_string(event_string):
        for c in all_subclasses(AbinitEvent):
            if c.__name__ == event_string or c.yaml_tag == event_string:
                return c
        raise ValueError(f"Cannot find event class associated to {event_string}.")
    raise ValueError(
        f"Cannot convert event_string of type {type(event_string)}. Should be a string."
    )


ResponseArgs = namedtuple(
    "ResponseArgs",
    ["detour", "addition", "replace", "stop_children", "stop_jobflow", "stored_data"],
)


@dataclass
class BaseAbinitMaker(Maker):
    """
    Base ABINIT job maker.

    Parameters
    ----------
    name : str
        The job name.
    pseudos : list of str, PseudoTable
        The pseudopotentials to use.
    """

    input_set_generator: AbinitInputSetGenerator
    name: str = "base abinit job"
    pseudos: Union[
        List[str], PseudoTable
    ] = pseudo_dojo.OfficialDojoTable.from_djson_file(
        os.path.join(
            pseudo_dojo.dojotable_absdir("ONCVPSP-PBE-PDv0.4"), "standard.djson"
        )
    )
    walltime: Optional[int] = None
    CRITICAL_EVENTS: Sequence[str] = ()
    dependencies: Optional[dict] = None
    extra_abivars: dict = field(default_factory=dict)

    # class variables
    structure_fixed: ClassVar[bool] = True

    def __post_init__(self):
        """Process post-init configuration."""
        self.critical_events = [
            as_event_class(ce_name) for ce_name in self.CRITICAL_EVENTS
        ]
        if self.input_set_generator is None:
            raise InitializationError("Input set generator is not set.")

    @job
    def make(
        self,
        structure: Optional[Structure] = None,
        prev_outputs: Optional[Any] = None,
        restart_from: Optional[Union[str]] = None,
        history: Optional[JobHistory] = None,
    ) -> Union[jobflow.Flow, jobflow.Job]:
        """
        Return an ABINIT jobflow.Job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_outputs : TODO: add description from sets.base
        restart_from : TODO: add description from sets.base
        history : JobHistory
            A JobHistory object containing the history of this job.
        """
        # Get the start time.
        start_time = time.time()

        if structure is None and prev_outputs is None and restart_from is None:
            raise RuntimeError(
                "At least one of structure, prev_outputs or "
                "restart_from should be defined."
            )

        if history is None:
            # Supposedly the first time the job is created
            history = JobHistory()
        elif restart_from is not None:
            # We want to log the restart only if the restart_from is due to
            # an automatic restart, not a restart from e.g. another scf or relax
            # with different parameters.
            history.log_restart()

        # Set up working directory
        workdir = os.getcwd()

        # Log information about the start of the job
        history.log_start(workdir=workdir, start_time=start_time)

        # Set up logging
        self.set_logger()

        # Load the atomate settings for abinit to get configuration parameters
        # TODO: how to allow for tuned parameters on a per-job basis ?
        #  (similar to fw_spec-passed settings)
        settings = AbinitAtomateSettings()
        get_abipy_manager(settings)

        # set walltime, if possible
        # TODO: see in set_walltime, where to put this walltime_command
        self.set_walltime()

        # Get abinit input
        abinit_input_set = self.get_abinit_input_set(
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
        )

        abinit_input_set.write_input(directory=workdir, make_dir=True, overwrite=False)

        # Run abinit
        self.run_abinit()

        # parse Abinit outputs
        run_number = history.run_number
        task_doc = AbinitTaskDocument.from_directory(
            workdir,
            critical_events=self.critical_events,
            run_number=run_number,
            structure_fixed=self.structure_fixed,
        )
        task_doc.task_label = self.name

        response_args = self.get_response_args(
            task_document=task_doc, history=history, max_restarts=settings.MAX_RESTARTS
        )

        dumpfn(jsanitize(task_doc.dict()), fn="task_document.json", indent=2)

        return Response(
            output=task_doc,
            detour=response_args.detour,
            addition=response_args.addition,
            replace=response_args.replace,
            stop_children=response_args.stop_children,
            stop_jobflow=response_args.stop_jobflow,
            stored_data=response_args.stored_data,
        )

    def set_walltime(self):
        """Set the walltime."""

    @staticmethod
    def set_logger():
        """Set a logger for pymatgen.io.abinit and abipy."""
        log_handler = logging.FileHandler("atomate2_abinit.log")
        log_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        logging.getLogger("pymatgen.io.abinit").addHandler(log_handler)
        logging.getLogger("abipy").addHandler(log_handler)
        logging.getLogger("atomate2").addHandler(log_handler)

    def get_abinit_input_set(
        self,
        structure: Optional[Structure] = None,
        prev_outputs=None,
        restart_from=None,
    ):
        """Set up AbinitInputSet.

        Parameters
        ----------
        structure : Structure
            Structure of this job.
        prev_outputs : TBD
            TBD
        restart_from : TBD
            restart from a directory, from a previous job, from a previous uuid,
            from a previous ...
        """
        gen_kwargs: Dict[str, Any] = {"extra_abivars": self.extra_abivars}

        if restart_from is not None:
            return self.input_set_generator.get_input_set(
                structure=structure,
                restart_from=restart_from,
                prev_outputs=prev_outputs,
                **gen_kwargs,
            )

        if self.input_set_generator is None and restart_from is None:
            raise RuntimeError(
                "Cannot create abinit input set from structure without"
                "input set generator."
            )

        return self.input_set_generator.get_input_set(
            structure=structure,
            restart_from=None,
            prev_outputs=prev_outputs,
            **gen_kwargs,
        )

    def run_abinit(self):
        """Execute abinit."""
        run_abinit(
            abinit_cmd="abinit",
            mpirun_cmd="mpirun",
            log_file_path=LOG_FILE_NAME,
            stderr_file_path=STDERR_FILE_NAME,
        )

    def get_response_args(
        self,
        task_document: AbinitTaskDocument,
        history: JobHistory,
        max_restarts: int = 5,
    ):
        """Get new job to restart abinit calculation."""
        if task_document.state == Status.SUCCESS:
            return ResponseArgs(
                detour=None,
                addition=None,
                replace=None,
                stop_children=False,
                stop_jobflow=False,
                stored_data=None,
            )

        if history.run_number > max_restarts:
            # TODO: check here if we should stop jobflow or children or
            #  if we should throw an error.
            unconverged_error = UnconvergedError(
                self,
                msg="Unconverged after {} runs.".format(history.run_number),
                abinit_input=task_document.abinit_input,
                history=history,
            )
            return ResponseArgs(
                detour=None,
                addition=None,
                replace=None,
                stop_children=True,
                stop_jobflow=True,
                stored_data={"error": unconverged_error},
            )

        logger.info(msg="Getting restart job.")

        new_job = self.make(
            structure=task_document.structure,
            restart_from=task_document.dir_name,
            history=history,
        )

        return ResponseArgs(
            detour=None,
            addition=None,
            replace=new_job,
            stop_children=False,
            stop_jobflow=False,
            stored_data=None,
        )
