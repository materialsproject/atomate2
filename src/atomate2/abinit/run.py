"""Functions to run ABINIT."""

from __future__ import annotations

import logging
import subprocess
import time

from atomate2.abinit.schemas.core import AbinitTaskDocument, Status
from atomate2.abinit.utils.common import (
    INPUT_FILE_NAME,
    LOG_FILE_NAME,
    STDERR_FILE_NAME,
)
from atomate2.abinit.utils.history import JobHistory

__all__ = ["run_abinit"]


SLEEP_TIME_STEP = 30


logger = logging.getLogger(__name__)


def run_abinit(
    abinit_cmd: str = "abinit",
    mpirun_cmd: str = None,
    log_file_path: str = LOG_FILE_NAME,
    stderr_file_path: str = STDERR_FILE_NAME,
    walltime: int = None,
):
    """Run ABINIT."""
    start_time = time.process_time()
    max_end_time = 0.0
    if walltime is not None:
        max_end_time = start_time + walltime

    if mpirun_cmd is not None:
        command = [mpirun_cmd, abinit_cmd, INPUT_FILE_NAME]
    else:
        command = [abinit_cmd, INPUT_FILE_NAME]
    with open(log_file_path, "w") as stdout, open(stderr_file_path, "w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)

        if walltime is not None:
            while True:
                time.sleep(SLEEP_TIME_STEP)
                if process.poll() is not None:
                    break
                current_time = time.process_time()
                remaining_time = max_end_time - current_time
                if remaining_time < 5 * SLEEP_TIME_STEP:
                    process.terminate()

        process.wait()


def get_replace_job(
    task_document: AbinitTaskDocument,
    history: JobHistory,
):
    """
    .

    Parameters
    ----------
    task_document : .TaskDocument
        An Abinit task document.
    history : JobHistory
        The history of the Abinit Job.

    Returns
    -------
    bool
        Whether to stop child jobs.
    """
    task_document.event_report
    history.log_end(workdir=task_document.dir_name)
    if task_document.state == Status.SUCCESS:
        # TODO: add convergence of custom parameters (this is used e.g. for
        #  dilatmx convergence)
        return None

    history.log_unconverged()
    # num_restarts = history.num_restarts
    # if num_restarts < self.settings.MAX_RESTARTS:
    #     pass
    #     # new_job = self.get_restart_job(output=output)
    # #                     response.replace = new_job
    #
    #
    # if task_document.state == "successful":
    #     return False
    #
    # if isinstance(handle_unsuccessful, bool):
    #     return handle_unsuccessful
    #
    # if handle_unsuccessful == "error":
    #     raise RuntimeError(
    #         "Job was not successful (perhaps your job did not converge within the "
    #         "limit of electronic/ionic iterations)!"
    #     )
    #
    # raise RuntimeError(f"Unknown option for defuse_unsuccessful: "
    #                    f"{handle_unsuccessful}")


# if self.report is not None:
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
#                     # TODO: check here if we should stop jobflow or children or if
#                     #  we should throw an error.
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
