"""Functions to run ABINIT."""

from __future__ import annotations

import logging
import subprocess
import time

from atomate2.abinit.utils.common import (
    INPUT_FILE_NAME,
    LOG_FILE_NAME,
    STDERR_FILE_NAME,
)

__all__ = ["run_abinit"]


SLEEP_TIME_STEP = 30


logger = logging.getLogger(__name__)


def run_abinit(
    abinit_cmd: str = "abinit",
    mpirun_cmd: str = None,
    log_file_path: str = LOG_FILE_NAME,
    stderr_file_path: str = STDERR_FILE_NAME,
    wall_time: int = None,
):
    """Run ABINIT."""
    start_time = time.process_time()
    max_end_time = 0.0
    if wall_time is not None:
        max_end_time = start_time + wall_time

    if mpirun_cmd is not None:
        command = [mpirun_cmd, abinit_cmd, INPUT_FILE_NAME]
    else:
        command = [abinit_cmd, INPUT_FILE_NAME]
    with open(log_file_path, "w") as stdout, open(stderr_file_path, "w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)

        if wall_time is not None:
            while True:
                time.sleep(SLEEP_TIME_STEP)
                if process.poll() is not None:
                    break
                current_time = time.process_time()
                remaining_time = max_end_time - current_time
                if remaining_time < 5 * SLEEP_TIME_STEP:
                    process.terminate()

        process.wait()
