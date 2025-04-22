"""Wrapper to invoke lammps from the CLI."""

import subprocess
from pathlib import Path

from atomate2 import SETTINGS


def run_lammps(
    lammps_input_file: str = "in.lammps",
    lammps_cmd: str = SETTINGS.LAMMPS_CMD,
    lammps_mpi_cmd: str | None = SETTINGS.MPI_CMD,
    lammps_suffix: list[str] | str | None = SETTINGS.LAMMPS_SUFFIX,
    lammps_pks: list[str] | str | None = SETTINGS.LAMMPS_PACKAGES,
    lammps_run_flags: list[str] | str | None = None,
    stdout_file: str | Path = "stdout.log",
    stderr_file: str | Path = "stderr.log",
) -> subprocess.Popen:
    """Run LAMMPS.

    Parameters
    ----------
        lammps_input_file: The path to the main input file to be passed to the
            LAMMPS executable with the `-in` command-line option.
        lammps_cmd: The name or path to the LAMMPS executable.
        lammps_mpi_cmd: The command to invoke MPI (e.g., `'mpirun'` or `'mpiexec'`).
            If None, invoke the `lammps_cmd` in serial mode.
        lammps_suffix: The suffix to use that applies style variants at runtime
            (see `LammpsSettings.LAMMPS_SUFFIX`).
        lammps_pks: The runtime packages and options to tell LAMMPS to use
            (see `LammpsSettings.LAMMPS_PACKAGES`).
        lammps_run_kwargs: Any additional arbitrary flags to invoke LAMMPS with.
        max_walltime_hours: The maximum walltime in hours to allow for the task. If
            provided, attempts will be made to cleanly end the calculation after this
            amount of time. Note: if using with a queueing system, this value should
            leave sufficient time for the clean-up of the calculation within the
            maximum walltime allocated to the job by the queue.
        stdout_file: The name of or path to a file in which to save the stdout stream.
        stderr_file: The name of or path to a file in which to save the stderr stream.

    """
    lammps_invocation: list[str] = []

    if lammps_suffix is not None:
        if isinstance(lammps_suffix, str):
            lammps_suffix = [lammps_suffix]
        for sf in lammps_suffix:
            lammps_invocation += ["-sf", sf]

    if lammps_pks is not None:
        if isinstance(lammps_pks, str):
            lammps_pks = [lammps_pks]
        for pk in lammps_pks:
            lammps_invocation += ["-pk", pk]

    if lammps_run_flags is not None:
        if isinstance(lammps_run_flags, str):
            lammps_run_flags = [lammps_run_flags]
        for flag in lammps_run_flags:
            lammps_invocation += [flag]

    if lammps_mpi_cmd is not None:
        lammps_invocation.extend([lammps_mpi_cmd, "-in", lammps_input_file])
    else:
        lammps_invocation.extend([lammps_cmd, "-in", lammps_input_file])

    with open(stdout_file, "a") as stdout, open(stderr_file, "a") as stderr:
        process = subprocess.Popen(
            lammps_invocation,
            stdout=stdout,
            stderr=stderr,
        )
        process.wait()
        return process
