"""SIESTA job for custodian execution.

This module provides a Job class that integrates SIESTA calculations
with the custodian library's error handling framework.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from custodian.custodian import Job

logger = logging.getLogger(__name__)


class SiestaJob(Job):
    """SIESTA job for custodian framework.

    This class inherits from custodian.custodian.Job and implements
    the required methods (setup, run, postprocess) for SIESTA calculations.

    The custodian library will handle:
    - Error detection via ErrorHandlers
    - Automatic correction application
    - Retry logic
    - Validation

    Parameters
    ----------
    siesta_cmd : str
        SIESTA command to execute
    output_file : str, optional
        Output file name (default: "siesta.out")
    suffix : str, optional
        Job name suffix for identification (default: "")
    backup_files : list[str], optional
        Files to backup before corrections (default: ["siesta.fdf"])

    Example
    -------
    >>> from custodian import Custodian
    >>> from atomate2.siesta.custodian.jobs import SiestaJob
    >>> from atomate2.siesta.custodian.handlers import DEFAULT_HANDLERS
    >>>
    >>> job = SiestaJob(siesta_cmd="siesta < siesta.fdf > siesta.out")
    >>> custodian = Custodian(
    ...     handlers=DEFAULT_HANDLERS,
    ...     jobs=[job],
    ...     max_errors=5,
    ... )
    >>> custodian.run()
    """

    def __init__(
        self,
        siesta_cmd: str,
        output_file: str = "siesta.out",
        suffix: str = "",
        backup_files: list[str] | None = None,
    ) -> None:
        """Initialize SiestaJob.

        Parameters
        ----------
        siesta_cmd : str
            SIESTA command to execute
        output_file : str, optional
            Output file name (default: "siesta.out")
        suffix : str, optional
            Job name suffix (default: "")
        backup_files : list[str], optional
            Files to backup (default: ["siesta.fdf"])
        """
        self.siesta_cmd = siesta_cmd
        self.output_file = output_file
        self.suffix = suffix
        self.backup_files = backup_files or ["siesta.fdf"]

    def setup(self, directory: str = "./") -> None:
        """Pre-job setup.

        Custodian calls this before running the job.
        We don't need to do anything here since input files
        are already written by the maker.

        Parameters
        ----------
        directory : str, optional
            Working directory (default: "./")
        """
        dir_path = Path(directory)
        logger.info(f"SiestaJob setup in {dir_path}")
        logger.info(f"Command: {self.siesta_cmd}")

    def run(self, directory: str = "./") -> subprocess.Popen:
        """Execute SIESTA calculation.

        Custodian calls this to run the job. We return a Popen
        process to enable monitoring (if handlers have is_monitor=True).

        Parameters
        ----------
        directory : str, optional
            Working directory (default: "./")

        Returns
        -------
        subprocess.Popen
            Running process (for monitoring support)
        """
        dir_path = Path(directory)
        logger.info(f"Running SIESTA in {dir_path}")

        # Environment with LUA_PATH derived from FLOS_PATH when needed
        # (Lua-driven runs: NEB, Lua relaxation scripts)
        from atomate2.siesta.run import get_siesta_run_env

        # Start SIESTA process
        # Return Popen for monitoring support
        return subprocess.Popen(  # noqa: S602 custodian must run the user's siesta command
            self.siesta_cmd,
            shell=True,
            cwd=dir_path,
            env=get_siesta_run_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def postprocess(self, directory: str = "./") -> None:
        """Post-job processing.

        Custodian calls this after successful job completion.
        We can do cleanup, analysis, etc. here.

        Parameters
        ----------
        directory : str, optional
            Working directory (default: "./")
        """
        dir_path = Path(directory)
        logger.info(f"SiestaJob postprocess in {dir_path}")

        # Optional: Could compress files here
        # Optional: Could do additional validation
        # For now, just log completion

    @property
    def name(self) -> str:
        """Job name for logging.

        Returns
        -------
        str
            Job identifier
        """
        return f"siesta{self.suffix}"


# Backward compatibility: Keep CustodianJob as alias for now
# Will be deprecated in favor of using Custodian orchestrator directly
CustodianJob = SiestaJob


def run_custodian_job(
    siesta_cmd: str,
    handlers: list | None = None,
    validators: list | None = None,
    max_errors: int = 5,
    directory: str = "./",
) -> None:
    """Run SIESTA with custodian (convenience wrapper).

    This is a helper function that creates a Custodian orchestrator
    and runs a SIESTA job with error handling.

    Parameters
    ----------
    siesta_cmd : str
        SIESTA command to execute
    handlers : list[ErrorHandler], optional
        Error handlers (default: DEFAULT_HANDLERS)
    validators : list[Validator], optional
        Output validators (default: [SiestaOutputValidator()])
    max_errors : int, optional
        Maximum errors allowed (default: 5)
    directory : str, optional
        Working directory (default: "./")

    Returns
    -------
    None
        Custodian will run and handle errors automatically

    Raises
    ------
    Various custodian exceptions on failure

    Example
    -------
    >>> from atomate2.siesta.custodian.jobs import run_custodian_job
    >>> run_custodian_job(
    ...     siesta_cmd="siesta < siesta.fdf > siesta.out",
    ...     max_errors=5,
    ... )
    """
    from custodian import Custodian

    from atomate2.siesta.custodian.handlers import DEFAULT_HANDLERS
    from atomate2.siesta.custodian.validators import SiestaOutputValidator

    # Setup defaults
    if handlers is None:
        handlers = DEFAULT_HANDLERS

    if validators is None:
        validators = [SiestaOutputValidator()]

    # Create job
    job = SiestaJob(siesta_cmd=siesta_cmd)

    # Create and run custodian
    custodian = Custodian(
        handlers=handlers,
        jobs=[job],
        validators=validators,
        max_errors=max_errors,
        directory=directory,
    )

    custodian.run()
