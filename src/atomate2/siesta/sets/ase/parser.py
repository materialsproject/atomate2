import logging
import re
import shutil
import tempfile

logger = logging.getLogger(__name__)


def parse_siesta_version(output: bytes) -> str:
    """
    Parse SIESTA version from command output.

    Args:
        output (bytes): Raw output from running the SIESTA executable.

    Returns
    -------
        str: The parsed SIESTA version string.

    Raises
    ------
        RuntimeError: If version information cannot be extracted from output.
    """
    logger.info("parse_siesta_version()")
    match = re.search(rb"Version\s*:\s*(\S+)", output)

    if match is None:
        raise RuntimeError(f"Could not get Siesta version info from output {output!r}")

    string = match.group(1).decode("ascii")
    return string


def get_siesta_version(executable: str) -> str:
    """
    Retrieve SIESTA version by running the executable in a temporary directory.

    Args:
        executable (str): Path to the SIESTA executable.

    Returns
    -------
        str: The SIESTA version string.

    Notes
    -----
        Creates a temporary directory to run the executable safely, capturing
        output to parse the version. Cleans up the directory afterward.
    """
    logger.info("get_siesta_version()")
    temp_dirname = tempfile.mkdtemp(prefix="siesta-version-check-")
    try:
        from subprocess import PIPE, Popen

        proc = Popen(
            [executable], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=temp_dirname
        )
        output, _ = proc.communicate()
        # SIESTA will exit with status 1 if no input is provided (e.g., missing Chemical_species_label).
    finally:
        shutil.rmtree(temp_dirname)

    return parse_siesta_version(output)
