"""Functions for manipulating QChem files."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from atomate2.common.files import copy_files, get_zfile, gunzip_files, rename_files
from atomate2.utils.file_client import FileClient, auto_fileclient
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


@auto_fileclient
def copy_qchem_outputs(
    src_dir: Path | str,
    src_host: str | None = None,
    additional_qchem_files: Sequence[str] = (),
    file_client: FileClient | None = None,
) -> None:
    """
    Copy QChem output files to the current directory.

    For folders containing multiple calculations (e.g., suffixed with opt_1, opt_2,
    etc), this function will only copy the files with the highest numbered suffix
    and the suffix will be removed. Additional qchem files will be also be copied
    with the same suffix applied.
    Lastly, this function will gunzip any gzipped files.

    Parameters
    ----------
    src_dir : str or Path
        The source directory.
    src_host : str or None
        The source hostname used to specify a remote filesystem. Can be given as
        either "username@remote_host" or just "remote_host" in which case the username
        will be inferred from the current user. If ``None``, the local filesystem will
        be used as the source.
    additional_qchem_files : list of str
        Additional files to copy.
    file_client : .FileClient
        A file client to use for performing file operations.
    """
    src_dir = strip_hostname(src_dir)  # TODO: Handle hostnames properly.

    logger.info(f"Copying QChem inputs from {src_dir}")
    opt_ext = get_largest_opt_extension(src_dir, src_host, file_client=file_client)
    directory_listing = file_client.listdir(src_dir, host=src_host)

    # find required files
    files = ("mol.qin", "mol.qout", *tuple(additional_qchem_files))
    required_files = [get_zfile(directory_listing, r + opt_ext) for r in files]

    copy_files(
        src_dir,
        src_host=src_host,
        include_files=required_files,
        file_client=file_client,
    )

    gunzip_files(
        include_files=required_files,
        allow_missing=True,
        file_client=file_client,
    )

    # rename files to remove opt extension
    if opt_ext:
        all_files = required_files
        files_to_rename = {
            file.name.replace(".gz", ""): file.name.replace(opt_ext, "").replace(
                ".gz", ""
            )
            for file in all_files
        }
        rename_files(files_to_rename, allow_missing=True, file_client=file_client)

    logger.info("Finished copying inputs")


@auto_fileclient
def get_largest_opt_extension(
    directory: Path | str,
    host: str | None = None,
    file_client: FileClient | None = None,
) -> str:
    """
    Get the largest numbered opt extension of files in a directory.

    For example, if listdir gives ["mol.qout.opt_0.gz", "mol.qout.opt_1.gz"],
    this function will return ".opt_1".

    Parameters
    ----------
    directory : str or Path
        A directory to search.
    host : str or None
        The hostname used to specify a remote filesystem. Can be given as either
        "username@remote_host" or just "remote_host" in which case the username will be
        inferred from the current user. If ``None``, the local filesystem will be used.
    file_client : .FileClient
        A file client to use for performing file operations.

    Returns
    -------
    str
        The opt extension or an empty string if there were not multiple relaxations.
    """
    opt_files = file_client.glob(Path(directory) / "*.opt*", host=host)
    if len(opt_files) == 0:
        return ""
    numbers = []
    for file in opt_files:
        match = re.search(r"\.opt_(\d+)", file.name)
        if match:
            numbers.append(match.group(1))

    if not numbers:
        return ""  # No matches found
    max_relax = max(numbers, key=int)
    return f".opt_{max_relax}"
