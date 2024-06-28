"""Functions for manipulating CP2K files."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pymatgen.io.cp2k.outputs import Cp2kOutput

from atomate2 import SETTINGS
from atomate2.common.files import copy_files, get_zfile, gunzip_files, rename_files
from atomate2.utils.file_client import FileClient, auto_fileclient
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Structure

    from atomate2.cp2k.sets.base import Cp2kInputGenerator


logger = logging.getLogger(__name__)


@auto_fileclient
def copy_cp2k_outputs(
    src_dir: Path | str,
    src_host: str | None = None,
    additional_cp2k_files: list[str] | None = None,
    restart_to_input: bool = True,
    file_client: FileClient | None = None,
) -> None:
    """
    Copy CP2K output files to the current directory.

    For folders containing multiple calculations (e.g., suffixed with relax1, relax2,
    etc), this function will only copy the files with the highest numbered suffix and
    the suffix will be removed. Additional cp2k files will be also be copied with the
    same suffix applied. Lastly, this function will gunzip any gzipped files.

    Parameters
    ----------
    src_dir : str or Path
        The source directory.
    src_host : str or None
        The source hostname used to specify a remote filesystem. Can be given as
        either "username@remote_host" or just "remote_host" in which case the username
        will be inferred from the current user. If ``None``, the local filesystem will
        be used as the source.
    additional_cp2k_files : list of str
        Additional files to copy
    restart_to_input : bool
        Move the cp2k restart file to by the cp2k input in the new directory
    file_client : .FileClient
        A file client to use for performing file operations.
    """
    src_dir = strip_hostname(src_dir)  # TODO: Handle hostnames properly.
    logger.info(f"Copying CP2K inputs from {src_dir}")
    relax_ext = get_largest_relax_extension(src_dir, src_host, file_client=file_client)
    directory_listing = file_client.listdir(src_dir, host=src_host)
    restart_file = None
    additional_cp2k_files = additional_cp2k_files or []

    # find required files
    o = Cp2kOutput(src_dir / get_zfile(directory_listing, "cp2k.out"), auto_load=False)
    o.parse_files()
    if restart_to_input:
        additional_cp2k_files += ("restart",)

    # copy files
    additional_cp2k_files += ("wfn",)
    files = ["cp2k.inp", "cp2k.out"]
    for f in set(additional_cp2k_files):
        if f in o.filenames and o.filenames.get(f):
            if isinstance(o.filenames[f], str):
                files.append(Path(o.filenames[f]).name)
            else:
                files.append(Path(o.filenames[f][-1]).name)
        else:
            files.append(Path(f).name)
    all_files = [
        get_zfile(directory_listing, r + relax_ext, allow_missing=True) for r in files
    ]
    all_files = [f for f in all_files if f]

    copy_files(
        src_dir,
        src_host=src_host,
        include_files=all_files,
        file_client=file_client,
    )

    gunzip_files(
        include_files=all_files,
        allow_missing=True,
        file_client=file_client,
    )

    # rename files to remove relax extension
    if relax_ext:
        files_to_rename = {
            file.name.replace(".gz", ""): file.name.replace(relax_ext, "").replace(
                ".gz", ""
            )
            for file in all_files
        }
        rename_files(files_to_rename, allow_missing=True, file_client=file_client)

    if restart_file:
        file_to_rename = restart_file.replace(".gz", "")
        rename_files({f"{file_to_rename}": "cp2k.inp"}, file_client=file_client)

    logger.info("Finished copying inputs")


@auto_fileclient
def get_largest_relax_extension(
    directory: Path | str,
    host: str | None = None,
    file_client: FileClient | None = None,
) -> str:
    """
    Get the largest numbered relax extension of files in a directory.

    For example, if listdir gives ["Cp2k-RESTART.wfn.relax1.gz",
    "Cp2k-RESTART.wfn.relax2.gz"], this function will return ".relax2".

    Parameters
    ----------
    directory : str or Path
        A directory to search.
    host : str or None
        The hostname used to specify a remote filesystem. Can be given as
        either "username@remote_host" or just "remote_host" in which case
        the username will be inferred from the current user. If ``None``,
        the local filesystem will be used.
    file_client : .FileClient
        A file client to use for performing file operations.

    Returns
    -------
    str
        The relax extension or an empty string if there were not multiple
        relaxations.
    """
    relax_files = file_client.glob(Path(directory) / "*.relax*", host=host)
    if len(relax_files) == 0:
        return ""

    numbers = [re.search(r".relax(\d+)", file.name).group(1) for file in relax_files]
    max_relax = max(numbers, key=int)
    return f".relax{max_relax}"


def write_cp2k_input_set(
    structure: Structure,
    input_set_generator: Cp2kInputGenerator,
    directory: str | Path = ".",
    from_prev: bool = False,
    apply_input_updates: bool = True,
    optional_files: dict | None = None,
    **kwargs,
) -> None:
    """
    Write CP2K input set.

    Parameters
    ----------
    structure : .Structure
        A structure.
    input_set_generator : .Cp2kInputGenerator
        A CP2K input set generator.
    directory : str or Path
        The directory to write the input files to.
    from_prev : bool
        Whether to initialize the input set from a previous calculation.
    apply_input_updates : bool
        Whether to apply incar updates given in the ~/.atomate2.yaml settings
        file.
    clean_prev : bool
        Remove previous inputs before writing new inputs.
    **kwargs
        Keyword arguments to pass to :obj:`.Cp2kInputSet.write_input`.
    """
    prev_dir = "." if from_prev else None
    cis = input_set_generator.get_input_set(
        structure, prev_dir=prev_dir, optional_files=optional_files
    )

    if apply_input_updates:
        cis.cp2k_input.update(SETTINGS.CP2K_INPUT_UPDATES)

    logger.info("Writing CP2K input set.")
    cis.write_input(directory, **kwargs)


@auto_fileclient
def cleanup_cp2k_outputs(
    directory: Path | str,
    host: str | None = None,
    file_patterns: Sequence[str] = ("*bak*",),
    file_client: FileClient | None = None,
) -> None:
    """
    Remove unnecessary files.

    Parameters
    ----------
    directory:
        Directory containing files
    host:
        File client host
    file_patterns:
        Glob patterns to find files for deletion. Default is to
        remove the "backup" wavefunctions.
    file_client:
        A file client to use for performing file operations.
    """
    files_to_delete = []
    for pattern in file_patterns:
        files_to_delete.extend(file_client.glob(Path(directory) / pattern, host=host))

    for file in files_to_delete:
        file_client.remove(file)
