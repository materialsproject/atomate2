"""Functions dealing with FHI-aims files"""
from __future__ import annotations

import logging
from glob import glob
from pathlib import Path
from typing import Sequence

from atomate2.common.files import copy_files, get_zfile, gunzip_files
from atomate2.utils.file_client import FileClient, auto_fileclient
from atomate2.utils.path import strip_hostname

from atomate2.aims.sets.base import AimsInputGenerator
from atomate2.aims.utils.msonable_atoms import MSONableAtoms

logger = logging.getLogger(__name__)

__all__ = ["copy_aims_outputs", "write_aims_input_set", "cleanup_aims_outputs"]


@auto_fileclient
def copy_aims_outputs(
    src_dir: Path | str,
    src_host: str | None = None,
    additional_aims_files: list[str] | None = None,
    restart_to_input: bool = True,
    file_client: FileClient | None = None,
):
    """
    Copy FHI-aims output files to the current directory (inspired by CP2K plugin).

    Parameters
    ----------
    src_dir : str or Path
        The source directory.
    src_host : str or None
        The source hostname used to specify a remote filesystem. Can be given as
        either "username@remote_host" or just "remote_host" in which case the username
        will be inferred from the current user. If ``None``, the local filesystem will
        be used as the source.
    additional_aims_files : list of str
        Additional files to copy
    restart_to_input : bool
        Move the aims restart files to by the aims input in the new directory
    file_client : .FileClient
        A file client to use for performing file operations.
    """
    src_dir = strip_hostname(src_dir)
    logger.info(f"Copying FHI-aims inputs from {src_dir}")
    directory_listing = file_client.listdir(src_dir, host=src_host)
    # additional files like bands, DOS, *.cube, whatever
    additional_files = additional_aims_files if additional_aims_files else []

    if restart_to_input:
        additional_files += ("hessian.aims", "geometry.in.next_step", "*.csc")

    # copy files
    files = ["aims.out", "*.json"]

    for pattern in set(additional_files):
        for f in glob((Path(src_dir) / pattern).as_posix()):
            files.append(Path(f).name)

    all_files = [get_zfile(directory_listing, r, allow_missing=True) for r in files]
    all_files = [f for f in all_files if f]

    copy_files(
        src_dir,
        src_host=src_host,
        include_files=all_files,
        file_client=file_client,
    )

    zipped_files = [f for f in all_files if f.name.endswith("gz")]

    gunzip_files(
        include_files=zipped_files,
        allow_missing=True,
        file_client=file_client,
    )

    logger.info("Finished copying inputs")


def write_aims_input_set(
    structure: MSONableAtoms,
    input_set_generator: AimsInputGenerator,
    directory: str | Path = ".",
    prev_dir: str | Path | None = None,
    **kwargs,
):
    """
    Write FHI-aims input set.

    Parameters
    ----------
    structure : .MSONableAtoms
        A structure.
    input_set_generator : .AimsInputGenerator
        An GHI-aims input set generator.
    directory : str or Path
        The directory to write the input files to.
    prev_dir : str or Path or None
        If the input set is to be initialized from a previous calculation,
        the previous calc directory
    **kwargs
        Keyword arguments to pass to :obj:`.AimsInputSet.write_input`.
    """
    properties = kwargs.get("properties", [])
    aims_is = input_set_generator.get_input_set(
        structure, prev_dir=prev_dir, properties=properties
    )

    logger.info("Writing FHI-aims input set.")
    aims_is.write_input(directory, **kwargs)


@auto_fileclient
def cleanup_aims_outputs(
    directory: Path | str,
    host: str | None = None,
    file_patterns: Sequence[str] = (),
    file_client: FileClient | None = None,
):
    """
    Remove unnecessary files.

    Parameters
    ----------
    directory:
        Directory containing files
    host:
        File client host
    file_patterns:
        Glob patterns to find files for deletion.
    file_client:
        A file client to use for performing file operations.
    """
    files_to_delete = []
    for pattern in file_patterns:
        files_to_delete.extend(file_client.glob(Path(directory) / pattern, host=host))

    for file in files_to_delete:
        file_client.remove(file)
