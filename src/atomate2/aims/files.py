"""Functions dealing with FHI-aims files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from atomate2.common.files import copy_files, gunzip_files
from atomate2.utils.file_client import FileClient, auto_fileclient
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Molecule, Structure
    from pymatgen.io.aims.sets.base import AimsInputGenerator

logger = logging.getLogger(__name__)


@auto_fileclient
def copy_aims_outputs(
    src_dir: Path | str,
    src_host: str | None = None,
    additional_aims_files: list[str] | None = None,
    restart_to_input: bool = False,
    file_client: FileClient | None = None,
) -> None:
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
    additional_aims_files : list[str]
        Additional files to copy
    restart_to_input : bool
        Move the aims restart files to by the aims input in the new directory
    file_client : .FileClient
        A file client to use for performing file operations.
    """
    src_dir = strip_hostname(src_dir)
    logger.info(f"Copying FHI-aims inputs from {src_dir}")

    # (no need to copy aims.out by default; it can be added to additional_aims_files
    # explicitly if needed)
    patterns = (
        ["hessian.aims", "geometry.in.next_step", "*.csc"] if restart_to_input else []
    )
    # additional files like bands, DOS, *.cube, whatever
    patterns += additional_aims_files or []
    gz_patterns = [f"{pattern}.gz" for pattern in patterns]

    copy_files(
        src_dir,
        src_host=src_host,
        include_files=patterns + gz_patterns,
        allow_missing=True,
        file_client=file_client,
    )

    # if both the gzipped and the plain version were copied, keep the gzipped one
    gunzip_files(
        include_files=gz_patterns,
        allow_missing=True,
        force=True,
        file_client=file_client,
    )

    logger.info("Finished copying inputs")


def write_aims_input_set(
    structure: Structure | Molecule,
    input_set_generator: AimsInputGenerator,
    directory: str | Path = ".",
    prev_dir: str | Path | None = None,
    **kwargs,
) -> None:
    """
    Write FHI-aims input set.

    Parameters
    ----------
    structure : Structure or Molecule
        A to write the input set for.
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
) -> None:
    """Remove unnecessary files.

    Parameters
    ----------
    directory: Path or str
        Directory containing files
    host: str or None
        File client host
    file_patterns: Sequence[str]
        Glob patterns to find files for deletion.
    file_client: .FileClient
        A file client to use for performing file operations.
    """
    files_to_delete = []
    for pattern in file_patterns:
        files_to_delete.extend(file_client.glob(Path(directory) / pattern, host=host))

    for file in files_to_delete:
        file_client.remove(file)
