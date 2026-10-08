"""Functions dealing with FHI-aims files."""

from __future__ import annotations

import logging
import warnings
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from atomate2.common.files import copy_files, get_zfile, gunzip_files
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
    src_dir : str | Path
        The source directory.
    src_host : str | None
        The source hostname used to specify a remote filesystem. Can be given as
        either "username@remote_host" or just "remote_host" in which case the username
        will be inferred from the current user. If ``None``, the local filesystem will
        be used as the source.
    additional_aims_files : list[str] | None
        Additional files to copy.
    restart_to_input : bool
        Move the aims restart files to by the aims input in the new directory
    file_client : .FileClient | None
        A file client to use for performing file operations. If ``None``, a default file
        client will be created.
    """
    if file_client is None:
        raise RuntimeError(
            "File client should have been set automatically. This should not happen."
        )

    src_dir = strip_hostname(src_dir)
    logger.info(f"Copying FHI-aims inputs from {src_dir}")
    directory_listing = file_client.listdir(src_dir, host=src_host)
    # additional files like bands, DOS, *.cube, whatever
    additional_files = additional_aims_files or []

    # copy files
    # (no need to copy aims.out by default; it can be added to additional_aims_files
    # explicitly if needed)
    files: list[str] = (
        ["hessian.aims", "geometry.in.next_step", "*.csc"] if restart_to_input else []
    )

    files += [
        Path(f).name
        for pattern in set(files + additional_files)
        for f in glob((Path(src_dir) / pattern).as_posix())
    ]

    all_files = [
        get_zfile(directory_listing, str(r), allow_missing=True) for r in files
    ]
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
    if file_client is None:
        raise RuntimeError(
            "File client should have been set automatically. This should not happen."
        )

    files_to_delete = []
    for pattern in file_patterns:
        files_to_delete.extend(file_client.glob(Path(directory) / pattern, host=host))

    for file in files_to_delete:
        file_client.remove(file)


def get_aims_zip_files_setting() -> bool | Literal["atomate"]:
    """
    Get the file compression setting for FHI-aims jobs.

    Uses ``AIMS_ZIP_FILES`` if set. Otherwise, falls back to ``VASP_ZIP_FILES``
    (deprecated for FHI-aims) if it was explicitly set, and to "atomate" if not.

    Returns
    -------
    bool or "atomate"
        The compression setting passed to ``gzip_output_folder``.
    """
    from atomate2 import SETTINGS

    if SETTINGS.AIMS_ZIP_FILES is not None:
        return SETTINGS.AIMS_ZIP_FILES

    if "VASP_ZIP_FILES" in SETTINGS.model_fields_set:
        warnings.warn(
            "Using VASP_ZIP_FILES to control compression of FHI-aims files is "
            "deprecated and will be removed in a future release. Please set "
            "AIMS_ZIP_FILES instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SETTINGS.VASP_ZIP_FILES

    return "atomate"
