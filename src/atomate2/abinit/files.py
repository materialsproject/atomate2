"""Functions for manipulating Abinit files."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from abipy.flowtk.utils import abi_extensions
from jobflow.core.job import job
from monty.serialization import loadfn

from atomate2 import SETTINGS
from atomate2.abinit.utils.common import INDIR_NAME
from atomate2.common.files import delete_files, gzip_files
from atomate2.utils.file_client import FileClient, auto_fileclient
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from collections.abc import Iterable

    from abipy.abio.inputs import AbinitInput
    from jobflow.core.reference import OutputReference
    from pymatgen.core.structure import Structure

    from atomate2.abinit.sets.anaddb import AnaddbInputGenerator
    from atomate2.abinit.sets.base import AbinitInputGenerator
    from atomate2.abinit.sets.mrgddb import MrgddbInputGenerator

__all__ = [
    "out_to_in",
    "fname2ext",
    "load_abinit_input",
    "write_abinit_input_set",
    "write_mrgddb_input_set",
    "write_anaddb_input_set",
    "del_gzip_files",
]

logger = logging.getLogger(__name__)

ALL_ABIEXTS = abi_extensions()


def fname2ext(filepath: Path | str) -> None | str:
    """Get the abinit extension of a given filename.

    This will return None if no extension is found.
    """
    filename = os.path.basename(filepath)
    if "_" not in filename:
        return None
    ext = filename.split("_")[-1].replace(".nc", "")
    if "1WF" in ext:
        ext = "1WF"
    if ext not in ALL_ABIEXTS:
        return None
    return ext


@auto_fileclient
def out_to_in(
    out_files: Iterable[tuple[str, str]],
    src_host: str | None = None,
    indir: Path | str = INDIR_NAME,
    file_client: FileClient | None = None,
    link_files: bool = True,
) -> None:
    """
    Copy or link abinit output files to the Abinit input directory.

    Parameters
    ----------
    out_files : list of tuples
        The list of (abinit output filepath, abinit input filename) to be copied
        or linked.
    src_host : str or None
        The source hostname used to specify a remote filesystem. Can be given as
        either "username@remote_host" or just "remote_host" in which case the username
        will be inferred from the current user. If ``None``, the local filesystem will
        be used as the source.
    indir : Path or str
        The input directory for Abinit input files.
    file_client : .FileClient
        A file client to use for performing file operations.
    link_files : bool
        Whether to link the files instead of copying them.
    """
    dest_dir = file_client.abspath(indir, host=None)

    for out_filepath, in_file in out_files:
        src_file = file_client.abspath(out_filepath, host=src_host)
        dest_file = os.path.join(dest_dir, in_file)
        if link_files and src_host is None:
            file_client.link(src_file, dest_file)
        else:
            file_client.copy(src_file, dest_file, src_host=src_host)


def load_abinit_input(
    dirpath: Path | str, fname: str = "abinit_input.json"
) -> AbinitInput:
    """Load the AbinitInput object from a given directory.

    Parameters
    ----------
    dirpath
        Directory to load the AbinitInput from.
    fname
        Name of the json file containing the AbinitInput.

    Returns
    -------
    AbinitInput
        The AbinitInput object.
    """
    dirpath = strip_hostname(dirpath)  # TODO: to FileCLient?
    abinit_input_file = os.path.join(dirpath, f"{fname}")
    if not os.path.exists(abinit_input_file):
        raise NotImplementedError(
            f"Cannot load AbinitInput from directory {dirpath} without {fname} file."
        )

    return loadfn(abinit_input_file)


def write_abinit_input_set(
    structure: Structure,
    input_set_generator: AbinitInputGenerator,
    prev_outputs: str | Path | list[str] | None = None,
    restart_from: str | Path | list[str] | None = None,
    directory: str | Path = ".",
) -> None:
    """Write the abinit inputs for a given structure using a given generator.

    Parameters
    ----------
    structure
        The structure for which the abinit inputs have to be written.
    input_set_generator
        The input generator used to write the abinit inputs.
    prev_outputs
        The list of previous directories needed for the calculation.
    restart_from
        The previous directory of the same calculation (in case of a restart).
        Note that this should be provided as a list of one directory.
    directory
        Directory in which to write the abinit inputs.
    """
    ais = input_set_generator.get_input_set(
        structure=structure,
        restart_from=restart_from,
        prev_outputs=prev_outputs,
    )
    if not ais.validate():
        raise RuntimeError("AbinitInputSet is not valid.")

    ais.write_input(directory=directory, make_dir=True, overwrite=False)


def write_mrgddb_input_set(
    input_set_generator: MrgddbInputGenerator,
    prev_outputs: str | Path | list[str] | None = None,
    directory: str | Path = ".",
) -> None:
    """Write the mrgddb input using a given generator.

    Parameters
    ----------
    input_set_generator
        The input generator used to write the mrgddb inputs.
    prev_outputs
        The list of previous directories needed for the calculation.
    directory
        Directory in which to write the abinit inputs.
    """
    mrgis = input_set_generator.get_input_set(
        prev_outputs=prev_outputs,
        workdir=directory,
    )
    if not mrgis.validate():
        raise RuntimeError(
            "MrgddbInputSet is not valid. Some previous outputs \
        do not exist."
        )

    mrgis.write_input(directory=directory, make_dir=True, overwrite=False)


def write_anaddb_input_set(
    structure: Structure,
    input_set_generator: AnaddbInputGenerator,
    prev_outputs: str | Path | list[str] | None = None,
    directory: str | Path = ".",
) -> None:
    """Write the anaddb input using a given generator.

    Parameters
    ----------
    input_set_generator
        The input generator used to write the anaddb inputs.
    prev_outputs
        The list of previous directories needed for the calculation.
    directory
        Directory in which to write the abinit inputs.
    """
    anais = input_set_generator.get_input_set(
        structure=structure,
        prev_outputs=prev_outputs,
    )
    if not anais.validate():
        raise RuntimeError(
            "AnaddbInputSet is not valid. Some previous outputs \
        do not exist."
        )

    anais.write_input(directory=directory, make_dir=True, overwrite=False)


# TODO: atm does not take the directories of the generate..., run_rf,
# store_inputs, and its own directory into account...
@job
def del_gzip_files(
    output: list | OutputReference,
    exclude_files_from_zip: list[str | Path] | None = None,
    delete: bool = True,
    exclude_files_from_del: list[str | Path] | None = None,
    include_files_to_del: list[str | Path] | None = None,
) -> None:
    r"""Delete and gzip files from a previous Job or Flow.

    Parameters
    ----------
    output
        The OutputReference or the list of OutputReference of the Job/Flow
        that needs its files deleted/compressed.
    exclude_files_from_zip
        Filenames to exclude from the compression.
        Supports glob file matching, e.g., "\*.dat".
    delete: bool = True,
        Activates the deletion of the files or not.
    exclude_files_from_del
        Filenames to exclude from the compression.
        Supports glob file matching, e.g., "\*.dat".
    include_files_to_del
        Filenames to include as a list of str or Path objects given relative to
        directory. Glob file paths are supported, e.g. "\*.dat". If ``None``, all files
        in the directory will be deleted.
    """
    dirs_to_zip = []
    if not isinstance(output, list):  # in case of a single Job
        output = [output]
    for o in output:
        with contextlib.suppress(TypeError, AttributeError):
            dirs_to_zip.append(o.dir_name)
        with contextlib.suppress(TypeError, AttributeError, KeyError):
            dirs_to_zip.extend(o["dirs"])
        with contextlib.suppress(TypeError, AttributeError, KeyError):
            dirs_to_zip.append(o["dir_name"])  # to zip run_rf and generate_perts

    recursiv_dirs_to_zip = []
    for dz in dirs_to_zip:
        recursiv_dirs_to_zip.append(Path(dz))
        for root, dirs, _ in os.walk(dz):
            recursiv_dirs_to_zip.extend([Path(root) / d for d in dirs])
    # to take its own directory into account
    recursiv_dirs_to_zip.append(Path(os.getcwd()))

    if delete:
        if include_files_to_del is None:
            include_files_to_del = SETTINGS.ABINIT_FILES_TO_DEL
        for dz in recursiv_dirs_to_zip:
            delete_files(
                directory=strip_hostname(dz),
                include_files=include_files_to_del,
                exclude_files=exclude_files_from_del,
                allow_missing=True,
            )

    for dz in recursiv_dirs_to_zip:
        gzip_files(
            directory=strip_hostname(dz),
            exclude_files=exclude_files_from_zip,
            force=True,
        )
