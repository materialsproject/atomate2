"""Functions for manipulating Abinit files."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from abipy.flowtk.utils import abi_extensions
from monty.serialization import loadfn

from atomate2.abinit.utils.common import INDIR_NAME
from atomate2.utils.file_client import FileClient, auto_fileclient

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from abipy.abio.inputs import AbinitInput
    from pymatgen.core.structure import Structure

    from atomate2.abinit.sets.base import AbinitInputGenerator


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
    abinit_input_file = os.path.join(dirpath, f"{fname}")
    if not os.path.exists(abinit_input_file):
        raise NotImplementedError(
            f"Cannot load AbinitInput from directory without {fname} file."
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
