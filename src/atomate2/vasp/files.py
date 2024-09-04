"""Functions for manipulating VASP files."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from atomate2 import SETTINGS
from atomate2.common.files import copy_files, get_zfile, gunzip_files, rename_files
from atomate2.utils.file_client import FileClient, auto_fileclient
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Structure

    from atomate2.vasp.sets.base import VaspInputGenerator


logger = logging.getLogger(__name__)


@auto_fileclient
def copy_vasp_outputs(
    src_dir: Path | str,
    src_host: str | None = None,
    additional_vasp_files: Sequence[str] = (),
    contcar_to_poscar: bool = True,
    force_overwrite: bool | str = False,
    file_client: FileClient | None = None,
) -> None:
    """
    Copy VASP output files to the current directory.

    For folders containing multiple calculations (e.g., suffixed with relax1, relax2,
    etc), this function will only copy the files with the highest numbered suffix and
    the suffix will be removed. Additional vasp files will be also be copied with the
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
    additional_vasp_files : list of str
        Additional files to copy, e.g. ["CHGCAR", "WAVECAR"].
    contcar_to_poscar : bool
        Move CONTCAR to POSCAR (original POSCAR is not copied).
    force_overwrite : bool or str
        How to handle overwriting existing files during the copy step. Accepts
        either a string or bool:

            - `"force"` or `True`: Overwrite existing files if they already exist.
            - `"raise"` or `False`: Raise an error if files already exist.
            - `"skip"` Skip files they already exist.
    file_client : .FileClient
        A file client to use for performing file operations.
    """
    src_dir = strip_hostname(src_dir)  # TODO: Handle hostnames properly.

    logger.info(f"Copying VASP inputs from {src_dir}")

    relax_ext = get_largest_relax_extension(src_dir, src_host, file_client=file_client)
    directory_listing = file_client.listdir(src_dir, host=src_host)

    # find required files
    files = ("INCAR", "OUTCAR", "CONTCAR", "vasprun.xml", *additional_vasp_files)
    required_files = [get_zfile(directory_listing, r + relax_ext) for r in files]

    # find optional files; do not fail if KPOINTS is missing, this might be KSPACING
    # note: POTCAR files never have the relax extension, whereas KPOINTS files should
    optional_files = []
    for file in ("POTCAR", "POTCAR.spec", "KPOINTS" + relax_ext):
        found_file = get_zfile(directory_listing, file, allow_missing=True)
        if found_file is not None:
            optional_files.append(found_file)

    # check at least one type of POTCAR file is included
    if len([f for f in optional_files if "POTCAR" in f.name]) == 0:
        raise FileNotFoundError(f"Could not find a POTCAR file in {src_dir!r} to copy")

    copy_files(
        src_dir,
        src_host=src_host,
        include_files=required_files + optional_files,
        file_client=file_client,
    )

    gunzip_files(
        include_files=required_files + optional_files,
        allow_missing=True,
        file_client=file_client,
        force=force_overwrite,
    )

    # rename files to remove relax extension
    if relax_ext:
        all_files = optional_files + required_files
        files_to_rename = {
            file.name.replace(".gz", ""): file.name.replace(relax_ext, "").replace(
                ".gz", ""
            )
            for file in all_files
        }
        rename_files(files_to_rename, allow_missing=True, file_client=file_client)

    if contcar_to_poscar:
        rename_files({"CONTCAR": "POSCAR"}, file_client=file_client)

    logger.info("Finished copying inputs")


@auto_fileclient
def get_largest_relax_extension(
    directory: Path | str,
    host: str | None = None,
    file_client: FileClient | None = None,
) -> str:
    """
    Get the largest numbered relax extension of files in a directory.

    For example, if listdir gives ["vasprun.xml.relax1.gz", "vasprun.xml.relax2.gz"],
    this function will return ".relax2".

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
        The relax extension or an empty string if there were not multiple relaxations.
    """
    relax_files = file_client.glob(Path(directory) / "*.relax*", host=host)
    if len(relax_files) == 0:
        return ""

    numbers = [re.search(r".relax(\d+)", file.name).group(1) for file in relax_files]
    max_relax = max(numbers, key=int)
    return f".relax{max_relax}"


def write_vasp_input_set(
    structure: Structure,
    input_set_generator: VaspInputGenerator,
    directory: str | Path = ".",
    from_prev: bool = False,
    apply_incar_updates: bool = True,
    potcar_spec: bool = False,
    clean_prev: bool = True,
    **kwargs,
) -> None:
    """
    Write VASP input set.

    Parameters
    ----------
    structure : .Structure
        A structure.
    input_set_generator : .VaspInputGenerator
        A VASP input set generator.
    directory : str or Path
        The directory to write the input files to.
    from_prev : bool
        Whether to initialize the input set from a previous calculation.
    apply_incar_updates : bool
        Whether to apply incar updates given in the ~/.atomate2.yaml settings file.
    potcar_spec : bool
        Whether to use the POTCAR.spec file instead of the POTCAR file.
    clean_prev : bool
        Remove previous KPOINTS, INCAR, POSCAR, and POTCAR before writing new inputs.
    **kwargs
        Keyword arguments that will be passed to :obj:`.VaspInputSet.write_input`.
    """
    prev_dir = "." if from_prev else None
    vis = input_set_generator.get_input_set(
        structure, prev_dir=prev_dir, potcar_spec=potcar_spec
    )

    if apply_incar_updates:
        vis.incar.update(SETTINGS.VASP_INCAR_UPDATES)

    if clean_prev:
        # remove previous inputs (prevents old KPOINTS file from overriding KSPACING)
        for filename in ("POSCAR", "KPOINTS", "POTCAR", "INCAR"):
            if Path(filename).exists():
                Path(filename).unlink()

    logger.info("Writing VASP input set.")
    vis.write_input(directory, **kwargs)
