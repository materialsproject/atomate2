"""Functions dealing with SIESTA files."""

from __future__ import annotations

import logging
import os
import shutil
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any

from monty.serialization import loadfn
from rich.console import Console

from atomate2.common.files import get_zfile, gunzip_files
from atomate2.siesta import SETTINGS
from atomate2.siesta.utils.file_client import FileClient, auto_fileclient
from atomate2.siesta.utils.path import strip_hostname

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Molecule, Structure

    from atomate2.siesta.sets.base import SiestaInputGenerator

logger = logging.getLogger(__name__)
console = Console()


@auto_fileclient
def copy_siesta_outputs(
    src_dir: Path | str,
    src_host: str | None = None,
    additional_siesta_files: list[str] | None = None,
    restart_to_input: bool = False,
    file_client: FileClient | None = None,
) -> None:
    """Copy SIESTA output files to the current directory.

    Inspired by CP2K plugin implementation.

    Parameters
    ----------
    src_dir : str or Path
        The source directory.
    src_host : str or None
        The source hostname used to specify a remote filesystem. Can be given as
        either "username@remote_host" or just "remote_host" in which case the username
        will be inferred from the current user. If ``None``, the local filesystem will
        be used as the source.
    additional_siesta_files : list[str]
        Additional files to copy.
    restart_to_input : bool
        Move the SIESTA restart files to be the SIESTA input in the new directory.
    file_client : .FileClient
        A file client to use for performing file operations.
    """
    src_dir = strip_hostname(src_dir)
    logger.info(f"Copying SIESTA inputs from {src_dir}")
    directory_listing = file_client.listdir(src_dir, host=src_host)

    # Convert to absolute paths for get_zfile to work properly
    directory_listing = [Path(src_dir) / f for f in directory_listing]

    # Additional files like bands, DOS, *.cube, whatever
    additional_files = additional_siesta_files or []

    # Copy files (no need to copy siesta.out by default; it can be added to
    # additional_siesta_files explicitly if needed)
    files_to_copy: list[str] = (
        ["siesta.DM", "*.STRUCT_OUT", "*.xyz"] if restart_to_input else []
    )

    # Add additional files
    files_to_copy += additional_files

    # For patterns with wildcards, expand them by globbing
    # For specific filenames, keep them as-is for get_zfile to find
    expanded_files: list[str] = []
    for pattern in set(files_to_copy):
        if "*" in pattern or "?" in pattern:
            # Wildcard pattern - expand it
            matches = glob((Path(src_dir) / pattern).as_posix())
            expanded_files += [Path(f).name for f in matches]
        else:
            # Specific filename - keep it for get_zfile
            expanded_files.append(pattern)

    logger.debug(f"Files to copy: {files_to_copy}")
    logger.debug(f"Expanded files: {expanded_files}")

    # Use get_zfile to find files (handles compressed versions)
    all_files = [
        get_zfile(directory_listing, str(r), allow_missing=True) for r in expanded_files
    ]
    all_files = [f for f in all_files if f]

    logger.debug(f"Files found by get_zfile: {all_files}")

    # Convert to relative paths and flatten (ignore subdirectories)
    # This copies files from siesta_compressed/ to current directory
    all_files_relative = []
    for f in all_files:
        rel_path = f.relative_to(src_dir) if f.is_absolute() else f
        # Use just the filename, not the full path (flattens siesta_compressed/)
        all_files_relative.append(Path(rel_path.name))

    logger.debug(f"Files to copy (flattened): {all_files_relative}")

    # Manually copy each file to flatten directory structure
    for src_file, dest_name in zip(all_files, all_files_relative, strict=False):
        src_path = src_file if src_file.is_absolute() else Path(src_dir) / src_file
        file_client.copy(src_path, dest_name, src_host=src_host)

    # Gunzip the copied files (use flattened names)
    zipped_files = [f for f in all_files_relative if f.name.endswith("gz")]

    gunzip_files(
        include_files=zipped_files,
        allow_missing=True,
        file_client=file_client,
    )
    logger.info("Finished copying SIESTA inputs")


def write_siesta_input_set(
    structure: Structure | Molecule,
    input_set_generator: SiestaInputGenerator,
    directory: str | Path = ".",
    prev_dir: str | Path | None = None,
    **kwargs,  # noqa: ARG001 kept for documented write_input passthrough API
) -> None:
    """Write SIESTA input set.

    Parameters
    ----------
    structure : Structure or Molecule
        A structure or molecule to write the input set for.
    input_set_generator : .SiestaInputGenerator
        A SIESTA input set generator.
    directory : str or Path
        The directory to write the input files to.
    prev_dir : str or Path or None
        If the input set is to be initialized from a previous calculation,
        the previous calc directory.
    **kwargs
        Keyword arguments to pass to :obj:`.SiestaInputSet.write_input`.
    """
    siesta_fdf_is = input_set_generator.get_input_set(
        structure,
        prev_dir=prev_dir,
    )
    siesta_fdf_is.write_siesta_fdf(structure=structure, directory=directory)

    # Handle Lua script files for FLOS (if MD.TypeOfRun is set to 'LUA')
    lua = siesta_fdf_is.siesta_input.parameters
    if lua["fdf_arguments"].get("MD.TypeOfRun") == "LUA":
        flos_file_name = lua["fdf_arguments"].get("Lua.Script")
        # Copy Lua script to the specified directory
        copy_file_from_flos(flos_file_name, Path(directory))


@auto_fileclient
def cleanup_siesta_outputs(
    directory: Path | str,
    host: str | None = None,
    file_patterns: Sequence[str] = (),
    file_client: FileClient | None = None,
) -> None:
    """Remove unnecessary files.

    Parameters
    ----------
    directory : Path or str
        Directory containing files.
    host : str or None
        File client host.
    file_patterns : Sequence[str]
        Glob patterns to find files for deletion.
    file_client : .FileClient
        A file client to use for performing file operations.
    """
    files_to_delete = []
    for pattern in file_patterns:
        files_to_delete.extend(file_client.glob(Path(directory) / pattern, host=host))

    for file in files_to_delete:
        file_client.remove(file)


def load_siesta_input(
    dirpath: Path | str, fname: str = "siesta_parameters.json"
) -> Any:
    """Load the SiestaInput object from a given directory.

    Parameters
    ----------
    dirpath : Path or str
        Directory to load the SiestaInput from.
    fname : str
        Name of the json file containing the SiestaInput.

    Returns
    -------
    SiestaInput
        The SiestaInput object.
    """
    siesta_input_file = os.path.join(dirpath, f"{fname}")
    if not os.path.exists(siesta_input_file):
        raise NotImplementedError(
            f"Cannot load SiestaInput from directory without {fname} file."
        )
    return loadfn(siesta_input_file)


def read_directly_from_siesta_out(
    file_path: str | Path, what: str = "Version", multiple: bool = False
) -> dict[str, str | list[str] | None]:
    """Read a specified keyword from the siesta.out file.

    Parameters
    ----------
    file_path : str or Path
        Path to the SIESTA output file.
    what : str
        The keyword to search for in the file.
    multiple : bool
        If True, returns a list of all occurrences of the keyword.
        If False, returns only the first occurrence.

    Returns
    -------
    dict
        A dictionary with the keyword and its corresponding extracted values.
        If multiple=True and no matches found, value will be None.
    """
    query_info: dict[str, str | list[str] | None]
    query_info = {what: []} if multiple else {}

    with open(file_path) as file:
        for line in file:
            if what in line:
                extracted_info = line.split(":")[1].strip()
                if multiple:
                    current_val = query_info.get(what)
                    if isinstance(current_val, list):
                        current_val.append(extracted_info)
                else:
                    query_info[what] = extracted_info
                    break

    # If multiple is True and no matches are found, return None
    if multiple and isinstance(query_info.get(what), list) and not query_info[what]:
        query_info[what] = None

    return query_info


def copy_file_from_flos(file_name: str, destination_dir: Path | str) -> None:
    """Copy a Lua script file from FLOS_PATH to destination directory.

    Parameters
    ----------
    file_name : str
        Name of the Lua script file to copy.
    destination_dir : Path or str
        Destination directory for the file.

    Raises
    ------
    EnvironmentError
        If FLOS_PATH is not set in SETTINGS.
    FileNotFoundError
        If the specified file does not exist in FLOS_PATH.
    """
    flos_path = SETTINGS.FLOS_PATH

    if not flos_path:
        console.print(
            "[bold red]EnvironmentError:[/bold red] "
            "FLOS_PATH is not set in the SETTINGS.",
            style="red",
        )
        raise OSError("FLOS_PATH is not configured")

    # Check for 'examples' subfolder
    examples_path = os.path.join(flos_path, "examples")
    if os.path.isdir(examples_path):
        source_file = os.path.join(examples_path, file_name)
    else:
        source_file = os.path.join(flos_path, file_name)

    if not os.path.exists(source_file):
        console.print(
            f"[bold red]FileNotFoundError:[/bold red] The file '{file_name}' "
            f"does not exist in FLOS_PATH.",
            style="red",
        )
        raise FileNotFoundError(f"File '{file_name}' not found in FLOS_PATH")

    # Create destination directory if needed
    os.makedirs(destination_dir, exist_ok=True)

    # Copy the file (but don't overwrite if it already exists)
    destination_file = os.path.join(destination_dir, file_name)
    if os.path.exists(destination_file):
        logger.info(
            f"FLOS file {file_name} already exists in {destination_dir}, skipping copy"
        )
    else:
        shutil.copy(source_file, destination_file)
        logger.info(f"Copied FLOS file {file_name} to {destination_dir}")


def extract_siesta_timing(dir_name: str | Path) -> float | None:
    """
    Extract wall time from SIESTA output file.

    Parameters
    ----------
    dir_name : str or Path
        Directory containing SIESTA output

    Returns
    -------
    float : Wall time in seconds, or None if not found
    """
    import gzip
    import re

    dir_path = Path(dir_name)

    # Check for siesta.times.gz first (most reliable)
    timing_files = [
        dir_path / "siesta_compressed" / "siesta.times.gz",
        dir_path / "siesta.times.gz",
        dir_path / "siesta.times",
    ]

    for timing_file in timing_files:
        if timing_file.exists():
            try:
                use_gzip = str(timing_file).endswith(".gz")
                open_func = gzip.open if use_gzip else open
                mode = "rt" if use_gzip else "r"

                with open_func(timing_file, mode) as f:
                    for line in f:
                        # Look for a line like:
                        # "timer: Total elapsed wall-clock time (sec) =  0.592"
                        match = re.search(
                            r"timer:\s+Total elapsed wall-clock time "
                            r"\(sec\)\s+=\s+([\d.]+)",
                            line,
                        )
                        if match:
                            return float(match.group(1))
            except Exception:  # noqa: BLE001, S112 best-effort timing parse
                continue

    # Fallback: check siesta.out files
    output_files = [
        dir_path / "siesta_compressed" / "siesta.out.gz",
        dir_path / "siesta.out.gz",
        dir_path / "siesta.out",
    ]

    for output_file in output_files:
        if output_file.exists():
            try:
                use_gzip = str(output_file).endswith(".gz")
                open_func = gzip.open if use_gzip else open
                mode = "rt" if use_gzip else "r"

                with open_func(output_file, mode) as f:
                    for line in f:
                        # Alternative timing patterns in siesta.out
                        match = re.search(
                            r"timer:\s+(?:Total elapsed |Elapsed )"
                            r"wall(?:-clock)? time \(sec\)\s+=\s+([\d.]+)",
                            line,
                        )
                        if match:
                            return float(match.group(1))
            except Exception:  # noqa: BLE001, S112 best-effort timing parse
                continue

    return None


def gzip_output_folder(
    directory: str | Path,
    setting: bool | str,
    files_list: list[str] | None = None,
    exclude_files: list[str] | None = None,
    move_to_subfolder: bool = True,
    subfolder_name: str = "siesta_compressed",
) -> None:
    """
    Zip the content of the SIESTA output folder based on the code setting.

    SIESTA-specific variant of :func:`atomate2.common.files.gzip_output_folder`
    that additionally supports excluding files from compression and moving the
    compressed files into a subfolder to keep the job directory tidy.

    Parameters
    ----------
    directory : str or Path
        Directory in which to gzip files.
    setting : bool or str
        The setting determining which files to zip. If True all the files in
        the directory will be zipped (except those in ``exclude_files``), if
        "atomate" only the files in ``files_list``, if False no file will be
        zipped.
    files_list : list of str or None
        List of files to be zipped in case setting is "atomate".
    exclude_files : list of str or None
        List of files to exclude from compression (always excluded regardless
        of setting).
    move_to_subfolder : bool
        If True, move compressed files to a subfolder (default: True).
    subfolder_name : str
        Name of the subfolder for compressed files.
    """
    from atomate2.common.files import gzip_files

    if setting == "atomate":
        gzip_files(
            directory=directory,
            include_files=files_list,
            exclude_files=exclude_files,
            allow_missing=True,
            force=True,
        )
    elif setting:
        gzip_files(directory=directory, exclude_files=exclude_files, force=True)

    # Move gzipped files to a subfolder to keep the job directory tidy
    if move_to_subfolder and setting:
        directory = Path(directory)
        subfolder = directory / subfolder_name
        subfolder.mkdir(exist_ok=True)

        for gz_file in directory.glob("*.gz"):
            if gz_file.is_file():
                shutil.move(str(gz_file), str(subfolder / gz_file.name))
