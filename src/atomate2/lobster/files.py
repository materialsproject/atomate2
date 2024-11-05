"""Module defining functions for manipulating lobster files."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from atomate2.common.files import copy_files, get_zfile, gunzip_files
from atomate2.utils.file_client import FileClient, auto_fileclient
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from pathlib import Path

LOBSTEROUTPUT_FILES = [
    "BWDF.lobster",
    "BWDFCOHP.lobster",
    "CHARGE.lobster",
    "CHARGE.LCFO.lobster",
    "COBICAR.lobster",
    "COBICAR.LCFO.lobster"
    "COHPCAR.lobster",
    "COHPCAR.LCFO.lobster",
    "COOPCAR.lobster",
    "COOPCAR.LCFO.lobster",
    "DOSCAR.lobster",
    "DOSCAR.LCFO.lobster",
    "DOSCAR.LSO.lobster",
    "GROSSPOP.lobster",
    "GROSSPOP.LCFO.lobster",
    "ICOBILIST.lobster",
    "ICOBILIST.LCFO.lobster",
    "ICOHPLIST.lobster",
    "ICOHPLIST.LCFO.lobster",
    "ICOOPLIST.lobster",
    "ICOOPLIST.LCFO.lobster",
    "IMOFELIST.lobster",
    "LCFO_Fragments.lobster",
    "lobsterout",
    "lobster.out",
    "projectionData.lobster",
    "POLARIZATION.lobster",
    "POSCAR.lobster",
    "POSCAR.lobster.vasp",
    "MadelungEnergies.lobster",
    "MOFECAR.lobster",
    "SitePotentials.lobster",
    "bandOverlaps.lobster",
]

VASP_OUTPUT_FILES = [
    "OUTCAR",
    "vasprun.xml",
    "CHG",
    "CHGCAR",
    "CONTCAR",
    "INCAR",
    "KPOINTS",
    "POSCAR",
    "POTCAR",
    "DOSCAR",
    "EIGENVAL",
    "IBZKPT",
    "OSZICAR",
    "WAVECAR",
    "XDATCAR",
]

logger = logging.getLogger(__name__)


@auto_fileclient
def copy_lobster_files(
    src_dir: Path | str,
    src_host: str | None = None,
    file_client: FileClient = None,
) -> None:
    """
    Copy Lobster files to current directory.

    This function will gunzip any gzipped files.

    Parameters
    ----------
    src_dir : Path or str
        The source directory.
    src_host : str or None
        The source hostname used to specify a remote filesystem. Can be given as
        either "username@remote_host" or just "remote_host" in which case the username
        will be inferred from the current user. If ``None``, the local filesystem will
        be used as the source.
    file_client : FileClient
        A file client to use for performing file operations.
    """
    src_dir = strip_hostname(src_dir)  # TODO: Handle hostnames properly.

    logger.info(f"Copying LOBSTER inputs from {src_dir}")
    directory_listing = file_client.listdir(src_dir, host=src_host)

    # find optional files
    files = []
    for file in VASP_OUTPUT_FILES:
        found_file = get_zfile(directory_listing, file, allow_missing=True)
        if found_file is not None:
            files.append(found_file)

    copy_files(
        src_dir,
        src_host=src_host,
        include_files=files,
        file_client=file_client,
    )

    gunzip_files(
        include_files=files,
        allow_missing=True,
        file_client=file_client,
    )

    logger.info("Finished copying inputs")
