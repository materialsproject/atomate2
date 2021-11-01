"""Module defining functions for manipulating amset files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from monty.serialization import dumpfn, loadfn

from atomate2.common.file import copy_files, get_zfile, gunzip_files, rename_files
from atomate2.settings import Settings
from atomate2.utils.file_client import FileClient, auto_fileclient
from atomate2.utils.path import strip_hostname

__all__ = ["copy_amset_files"]


logger = logging.getLogger(__name__)


@auto_fileclient
def copy_amset_files(
    src_dir: Union[Path, str],
    src_host: Optional[str] = None,
    file_client: FileClient = None,
):
    """
    Copy AMSET files to current directory.

    This function will gunzip any gzipped files.

    Parameters
    ----------
    src_dir
        The source directory.
    src_host
        The source hostname used to specify a remote filesystem. Can be given as
        either "username@remote_host" or just "remote_host" in which case the username
        will be inferred from the current user. If ``None``, the local filesystem will
        be used as the source.
    file_client
        A file client to use for performing file operations.
    """
    src_dir = strip_hostname(src_dir)  # TODO: Handle hostnames properly.

    logger.info(f"Copying AMSET inputs from {src_dir}")
    directory_listing = file_client.listdir(src_dir, host=src_host)

    # find required files
    required_files = [get_zfile(directory_listing, r) for r in ("settings.yaml",)]

    # find optional files
    optional_files = []
    found_vasprun = False
    for file in (
        "vasprun.xml",
        "band_structure_data.json",
        "wavefunction.h5",
        "deformation.h5",
        "transport.json",
    ):
        found_file = get_zfile(directory_listing, file, allow_missing=True)
        if found_file is not None:
            optional_files.append(found_file)

            if "vasprun" in file or "band_structure_data" in file:
                found_vasprun = True

    # check at least one of vasprun or band_structure_data is found
    if not found_vasprun:
        raise FileNotFoundError(
            "Could not find vasprun.xml or band_structure_data.json file to copy."
        )

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
    )

    rename_files({"transport.json": "transport.prev.json"}, allow_missing=True)
    logger.info("Finished copying inputs")


def write_amset_settings(settings_updates: Dict, from_prev: bool = False):
    """
    Write AMSET settings to file.

    This function will also apply any settings specified in
    :obj:`.Settings.AMSET_SETTINGS_UPDATE`.

    Parameters
    ----------
    settings_updates
        A dictionary of settings to write.
    from_prev
        Whether apply the settings on top of an existing settings.yaml file in the
        current directory.
    """
    if from_prev:
        settings = loadfn("settings.yaml")
        settings.update(settings_updates)
    else:
        settings = settings_updates

    settings.update(Settings.AMSET_SETTINGS_UPDATE)

    dumpfn(settings, "settings.yaml")
