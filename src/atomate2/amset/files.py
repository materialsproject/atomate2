"""Module defining functions for manipulating amset files."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from monty.serialization import loadfn

from atomate2 import SETTINGS
from atomate2.common.files import copy_files, get_zfile, gunzip_files, rename_files
from atomate2.utils.file_client import FileClient, auto_fileclient
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


@auto_fileclient
def copy_amset_files(
    src_dir: Path | str,
    src_host: str | None = None,
    file_client: FileClient = None,
) -> None:
    """
    Copy AMSET files to current directory.

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

    logger.info(f"Copying AMSET inputs from {src_dir}")
    directory_listing = file_client.listdir(src_dir, host=src_host)

    # find optional files
    files = []
    for file in (
        "settings.yaml",
        "vasprun.xml",
        "band_structure_data.json",
        "wavefunction.h5",
        "deformation.h5",
        "transport.json",
    ):
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

    rename_files({"transport.json": "transport.prev.json"}, allow_missing=True)
    logger.info("Finished copying inputs")


def write_amset_settings(settings_updates: dict, from_prev: bool = False) -> None:
    """
    Write AMSET settings to file.

    This function will also apply any settings specified in
    :obj:`.Atomate2Settings.AMSET_SETTINGS_UPDATE`.

    Parameters
    ----------
    settings_updates : dict
        A dictionary of settings to write.
    from_prev : bool
        Whether apply the settings on top of an existing settings.yaml file in the
        current directory.
    """
    from amset.io import write_settings

    if from_prev:
        settings = loadfn("settings.yaml")
        settings.update(settings_updates)
    else:
        settings = settings_updates

    if SETTINGS.AMSET_SETTINGS_UPDATE is not None:
        settings.update(SETTINGS.AMSET_SETTINGS_UPDATE)

    write_settings(settings, "settings.yaml")
