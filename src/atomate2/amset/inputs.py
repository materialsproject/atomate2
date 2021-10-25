"""Module defining functions to write amset inputs."""

from typing import Dict

from monty.serialization import dumpfn, loadfn

from atomate2.settings import Settings

__all__ = ["write_amset_settings"]


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
