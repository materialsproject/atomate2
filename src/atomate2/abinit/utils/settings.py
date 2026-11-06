"""Settings for ABINIT flows in atomate2."""

from typing import Any

from abipy.flowtk.tasks import TaskManager

__all__ = ["get_abipy_manager"]


def get_abipy_manager(settings: Any) -> TaskManager:
    """
    Get the AbiPy manager for managing ABINIT calculations.

    Retrieves the TaskManager either from a file specified in settings or
    from the user's default configuration.

    Parameters
    ----------
    settings : Any
        ABINIT atomate2 settings containing ABIPY_MANAGER_FILE attribute.

    Returns
    -------
    TaskManager
        The AbiPy manager (TaskManager) object for managing ABINIT tasks.
    """
    if settings.ABIPY_MANAGER_FILE:
        return TaskManager.from_file(settings.ABIPY_MANAGER_FILE)
    return TaskManager.from_user_config()
