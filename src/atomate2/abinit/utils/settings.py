"""Settings for abinit flows in atomate2."""

from typing import Any

from abipy.flowtk.tasks import TaskManager


def get_abipy_manager(settings: Any) -> TaskManager:
    """Get abipy manager.

    Parameters
    ----------
    settings
        Abinit atomate2 settings.
    """
    if settings.ABIPY_MANAGER_FILE:
        return TaskManager.from_file(settings.ABIPY_MANAGER_FILE)
    return TaskManager.from_user_config()
