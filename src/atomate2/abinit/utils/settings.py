"""Settings for abinit flows in atomate2."""

from abipy.flowtk.tasks import TaskManager


def get_abipy_manager(settings):
    """Get abipy manager.

    Parameters
    ----------
    settings
        Abinit atomate2 settings.
    """
    if settings.ABIPY_MANAGER_FILE:
        return TaskManager.from_file(settings.ABIPY_MANAGER_FILE)
    return TaskManager.from_user_config()
