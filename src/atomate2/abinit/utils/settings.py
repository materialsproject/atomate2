"""Settings for abinit flows in atomate2."""

from abipy.flowtk.tasks import TaskManager
from pydantic import BaseSettings, Field


class AbinitAtomateSettings(BaseSettings):
    """Settings for abinit in atomate2."""

    MPIRUN_CMD: str = Field("mpirun", description="Mpirun command.")

    # ABINIT specific settings
    ABINIT_CMD: str = Field("abinit", description="Abinit command.")
    MRGDDB_CMD: str = Field("mrgddb", description="Mrgddb command.")
    ANADDB_CMD: str = Field("anaddb", description="Anaddb command.")
    COPY_DEPS: bool = Field(
        False,
        description="Copy (True) or link file dependencies between jobs.",
    )
    AUTOPARAL: bool = Field(
        False,
        description="Use autoparal to determine optimal parallel configuration.",
    )
    ABIPY_MANAGER_FILE: str = Field(
        None,
        description="Config file for task manager of abipy.",
    )
    MAX_RESTARTS: int = Field(5, description="Maximum number of restarts of a job.")


def get_abipy_manager(settings):
    """Get abipy manager.

    Parameters
    ----------
    settings
        Abinit atomate2 settings.
    """
    if settings.ABIPY_MANAGER_FILE:
        return TaskManager.from_file(settings.ABIPY_MANAGER_FILE)
    try:
        return TaskManager.from_user_config()
    except RuntimeError:
        # logger.warning("Couldn't load the abipy task manager.")
        return None
