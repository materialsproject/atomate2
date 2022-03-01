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


# SETTINGS = AbinitAtomateSettings()
# print(SETTINGS.dict())
# print(get_abipy_manager(SETTINGS))
# print(SETTINGS.ABIPY_MANAGER)

# class FWTaskManager(object):
#     """
#     Object containing the configuration parameters and policies to run abipy.
#     The policies needed for the abinit FW will always be available through default values. These can be overridden
#     also setting the parameters in the spec.
#     The standard abipy task manager is contained as an object on its own that can be used to run the autoparal or
#     factories if needed.
#     The rationale behind this choice, instead of subclassing, is to not force the user to fill the qadapter part
#     of the task manager, which is needed only for the autoparal, but is required in the TaskManager initialization.
#     Wherever the TaskManager is needed just pass the ftm.task_manager.
#     The TaskManager part can be loaded from an external manager.yml file using the "abipy_manager" key in fw_policy.
#     This is now the preferred choice. If this value is not defined, it will be loaded with TaskManager.from_user_config
#     """
#
#     YAML_FILE = "fw_manager.yaml"
#     USER_CONFIG_DIR = TaskManager.USER_CONFIG_DIR # keep the same as the standard TaskManager
#
#     fw_policy_defaults = dict(rerun_same_dir=False,
#                               max_restarts=10,
#                               autoparal=False,
#                               abinit_cmd='abinit',
#                               mrgddb_cmd='mrgddb',
#                               anaddb_cmd='anaddb',
#                               cut3d_cmd='cut3d',
#                               mpirun_cmd='mpirun',
#                               copy_deps=False,
#                               walltime_command=None,
#                               continue_unconverged_on_rerun=True,
#                               allow_local_restart=False,
#                               timelimit_buffer=120,
#                               short_job_timelimit=600,
#                               recover_previous_job=True,
#                               abipy_manager=None)
#     FWPolicy = namedtuple("FWPolicy", fw_policy_defaults.keys())
#
#     def __init__(self, **kwargs):
#         self._kwargs = copy.deepcopy(kwargs)
#
#         fw_policy = kwargs.pop('fw_policy', {})
#         unknown_keys = set(fw_policy.keys()) - set(self.fw_policy_defaults.keys())
#         if unknown_keys:
#             msg = "Unknown key(s) present in fw_policy: {}".format(", ".join(unknown_keys))
#             logger.error(msg)
#             raise RuntimeError(msg)
#         fw_policy = dict(self.fw_policy_defaults, **fw_policy)
#
#         # make a namedtuple for easier access to the attributes
#         self.fw_policy = self.FWPolicy(**fw_policy)
#
#         #TODO consider if raising an exception if it's requested when not available
#         # create the task manager only if possibile
#         if 'qadapters' in kwargs:
#             self.task_manager = TaskManager.from_dict(kwargs)
#             msg = "Loading the abipy TaskManager from inside the fw_manager.yaml file is deprecated. " \
#                   "Use a separate file"
#             logger.warning(msg)
#             warnings.warn(msg, DeprecationWarning, stacklevel=2)
#         else:
#             if self.fw_policy.abipy_manager:
#                 self.task_manager = TaskManager.from_file(self.fw_policy.abipy_manager)
#             else:
#                 try:
#                     self.task_manager = TaskManager.from_user_config()
#                 except Exception:
#                     logger.warning("Couldn't load the abipy task manager.")
#                     self.task_manager = None
#
#     @classmethod
#     def from_user_config(cls, fw_policy=None):
#         """
#         Initialize the manager using the dict in the following order of preference:
#         - the "fw_manager.yaml" file in the folder where the command is executed
#         - a yaml file pointed by the "FW_TASK_MANAGER"
#         - the "fw_manager.yaml" in the ~/.abinit/abipy folder
#         - if no file available, fall back to default values
#         """
#
#         if fw_policy is None:
#             fw_policy = {}
#
#         # Try in the current directory then in user configuration directory.
#         paths = [os.path.join(os.getcwd(), cls.YAML_FILE), os.getenv("FW_TASK_MANAGER"),
#                  os.path.join(cls.USER_CONFIG_DIR, cls.YAML_FILE)]
#
#         config = {}
#         for path in paths:
#             if path and os.path.exists(path):
#                 with io.open(path, "rt", encoding="utf-8") as fh:
#                     config = yaml.safe_load(fh)
#                 logger.info("Reading manager from {}.".format(path))
#                 break
#
#         return cls(**config)
#
#     @classmethod
#     def from_file(cls, path):
#         """Read the configuration parameters from the Yaml file filename."""
#         with open(path, "rt") as fh:
#             d = yaml.safe_load(fh)
#         return cls(**d)
#
#     def has_task_manager(self):
#         return self.task_manager is not None
#
#     def update_fw_policy(self, d):
#         self.fw_policy = self.fw_policy._replace(**d)
