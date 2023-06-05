"""Settings for atomate2."""

from pathlib import Path
from typing import Optional, Tuple, Union

from pydantic import BaseSettings, Field, root_validator

_DEFAULT_CONFIG_FILE_PATH = "~/.atomate2.yaml"

__all__ = ["Atomate2Settings"]


class Atomate2Settings(BaseSettings):
    """
    Settings for atomate2.

    The default way to modify these is to modify ~/.atomate2.yaml. Alternatively,
    the environment variable ATOMATE2_CONFIG_FILE can be set to point to a yaml file
    with atomate2 settings.

    Lastly, the variables can be modified directly through environment variables by
    using the "ATOMATE2" prefix. E.g. ATOMATE2_SCRATCH_DIR = path/to/scratch.
    """

    CONFIG_FILE: str = Field(
        _DEFAULT_CONFIG_FILE_PATH, description="File to load alternative defaults from."
    )

    # general settings
    SYMPREC: float = Field(
        0.1, description="Symmetry precision for spglib symmetry finding."
    )
    CUSTODIAN_SCRATCH_DIR: str = Field(
        None, description="Path to scratch directory used by custodian."
    )

    # VASP specific settings
    VASP_CMD: str = Field(
        "vasp_std", description="Command to run standard version of VASP."
    )
    VASP_GAMMA_CMD: str = Field(
        "vasp_gam", description="Command to run gamma-only version of VASP."
    )
    VASP_NCL_CMD: str = Field(
        "vasp_ncl", description="Command to run non-collinear version of VASP."
    )
    VASP_MIN_VERSION: float = Field(
        5.4,
        description="Minimum version of VASP you plan to run. Used for INCAR validation.",
    )
    VASP_VDW_KERNEL_DIR: str = Field(None, description="Path to VDW VASP kernel.")
    VASP_INCAR_UPDATES: dict = Field(
        default_factory=dict, description="Updates to apply to VASP INCAR files."
    )
    VASP_RELAX_MAX_FORCE: float = Field(
        0.25,
        description="Maximum force allowed on each atom for successful structure "
        "optimization",
    )
    VASP_VOLUME_CHANGE_WARNING_TOL: float = Field(
        0.2,
        description="Maximum volume change allowed in VASP relaxations before the "
        "calculation is tagged with a warning",
    )
    VASP_HANDLE_UNSUCCESSFUL: Union[str, bool] = Field(
        "error",
        description="Three-way toggle on what to do if the job looks OK but is actually"
        " unconverged (either electronic or ionic). - True: mark job as COMPLETED, but "
        "stop children. - False: do nothing, continue with workflow as normal. 'error':"
        " throw an error",
    )
    VASP_CUSTODIAN_MAX_ERRORS: int = Field(
        5, description="Maximum number of errors to correct before custodian gives up"
    )
    VASP_STORE_VOLUMETRIC_DATA: Optional[Tuple[str]] = Field(
        None, description="Store data from these files in database if present"
    )
    VASP_STORE_ADDITIONAL_JSON: bool = Field(
        True,
        description="Ingest any additional JSON data present into database when "
        "parsing VASP directories useful for storing duplicate of FW.json",
    )
    VASP_RUN_BADER: bool = Field(
        False,
        description="Whether to run the Bader program when parsing VASP calculations."
        "Requires the bader executable to be on the path.",
    )

    LOBSTER_CMD: str = Field(
        default="lobster", description="Command to run standard version of VASP."
    )

    LOBSTER_CUSTODIAN_MAX_ERRORS: int = Field(
        5, description="Maximum number of errors to correct before custodian gives up"
    )

    CP2K_CMD: str = Field(
        "cp2k.psmp", description="Command to run the MPI version of cp2k"
    )
    CP2K_RUN_BADER: bool = Field(
        False,
        description="Whether to run the Bader program when parsing CP2K calculations."
        "Requires the bader executable to be on the path.",
    )
    CP2K_INPUT_UPDATES: dict = Field(
        default_factory=dict, description="Updates to apply to cp2k input files."
    )
    CP2K_RELAX_MAX_FORCE: float = Field(
        0.25,
        description="Maximum force allowed on each atom for successful structure "
        "optimization",
    )
    CP2K_VOLUME_CHANGE_WARNING_TOL: float = Field(
        0.2,
        description="Maximum volume change allowed in CP2K relaxations before the "
        "calculation is tagged with a warning",
    )
    CP2K_HANDLE_UNSUCCESSFUL: Union[str, bool] = Field(
        "error",
        description="Three-way toggle on what to do if the job looks OK but is actually"
        " unconverged (either electronic or ionic). - True: mark job as COMPLETED, but "
        "stop children. - False: do nothing, continue with workflow as normal. 'error':"
        " throw an error",
    )
    CP2K_CUSTODIAN_MAX_ERRORS: int = Field(
        5, description="Maximum number of errors to correct before custodian gives up"
    )
    CP2K_STORE_VOLUMETRIC_DATA: Optional[Tuple[str]] = Field(
        None, description="Store data from these files in database if present"
    )
    CP2K_STORE_ADDITIONAL_JSON: bool = Field(
        True,
        description="Ingest any additional JSON data present into database when "
        "parsing CP2K directories useful for storing duplicate of FW.json",
    )

    # Elastic constant settings
    ELASTIC_FITTING_METHOD: str = Field(
        "finite_difference", description="Elastic constant fitting method"
    )

    # AMSET settings
    AMSET_SETTINGS_UPDATE: dict = Field(
        None, description="Additional settings applied to AMSET settings file."
    )

    class Config:
        """Pydantic config settings."""

        env_prefix = "atomate2_"

    @root_validator(pre=True)
    def load_default_settings(cls, values):
        """
        Load settings from file or environment variables.

        Loads settings from a root file if available and uses that as defaults in
        place of built-in defaults.

        This allows setting of the config file path through environment variables.
        """
        from monty.serialization import loadfn

        config_file_path: str = values.get("CONFIG_FILE", _DEFAULT_CONFIG_FILE_PATH)

        new_values = {}
        if Path(config_file_path).expanduser().exists():
            new_values.update(loadfn(Path(config_file_path).expanduser()))

        new_values.update(values)
        return new_values
