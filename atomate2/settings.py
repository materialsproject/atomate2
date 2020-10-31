from pathlib import Path
from typing import Optional, Tuple, Union

from monty.serialization import loadfn
from pydantic import BaseSettings, Field, root_validator

DEFAULT_CONFIG_FILE_PATH = str(Path.home() / ".atomate.json")


class Settings(BaseSettings):
    """
    Settings for atomate2.

    The default way to modify these is to modify ~/.atomate.yaml. Alternatively,
    the environment variable ATOMATE_CONFIG_FILE can be set to point to a yaml file with
    atomate2 settings.

    Lastly, the variables can be modified directly though environment variables by
    using the ATOMATE_ prefix. E..g., ATOMATE_SCRATCH_DIR = path/to/scratch.
    """

    config_file: str = Field(
        DEFAULT_CONFIG_FILE_PATH, description="File to load alternative defaults from"
    )

    # general settings
    scratch_dir: str = Field(
        ">>scratch_dir<<", description="Path to scratch directory (supports env_check)"
    )
    db_file: str = Field(
        ">>db_file<<", description="Path to database file (supports env_check)"
    )

    # VASP specific settings
    vasp_cmd: str = Field(
        ">>vasp_gamma_cmd<<", description="Command to run standard version of VASP"
    )
    vasp_gamma_cmd: str = Field(
        ">>vasp_gamma_cmd<<", description="Command to run gamma only version of VASP"
    )
    vasp_vdw_kernel_dir: str = Field(
        ">>vdw_kernel_dir<<", description="Path to VDW VASP kernel"
    )
    vasp_add_namefile: bool = Field(
        True, description="Whether vasp.powerups.add_common_powerups adds a namefile"
    )
    vasp_add_smallgap_kpoint_multiply: bool = Field(
        True,
        description="Whether vasp.powerups.add_common_powerups adds a small gap "
        "multiply task for static and NSCF calculations",
    )
    vasp_add_modify_incar: bool = Field(
        False,
        description="Whether vasp.powerups.add_common_powerups adds a modify incar "
        "task",
    )
    vasp_add_stability_check: bool = Field(
        False,
        description="Whether vasp.powerups.add_common_powerups adds a stability check "
        "task for structure optimization calculations",
    )
    vasp_add_wf_metadata: bool = Field(
        False,
        description="Whether vasp.powerups.add_common_powerups adds structure metadata "
        "to a workflow",
    )
    vasp_half_kpoints_first_relax: bool = Field(
        False,
        description="Whether to use only half the k-point density in the initial"
        "relaxation of a structure optimization for faster performance",
    )
    vasp_relax_max_force: float = Field(
        0.25,
        description="Maximum force allowed on each atom for successful structure "
        "optimization",
    )
    vasp_volume_charge_warning_tol: float = Field(
        0.2,
        description="Maximum volume change allowed in VASP relaxations before the "
        "calculation is tagged with a warning",
    )
    vasp_defuse_unsuccessful: Union[str, bool] = Field(
        "fizzle",
        description="Three-way toggle on what to do if the job looks OK but is actually"
        "unconverged (either electronic or ionic). "
        "True -> mark job as COMPLETED, but defuse children. "
        "False --> do nothing, continue with workflow as normal. "
        "'fizzle' --> throw an error (mark this job as FIZZLED)",
    )
    vasp_custodian_max_errors: int = Field(
        5, description="Maximum number of errors to correct before custodian gives up"
    )
    vasp_store_volumetric_data: Optional[Tuple[str]] = Field(
        None, description="Store data from these files in database if present"
    )
    vasp_store_additional_json: bool = Field(
        False,
        description="Ingest any additional JSON data present into database when "
        "parsing VASP directories useful for storing duplicate of FW.json",
    )
    vasp_run_bader: bool = Field(
        False,
        description="Whether to run the Bader program when parsing VASP calculations."
        "Requires the bader command to be on the path.",
    )

    class Config:
        env_prefix = "atomate"

    @root_validator(pre=True)
    def load_default_settings(cls, values):
        """
        Loads settings from a root file if available and uses that as defaults in
        place of built in defaults.

        This allows setting of the config file path through environment variables.
        """
        config_file_path: str = values.get("config_file", DEFAULT_CONFIG_FILE_PATH)

        new_values = {}
        if Path(config_file_path).exists():
            new_values.update(loadfn(config_file_path))

        return new_values


settings = Settings()
