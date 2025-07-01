"""Settings for atomate2."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_CONFIG_FILE_PATH = "~/.atomate2.yaml"
_ENV_PREFIX = "atomate2_"


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
    PHONON_SYMPREC: float = Field(
        1e-3, description="Symmetry precision for spglib symmetry finding."
    )
    SYMPREC: float = Field(
        0.1, description="Symmetry precision for spglib symmetry finding."
    )
    BANDGAP_TOL: float = Field(
        1e-4,
        description="Tolerance for determining if a material is a semiconductor or "
        "metal",
    )
    CUSTODIAN_SCRATCH_DIR: str | None = Field(
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
    VASP_VDW_KERNEL_DIR: str | None = Field(
        None, description="Path to VDW VASP kernel."
    )
    VASP_INCAR_UPDATES: dict = Field(
        default_factory=dict, description="Updates to apply to VASP INCAR files."
    )
    VASP_VOLUME_CHANGE_WARNING_TOL: float = Field(
        0.2,
        description="Maximum volume change allowed in VASP relaxations before the "
        "calculation is tagged with a warning",
    )
    VASP_HANDLE_UNSUCCESSFUL: Union[bool, Literal["error"]] = Field(
        "error",
        description="Three-way toggle on what to do if the job looks OK but is actually"
        " unconverged (either electronic or ionic). - True: mark job as COMPLETED, but "
        "stop children. - False: do nothing, continue with workflow as normal. 'error':"
        " throw an error",
    )
    VASP_CUSTODIAN_MAX_ERRORS: int = Field(
        5, description="Maximum number of errors to correct before custodian gives up"
    )
    VASP_STORE_VOLUMETRIC_DATA: tuple[str] | None = Field(
        None, description="Store data from these files in database if present"
    )
    VASP_STORE_ADDITIONAL_JSON: bool = Field(
        default=True,
        description="Ingest any additional JSON data present into database when "
        "parsing VASP directories useful for storing duplicate of FW.json",
    )
    VASP_RUN_BADER: bool = Field(
        default=False,
        description="Whether to run the Bader program when parsing VASP calculations."
        "Requires the bader executable to be on the path.",
    )
    VASP_RUN_DDEC6: bool = Field(
        default=False,
        description="Whether to run the DDEC6 program when parsing VASP calculations."
        "Requires the chargemol executable to be on the path.",
    )
    DDEC6_ATOMIC_DENSITIES_DIR: str | None = Field(
        default=None,
        description="Directory where the atomic densities are stored.",
        # TODO uncomment below once that functionality is actually implemented
        # If not set, pymatgen tries to auto-download the densities and extract them
        # into ~/.cache/pymatgen/ddec
    )

    VASP_ZIP_FILES: Union[bool, Literal["atomate"]] = Field(
        "atomate",
        description="Determine if the files in folder are being compressed. If True "
        "all the files are compressed. If 'atomate' only a selection of files related "
        "to the simulation will be compressed. If False no file is compressed.",
    )
    VASP_INHERIT_INCAR: bool = Field(
        default=False,
        description="Whether to inherit INCAR settings from previous calculation. "
        "This might be useful to port Custodian fixes to child jobs but can also be "
        "dangerous e.g. when switching from GGA to meta-GGA or relax to static jobs."
        "Can be overridden on a per-job basis via the inherit_incar keyword of "
        "VaspInputGenerator.",
    )

    LOBSTER_CMD: str = Field(
        default="lobster", description="Command to run standard version of VASP."
    )

    LOBSTER_CUSTODIAN_MAX_ERRORS: int = Field(
        5, description="Maximum number of errors to correct before custodian gives up"
    )

    LOBSTER_ZIP_FILES: Union[bool, Literal["atomate"]] = Field(
        "atomate",
        description="Determine if the files in folder are being compressed. If True "
        "all the files are compressed. If 'atomate' only a selection of files related "
        "to the simulation will be compressed. If False no file is compressed.",
    )

    CP2K_CMD: str = Field(
        "cp2k.psmp", description="Command to run the MPI version of cp2k"
    )
    CP2K_RUN_BADER: bool = Field(
        default=False,
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
    CP2K_STORE_VOLUMETRIC_DATA: tuple[str] | None = Field(
        None, description="Store data from these files in database if present"
    )
    CP2K_STORE_ADDITIONAL_JSON: bool = Field(
        default=True,
        description="Ingest any additional JSON data present into database when "
        "parsing CP2K directories useful for storing duplicate of FW.json",
    )

    CP2K_ZIP_FILES: Union[bool, Literal["atomate"]] = Field(
        default=True,
        description="Determine if the files in folder are being compressed. If True "
        "all the files are compressed. If 'atomate' only a selection of files related "
        "to the simulation will be compressed. If False no file is compressed.",
    )

    # FHI-aims settings
    AIMS_CMD: str = Field(
        "aims.x > aims.out", description="The default command used run FHI-aims"
    )

    # Elastic constant settings
    ELASTIC_FITTING_METHOD: str = Field(
        "finite_difference", description="Elastic constant fitting method"
    )

    # AMSET settings
    AMSET_SETTINGS_UPDATE: dict | None = Field(
        None, description="Additional settings applied to AMSET settings file."
    )

    # ABINIT settings
    ABINIT_MPIRUN_CMD: str | None = Field(None, description="Mpirun command.")
    ABINIT_CMD: str = Field("abinit", description="Abinit command.")
    ABINIT_MRGDDB_CMD: str = Field("mrgddb", description="Mrgddb command.")
    ABINIT_ANADDB_CMD: str = Field("anaddb", description="Anaddb command.")
    ABINIT_COPY_DEPS: bool = Field(
        default=False,
        description="Copy (True) or link file dependencies between jobs.",
    )
    ABINIT_AUTOPARAL: bool = Field(
        default=False,
        description="Use autoparal to determine optimal parallel configuration.",
    )
    ABINIT_ABIPY_MANAGER_FILE: str | None = Field(
        None,
        description="Config file for task manager of abipy.",
    )
    ABINIT_MAX_RESTARTS: int = Field(
        5, description="Maximum number of restarts of a job."
    )

    model_config = SettingsConfigDict(env_prefix=_ENV_PREFIX)

    # QChem specific settings

    QCHEM_CMD: str = Field(
        "qchem", description="Command to run standard version of qchem."
    )

    QCHEM_CUSTODIAN_MAX_ERRORS: int = Field(
        5, description="Maximum number of errors to correct before custodian gives up"
    )

    QCHEM_MAX_CORES: int = Field(4, description="Maximum number of cores for QCJob")

    QCHEM_HANDLE_UNSUCCESSFUL: Union[str, bool] = Field(
        "fizzle",
        description="Three-way toggle on what to do if the job looks OK but is actually"
        " unconverged (either electronic or ionic). - True: mark job as COMPLETED, but "
        "stop children. - False: do nothing, continue with workflow as normal. 'error':"
        " throw an error",
    )

    QCHEM_STORE_ADDITIONAL_JSON: bool = Field(
        default=True,
        description="Ingest any additional JSON data present into database when "
        "parsing QChem directories useful for storing duplicate of FW.json",
    )

    @model_validator(mode="before")
    @classmethod
    def load_default_settings(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load settings from file or environment variables.

        Loads settings from a root file if available and uses that as defaults in
        place of built-in defaults.

        This allows setting of the config file path through environment variables.
        """
        from monty.serialization import loadfn

        config_file_path = values.get(key := "CONFIG_FILE", _DEFAULT_CONFIG_FILE_PATH)
        env_var_name = f"{_ENV_PREFIX.upper()}{key}"
        config_file_path = Path(config_file_path).expanduser()

        new_values = {}
        if config_file_path.exists():
            if config_file_path.stat().st_size == 0:
                warnings.warn(
                    f"Using {env_var_name} at {config_file_path} but it's empty",
                    stacklevel=2,
                )
            else:
                try:
                    new_values.update(loadfn(config_file_path))
                except ValueError:
                    raise SyntaxError(
                        f"{env_var_name} at {config_file_path} is unparsable"
                    ) from None
        # warn if config path is not the default but file doesn't exist
        elif config_file_path != Path(_DEFAULT_CONFIG_FILE_PATH).expanduser():
            warnings.warn(
                f"{env_var_name} at {config_file_path} does not exist", stacklevel=2
            )

        return new_values | values
