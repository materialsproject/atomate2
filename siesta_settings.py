"""Standalone SIESTA settings for atomate2siesta.

This module provides settings configuration independent of the main atomate2 package.
Settings can be loaded from:
- YAML configuration file (~/.atomate2siesta.yaml by default)
- Environment variables (with atomate2_ prefix)
- Direct instantiation with custom values
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_FILE_PATH = "~/.atomate2siesta.yaml"
_ENV_PREFIX = "atomate2_"


class ExtendedAtomate2Settings(BaseSettings):
    """Extended settings for atomate2siesta with SIESTA-specific parameters.

    This class is independent of the main atomate2 package and provides
    standalone configuration management for SIESTA workflows.

    Settings are loaded in the following priority order:
    1. Explicitly provided values
    2. Environment variables (with atomate2_ prefix)
    3. Configuration file (~/.atomate2siesta.yaml)
    4. Built-in defaults
    """

    CONFIG_FILE: str = Field(
        _DEFAULT_CONFIG_FILE_PATH, description="File to load alternative defaults from."
    )

    SIESTA_CMD: str = Field(
        "siesta < siesta.fdf > siesta.out",
        description="The default command used to run SIESTA",
    )

    VIBRA_CMD: str = Field(
        "vibra < siesta.fdf > siesta.vibra.out",
        description="The default command used to run Vibra",
    )

    OPTICAL_INPUT_CMD: str = Field(
        "optical_input < siesta.EPSIMG",
        description="The default command used to run optical_input",
    )

    OPTICAL_CMD: str = Field(
        "optical < siesta.EPSIMG",
        description="The default command used to run optical",
    )

    SIESTA_SHOW_BANNER: bool = Field(
        True,
        description="Whether to display the welcome banner and logo on module import",
    )

    SIESTA_SHOW_PARAMETER_EVOLUTION: Literal[
        "none", "user", "diff", "summary", "full"
    ] = Field(
        "summary",
        description=(
            "Control parameter evolution display level:\n"
            "  - 'none': No parameter tracking display\n"
            "  - 'user': Show only initial user-provided parameters\n"
            "  - 'diff': Show only changes (added/modified by dataclasses and powerups)\n"
            "  - 'summary': Show initial + changes summary (default)\n"
            "  - 'full': Show all stages with complete final parameter table"
        ),
    )

    SIESTA_SHOW_DOCSTRINGS: bool = Field(
        True,
        description="Whether to display FlowMaker docstrings in Rich panels when .make() is called",
    )

    SIESTA_ZIP_FILES: Union[bool, Literal["atomate"]] = Field(
        "atomate",
        description=(
            "Determine if the files in the folder are being compressed. If True "
            "all the files are compressed. If 'atomate' only a selection of files related "
            "to the simulation will be compressed. If False no file is compressed."
        ),
    )

    SIESTA_PP_PATH: Optional[str] = Field(
        None, description="The path where files for pseudos are stored."
    )

    FLOS_PATH: Optional[str] = Field(
        None, description="The path where files for FLOS are stored."
    )

    SYMPREC: float = Field(
        0.1, description="Symmetry precision for spglib symmetry finding."
    )

    PHONON_SYMPREC: float = Field(
        1e-4, description="Symmetry precision for phonon calculations."
    )

    ELASTIC_FITTING_METHOD: str = Field(
        "finite_difference",
        description=(
            "Method used for fitting elastic tensors. Options: "
            "'finite_difference' (for 2nd or 3rd order), "
            "'pseudoinverse', or 'independent'."
        ),
    )

    model_config = SettingsConfigDict(env_prefix=_ENV_PREFIX)

    @model_validator(mode="before")
    @classmethod
    def load_default_settings(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load settings from file or environment variables.

        Loads settings from a root file if available and uses that as defaults in
        place of built-in defaults.

        This allows setting of the config file path through environment variables.
        """
        from monty.serialization import loadfn

        logger.info("ExtendedAtomate2Settings.load_default_settings()")
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
