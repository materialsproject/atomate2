"""
Module defining base SIESTA input set and generator.

class ExternalControlAndScripting

Based on User's Guide Siesta 5.4.0
Section: 10 External Control of Siesta
"""

# Metadata

__all__ = ["ExternalControlAndScripting"]

import logging
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class ExternalControlAndScripting(FDFDataclass):
    """
    Data class to manage external control and scripting options for SIESTA input.

    This class handles Lua scripting configuration for SIESTA simulations,
    integrating with `sisl` for FDF input generation, `pymatgen` for structure
    handling, and `ase` for atomic structure manipulations. It provides methods
    to validate and generate FDF blocks for SIESTA's Lua scripting capabilities,
    ensuring compatibility with Atomate2 workflows.

    Attributes
    ----------
        use_lua_scripting (bool): Flag to enable Lua scripting for advanced
            simulation workflow control.
        lua_script (str): Filename of the main Lua script to be executed by
            SIESTA.
        lua_fdf_arguments (OrderedDict[str, Any]): Dictionary of FDF flags
            related to Lua scripting.
        _user_params (Optional[Dict[str, Any]]): Temporary storage for user
            parameters to validate scripting settings.
    """

    # ------------------------------
    # 10 External Control of Siesta
    # ------------------------------

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.ERROR
    )  # Default to show info & errors messages

    _user_params: dict[str, Any] | None = field(
        default=None,
        init=False,  # Prevent user from setting via constructor
        metadata={
            "description": (
                "Temporary storage for user parameters to validate Lua "
                "scripting settings."
            ),
            "SIESTA keyword": None,
        },
    )
    use_lua_scripting: bool = field(
        default=False,
        metadata={
            "description": (
                "A wrapper-level flag to enable the use of Lua scripting for "
                "advanced control over the simulation workflow."
            ),
            "SIESTA keyword": None,
        },
    )
    lua_script: str = field(
        default="",
        metadata={
            "description": (
                "The filename of the main Lua script to be executed by SIESTA."
            ),
            "SIESTA keyword": "Lua.Script",
        },
    )
    lua_fdf_arguments: OrderedDict[str, Any] = field(
        default_factory=OrderedDict,
        metadata={
            "description": "A dictionary for FDF flags related to Lua scripting.",
            "SIESTA keyword": None,
        },
    )

    _registered: ClassVar[bool]

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "Lua.Script",
                # Note: MD.TypeOfRun is NOT registered here
                # It's handled by MolecularDynamicsAndRelaxation
            )
            self.__class__._registered = True  # noqa: SLF001 class-level flag

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["lua.script", "lua_script"]:
                self.lua_script = str(value)
                self.use_lua_scripting = True  # Auto-enable if script provided

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters

        Note:
            MD.TypeOfRun is NOT written here. It remains in the
            MolecularDynamicsAndRelaxation section, where it's set to "LUA"
            when Lua scripting is enabled. This method only writes Lua.Script.
        """
        fdf: dict[str, Any] = {}

        if self.use_lua_scripting and self.lua_script:
            # Only write Lua.Script here
            # MD.TypeOfRun stays in MolecularDynamicsAndRelaxation section
            fdf["Lua.Script"] = self.lua_script

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have Lua scripting parameters
        # These are SIESTA-specific external control options
        return {}

    @classmethod
    def setup_lua_settings(
        cls, user_params: dict[str, Any] | None = None
    ) -> "ExternalControlAndScripting":
        """
        Create and configure an ExternalControlAndScripting instance.

        Based on user parameters, retaining all default values for unspecified
        fields. Processes user-provided parameters (case-insensitive, may
        include dots) to configure Lua scripting settings. Automatically sets
        `use_lua_scripting` to True if `lua.script` is provided in
        `user_params`. Issues warnings for invalid keys and skips them. The
        method validates the configuration and generates the FDF block, making
        it ready for integration with SIESTA input files via `sisl` and
        Atomate2 workflows.

        Args:
            user_params (Optional[Dict[str, Any]]): Dictionary of user-defined
                parameters (case-insensitive, may include dots). Expected key:
                "lua.script" for the Lua script filename. If None or empty, all
                default values are used.

        Returns
        -------
            ExternalControlAndScripting: Configured instance with all fields
                (default and user-specified) and FDF arguments.

        Example:
            >>> lua_settings = ExternalControlAndScripting.setup_lua_settings(
            ...     {"lua.script": "run.lua"}
            ... )
            >>> print(lua_settings.lua_fdf_arguments)
            OrderedDict([('#ExternalControlAndScripting',
            'ExternalControlAndScripting'), ('Lua.Script', 'run.lua')])
            >>> lua_settings = ExternalControlAndScripting.setup_lua_settings()
            >>> print(lua_settings.lua_fdf_arguments)
            OrderedDict([])
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]ExternalControlAndScripting.setup_lua_settings()[/green]"
            )

        # Initialize instance with defaults
        lua_settings_instance = cls()

        # Store user_params for validate method
        lua_settings_instance._user_params = user_params or {}

        # Handle case where user_params is None or empty
        if not lua_settings_instance._user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default "
                    "ExternalControlAndScripting values.[/blue]"
                )
                console.print(
                    f"[blue]user_params: {lua_settings_instance._user_params}[/blue]"
                )
        elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]user_params: {lua_settings_instance._user_params}[/blue]"
            )

        # Check for lua.script in user_params (case-insensitive) and set
        # use_lua_scripting accordingly
        lua_script_value = None
        for key, value in lua_settings_instance._user_params.items():
            if key.lower() == "lua.script":
                lua_script_value = value
                break

        if lua_script_value:
            lua_settings_instance.use_lua_scripting = True
            lua_settings_instance.lua_script = lua_script_value
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Found lua.script in user_params; setting "
                    f"use_lua_scripting=True, "
                    f"lua_script={lua_settings_instance.lua_script}[/blue]"
                )

        # Get valid attribute names (lowercase for comparison), excluding _user_params
        lua_settings_attributes = {
            field.name.lower()
            for field in fields(cls)
            if not field.name.startswith("_")
        }
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]Available ExternalControlAndScripting attributes: "
                f"{lua_settings_attributes}[/blue]"
            )

        # Process remaining user parameters
        for key, value in lua_settings_instance._user_params.items():
            # Skip lua.script as it's already handled
            if key.lower() == "lua.script":
                continue

            # Normalize key: convert to lowercase and replace dots with underscores
            key_normalized = key.lower().replace(".", "_")
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Processing key: {key} -> normalized: "
                    f"{key_normalized}, value: {value}[/blue]"
                )

            # Skip _user_params if provided by user
            if key_normalized == "_user_params":
                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                    console.print(
                        f"[yellow]Ignoring user-provided '{key}'; "
                        f"it is internal.[/yellow]"
                    )
                continue

            # Check if normalized key matches any attribute
            if key_normalized in lua_settings_attributes:
                # Find the original attribute name (preserving case)
                original_key = next(
                    field.name
                    for field in fields(cls)
                    if field.name.lower() == key_normalized
                )
                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                    console.print(
                        f"[blue]Matched ExternalControlAndScripting field: "
                        f"{original_key} = {value}[/blue]"
                    )

                # Handle type conversion for specific fields
                if original_key == "lua_fdf_arguments" and isinstance(value, dict):
                    setattr(lua_settings_instance, original_key, OrderedDict(value))
                elif original_key == "use_lua_scripting":
                    setattr(lua_settings_instance, original_key, bool(value))
                else:
                    setattr(lua_settings_instance, original_key, value)
            elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                console.print(
                    f"[yellow]Key '{key}' does not match any "
                    f"ExternalControlAndScripting field, skipping.[/yellow]"
                )

        # Validate settings
        try:
            lua_settings_instance.validate()
        except ValueError as e:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.ERROR.value:
                console.print(f"[red]Validation failed: {e}[/red]")
            raise

        # Generate FDF block
        lua_settings_instance.generate_scripting_block()

        # Clear _user_params after validation to avoid memory leaks
        lua_settings_instance._user_params = None

        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]Validation & Generation: "
                "[yellow]ExternalControlAndSetting[/yellow] Successful![/green]"
            )

        return lua_settings_instance

    def validate(self) -> None:
        """
        Validate the external control and scripting options for SIESTA.

        Ensures that if Lua scripting is enabled, a valid Lua script filename is
        provided. Raises an error if the configuration is invalid, ensuring
        compatibility with SIESTA's input requirements.

        Raises
        ------
            ValueError: If `use_lua_scripting` is True but no `lua_script` is
                specified.

        Example:
            >>> scripting = ExternalControlAndScripting(
            ...     use_lua_scripting=True, lua_script=""
            ... )
            >>> scripting.validate()
            ValueError: At least one Lua script must be specified if Lua
            scripting is enabled.
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]ExternalControlAndScripting.validate()[/green]")
        if self.use_lua_scripting and not self.lua_script:
            raise ValueError(
                "At least one Lua script must be specified if Lua scripting is enabled."
            )
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]Validation: "
                "[yellow]ExternalControlAndScripting[/yellow] Successful![/green]"
            )

    def generate_scripting_block(self) -> None:
        """
        Generate the scripting options block for the SIESTA FDF file.

        Populates `lua_fdf_arguments` with the necessary FDF entries for Lua
        scripting, including a comment header and, if enabled, `Lua.Script`.
        This method ensures compatibility with `sisl` for FDF file generation
        and Atomate2 workflows.

        Note:
            MD.TypeOfRun is NOT written here. It's written in the
            MolecularDynamicsAndRelaxation section, where it's set to "LUA" when
            Lua scripting is enabled.

        Example:
            >>> scripting = ExternalControlAndScripting(
            ...     use_lua_scripting=True, lua_script="run.lua"
            ... )
            >>> scripting.generate_scripting_block()
            >>> print(scripting.lua_fdf_arguments)
            OrderedDict([('#ExternalControlAndScripting',
            'ExternalControlAndScripting'), ('Lua.Script', 'run.lua')])
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]ExternalControlAndScripting.generate_scripting_block()[/green]"
            )
        self.lua_fdf_arguments = OrderedDict()

        # Collect parameters (only if scripting is enabled)
        if self.use_lua_scripting:
            # Add section header first (ensures proper ordering in FDF file)
            self.lua_fdf_arguments["#ExternalControlAndScripting"] = (
                "ExternalControlAndScripting"
            )
            # Only write Lua.Script
            # MD.TypeOfRun stays in MolecularDynamicsAndRelaxation section
            self.lua_fdf_arguments["Lua.Script"] = f"{self.lua_script}"
