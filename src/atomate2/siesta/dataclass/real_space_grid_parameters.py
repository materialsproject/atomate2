"""
Module defining base SIESTA input set and generator.

class RealSpaceGridParameters

Based on User's Guide Siesta 5.4.0
Section:  6.10 The real-space grid and the eggbox-effect
"""

# Metadata

__all__ = ["RealSpaceGridParameters"]

from dataclasses import dataclass, field, fields
from typing import Dict, Any
from typing import Optional
from collections import OrderedDict


from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.dataclass.units import parse_energy
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

import logging

logger = logging.getLogger(__name__)


@dataclass
class RealSpaceGridParameters(FDFDataclass):
    """
    Data class to manage real-space grid and eggbox effect parameters for SIESTA input.
    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.INFO
    )  # Default to show info messages

    # -----------------------------------------------
    # 6.10 The real-space grid and the eggbox-effect
    # -----------------------------------------------

    _user_params: Optional[Dict[str, Any]] = field(
        default=None,
        init=False,  # Prevent user from setting via constructor
        metadata={
            "description": "Temporary storage for user parameters to validate grid settings.",
            "SIESTA keyword": None,
        },
    )
    mesh_cutoff: float = field(
        default=100.0,
        metadata={
            "description": "Sets the energy cutoff (in Rydberg) that determines the fineness of the real-space grid. This is a crucial accuracy parameter.",
            "SIESTA keyword": "Mesh.Cutoff",
            "unit": "Ry",
        },
    )
    mesh_sizes_block: Dict[int, Any] = field(
        default_factory=dict,
        metadata={
            "description": "A block to manually specify the number of grid points along each lattice vector, offering direct control over the grid dimensions as an alternative to 'Mesh.Cutoff'.",
            "SIESTA keyword": "%block Mesh.Sizes",
        },
    )
    mesh_subdivisions: int = field(
        default=2,
        metadata={
            "description": "For parallel calculations, this specifies the number of subdivisions of the real-space grid for domain decomposition.",
            "SIESTA keyword": "Mesh.SubDivisions",
        },
    )
    grid_cell_sampling: Dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": "A block to define a non-uniform real-space grid, allowing for different grid density in different regions of the simulation cell.",
            "SIESTA keyword": "%block Grid.CellSampling",
        },
    )
    eggbox_remove_block: Optional[Dict[float, Any]] = field(
        default_factory=dict,
        metadata={
            "description": "A block to enable and configure a scheme to mitigate the 'egg-box' effect, which is an artificial energy corrugation due to the discrete real-space grid.",
            "SIESTA keyword": "%block EggboxRemove",
        },
    )
    eggbox_scale: float = field(
        default=1.0,
        metadata={
            "description": "An energy scale parameter (in eV) used by the 'EggboxRemove' correction scheme.",
            "SIESTA keyword": "EggboxScale",
            "unit": "eV",
        },
    )
    grid_fdf_arguments: OrderedDict[str, Any] = field(
        default_factory=OrderedDict,
        metadata={
            "description": "A dictionary for FDF flags related to real-space grid parameters.",
            "SIESTA keyword": None,
        },
    )

    def __post_init__(self):
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "Mesh.Cutoff",
                "Mesh.SubDivisions",
                "%block Mesh.Sizes",
                "%block Grid.CellSampling",
                "%block EggboxRemove",
                "EggboxScale",
            )
            self.__class__._registered = True

    @classmethod
    def setup_grid_settings(
        cls, user_params: Optional[Dict[str, Any]] = None
    ) -> "RealSpaceGridParameters":
        """
        Create and configure a RealSpaceGridParameters instance based on user parameters, retaining all default values for unspecified fields.
        Issues warnings for invalid keys and skips them.

        Args:
            user_params (dict, optional): Dictionary of user-defined parameters (case-insensitive, may include dots).
                                         If None or empty, all default RealSpaceGridParameters values are used.

        Returns:
            RealSpaceGridParameters: Configured instance with all fields (default and user-specified) and FDF arguments.
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]RealSpaceGridParameters.setup_grid_settings()[/green]"
            )

        # Initialize instance with defaults
        grid_settings_instance = cls()

        # Store user_params for validate method
        grid_settings_instance._user_params = user_params if user_params else {}

        # Handle case where user_params is None or empty
        if not grid_settings_instance._user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default RealSpaceGridParameters values.[/blue]"
                )
                console.print(
                    f"[blue]user_params: {grid_settings_instance._user_params}[/blue]"
                )
        else:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]user_params: {grid_settings_instance._user_params}[/blue]"
                )

            # Get valid attribute names (lowercase for comparison), excluding _user_params
            grid_settings_attributes = {
                field.name.lower()
                for field in fields(cls)
                if not field.name.startswith("_")
            }
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Available RealSpaceGridParameters attributes: {grid_settings_attributes}[/blue]"
                )

            # Process user parameters
            for key, value in grid_settings_instance._user_params.items():
                # Normalize key: convert to lowercase and replace dots with underscores
                key_normalized = key.lower().replace(".", "_")
                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                    console.print(
                        f"[blue]Processing key: {key} -> normalized: {key_normalized}, value: {value}[/blue]"
                    )

                # Skip _user_params if provided by user
                if key_normalized == "_user_params":
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Ignoring user-provided '{key}'; it is internal.[/yellow]"
                        )
                    continue

                # Check if normalized key matches any attribute
                if key_normalized in grid_settings_attributes:
                    # Find the original attribute name (preserving case)
                    original_key = next(
                        field.name
                        for field in fields(cls)
                        if field.name.lower() == key_normalized
                    )
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                        console.print(
                            f"[blue]Matched RealSpaceGridParameters field: {original_key} = {value}[/blue]"
                        )

                    # Handle type conversion for specific fields
                    if original_key in [
                        "mesh_sizes_block",
                        "grid_cell_sampling",
                        "eggbox_remove_block",
                    ] and isinstance(value, dict):
                        setattr(grid_settings_instance, original_key, value)
                    elif original_key == "mesh_cutoff":
                        # Handle values with units like "300 Ry" or just numbers
                        if isinstance(value, str):
                            import re

                            match = re.search(r"[\d.]+", value)
                            if match:
                                setattr(
                                    grid_settings_instance,
                                    original_key,
                                    float(match.group()),
                                )
                            else:
                                setattr(
                                    grid_settings_instance, original_key, float(value)
                                )
                        else:
                            setattr(grid_settings_instance, original_key, float(value))
                    elif original_key == "mesh_subdivisions":
                        setattr(grid_settings_instance, original_key, int(value))
                    elif original_key == "eggbox_scale":
                        if isinstance(value, str):
                            import re

                            match = re.search(r"[\d.]+", value)
                            if match:
                                setattr(
                                    grid_settings_instance,
                                    original_key,
                                    float(match.group()),
                                )
                            else:
                                setattr(
                                    grid_settings_instance, original_key, float(value)
                                )
                        else:
                            setattr(grid_settings_instance, original_key, float(value))
                    else:
                        setattr(grid_settings_instance, original_key, value)
                else:
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Key '{key}' does not match any RealSpaceGridParameters field, skipping.[/yellow]"
                        )

        # Validate settings
        try:
            grid_settings_instance.validate()
        except ValueError as e:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.ERROR.value:
                console.print(f"[red]Validation failed: {e}[/red]")
            raise

        # Generate FDF block
        grid_settings_instance.generate_grid_block()

        # Clear _user_params after validation to avoid memory leaks
        grid_settings_instance._user_params = None

        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]Validation & Generation: [yellow]RealSpaceGridParameters[/yellow] Successful![/green]"
            )

        return grid_settings_instance

    def validate(self):
        """
        Validates the real-space grid parameters.

        Raises:
            ValueError: If mesh_cutoff, mesh_subdivisions, or eggbox_scale are invalid.
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]RealSpaceGridParameters.validate()[/green]")

        # Validate mesh_cutoff
        if not isinstance(self.mesh_cutoff, (int, float)) or self.mesh_cutoff <= 0:
            raise ValueError(
                f"Mesh.Cutoff must be a positive number, got '{self.mesh_cutoff}'"
            )

        # Validate mesh_subdivisions
        if not isinstance(self.mesh_subdivisions, int) or self.mesh_subdivisions <= 0:
            raise ValueError(
                f"Mesh.SubDivisions must be a positive integer, got '{self.mesh_subdivisions}'"
            )

        # Validate eggbox_scale
        if not isinstance(self.eggbox_scale, (int, float)) or self.eggbox_scale <= 0:
            raise ValueError(
                f"EggboxScale must be a positive number, got '{self.eggbox_scale}'"
            )

        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]Validation: [yellow]RealSpaceGridParameters[/yellow] Successful![/green]"
            )

    def update_from_fdf(self, fdf_dict: Dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)

        Examples:
            >>> grid = RealSpaceGridParameters()
            >>> grid.update_from_fdf({"Mesh.Cutoff": "450 Ry"})
            >>> grid.mesh_cutoff
            450.0
            >>> grid.update_from_fdf({"Mesh.Cutoff": "10 eV"})
            >>> grid.mesh_cutoff  # Converted to Ry
            0.735...
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]RealSpaceGridParameters.update_from_fdf()[/green]")

        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower == "mesh.cutoff":
                # Parse energy with unit conversion (e.g., "450 Ry" or "10 eV")
                self.mesh_cutoff = parse_energy(value, target_unit="Ry")

            elif key_lower == "mesh.subdivisions":
                self.mesh_subdivisions = int(value)

            elif key_lower == "%block mesh.sizes":
                if isinstance(value, dict):
                    self.mesh_sizes_block = value

            elif key_lower == "%block grid.cellsampling":
                if isinstance(value, dict):
                    self.grid_cell_sampling = value

            elif key_lower == "%block eggboxremove":
                if isinstance(value, dict):
                    self.eggbox_remove_block = value

            elif key_lower == "eggboxscale":
                # Parse energy with unit conversion (e.g., "1.0 eV")
                self.eggbox_scale = parse_energy(value, target_unit="eV")

    def generate_fdf(self) -> Dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns:
            Dictionary of FDF parameters with proper units

        Examples:
            >>> grid = RealSpaceGridParameters(mesh_cutoff=450.0)
            >>> fdf = grid.generate_fdf()
            >>> fdf["Mesh.Cutoff"]
            '450.0 Ry'
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]RealSpaceGridParameters.generate_fdf()[/green]")

        fdf = OrderedDict()
        fdf["#RealSpaceGridParameters"] = "RealSpaceGridParameters"

        # Mesh.Cutoff - always write with default marker
        if self.mesh_cutoff == 100.0:
            fdf["Mesh.Cutoff"] = f"{self.mesh_cutoff} Ry  # SIESTA DEFAULT VALUE"
        else:
            fdf["Mesh.Cutoff"] = f"{self.mesh_cutoff} Ry"

        # Mesh.SubDivisions - always write with default marker
        if self.mesh_subdivisions == 2:
            fdf[
                "Mesh.SubDivisions"
            ] = f"{self.mesh_subdivisions}  # SIESTA DEFAULT VALUE"
        else:
            fdf["Mesh.SubDivisions"] = f"{self.mesh_subdivisions}"

        # EggboxScale - always write with default marker
        if self.eggbox_scale == 1.0:
            fdf["EggboxScale"] = f"{self.eggbox_scale} eV  # SIESTA DEFAULT VALUE"
        else:
            fdf["EggboxScale"] = f"{self.eggbox_scale} eV"

        # Add blocks if provided
        if self.mesh_sizes_block:
            fdf["%block Mesh.Sizes"] = self.mesh_sizes_block
        if self.grid_cell_sampling:
            fdf["%block Grid.CellSampling"] = self.grid_cell_sampling
        if self.eggbox_remove_block:
            fdf["%block EggboxRemove"] = self.eggbox_remove_block

        return fdf

    def to_ase(self) -> Dict[str, Any]:
        """
        Generate ASE-format parameters (optional).

        Returns:
            Dictionary of ASE parameters (ASE uses 'mesh' instead of 'Mesh.Cutoff')

        Examples:
            >>> grid = RealSpaceGridParameters(mesh_cutoff=450.0)
            >>> ase_params = grid.to_ase()
            >>> ase_params["mesh"]
            450.0
        """
        # ASE uses 'mesh' parameter name instead of 'Mesh.Cutoff'
        return {"mesh": self.mesh_cutoff}

    def generate_grid_block(self):
        """
        Generates the real-space grid parameters block for the FDF file.

        This is a wrapper around generate_fdf() to maintain backward compatibility
        with code that calls this method directly (e.g., setup_real_space_grid_parameters()).

        By calling generate_fdf(), we ensure:
        - Single source of truth for FDF generation
        - Proper "# SIESTA DEFAULT VALUE" markers on default parameters
        - Consistency with user_params, powerups, and tier presets
        - DRY principle (no parameter duplication)
        - Values updated via update_from_fdf() are properly reflected
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]RealSpaceGridParameters.generate_grid_block()[/green]"
            )

        # Call generate_fdf() which uses the current dataclass attributes
        # (these have been updated from user_params/powerups/tiers via update_from_fdf())
        fdf = self.generate_fdf()

        # Add comment header
        fdf_with_header = OrderedDict()
        fdf_with_header["#RealSpaceGridParameters"] = "RealSpaceGridParameters"
        fdf_with_header.update(fdf)

        self.grid_fdf_arguments = fdf_with_header
