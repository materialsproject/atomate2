"""
Module defining base SIESTA input set and generator.

class SpinSettings

Based on User's Guide Siesta 5.4.0
Section: 6.7 Spin polarization
         6.8 Spin-Orbit coupling
"""

# Metadata

__all__ = ["SpinSettings"]

import logging
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class SpinSettings(FDFDataclass):
    """
    Data class to manage spin polarization and spin-orbit coupling for SIESTA input.
    Spin non-polarized
    Spin.Fix false
    %block Spin.Spiral 〈None〉
    Spin.Spiral.Scale 〈None〉
    SingleExcitation false
    Spin.OrbitStrength 1.0
    WriteOrbMom false
    SOC.Split.SR.SO true

    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.INFO
    )  # Default to show info messages

    _performe_spin_polarized: bool = field(
        default=False,
        metadata={
            "description": "A wrapper-level flag to enable spin-polarized calculations. If true, this will activate the appropriate 'Spin' keyword setting.",
            "SIESTA keyword": None,
        },
    )

    spin: str = field(
        default="non-polarized",
        metadata={
            "description": "Sets the global spin configuration. Options: 'non-polarized', 'polarized' (collinear), 'non-collinear', 'spin-orbit'.",
            "SIESTA keyword": "Spin",
        },
    )

    dm_init: str = field(
        default="atomic",
        metadata={
            "description": "Specifies the method for initializing the Density Matrix (DM) at the start of the calculation.",
            "SIESTA keyword": "DM.Init",
        },
    )

    dm_init_spin_af: bool = field(
        default=False,
        metadata={
            "description": "A boolean flag to initialize the spin density in a simple antiferromagnetic (AF) configuration.",
            "SIESTA keyword": "DM.InitSpin.AF",
        },
    )

    dm_init_spin_block: list[str] | None = field(
        default=None,
        metadata={
            "description": "A block to explicitly specify the initial spin moment (a 3D vector for non-collinear cases) for each atom. Each element is a string like '1  +0.5'.",
            "SIESTA keyword": "%block DM.InitSpin",
        },
    )

    spin_fix: bool = field(
        default=False,
        metadata={
            "description": "A flag to fix the total spin moment of the system during the self-consistency cycle.",
            "SIESTA keyword": "Spin.Fix",
        },
    )

    spin_total: float = field(
        default=0.0,
        metadata={
            "description": "The target total spin moment (in units of h-bar/2) to be enforced when 'Spin.Fix' is enabled.",
            "SIESTA keyword": "Spin.Total",
        },
    )

    spin_spiral_block: dict[str, Any] | None = field(
        default_factory=dict,
        metadata={
            "description": "A block to define the q-vector of a spin spiral for non-collinear calculations.",
            "SIESTA keyword": "%block Spin.Spiral",
        },
    )

    spin_spiral_scale: list[str] | None = field(
        default_factory=list,
        metadata={
            "description": "A block to scale the q-vector of the spin spiral, which can be used to define a path in q-space.",
            "SIESTA keyword": "%block Spin.Spiral.Scale",
        },
    )

    single_excitation: bool = field(
        default=False,
        metadata={
            "description": "Enables the calculation of single-particle excitations, typically used in methods like TD-DFT.",
            "SIESTA keyword": "SingleExcitation",
        },
    )

    spin_orbit_strength: float = field(
        default=1.0,
        metadata={
            "description": "A scaling factor (from 0.0 to 1.0) for the strength of the spin-orbit coupling interaction.",
            "SIESTA keyword": "Spin.OrbitStrength",
        },
    )

    write_orb_mom: bool = field(
        default=False,
        metadata={
            "description": "A flag to enable the writing of orbital moments to the output files.",
            "SIESTA keyword": "WriteOrbMom",
        },
    )

    soc_split_sr_so: bool = field(
        default=True,
        metadata={
            "description": "A technical flag for handling the interplay between scalar-relativistic (SR) and spin-orbit (SO) components in the pseudopotentials.",
            "SIESTA keyword": "SOC.Split.SR.SO",
        },
    )

    spin_fdf_arguments: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "A dictionary for any additional or arbitrary FDF (Flexible Data Format) flags related to spin. This allows for using keywords not explicitly defined elsewhere.",
            "SIESTA keyword": None,
        },
    )

    comments: str = field(
        default="SpinSettings",
        metadata={
            "description": "User-provided comments to be included as a comment block in the FDF file.",
            "SIESTA keyword": None,
        },
    )

    def __post_init__(self):
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "Spin",
                "DM.Init",
                "DM.InitSpin.AF",
                "%block DM.InitSpin",
                "Spin.Fix",
                "Spin.Total",
                "%block Spin.Spiral",
                "%block Spin.Spiral.Scale",
                "SingleExcitation",
                "Spin.OrbitStrength",
                "WriteOrbMom",
                "SOC.Split.SR.SO",
            )
            self.__class__._registered = True

    @classmethod
    def setup_spin_settings(
        cls,
        user_params: dict[str, Any] | None = None,
        structure: Any | None = None,
        magnetic_ordering: str = "antiferromagnetic",
    ) -> "SpinSettings":
        """
        Create and configure a SpinSettings instance based on user parameters, retaining all default values for unspecified fields.
        The _performe_spin_polarized field is derived from the spin setting and cannot be directly set by user_params.

        Args:
            user_params (dict, optional): Dictionary of user-defined parameters (case-insensitive, may include dots).
                                         If None or empty, all default SpinSettings values are used.
            structure (Structure or Molecule, optional): Pymatgen structure with magnetic moments in site_properties["magmom"].
                                                        If provided and magmom property exists, automatically generates DM.InitSpin block.
            magnetic_ordering (str): Magnetic ordering type for auto-generation. Options:
                                    - "ferromagnetic" or "FM": All moments aligned (same sign)
                                    - "antiferromagnetic" or "AFM": Alternating moments (opposite signs)
                                    - "custom": Use exact values from structure.magmom (with signs)
                                    Default: "antiferromagnetic"

        Returns
        -------
            SpinSettings: Configured SpinSettings instance with all fields (default and user-specified) and FDF arguments.
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]SpinSettings.setup_spin_settings()[/green]")

        # Initialize SpinSettings instance with defaults
        spin_settings_instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default SpinSettings values.[/blue]"
                )
        else:
            # Get valid SpinSettings attribute names (lowercase for comparison), excluding _performe_spin_polarized
            spin_settings_attributes = {
                field.name.lower()
                for field in fields(cls)
                if not field.name.startswith("_")
            }
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Available SpinSettings attributes: {spin_settings_attributes}[/blue]"
                )

            # Process user parameters
            for key, value in user_params.items():
                # Normalize key: convert to lowercase and replace dots with underscores
                key_normalized = key.lower().replace(".", "_")
                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                    console.print(
                        f"[blue]Processing key: {key} -> normalized: {key_normalized}, value: {value}[/blue]"
                    )

                # Skip _performe_spin_polarized if provided by user
                if key_normalized == "_performe_spin_polarized":
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            "[yellow]Ignoring user-provided '_performe_spin_polarized'; it is derived from 'spin'.[/yellow]"
                        )
                    continue

                # Check if normalized key matches any SpinSettings attribute
                if key_normalized in spin_settings_attributes:
                    # Find the original attribute name (preserving case)
                    original_key = next(
                        field.name
                        for field in fields(cls)
                        if field.name.lower() == key_normalized
                    )
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                        console.print(
                            f"[blue]Matched SpinSettings field: {original_key} = {value}[/blue]"
                        )

                    # Handle type conversion for specific fields
                    if original_key in [
                        "dm_init_spin_block",
                        "spin_spiral_block",
                    ] and isinstance(value, dict):
                        setattr(spin_settings_instance, original_key, value)
                    elif original_key == "spin_spiral_scale" and isinstance(
                        value, (list, tuple)
                    ):
                        setattr(spin_settings_instance, original_key, list(value))
                    elif original_key in [
                        "dm_init_spin_af",
                        "spin_fix",
                        "single_excitation",
                        "write_orb_mom",
                        "soc_split_sr_so",
                    ]:
                        if isinstance(value, str):
                            value = value.lower() in ("true", "t", "1", "yes")
                        setattr(spin_settings_instance, original_key, bool(value))
                    elif original_key in ["spin_total", "spin_orbit_strength"]:
                        setattr(spin_settings_instance, original_key, float(value))
                    elif original_key == "spin":
                        allowed_spin = [
                            "non-polarized",
                            "polarized",
                            "non-colinear",
                            "spin-orbit",
                            "spin+onsite",
                        ]
                        if value.lower() not in allowed_spin:
                            if (
                                cls.CONSOLE_VERBOSITY.value
                                >= VerbosityLevel.WARNING.value
                            ):
                                console.print(
                                    f"[red]Invalid spin value [bold]'{value}'[/bold] for [bold]{original_key}[/bold].[/red]"
                                )
                                # raise ValueError(f"allowed spin values are {allowed_spin=}")
                                console.print(
                                    f"[red]allowed spin values are [bold]{allowed_spin=}[/bold][/red]"
                                )
                                raise ValueError
                            continue
                        setattr(spin_settings_instance, original_key, value)
                    else:
                        setattr(spin_settings_instance, original_key, value)
                elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                    console.print(
                        f"[yellow]Key '{key}' does not match any SpinSettings field, skipping.[/yellow]"
                    )

        # Derive _performe_spin_polarized from spin
        spin_settings_instance._performe_spin_polarized = (
            spin_settings_instance.spin.lower() != "non-polarized"
        )
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]Derived _performe_spin_polarized: {spin_settings_instance._performe_spin_polarized}[/blue]"
            )

        # =====================================================================
        # AUTO-GENERATE DM.InitSpin FROM STRUCTURE MAGNETIC MOMENTS
        # =====================================================================
        if structure is not None and hasattr(structure, "site_properties"):
            magmoms = structure.site_properties.get("magmom")
            if magmoms is not None and len(magmoms) > 0:
                # Check if any non-zero magnetic moments exist
                has_nonzero_magmoms = any(abs(m) > 1e-6 for m in magmoms)

                if has_nonzero_magmoms:
                    # Check if user already provided DM.InitSpin (don't override)
                    user_provided_init_spin = False
                    if user_params:
                        # Check in fdf_arguments
                        fdf_args = user_params.get("fdf_arguments", {})
                        if "DM.InitSpin" in fdf_args or "dm_init_spin" in fdf_args:
                            user_provided_init_spin = True
                        # Check in user_params directly
                        if "dm_init_spin_block" in {
                            k.lower() for k in user_params.keys()
                        }:
                            user_provided_init_spin = True

                    if not user_provided_init_spin:
                        # Check for sign-only format parameter (NEW v1.0.0!)
                        dm_init_spin_format = "numeric"  # default
                        if user_params:
                            # Check for internal parameter (with a2s_ or atomate2siesta_ prefix)
                            for key in [
                                "a2s_dm_init_spin_format",
                                "atomate2siesta_dm_init_spin_format",
                            ]:
                                if key in user_params:
                                    dm_init_spin_format = user_params[key].lower()
                                    break

                        # Auto-generate DM.InitSpin block with comments
                        dm_init_spin_lines = []

                        for i, moment in enumerate(magmoms):
                            abs_moment = abs(moment)

                            # Get atom information for comment
                            site = structure[i]
                            species = site.specie.symbol
                            coords = site.coords
                            comment = f"# {species} atom {i + 1} at ({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f})"

                            if abs_moment < 1e-6:
                                # Zero moment - skip it (cleaner DM.InitSpin)
                                continue
                            # Apply magnetic ordering
                            if magnetic_ordering.lower() in ["ferromagnetic", "fm"]:
                                # All moments same sign (positive)
                                final_moment = abs_moment
                            elif magnetic_ordering.lower() in [
                                "antiferromagnetic",
                                "afm",
                            ]:
                                # Alternate signs
                                final_moment = abs_moment if i % 2 == 0 else -abs_moment
                            elif magnetic_ordering.lower() == "custom":
                                # Use exact value from structure (preserve sign)
                                final_moment = moment
                            else:
                                if (
                                    cls.CONSOLE_VERBOSITY.value
                                    >= VerbosityLevel.WARNING.value
                                ):
                                    console.print(
                                        f"[yellow]Unknown magnetic_ordering '{magnetic_ordering}', using ferromagnetic[/yellow]"
                                    )
                                final_moment = abs_moment

                            # Format output: numeric vs sign-only (NEW v1.0.0!)
                            if dm_init_spin_format == "sign_only":
                                # Sign-only: just "+" or "-" (SIESTA determines magnitude)
                                sign = "+" if final_moment > 0 else "-"
                                dm_init_spin_lines.append(f"{i + 1}  {sign}  {comment}")
                            else:
                                # Numeric (default): full moment value
                                dm_init_spin_lines.append(
                                    f"{i + 1}  {final_moment:+.1f}  {comment}"
                                )

                        # Store as list of strings (FDF block format)
                        spin_settings_instance.dm_init_spin_block = dm_init_spin_lines

                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
                            n_atoms = len(magmoms)
                            n_magnetic = sum(1 for m in magmoms if abs(m) > 1e-6)
                            console.print(
                                f"[green]✓ Auto-generated DM.InitSpin block: {n_magnetic}/{n_atoms} magnetic atoms ({magnetic_ordering} ordering)[/green]"
                            )

        # Validate spin settings
        try:
            spin_settings_instance.validate()
        except ValueError as e:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.ERROR.value:
                console.print(f"[red]Validation failed: {e}[/red]")
            raise

        # Generate FDF spin block with all relevant fields
        spin_settings_instance.generate_spin_block()

        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]Validation & Generation: [yellow]SpinSettings[/yellow] Successful![/green]"
            )

        return spin_settings_instance

    def validate(self):
        """
        Validates the spin settings.
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]SpinSettings.validate()[/green]")
        if self._performe_spin_polarized:
            allowed_spin = [
                "non-polarized",
                "polarized",
                "non-colinear",
                "spin-orbit",
                "spin+onsite",
            ]
            if self.spin not in allowed_spin:
                raise ValueError(
                    f"Invalid spin type '{self.spin}'. Allowed values are: {allowed_spin}"
                )
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]Validation: [yellow]SpinSettings[/yellow] Successful![/green]"
            )

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]SpinSettings.update_from_fdf()[/green]")

        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower == "spin":
                self.spin = str(value).lower()
            elif key_lower in ["dm.init", "dm_init"]:
                self.dm_init = str(value).lower()
            elif key_lower in ["dm.initspin.af", "dm_init_spin_af"]:
                self.dm_init_spin_af = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower == "%block dm.initspin":
                if isinstance(value, list):
                    self.dm_init_spin_block = value
            elif key_lower in ["spin.fix", "spin_fix"]:
                self.spin_fix = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["spin.total", "spin_total"]:
                self.spin_total = float(value) if isinstance(value, str) else value
            elif key_lower == "%block spin.spiral":
                if isinstance(value, dict):
                    self.spin_spiral_block = value
            elif key_lower == "%block spin.spiral.scale":
                if isinstance(value, list):
                    self.spin_spiral_scale = value
            elif key_lower in ["singleexcitation", "single_excitation"]:
                self.single_excitation = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["spin.orbitstrength", "spin_orbit_strength"]:
                self.spin_orbit_strength = (
                    float(value) if isinstance(value, str) else value
                )
            elif key_lower in ["writeorbmom", "write_orb_mom"]:
                self.write_orb_mom = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower in ["soc.split.sr.so", "soc_split_sr_so"]:
                self.soc_split_sr_so = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]SpinSettings.generate_fdf()[/green]")

        fdf: dict[str, Any] = OrderedDict()
        fdf["#SpinSettings"] = "SpinSettings"

        # Spin - always write with default marker
        if self.spin == "non-polarized":
            fdf["Spin"] = f"{self.spin}  # SIESTA DEFAULT VALUE"
        else:
            fdf["Spin"] = self.spin

        # DM.Init - always write with default marker
        if self.dm_init == "atomic":
            fdf["DM.Init"] = f"{self.dm_init}  # SIESTA DEFAULT VALUE"
        else:
            fdf["DM.Init"] = self.dm_init

        # DM.InitSpin.AF - always write with default marker
        if not self.dm_init_spin_af:
            fdf["DM.InitSpin.AF"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["DM.InitSpin.AF"] = "true"

        # DM.InitSpin block - write if provided (no default marker, it's a block)
        if self.dm_init_spin_block:
            fdf["%block DM.InitSpin"] = self.dm_init_spin_block

        # Spin.Fix - always write with default marker
        if not self.spin_fix:
            fdf["Spin.Fix"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["Spin.Fix"] = "true"

        # Spin.Total - always write with default marker
        if self.spin_total == 0.0:
            fdf["Spin.Total"] = f"{self.spin_total}  # SIESTA DEFAULT VALUE"
        else:
            fdf["Spin.Total"] = str(self.spin_total)

        # Spin.Spiral blocks - write if provided (no default marker, they're blocks)
        if self.spin_spiral_block:
            fdf["%block Spin.Spiral"] = self.spin_spiral_block

        if self.spin_spiral_scale:
            fdf["%block Spin.Spiral.Scale"] = self.spin_spiral_scale

        # SingleExcitation - always write with default marker
        if not self.single_excitation:
            fdf["SingleExcitation"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["SingleExcitation"] = "true"

        # Spin.OrbitStrength - always write with default marker
        if self.spin_orbit_strength == 1.0:
            fdf["Spin.OrbitStrength"] = (
                f"{self.spin_orbit_strength}  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["Spin.OrbitStrength"] = str(self.spin_orbit_strength)

        # WriteOrbMom - always write with default marker
        if not self.write_orb_mom:
            fdf["WriteOrbMom"] = "false  # SIESTA DEFAULT VALUE"
        else:
            fdf["WriteOrbMom"] = "true"

        # SOC.Split.SR.SO - always write with default marker
        if self.soc_split_sr_so:
            fdf["SOC.Split.SR.SO"] = "true  # SIESTA DEFAULT VALUE"
        else:
            fdf["SOC.Split.SR.SO"] = "false"

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have direct spin parameter equivalents
        # Most spin settings are SIESTA-specific
        return {}

    def generate_spin_block(self):
        """
        Generates the spin-related block for the FDF file, including all relevant fields (default and user-specified).

        This is a wrapper around generate_fdf() to maintain backward compatibility
        with code that calls this method directly (e.g., setup_spin_settings()).

        By calling generate_fdf(), we ensure:
        - Single source of truth for FDF generation
        - Proper "# SIESTA DEFAULT VALUE" markers on default parameters
        - Consistency with user_params, powerups, and tier presets
        - DRY principle (no parameter duplication)
        - Values updated via update_from_fdf() are properly reflected
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]SpinSettings.generate_spin_block()[/green]")

        # Call generate_fdf() which uses the current dataclass attributes
        # (these have been updated from user_params/powerups/tiers via update_from_fdf())
        fdf = self.generate_fdf()

        # Add comment header
        fdf_with_header = OrderedDict()
        if self.comments:
            fdf_with_header["#SpinSettings"] = self.comments
        fdf_with_header.update(fdf)

        self.spin_fdf_arguments = fdf_with_header

    # old version
    # def validate(self):
    #     """
    #     Validates the spin settings.
    #     """
    #     console=Console()
    #     logger.info("SpinSettings.validate()")
    #     if self.performe_spin_polarized:
    #         allowed_spin = ['non-polarized','polarized','non-colinear','spin-orbit','spin+onsite']
    #         if self.spin not in allowed_spin:
    #             raise ValueError(f"Invalid spin type '{self.spin}'. Allowed values are: {allowed_spin}")
    #         if self.initial_spin_moments is None:
    #             raise ValueError("Initial spin moments must be provided for spin-polarized calculations.")
    #     # print("Validation: SpinSettings DONE!")
    #     console.print(f"[green]Validation: [yellow]SpinSettings[/yellow] Successful![/green]")

    # def generate_spin_block(self):
    #     """
    #     Generates the spin-related block for the FDF file.
    #     """
    #     logger.info("SpinSettings.generate_spin_block()")

    #     self.spin_fdf_arguments = OrderedDict([
    #         ("#SpinSettings", self.comments if self.comments else "SpinSettings"),
    #         ("Spin", f"{self.spin}"),
    #         ("DM.Init", f"{self.dm_init}"),
    #         ("DM.InitSpin.AF", f"{self.dm_init_spin_af}"),
    #         ("Spin.Fix", f"{self.spin_fix}"),
    #         ("Spin.Total", f"{self.spin_total}"),
    #         ("Spin.Spiral.Scale", f"{self.spin_spiral_scale}"),
    #         ("SingleExcitation", f"{self.single_excitation}"),
    #         ("Spin.OrbitStrength", f"{self.spin_orbit_strength}"),
    #         ("WriteOrbMom", f"{self.write_orb_mom}"),
    #         ("SOC.Split.SR.SO", f"{self.soc_split_sr_so}"),
    #     ])
