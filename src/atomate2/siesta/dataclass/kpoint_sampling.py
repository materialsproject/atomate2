"""
Module defining base SIESTA input set and generator.

class KPointSampling

Based on User's Guide Siesta 5.4.0
Section: 6.5 k-point sampling
"""

# Metadata

__all__ = ["KPointSampling"]

import logging
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Any

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Kpoints

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.dataclass.units import parse_length
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class KPointSampling(FDFDataclass):
    """Data class to manage k-point sampling for SIESTA input."""

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.ERROR
    )  # Default to show info & errors messages

    _user_params: dict[str, Any] | None = field(
        default=None,
        init=False,  # Prevent user from setting via constructor
        metadata={
            "description": (
                "Temporary storage for user parameters to validate k-point settings."
            ),
            "SIESTA keyword": None,
        },
    )
    _use_absolute_kpts: bool = field(
        default=False,
        init=False,  # Set internally based on user input
        metadata={
            "description": (
                "If True, use absolute k-point grid "
                "(don't scale with cell size). Used for phonon supercells."
            ),
            "SIESTA keyword": None,
        },
    )
    _user_requested_cutoff: bool = field(
        default=False,
        init=False,  # Set by update_from_fdf when user provides kgrid.Cutoff
        metadata={
            "description": "If True, user explicitly requested kgrid.Cutoff method.",
            "SIESTA keyword": None,
        },
    )
    k_points: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(1, 1, 1)],
        metadata={
            "description": (
                "Defines the dimensions of the Monkhorst-Pack k-point grid "
                "for sampling the Brillouin zone."
            ),
            "SIESTA keyword": "%block kgrid_Monkhorst_Pack",
        },
    )
    k_shift: tuple[float, float, float] = field(
        default=(0.0, 0.0, 0.0),
        metadata={
            "description": (
                "A displacement vector applied to the entire k-point grid. "
                "This is specified within the kgrid_Monkhorst_Pack block."
            ),
            "SIESTA keyword": "%block kgrid_Monkhorst_Pack",
        },
    )
    kgrid_cutoff: float | None = field(
        default=10.0,
        metadata={
            "description": (
                "A real-space cutoff (in Angstroms) used to automatically "
                "generate a k-point grid of commensurate density. "
                "This is an alternative to specifying k_points manually."
            ),
            "SIESTA keyword": "kgrid.Cutoff",
        },
    )
    kpoint_fdf_arguments: OrderedDict[str, Any] = field(
        default_factory=OrderedDict,
        metadata={
            "description": "A dictionary for FDF flags related to k-point sampling.",
            "SIESTA keyword": None,
        },
    )

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "%block kgrid.Monkhorst.Pack",
                "kgrid.Cutoff",
            )
            self.__class__._registered = True  # noqa: SLF001  one-time class registration flag

    def validate(self) -> None:
        """
        Validate the k-point sampling settings.

        Raises
        ------
            ValueError: If k_points, k_shift, or kgrid_cutoff are invalid.
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]KPointSampling.validate()[/green]")

        # Validate k_points
        if not self.k_points or not all(
            isinstance(k, (list, tuple)) and len(k) == 3 for k in self.k_points
        ):
            raise ValueError("k_points must be a non-empty list of 3D integer tuples")
        if not all(
            all(isinstance(v, int) and v >= 1 for v in k) for k in self.k_points
        ):
            raise ValueError("k_points must contain positive integers")

        # Validate k_shift
        if not isinstance(self.k_shift, (list, tuple)) or len(self.k_shift) != 3:
            raise ValueError("k_shift must be a 3D float tuple")
        if not all(isinstance(v, (int, float)) for v in self.k_shift):
            raise ValueError("k_shift values must be numbers")

        # Validate kgrid_cutoff
        if self.kgrid_cutoff is not None and (
            not isinstance(self.kgrid_cutoff, (int, float)) or self.kgrid_cutoff <= 0
        ):
            raise ValueError(
                "kgrid.Cutoff must be a positive number or None, "
                f"got '{self.kgrid_cutoff}'"
            )

        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]Validated: kgrid_cutoff={self.kgrid_cutoff}, "
                f"k_points={self.k_points}, k_shift={self.k_shift}[/blue]"
            )

        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]Validation: [yellow]KPointSampling[/yellow] Successful![/green]"
            )

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)

        Note:
            Users typically use 'kpts' shorthand (e.g., kpts=[4,4,4]) which maps to
            the SIESTA FDF parameter %block kgrid.Monkhorst.Pack
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]KPointSampling.update_from_fdf()[/green]")

        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["a2s_kpts", "atomate2siesta_kpts", "kpts", "k_points"]:
                # Handle shorthand: a2s_kpts=[4,4,4] or kpts=[4,4,4] or kpts=[[4,4,4]]
                if isinstance(value, (list, tuple)):
                    if len(value) == 3 and all(isinstance(v, int) for v in value):
                        # Single k-point grid: [4,4,4]
                        self.k_points = [tuple(value)]
                    elif all(
                        isinstance(k, (list, tuple)) and len(k) == 3 for k in value
                    ):
                        # List of k-point grids: [[4,4,4], [6,6,6]]
                        self.k_points = [tuple(k) for k in value]

            elif key_lower == "%block kgrid.monkhorst.pack":
                # Direct SIESTA FDF format (advanced users)
                # Format: [[n1, 0, 0, s1], [0, n2, 0, s2], [0, 0, n3, s3]]
                if isinstance(value, list) and len(value) >= 3:
                    # Extract diagonal elements (n1, n2, n3) from 3x4 matrix
                    try:
                        n1 = int(value[0][0])
                        n2 = int(value[1][1])
                        n3 = int(value[2][2])
                        self.k_points = [(n1, n2, n3)]
                        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                            console.print(
                                f"[blue]Parsed %block kgrid.monkhorst.pack: "
                                f"k_points = {self.k_points}[/blue]"
                            )
                    except (IndexError, ValueError, TypeError) as e:
                        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                "[yellow]Failed to parse "
                                f"%block kgrid.monkhorst.pack: {e}. "
                                "Using default k-points.[/yellow]"
                            )

            elif key_lower == "kgrid.cutoff":
                # Parse length with unit conversion
                self.kgrid_cutoff = parse_length(value, target_unit="Ang")
                self._user_requested_cutoff = (
                    True  # Mark that user explicitly wants cutoff
                )

            elif (
                key_lower == "k_shift"
                and isinstance(value, (list, tuple))
                and len(value) == 3
            ):
                self.k_shift = tuple(float(v) for v in value)

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters

        Note:
            This outputs %block kgrid.Monkhorst.Pack which is the SIESTA FDF format.
            Internally we use k_points for convenience.

            Uses EITHER kgrid.Cutoff OR kgrid.Monkhorst.Pack, not both:
            - If k_points != default [(1,1,1)], use Monkhorst.Pack
            - Otherwise if kgrid_cutoff is set, use kgrid.Cutoff
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]KPointSampling.generate_fdf()[/green]")

        fdf = OrderedDict()
        fdf["#KPointSampling"] = "KPointSampling"

        # Determine which k-point method to use
        # Priority: User-requested cutoff > explicit k-points > default cutoff
        if self._user_requested_cutoff:
            # User explicitly requested kgrid.Cutoff - always write with default marker
            if self.kgrid_cutoff == 10.0:
                fdf["kgrid.Cutoff"] = f"{self.kgrid_cutoff} Ang  # SIESTA DEFAULT VALUE"
            else:
                fdf["kgrid.Cutoff"] = f"{self.kgrid_cutoff} Ang"
        elif self._use_absolute_kpts:
            # User explicitly provided k-points
            # (via a2s_kpts, kpts, or k_points parameter)
            # Write Monkhorst.Pack block even if k_points == [(1,1,1)]
            # SIESTA format:
            # %block kgrid.Monkhorst.Pack
            #   4  0  0  0.0
            #   0  4  0  0.0
            #   0  0  4  0.0
            # %endblock kgrid.Monkhorst.Pack
            k = self.k_points[0] if self.k_points else (1, 1, 1)
            # Format as list of strings (required by FDF writer)
            kgrid_block = [
                f"{k[0]} 0 0 {self.k_shift[0]}",
                f"0 {k[1]} 0 {self.k_shift[1]}",
                f"0 0 {k[2]} {self.k_shift[2]}",
            ]
            fdf["%block kgrid.Monkhorst.Pack"] = kgrid_block
        elif self.kgrid_cutoff is not None:
            # Use kgrid.Cutoff (only if no explicit k-points)
            # - always write with default marker
            if self.kgrid_cutoff == 10.0:
                fdf["kgrid.Cutoff"] = f"{self.kgrid_cutoff} Ang  # SIESTA DEFAULT VALUE"
            else:
                fdf["kgrid.Cutoff"] = f"{self.kgrid_cutoff} Ang"
        else:
            # Write default k-points as Monkhorst.Pack with default marker
            k = self.k_points[0] if self.k_points else (1, 1, 1)
            kgrid_block = [
                f"{k[0]} 0 0 {self.k_shift[0]}  # SIESTA DEFAULT VALUE",
                f"0 {k[1]} 0 {self.k_shift[1]}  # SIESTA DEFAULT VALUE",
                f"0 0 {k[2]} {self.k_shift[2]}  # SIESTA DEFAULT VALUE",
            ]
            fdf["%block kgrid.Monkhorst.Pack"] = kgrid_block

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
            (ASE uses 'kpts' instead of kgrid.Monkhorst.Pack)
        """
        # ASE uses 'kpts' parameter name
        if self.k_points and len(self.k_points) > 0:
            return {"kpts": list(self.k_points[0])}
        return {"kpts": [1, 1, 1]}

    @classmethod
    def setup_kpoint_settings(
        cls, structure: Structure, user_params: dict[str, Any] | None = None
    ) -> "KPointSampling":
        """
        Create and configure a KPointSampling instance from user params.

        Retains all default values for unspecified fields, issues warnings for
        invalid keys and skips them, and prioritizes kgrid.Cutoff if provided.

        Args:
            structure (pymatgen.core.Structure): The structure for which
                k-points are generated.
            user_params (dict, optional): Dictionary of user-defined parameters
                (case-insensitive, may include dots). Supported keys: kpts or
                k_points, k_shift, kgrid.cutoff, k_density (default 1000).

        Returns
        -------
            KPointSampling: Configured instance with all fields (default and
                user-specified) and FDF arguments.
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]KPointSampling.setup_kpoint_settings()[/green]")

        # Initialize instance with defaults
        kpoint_settings_instance = cls()
        kpoint_settings_instance._user_params = user_params or {}

        # Check for kgrid.cutoff first
        kgrid_cutoff = None
        k_density = 1000
        if kpoint_settings_instance._user_params:
            for key in kpoint_settings_instance._user_params:
                if key.lower() in ["kgrid.cutoff", "kgrid_cutoff"]:
                    try:
                        kgrid_cutoff = float(kpoint_settings_instance._user_params[key])
                        if kgrid_cutoff <= 0:
                            if (
                                cls.CONSOLE_VERBOSITY.value
                                >= VerbosityLevel.WARNING.value
                            ):
                                console.print(
                                    f"[yellow]Invalid kgrid.cutoff '{kgrid_cutoff}'; "
                                    "must be positive, using default.[/yellow]"
                                )
                            kgrid_cutoff = None
                        else:
                            kpoint_settings_instance.kgrid_cutoff = kgrid_cutoff
                            if (
                                cls.CONSOLE_VERBOSITY.value
                                >= VerbosityLevel.DEBUG.value
                            ):
                                console.print(
                                    "[blue]Using user-provided kgrid.cutoff: "
                                    f"{kgrid_cutoff}[/blue]"
                                )
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                "[yellow]Invalid kgrid.cutoff "
                                f"'{kpoint_settings_instance._user_params[key]}'; "
                                "skipping.[/yellow]"
                            )
                elif key.lower() in ["k_density", "k.density"]:
                    try:
                        k_density = float(kpoint_settings_instance._user_params[key])
                        if k_density <= 0:
                            if (
                                cls.CONSOLE_VERBOSITY.value
                                >= VerbosityLevel.WARNING.value
                            ):
                                console.print(
                                    f"[yellow]Invalid k_density '{k_density}'; "
                                    "must be positive, using default 1000.[/yellow]"
                                )
                            k_density = 1000
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                            console.print(
                                "[blue]Using user-provided k_density: "
                                f"{k_density}[/blue]"
                            )
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                "[yellow]Invalid k_density "
                                f"'{kpoint_settings_instance._user_params[key]}'; "
                                "using default 1000.[/yellow]"
                            )
                        k_density = 1000

        # Handle k-point generation
        kpts_key = None
        if kpoint_settings_instance._user_params:
            # Check for prefixed versions first, then legacy
            kpts_key = (
                "a2s_kpts"
                if "a2s_kpts" in kpoint_settings_instance._user_params
                else "atomate2siesta_kpts"
                if "atomate2siesta_kpts" in kpoint_settings_instance._user_params
                else "kpts"
                if "kpts" in kpoint_settings_instance._user_params
                else "k_points"
                if "k_points" in kpoint_settings_instance._user_params
                else None
            )

        if kpts_key:
            kpts_value = kpoint_settings_instance._user_params[kpts_key]
            if (
                isinstance(kpts_value, (list, tuple))
                and len(kpts_value) == 3
                and all(isinstance(v, (int, float)) for v in kpts_value)
            ):
                try:
                    kpoint_settings_instance.k_points = [
                        tuple(int(v) for v in kpts_value)
                    ]
                    # Mark as absolute k-points (don't scale with structure size)
                    kpoint_settings_instance._use_absolute_kpts = True
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                        console.print(
                            f"[blue]Using user-provided {kpts_key} "
                            f"(absolute grid): {kpoint_settings_instance.k_points}"
                            "[/blue]"
                        )
                except (ValueError, TypeError):
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Invalid {kpts_key} format '{kpts_value}', "
                            "using default k_points [(1, 1, 1)][/yellow]"
                        )
            elif isinstance(kpts_value, list) and all(
                isinstance(k, (list, tuple)) and len(k) == 3 for k in kpts_value
            ):
                try:
                    kpoint_settings_instance.k_points = [
                        tuple(int(v) for v in k) for k in kpts_value
                    ]
                    # Mark as absolute k-points (don't scale with structure size)
                    kpoint_settings_instance._use_absolute_kpts = True
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                        console.print(
                            f"[blue]Using user-provided {kpts_key} "
                            f"(absolute grid): {kpoint_settings_instance.k_points}"
                            "[/blue]"
                        )
                except (ValueError, TypeError):
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Invalid {kpts_key} format '{kpts_value}', "
                            "using default k_points [(1, 1, 1)][/yellow]"
                        )
            elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                console.print(
                    f"[yellow]Invalid {kpts_key} format '{kpts_value}', "
                    "using default k_points [(1, 1, 1)][/yellow]"
                )
        elif not kgrid_cutoff:  # Only generate k-points if kgrid.cutoff is not set
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No kpts/k_points provided, "
                    "generating k-points with k_density.[/blue]"
                )
            try:
                kpoints = Kpoints.automatic_density_by_vol(structure, k_density)
                if kpoints.kpts and len(kpoints.kpts[0]) == 3:
                    kpoint_settings_instance.k_points = [
                        tuple(int(max(1, round(v))) for v in kpoints.kpts[0])
                    ]
                else:
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            "[yellow]Automatic k-point generation failed; "
                            "using default k_points [(1, 1, 1)][/yellow]"
                        )
                    kpoint_settings_instance.k_points = [(1, 1, 1)]
            except Exception as e:  # noqa: BLE001  fall back to default k-points on any generation error
                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                    console.print(
                        f"[yellow]Error in k-point generation: {e}; "
                        "using default k_points [(1, 1, 1)][/yellow]"
                    )
                kpoint_settings_instance.k_points = [(1, 1, 1)]

        # Process other user parameters
        kpoint_settings_attributes = {
            field.name.lower()
            for field in fields(cls)
            if not field.name.startswith("_")
        }
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                "[blue]Available KPointSampling attributes: "
                f"{kpoint_settings_attributes}[/blue]"
            )

        for key, value in kpoint_settings_instance._user_params.items():
            if key.lower() in [
                "fdf_arguments",
                "k_density",
                "k.density",
                "kpts",
                "k_points",
                "kgrid.cutoff",
                "kgrid_cutoff",
            ]:
                continue
            key_normalized = key.lower().replace(".", "_")
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Processing key: {key} -> "
                    f"normalized: {key_normalized}, value: {value}[/blue]"
                )
            if key_normalized == "_user_params":
                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                    console.print(
                        f"[yellow]Ignoring user-provided '{key}'; "
                        "it is internal.[/yellow]"
                    )
                continue
            if key_normalized in kpoint_settings_attributes:
                original_key = next(
                    field.name
                    for field in fields(cls)
                    if field.name.lower() == key_normalized
                )
                if (
                    original_key == "k_shift"
                    and isinstance(value, (list, tuple))
                    and len(value) == 3
                ):
                    try:
                        kpoint_settings_instance.k_shift = tuple(
                            float(v) for v in value
                        )
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Invalid k_shift format for '{key}', "
                                "skipping.[/yellow]"
                            )
                else:
                    setattr(kpoint_settings_instance, original_key, value)
            elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                console.print(
                    f"[yellow]Key '{key}' does not match any "
                    "KPointSampling field, skipping.[/yellow]"
                )

        # Validate settings
        try:
            kpoint_settings_instance.validate()
        except ValueError as e:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.ERROR.value:
                console.print(f"[red]Validation failed: {e}[/red]")
            raise

        # Generate FDF block
        kpoint_settings_instance.generate_kpoint_block()
        kpoint_settings_instance._user_params = None

        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]Validation & Generation: "
                "[yellow]KPointSampling[/yellow] Successful![/green]"
            )

        return kpoint_settings_instance

    def generate_kpoint_block(self) -> None:
        """
        Generate the k-point sampling block for the FDF file.

        Prioritizes kgrid.Cutoff if explicitly set.

        This is a wrapper around generate_fdf() to maintain backward compatibility
        with code that calls this method directly (e.g., setup_kpoint_sampling()).

        By calling generate_fdf(), we ensure:
        - Single source of truth for FDF generation
        - Proper "# SIESTA DEFAULT VALUE" markers on default parameters
        - Consistency with user_params, powerups, and tier presets
        - DRY principle (no parameter duplication)
        - Values updated via update_from_fdf() are properly reflected
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]KPointSampling.generate_kpoint_block()[/green]")

        # Call generate_fdf() which uses the current dataclass attributes
        # (these have been updated from user_params/powerups/tiers
        # via update_from_fdf())
        self.kpoint_fdf_arguments = self.generate_fdf()
