"""
Module defining base SIESTA input set and generator.

class PhononCalculations

Based on User's Guide Siesta 5.4.0
Section:  7.8 Phonon calculations
"""

# Metadata

__all__ = ["PhononCalculations"]

import logging
from dataclasses import dataclass, field, fields
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class PhononCalculations(FDFDataclass):
    """
    A class for setting up and validating phonon-related calculations in SIESTA.

    This class manages the generation of the phonon-related input parameters
    for force constant (FC) calculations and eigenvector extraction in SIESTA.
    It ensures the correct setup of the molecular dynamics (MD) options required
    for phonon calculations and provides validation to ensure the input conforms
    to the allowed options for phonon calculations.

    Parameters
    ----------
    md_type_of_run : str
        The type of molecular dynamics run, typically set to "FC" for force constant
        calculations in phonon analyses. Default is "FC".
    md_fc_displ : float
        The displacement value (in Bohr) used in force constant calculations. Default is 0.04.
    md_fc_first : int
        The index of the first atom involved in force constant calculations. Default is 1.
    md_fc_last : int or None
        The index of the last atom involved in force constant calculations. If set to None,
        it defaults to the value of `md_fc_first`. Default is None.
    eigenvectors : bool
        A flag indicating whether eigenvector calculations should be performed. Default is True.
    phonon_fdf_arguments : dict
        A dictionary storing the phonon-related FDF (input) arguments for SIESTA.
        This dictionary is populated after calling `generate_phonon_block`.

    Methods
    -------
    validate():
        Validates the input parameters for phonon calculations, ensuring that the
        `md_type_of_run` is allowed for phonon calculations.

    generate_phonon_block():
        Generates the necessary FDF arguments for phonon calculations, including
        the molecular dynamics options such as displacement, first and last atoms,
        and eigenvector flag.
    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.ERROR
    )  # Default to show errors only

    # ------------------------
    # 7.8 Phonon calculations
    # ------------------------

    # For FC
    # md_type_of_run: str = "FC"  # For Phonon Calculations
    # md_fc_displ: float = 0.04   # MD.FCDispl 0.04 Bohr
    # md_fc_first: int = 1        # MD.FCFirst 1
    # md_fc_last: int = None      # MD.FCLast 〈MD.FCFirst〉
    # eigenvectors: bool = True   # Eigenvectors
    # phonon_fdf_arguments: Dict[float,Any]= field(default_factory=dict)
    md_type_of_run: str = field(
        default="FC",
        metadata={
            "description": "Sets the calculation type to compute forces for a finite-difference phonon calculation.",
            "SIESTA keyword": "MD.TypeOfRun",
        },
    )

    md_fc_displ: float = field(
        default=0.04,
        metadata={
            "description": "The size of the atomic displacement (in Bohr) used to calculate the force constants for phonon calculations.",
            "SIESTA keyword": "MD.FCDispl",
            "unit": "Bohr",
        },
    )

    md_fc_first: int = field(
        default=1,
        metadata={
            "description": "The index of the first atom to be displaced for the force-constants calculation.",
            "SIESTA keyword": "MD.FCFirst",
        },
    )

    md_fc_last: int = field(
        default=None,
        metadata={
            "description": "The index of the last atom to be displaced for the force-constants calculation. Defaults to the value of MD.FCFirst.",
            "SIESTA keyword": "MD.FCLast",
        },
    )

    eigenvectors: bool = field(
        default=True,
        metadata={
            "description": "A flag to control the output of eigenvectors from the diagonalization of the dynamical matrix.",
            "SIESTA keyword": "Eigenvectors",
        },
    )

    comments: str = field(
        default="PhononCalculations Settings",
        metadata={
            "description": "Comment header for this dataclass section in the FDF file.",
            "SIESTA keyword": None,
        },
    )

    phonon_fdf_arguments: dict[float, Any] = field(
        default_factory=dict,
        metadata={
            "description": "A dictionary for any additional or arbitrary FDF flags related to phonon calculations.",
            "SIESTA keyword": None,
        },
    )

    def __post_init__(self):
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "MD.TypeOfRun",
                "MD.FCDispl",
                "MD.FCFirst",
                "MD.FCLast",
                "Eigenvectors",
            )
            self.__class__._registered = True

    def validate(self):
        """
        Validate phonon calculation parameters.

        Checks that phonon calculation settings (MD type, displacement, atom indices)
        are properly configured for force constant calculations. Ensures MD.TypeOfRun
        is set to "FC" (force constants) which is required for phonon calculations.

        Raises
        ------
        ValueError
            If MD.TypeOfRun is not "FC" or phonon parameters are invalid
        """
        logger.info("PhononCalculations.validate()")
        # Allowed Phonon
        allowed_md_type_of_run_phonon = ["FC"]
        # if self.perform_fc and self.md_type_of_run not in allowed_md_type_of_run_phonon:
        if self.md_type_of_run not in allowed_md_type_of_run_phonon:
            raise ValueError(
                f"Invalid MD FC  '{self.md_type_of_run}'. Allowed values are: {allowed_md_type_of_run_phonon}"
            )
        print("Validation: PhononCalculations DONE!")

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        from atomate2.siesta.dataclass.units import parse_length

        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["md.typeofrun", "md_type_of_run"]:
                self.md_type_of_run = str(value)
            elif key_lower in ["md.fcdispl", "md_fc_displ"]:
                # Parse length with units (default Ang or Bohr)
                self.md_fc_displ = parse_length(value, target_unit="Ang")
            elif key_lower in ["md.fcfirst", "md_fc_first"]:
                self.md_fc_first = int(value)
            elif key_lower in ["md.fclast", "md_fc_last"]:
                self.md_fc_last = int(value) if value is not None else None
            elif key_lower in ["eigenvectors"]:
                self.eigenvectors = (
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
        fdf: dict[str, Any] = {}

        if self.md_type_of_run == "FC":
            fdf["MD.TypeOfRun"] = self.md_type_of_run
            fdf["MD.FCDispl"] = f"{self.md_fc_displ} Ang"
            fdf["MD.FCFirst"] = str(self.md_fc_first)
            if self.md_fc_last is not None:
                fdf["MD.FCLast"] = str(self.md_fc_last)
            if self.eigenvectors:
                fdf["Eigenvectors"] = "true"

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have phonon calculation parameters
        # These are SIESTA-specific force constant calculation settings
        return {}

    def generate_phonon_block(self):
        """
        Generates the molecular dynamics options block for the FDF file.
        """
        logger.info("PhononCalculations.generate_phonon_block()")

        # Add comment header
        if self.comments:
            self.phonon_fdf_arguments["#PhononCalculations"] = self.comments

        self.phonon_fdf_arguments.update(
            {
                "MD.TypeOfRun": f"{self.md_type_of_run}",
                "MD.FCDispl": f"{self.md_fc_displ} Ang",
                "MD.FCFirst": f"{self.md_fc_first}",
                "MD.FCLast": f"{self.md_fc_last}",
                "Eigenvectors": f"{self.eigenvectors}",
            }
        )

    @classmethod
    def setup_phonon_settings(
        cls, user_params: dict[str, Any] | None = None, **kwargs
    ) -> "PhononCalculations":
        """
        Create and configure a PhononCalculations instance with full parameter parsing.

        This method handles proper key normalization, type conversion, and fuzzy matching
        to configure phonon calculation settings from user parameters.

        Args:
            user_params: Dictionary of user-defined parameters (case-insensitive, may include dots).
                        If None or empty, all default values are used.
            **kwargs: Additional keyword arguments to override or supplement user_params.

        Returns
        -------
            PhononCalculations: Configured instance with all fields set.
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]PhononCalculations.setup_phonon_settings()[/green]")

        # Initialize instance with defaults
        instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default Phonon values.[/blue]"
                )
            return instance

        # Get valid attribute names (lowercase for comparison)
        phonon_attributes = {
            field.name.lower()
            for field in fields(cls)
            if not field.name.startswith("_") and field.name != "CONSOLE_VERBOSITY"
        }
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(
                f"[blue]Available PhononCalculations attributes: {phonon_attributes}[/blue]"
            )

        # Process user parameters
        import re

        for key, value in user_params.items():
            # Normalize key: handle camelCase properly
            key_with_underscores = re.sub(r"([a-z])([A-Z])", r"\1_\2", key)
            key_normalized = key_with_underscores.replace(".", "_").lower()

            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Processing key: {key} -> {key_normalized}, value: {value}[/blue]"
                )

            # Check if normalized key matches any attribute
            matched_attr = None
            if key_normalized in phonon_attributes:
                matched_attr = key_normalized
            else:
                # Fuzzy match: remove all underscores and compare
                key_no_underscores = key_normalized.replace("_", "")
                for attr in phonon_attributes:
                    if attr.replace("_", "") == key_no_underscores:
                        matched_attr = attr
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                            console.print(
                                f"[blue]Fuzzy matched: {key_normalized} -> {attr}[/blue]"
                            )
                        break

            if matched_attr:
                # Find the original attribute name (preserving case)
                original_key = next(
                    field.name
                    for field in fields(cls)
                    if field.name.lower() == matched_attr
                )

                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                    console.print(
                        f"[blue]Matched field: {original_key} = {value}[/blue]"
                    )

                # Handle type conversion for specific field types
                # Dict fields
                if original_key == "phonon_fdf_arguments":
                    if isinstance(value, dict):
                        setattr(instance, original_key, value)
                    elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Invalid type for {original_key}: expected dict, got {type(value)}[/yellow]"
                        )

                # Boolean fields
                elif original_key == "eigenvectors":
                    if isinstance(value, str):
                        value = value.lower() in ("true", "t", "1", "yes")
                    setattr(instance, original_key, bool(value))

                # Integer fields (md_fc_first, md_fc_last can be None)
                elif original_key in ["md_fc_first", "md_fc_last"]:
                    try:
                        if value is not None:
                            setattr(instance, original_key, int(value))
                        else:
                            setattr(instance, original_key, None)
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Could not convert {original_key}={value} to int[/yellow]"
                            )

                # Float fields
                elif original_key == "md_fc_displ":
                    try:
                        setattr(instance, original_key, float(value))
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Could not convert {original_key}={value} to float[/yellow]"
                            )

                # String fields
                elif original_key == "md_type_of_run":
                    setattr(instance, original_key, str(value))

                # Default: direct assignment
                else:
                    setattr(instance, original_key, value)

            elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                console.print(
                    f"[yellow]Unrecognized parameter: {key} (normalized: {key_normalized})[/yellow]"
                )

        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]PhononCalculations instance configured successfully.[/green]"
            )

        return instance
