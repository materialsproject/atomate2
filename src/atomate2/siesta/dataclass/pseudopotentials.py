"""
Module defining base SIESTA input set and generator.

class Pseudopotentials

Based on User's Guide Siesta 5.4.0
Section:  6.2 Pseudopotentials + atomate2siesta
"""

# Metadata

__all__ = ["Pseudopotentials"]

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class Pseudopotentials(FDFDataclass):
    """
    Data class to manage pseudopotential path for SIESTA input.

    Supports automatic pseudopotential directory selection based on XC functional.

    Example directory structure:
        /Users/user/.siesta/pseudos/
            ONCVPSP-PBE-SR-PDv0.4-Standard/
            ONCVPSP-PBEsol-FR-PDv0.4-Standard/
            ...

    The class can work in two modes:
    1. **Explicit mode**: Set `pseudo_path` directly to a specific directory
    2. **Automatic mode**: Set `pseudo_base_path` and let the class construct the
       full path based on XC functional and other parameters
    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.ERROR
    )  # Default to show info & errors messages

    # --------------------------------------
    # 6.2 Pseudopotentials + atomate2siesta
    # --------------------------------------
    # Direct path (takes precedence if set)
    pseudo_path: str | None = field(
        default=None,
        metadata={
            "description": (
                "Direct path to pseudopotential directory. If set, overrides "
                "automatic path construction."
            ),
            "SIESTA keyword": "SIESTA_PP_PATH",
        },
    )

    # Base path for automatic construction
    pseudo_base_path: str | None = field(
        default=None,
        metadata={
            "description": (
                "Base directory containing pseudopotential subdirectories "
                "(e.g., /Users/user/.siesta/pseudos)."
            ),
            "SIESTA keyword": None,
        },
    )

    # Pseudopotential family and version
    pseudo_family: str = field(
        default="ONCVPSP",
        metadata={
            "description": "Pseudopotential family name (e.g., ONCVPSP, pseudo-dojo).",
            "SIESTA keyword": None,
        },
    )
    pseudo_version: str = field(
        default="0.4",
        metadata={
            "description": "Pseudopotential version (e.g., 0.4, 0.5).",
            "SIESTA keyword": None,
        },
    )
    pseudo_quality: str = field(
        default="Standard",
        metadata={
            "description": "Pseudopotential quality level (Standard or Stringent).",
            "SIESTA keyword": None,
        },
    )
    pseudo_relativistic: str = field(
        # SR is more compatible (FR has LJ projector issues in some SIESTA versions)
        default="SR",
        metadata={
            "description": (
                "Relativistic treatment: SR (Scalar Relativistic) or FR "
                "(Fully Relativistic)."
            ),
            "SIESTA keyword": None,
        },
    )

    # XC functional information (used for automatic path construction)
    # from 6.2 Pseudopotentials
    xc_functional: str | None = field(
        default=None,
        metadata={
            "description": (
                "XC functional family (LDA, GGA, VDW). Extracted from "
                "ExchangeCorrelationFunctionals."
            ),
            "SIESTA keyword": None,
        },
    )
    # from 6.2 Pseudopotentials
    xc_authors: str | None = field(
        default=None,
        metadata={
            "description": (
                "XC authors/parametrization (PBE, PBEsol, etc.). Extracted "
                "from ExchangeCorrelationFunctionals."
            ),
            "SIESTA keyword": None,
        },
    )

    # Internal fields
    _user_params: dict[str, Any] | None = field(
        default=None,
        init=False,
        metadata={
            "description": (
                "Temporary storage for user parameters to validate "
                "pseudopotential settings."
            ),
            "SIESTA keyword": None,
        },
    )
    pseudo_fdf_arguments: OrderedDict[str, Any] = field(
        default_factory=OrderedDict,
        metadata={
            "description": (
                "A dictionary for FDF flags related to pseudopotential settings."
            ),
            "SIESTA keyword": None,
        },
    )

    _registered: ClassVar[bool]

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "SIESTA_PP_PATH",  # Can be in FDF file (also env var)
                "PS.lmax",
                "PS.KBprojectors",
                "%block PS.lmax",
            )
            self.__class__._registered = True  # noqa: SLF001 class-level flag

    def construct_pseudo_path(self) -> str | None:
        """
        Automatically construct the pseudopotential path from base_path and XC info.

        Priority order:
        1. If `pseudo_path` is explicitly set, use it (no construction needed)
        2. If `pseudo_base_path` and XC info are available, construct the path
        3. Otherwise, return None

        Returns
        -------
            str or None: Constructed pseudopotential path, or None if it
                cannot be constructed

        Example:
            pseudo_base_path = "/Users/user/.siesta/pseudos"
            xc_authors = "PBEsol"
            pseudo_family = "ONCVPSP"
            pseudo_relativistic = "FR"
            pseudo_version = "0.4"
            pseudo_quality = "Standard"

            → "/Users/user/.siesta/pseudos/ONCVPSP-PBEsol-FR-PDv0.4-Standard"
        """
        # If pseudo_path is explicitly set, use it
        if self.pseudo_path:
            if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Using explicitly set pseudo_path: {self.pseudo_path}[/blue]"
                )
            return self.pseudo_path

        # Need base_path and xc_authors to construct
        if not self.pseudo_base_path or not self.xc_authors:
            if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                if not self.pseudo_base_path:
                    console.print(
                        "[yellow]Cannot construct pseudo_path: "
                        "pseudo_base_path not set[/yellow]"
                    )
                if not self.xc_authors:
                    console.print(
                        "[yellow]Cannot construct pseudo_path: "
                        "xc_authors not set[/yellow]"
                    )
            return None

        # Normalize XC name (handle case variations)
        xc_name = self._normalize_xc_name(self.xc_authors)

        # Construct directory name: {family}-{XC}-{rel}-PDv{version}-{quality}
        # Example: ONCVPSP-PBEsol-FR-PDv0.4-Standard
        dir_name = (
            f"{self.pseudo_family}-{xc_name}-{self.pseudo_relativistic}"
            f"-PDv{self.pseudo_version}-{self.pseudo_quality}"
        )

        # Construct full path
        constructed_path = os.path.join(self.pseudo_base_path, dir_name)

        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(f"[green]Constructed pseudo_path: {constructed_path}[/green]")

        return constructed_path

    @staticmethod
    def _normalize_xc_name(xc_authors: str) -> str:
        """
        Normalize XC functional name to match pseudo directory naming convention.

        Args:
            xc_authors: XC authors string (e.g., "pbesol", "PBEsol", "PBE")

        Returns
        -------
            str: Normalized XC name (e.g., "PBEsol", "PBE")
        """
        # Common XC functional name mappings
        xc_mapping = {
            "pbe": "PBE",
            "pbesol": "PBEsol",
            "pw91": "PW91",
            "rpbe": "RPBE",
            "revpbe": "revPBE",
            "wc": "WC",
            "am05": "AM05",
            "blyp": "BLYP",
            "pz": "PZ",
            "ca": "CA",
            "pw92": "PW92",
        }

        xc_lower = xc_authors.lower()
        return xc_mapping.get(xc_lower, xc_authors)  # Return original if not in mapping

    @staticmethod
    def extract_xc_from_psml(psml_file: str) -> dict[str, str] | None:
        """
        Extract XC functional information from a PSML pseudopotential file.

        Args:
            psml_file: Path to .psml file

        Returns
        -------
            dict or None: Dictionary with XC information:
                - xc_type: Full XC description (e.g., "GGA -- Perdew-Burke-Ernzerhof")
                - xc_family: Functional family (e.g., "GGA", "LDA")
                - xc_name: Functional name (e.g., "PBE", "PBEsol")
            Returns None if XC info cannot be extracted.

        Example:
            >>> xc_info = Pseudopotentials.extract_xc_from_psml("/path/to/Si.psml")
            >>> print(xc_info["xc_name"])
            'PBE'
        """
        try:
            from xml.etree import ElementTree as ET

            tree = ET.parse(psml_file)  # noqa: S314 trusted local pseudo file
            root = tree.getroot()
            ns = {"psml": "http://esl.cecam.org/PSML/ns/1.1"}

            # Find exchange-correlation section
            xc_elem = root.find(".//psml:exchange-correlation", namespaces=ns)
            if xc_elem is None:
                return None

            # Get oncvpsp XC type annotation
            annotation = xc_elem.find(".//psml:annotation", namespaces=ns)
            if annotation is None:
                return None

            xc_type = annotation.get("oncvpsp-xc-type", "")
            if not xc_type:
                return None

            # Parse XC type: "GGA -- Perdew-Burke-Ernzerhof" or "LDA -- Perdew-Zunger"
            if " -- " in xc_type:
                xc_family, xc_full_name = xc_type.split(" -- ", 1)
                xc_family = xc_family.strip()
                xc_full_name = xc_full_name.strip()

                # Map full names to short names
                xc_name_mapping = {
                    "Perdew-Burke-Ernzerhof": "PBE",
                    "Perdew-Burke-Ernzerhof revised": "revPBE",
                    "Perdew-Wang 91": "PW91",
                    "Perdew-Zunger": "PZ",
                    "Ceperley-Alder": "CA",
                    "Wu-Cohen": "WC",
                }

                xc_name = xc_name_mapping.get(xc_full_name, xc_full_name)

                return {
                    "xc_type": xc_type,
                    "xc_family": xc_family,
                    "xc_name": xc_name,
                }

            return None  # noqa: TRY300 explicit control flow

        except Exception as e:  # noqa: BLE001 fall back to None on any parse error
            if Pseudopotentials.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[yellow]Could not extract XC from {psml_file}: {e}[/yellow]"
                )
            return None

    @staticmethod
    def validate_xc_consistency(
        pseudo_path: str,
        xc_functional: str,
        xc_authors: str,
        structure: "Structure",
    ) -> None:
        """
        Validate that XC functional in FDF matches pseudopotential files.

        Checks all element-specific .psml files in pseudo_path and warns if
        there's a mismatch between the FDF XC settings and pseudo XC.

        Args:
            pseudo_path: Path to pseudopotential directory
            xc_functional: XC functional family from FDF (e.g., "GGA")
            xc_authors: XC authors from FDF (e.g., "PBE", "PW91")
            structure: Pymatgen Structure object

        Warnings
        --------
            Prints colored warning if XC mismatch detected
        """
        if not pseudo_path or not os.path.isdir(pseudo_path):
            return

        # Get unique elements from structure
        elements = {site.species_string for site in structure}

        # Check each element's pseudopotential
        mismatches = []
        for element in elements:
            psml_file = os.path.join(pseudo_path, f"{element}.psml")
            if not os.path.exists(psml_file):
                continue

            xc_info = Pseudopotentials.extract_xc_from_psml(psml_file)
            if not xc_info:
                continue

            # Check XC family (GGA vs LDA)
            # Strip default marker comments before comparing
            xc_functional_clean = xc_functional.split("#", maxsplit=1)[0].strip()
            if xc_info["xc_family"].upper() != xc_functional_clean.upper():
                mismatches.append(
                    f"{element}: FDF has {xc_functional}, "
                    f"pseudo has {xc_info['xc_family']}"
                )

            # Check XC authors (PBE, PW91, etc.)
            # Normalize for comparison
            # Strip default marker comments before comparing
            xc_authors_clean = xc_authors.split("#", maxsplit=1)[0].strip()
            fdf_xc_norm = xc_authors_clean.upper().replace("-", "")
            pseudo_xc_norm = xc_info["xc_name"].upper().replace("-", "")

            if fdf_xc_norm != pseudo_xc_norm:
                mismatches.append(
                    f"{element}: FDF has {xc_authors}, "
                    f"pseudo has {xc_info['xc_name']} ({xc_info['xc_type']})"
                )

        # Print warnings if mismatches found
        if mismatches:
            # Always show XC mismatch warnings (this is a critical validation)
            console.print(
                "\n[bold yellow]⚠️  XC Functional Mismatch Warning[/bold yellow]"
            )
            console.print(
                "[yellow]The XC functional in your FDF does not match the "
                "pseudopotentials:[/yellow]\n"
            )
            for mismatch in mismatches:
                console.print(f"  [yellow]• {mismatch}[/yellow]")
            console.print(
                "\n[yellow]This may lead to inconsistent results. "
                "Consider either:[/yellow]"
            )
            console.print(
                "  [yellow]1. Using pseudopotentials matching "
                f"XC.Authors = {xc_authors}[/yellow]"
            )
            console.print(
                "  [yellow]2. Changing XC.Authors to match your "
                "pseudopotentials[/yellow]\n"
            )

    @staticmethod
    def parse_pseudo_path(pseudo_path: str) -> dict[str, str] | None:
        """
        Parse pseudopotential path to extract XC and other metadata from directory name.

        Supports directory naming pattern: {family}-{XC}-{rel}-PDv{version}-{quality}
        Example: ONCVPSP-PBEsol-FR-PDv0.4-Standard

        Args:
            pseudo_path: Full path to pseudopotential directory

        Returns
        -------
            dict or None: Dictionary with extracted metadata:
                - xc_authors: XC functional (e.g., "PBEsol", "PBE")
                - xc_functional: XC family (e.g., "GGA", "LDA")
                - pseudo_relativistic: Relativistic treatment (e.g., "SR", "FR")
                - pseudo_family: Pseudopotential family (e.g., "ONCVPSP")
                - pseudo_version: Version (e.g., "0.4")
                - pseudo_quality: Quality level (e.g., "Standard", "Stringent")
            Returns None if path doesn't match expected pattern.

        Example:
            >>> path = "/Users/user/.siesta/pseudos/ONCVPSP-PBEsol-FR-PDv0.4-Standard"
            >>> metadata = Pseudopotentials.parse_pseudo_path(path)
            >>> print(metadata["xc_authors"])
            'PBEsol'
        """
        import re

        # Get directory name from full path
        dir_name = os.path.basename(pseudo_path.rstrip("/"))

        # Pattern: {family}-{XC}-{rel}-PDv{version}-{quality}
        # Example: ONCVPSP-PBEsol-FR-PDv0.4-Standard
        pattern = r"^([^-]+)-([^-]+)-(SR|FR)-PDv([^-]+)-(.+)$"
        match = re.match(pattern, dir_name)

        if not match:
            if Pseudopotentials.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                console.print(
                    f"[yellow]Could not parse pseudo_path: {pseudo_path}[/yellow]"
                )
                console.print(
                    "[yellow]Expected pattern: "
                    "FAMILY-XC-REL-PDvVERSION-QUALITY[/yellow]"
                )
            return None

        family, xc_authors, rel, version, quality = match.groups()

        # Determine XC functional family from authors
        # LDA functionals
        lda_functionals = {"PZ", "CA", "PW92"}
        # Everything else is GGA (for now)
        xc_functional = "LDA" if xc_authors.upper() in lda_functionals else "GGA"

        metadata = {
            "xc_authors": xc_authors,
            "xc_functional": xc_functional,
            "pseudo_relativistic": rel,
            "pseudo_family": family,
            "pseudo_version": version,
            "pseudo_quality": quality,
        }

        if Pseudopotentials.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(f"[blue]Parsed pseudo_path: {pseudo_path}[/blue]")
            console.print(f"[blue]Extracted metadata: {metadata}[/blue]")

        return metadata

    @classmethod
    def setup_pseudos(
        cls,
        user_params: dict[str, Any] | None = None,
        default_pseudo_path: str | None = None,
    ) -> "Pseudopotentials":
        """
        Set up pseudopotential settings using user parameters and a default pseudo path.

        Args:
            user_params (dict, optional): User-provided parameters
                (case-insensitive, may include dots). Supported key:
                pseudo_path.
            default_pseudo_path (str, optional): Default path to
                pseudopotential files if not specified in user_params.

        Returns
        -------
            Pseudopotentials: Configured instance with pseudo_path and FDF arguments.
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]Pseudopotentials.setup_pseudos()[/green]")

        pseudo_settings = cls()
        user_params_ = OrderedDict(user_params) if user_params else OrderedDict()

        # Use default_pseudo_path as pseudo_base_path if neither is set
        # This allows auto-construction to work
        if (
            "pseudo_path" not in user_params_
            and "pseudo_base_path" not in user_params_
            and default_pseudo_path
        ):
            user_params_["pseudo_base_path"] = default_pseudo_path
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Using default_pseudo_path as pseudo_base_path: "
                    f"{default_pseudo_path}[/blue]"
                )

        pseudo_settings._user_params = user_params_

        if pseudo_settings._user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]user_params: {pseudo_settings._user_params}[/blue]"
                )

            pseudo_attributes = {
                field.name.lower()
                for field in fields(cls)
                if not field.name.startswith("_")
            }
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Available Pseudopotentials attributes: "
                    f"{pseudo_attributes}[/blue]"
                )

            for key, value in pseudo_settings._user_params.items():
                # Strip atomate2siesta prefix (a2s_ or atomate2siesta_)
                key_lower = key.lower()
                if key_lower.startswith("a2s_"):
                    key_stripped = key_lower[4:]  # Remove "a2s_"
                elif key_lower.startswith("atomate2siesta_"):
                    key_stripped = key_lower[15:]  # Remove "atomate2siesta_"
                else:
                    key_stripped = key_lower

                key_normalized = key_stripped.replace(".", "_")
                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                    console.print(
                        f"[blue]Processing key: {key} -> stripped: "
                        f"{key_stripped} -> normalized: {key_normalized}, "
                        f"value: {value}[/blue]"
                    )

                if key_normalized == "_user_params":
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                        console.print(
                            f"[yellow]Ignoring user-provided '{key}'; "
                            f"it is internal.[/yellow]"
                        )
                    continue

                if key_normalized in pseudo_attributes:
                    original_key = next(
                        field.name
                        for field in fields(cls)
                        if field.name.lower() == key_normalized
                    )
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                        console.print(
                            f"[blue]Matched Pseudopotentials field: "
                            f"{original_key} = {value}[/blue]"
                        )

                    # Set the attribute directly (works for all fields)
                    try:
                        setattr(pseudo_settings, original_key, value)
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                            console.print(f"[blue]Set {original_key} = {value}[/blue]")
                    except Exception as e:  # noqa: BLE001 report and skip bad value
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Failed to set '{original_key}': {e}[/yellow]"
                            )
                elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                    console.print(
                        f"[yellow]Key '{key}' does not match any "
                        f"Pseudopotentials field, skipping.[/yellow]"
                    )

        # Automatically construct pseudo_path if not explicitly set
        if not pseudo_settings.pseudo_path:
            constructed_path = pseudo_settings.construct_pseudo_path()
            if constructed_path:
                pseudo_settings.pseudo_path = constructed_path
                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
                    console.print(
                        f"[green]Auto-constructed pseudo_path: "
                        f"{constructed_path}[/green]"
                    )
            elif pseudo_settings.pseudo_base_path or pseudo_settings.xc_authors:
                # We have some info but couldn't construct - warn user
                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                    console.print(
                        "[yellow]Could not auto-construct pseudo_path. "
                        "Ensure both pseudo_base_path and xc_authors are set, "
                        "or provide pseudo_path directly.[/yellow]"
                    )

        try:
            pseudo_settings.validate()
        except ValueError as e:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.ERROR.value:
                console.print(f"[red]Validation failed: {e}[/red]")
            raise

        pseudo_settings.generate_pseudo_fdf()
        pseudo_settings._user_params = None

        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]Validation & Generation: "
                "[yellow]Pseudopotentials[/yellow] Successful![/green]"
            )

        return pseudo_settings

    def validate(self) -> None:
        """
        Validate the pseudopotential settings.

        Raises
        ------
            ValueError: If pseudo_path is invalid.
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]Pseudopotentials.validate()[/green]")

        if self.pseudo_path and not os.path.isdir(self.pseudo_path):
            raise ValueError(
                f"Pseudopotential path '{self.pseudo_path}' does not exist "
                f"or is not a directory."
            )

        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
            console.print(f"[blue]Validated: pseudo_path={self.pseudo_path}[/blue]")

        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]Validation: "
                "[yellow]Pseudopotentials[/yellow] Successful![/green]"
            )

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["siesta_pp_path", "pseudo_path"]:
                self.pseudo_path = str(value)

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters

        Note:
            SIESTA_PP_PATH can be specified in FDF file (though it's also an env var).
            Users can pass pseudo_path to override automatic path construction.
        """
        fdf: dict[str, Any] = {}

        # Construct or use explicit path
        pseudo_path = self.construct_pseudo_path()
        if pseudo_path:
            fdf["SIESTA_PP_PATH"] = pseudo_path

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE handles pseudopotentials through species parameter
        # Path is typically set via environment variable
        return {}

    def generate_pseudo_fdf(self) -> None:
        """Generate the pseudopotential-related FDF arguments for SIESTA input."""
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]Pseudopotentials.generate_pseudo_fdf()[/green]")

        self.pseudo_fdf_arguments = OrderedDict()
        self.pseudo_fdf_arguments["#Pseudopotentials"] = "Pseudopotentials"

        if self.pseudo_path:
            self.pseudo_fdf_arguments["SIESTA_PP_PATH"] = self.pseudo_path
            if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(f"[blue]Set SIESTA_PP_PATH: {self.pseudo_path}[/blue]")
