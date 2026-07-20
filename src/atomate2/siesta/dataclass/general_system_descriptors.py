"""
Module defining base SIESTA input set and generator.

class GeneralSystemDescriptors

Based on User's Guide Siesta 5.4.0
Section: 6.1 General system descriptors

TODO: Need to check what we need it from here
"""

# Metadata

__all__ = ["GeneralSystemDescriptors"]

import logging
from dataclasses import dataclass, field
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.utils.common import console

logger = logging.getLogger(__name__)


@dataclass
class GeneralSystemDescriptors(FDFDataclass):
    """Data class to store general system descriptors for SIESTA input."""

    # -------------------------------
    # 6.1 General system descriptors
    # -------------------------------

    system_label: str = field(
        default="siesta",
        metadata={
            "description": "Nickname for the system, used in output files.",
            "SIESTA keyword": "SystemLabel",
        },
    )  # Nickname for the system, used in output files

    system_name: str = field(
        default="siesta",
        metadata={
            "description": "Descriptive name of the system.",
            "SIESTA keyword": "SystemName",
        },
    )  # Descriptive name of the system

    number_of_species: int = field(
        default=0,
        metadata={
            "description": "Number of different atomic species in the simulation",
            "SIESTA keyword": "NumberOfSpecies",
        },
    )  # Number of different atomic species in the simulation

    number_of_atoms: int = field(
        default=0,
        metadata={
            "description": "Number of atoms in the simulation",
            "SIESTA keyword": "NumberOfAtoms",
        },
    )  # Number of atoms in the simulation

    chemical_species_label: dict[int, str] = field(
        default_factory=dict,
        metadata={
            "description": "Mapping of species number to chemical label",
            "SIESTA keyword": "%block ChemicalSpeciesLabel",
        },
    )  # Mapping of species number to chemical label

    synthetic_atoms: dict[int, list[float]] = field(
        default_factory=dict,
        metadata={
            "description": "Information for synthetic atoms, "
            "such as pseudopotential parameters",
            "SIESTA keyword": " %block SyntheticAtoms",
        },
    )  # Information for synthetic atoms

    atomic_mass: dict[int, float] = field(
        default_factory=dict,
        metadata={
            "description": "Custom atomic masses for different species",
            "SIESTA keyword": "%block AtomicMass",
        },
    )  # Custom atomic masses for different species

    _comments: str = field(
        default="",
        metadata={
            "description": "User-provided comments to be included as a comment "
            "block in the FDF file.",
            "SIESTA keyword": None,
        },
    )
    # system_label: str = "siesta"  # Nickname for the system, used in output files
    # system_name: str = "siesta"  # Descriptive name of the system
    # number_of_species: int = 0  # Number of different atomic species in the simulation
    # number_of_atoms: int = 0  # Number of atoms in the simulation
    # Mapping of species number to chemical label:
    # chemical_species_label: Dict[int, str] = field(default_factory=dict)
    # Information for synthetic atoms:
    # synthetic_atoms: Dict[int, List[float]] = field(default_factory=dict)
    # Custom atomic masses for different species:
    # atomic_mass: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "SystemLabel",
                "SystemName",
                "NumberOfSpecies",
                "NumberOfAtoms",
                "%block ChemicalSpeciesLabel",
                "%block SyntheticAtoms",
                "%block AtomicMass",
                "LatticeConstant",
                "AtomicCoordinatesFormat",
            )
            self.__class__._registered = True  # noqa: SLF001 class-level registration guard

    def validate_label_and_name(self) -> None:
        """Force label and name to be siesta."""
        logger.info("GeneralSystemDescriptors.validate_label_and_name()")
        if (self.system_label != "siesta") or (self.system_name != "siesta"):
            console.print(
                "[red] system label & name should be siesta ... "
                "(Maybe i'll change later this... [/red]"
            )
            # ("system label & name should be siesta ...
            # (Maybe i'll change later this...)")
            raise ValueError

    def validate(self) -> None:
        """Validate the general system descriptors."""
        logger.info("GeneralSystemDescriptors.validate()")
        if self.number_of_species != len(self.chemical_species_label):
            raise ValueError(
                "Number of species does not match the number of entries "
                "in chemical_species_label."
            )
        if self.number_of_atoms <= 0:
            raise ValueError("Number of atoms must be greater than 0.")
        # print(f"Validated: {self.number_of_species=} {self.number_of_atoms=}")

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            if key_lower in ["systemlabel", "system_label"]:
                self.system_label = str(value)
            elif key_lower in ["systemname", "system_name"]:
                self.system_name = str(value)
            elif key_lower in ["numberofspecies", "number_of_species"]:
                self.number_of_species = int(value)
            elif key_lower in ["numberofatoms", "number_of_atoms"]:
                self.number_of_atoms = int(value)
            elif key_lower == "%block chemicalspecieslabel" and isinstance(value, dict):
                self.chemical_species_label = value

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters
        """
        fdf: dict[str, Any] = {}
        fdf["#GeneralSystem"] = "General System Descriptors"

        fdf["SystemLabel"] = self.system_label
        fdf["SystemName"] = self.system_name
        fdf["NumberOfSpecies"] = str(self.number_of_species)
        fdf["NumberOfAtoms"] = str(self.number_of_atoms)

        if self.chemical_species_label:
            fdf["%block ChemicalSpeciesLabel"] = self.chemical_species_label

        if self.synthetic_atoms:
            fdf["%block SyntheticAtoms"] = self.synthetic_atoms

        if self.atomic_mass:
            fdf["%block AtomicMass"] = self.atomic_mass

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't use system labels/descriptors
        # These are SIESTA-specific metadata
        return {}

    @classmethod
    def setup_system_descriptors(
        cls, user_params: dict[str, Any] | None = None
    ) -> "GeneralSystemDescriptors":
        """
        Create and configure a GeneralSystemDescriptors instance.

        Note: This module is primarily auto-populated from the Structure object.
        User parameters are rarely needed for system descriptors.

        Args:
            user_params: Dictionary of user-defined parameters (optional)

        Returns
        -------
            GeneralSystemDescriptors: Configured instance
        """
        logger.info("GeneralSystemDescriptors.setup_system_descriptors()")
        instance = cls()

        # Simple parameter assignment if provided
        # Note: number_of_species, number_of_atoms, chemical_species_label
        # are typically set automatically from the Structure object
        if user_params:
            for key, value in user_params.items():
                key_normalized = key.lower().replace(".", "_")
                if hasattr(instance, key_normalized):
                    setattr(instance, key_normalized, value)

        return instance
