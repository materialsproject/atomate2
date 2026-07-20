"""Schemas for phonon calculations."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pymatgen.core import Structure


class ThermalProperties(BaseModel):
    """Thermal properties from phonopy."""

    temperatures: list[float] = Field(description="Temperatures in Kelvin")
    free_energy: list[float] = Field(description="Helmholtz free energy in eV")
    entropy: list[float] = Field(description="Entropy in eV/K")
    heat_capacity: list[float] = Field(
        description="Heat capacity at constant volume (Cv) in eV/K"
    )


class PhononDocument(BaseModel):
    """Document containing phonon calculation results."""

    # Structure information
    structure: Structure = Field(description="Original unit cell structure")
    supercell_matrix: list[list[int]] = Field(
        description="Supercell matrix used for phonon calculation"
    )

    # Calculation parameters
    displacement: float = Field(description="Atomic displacement distance in Angstroms")
    symprec: float = Field(description="Symmetry precision used in phonopy")
    n_displacements: int = Field(
        description="Number of displacement calculations performed"
    )

    # Force constants
    force_constants: list[list[float]] = Field(
        description="Force constants matrix from phonopy"
    )

    # Phonon frequencies
    has_imaginary_frequencies: bool = Field(
        description="Whether imaginary frequencies are present (structural instability)"
    )
    min_frequency: float = Field(description="Minimum phonon frequency in THz")
    max_frequency: float = Field(description="Maximum phonon frequency in THz")

    # Thermal properties (optional)
    thermal_properties: ThermalProperties | None = Field(
        None, description="Thermal properties (Cv, entropy, free energy) vs temperature"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
