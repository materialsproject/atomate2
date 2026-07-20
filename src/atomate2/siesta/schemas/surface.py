"""Schemas for surface energy calculations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TerminationData(BaseModel):
    """Data for a single surface termination."""

    termination: str = Field(..., description="Termination species or label")
    surface_energy: float = Field(..., description="Surface energy in eV/Å²")
    surface_energy_Jm2: float = Field(..., description="Surface energy in J/m²")
    relative_energy: float = Field(
        ..., description="Relative to lowest energy termination (eV/Å²)"
    )
    is_lowest: bool = Field(
        ..., description="Whether this is the lowest energy termination"
    )

    # Energetics
    slab_energy: float = Field(..., description="Total energy of slab (eV)")
    n_formula_units: float = Field(..., description="Number of formula units in slab")

    # Geometry
    surface_area: float = Field(..., description="Surface area in Ŵ²")
    n_atoms: int = Field(..., description="Number of atoms in slab")
    thickness: float = Field(default=0.0, description="Slab thickness in Å")

    # Composition
    composition: dict[str, int] = Field(..., description="Composition dictionary")
    bottom_composition: dict[str, int] = Field(
        default_factory=dict, description="Bottom layer composition"
    )
    top_composition: dict[str, int] = Field(
        default_factory=dict, description="Top layer composition"
    )

    # Metadata
    is_symmetric: bool = Field(default=False, description="Whether slab is symmetric")
    z_position: float = Field(
        default=0.0, description="Z-position of termination layer"
    )


class SurfaceEnergyDocument(BaseModel):
    """Complete surface energy calculation results."""

    # Input parameters
    miller_indices: tuple[int, int, int] = Field(
        ..., description="Miller indices (h,k,l)"
    )
    formula_units_per_cell: int = Field(
        ..., description="Formula units in bulk unit cell"
    )

    # Bulk properties
    bulk_energy: float = Field(..., description="Bulk energy per formula unit (eV)")
    bulk_energy_per_atom: float = Field(
        default=0.0, description="Bulk energy per atom (eV)"
    )

    # Surface terminations
    terminations: list[TerminationData] = Field(
        default_factory=list, description="Data for each termination"
    )
    lowest_termination: str = Field(..., description="Lowest energy termination label")

    # Statistics
    n_terminations: int = Field(default=0, description="Number of terminations")
    energy_spread: float = Field(
        default=0.0, description="Max - min surface energy (eV/Ų)"
    )

    # Optional metadata
    slab_directory: str = Field(
        default="", description="Directory containing slab files"
    )
    calculation_method: str = Field(default="SIESTA", description="DFT code used")

    class Config:
        """Pydantic configuration."""

        extra = "allow"
