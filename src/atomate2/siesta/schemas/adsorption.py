"""Schemas for adsorption site calculations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AdsorptionSiteResult(BaseModel):
    """Results for a single adsorption site."""

    site_x: float = Field(description="X position in fractional coordinates")
    site_y: float = Field(description="Y position in fractional coordinates")
    site_x_cart: float = Field(description="X position in Cartesian coordinates (Å)")
    site_y_cart: float = Field(description="Y position in Cartesian coordinates (Å)")
    adsorption_energy: float = Field(description="Adsorption energy (eV)")
    adsorption_energy_per_area: float = Field(
        description="Adsorption energy per unit area (eV/Ų)"
    )
    total_energy: float = Field(description="Total energy of slab+adsorbate (eV)")
    slab_energy: float = Field(description="Energy of clean slab (eV)")
    adsorbate_energy: float = Field(description="Energy of isolated adsorbate (eV)")
    surface_area: float = Field(description="Surface area of slab (Ų)")
    height: float = Field(description="Initial adsorbate height above surface (Å)")
    n_atoms: int = Field(description="Total number of atoms in combined system")
    n_slab_atoms: int = Field(description="Number of atoms in slab")
    n_adsorbate_atoms: int = Field(description="Number of atoms in adsorbate")


class AdsorptionScanDocument(BaseModel):
    """Document containing complete adsorption site scan results."""

    slab_formula: str = Field(description="Chemical formula of the slab")
    adsorbate_formula: str = Field(description="Chemical formula of the adsorbate")
    miller_indices: tuple[int, int, int] | None = Field(
        None, description="Miller indices of the surface"
    )
    grid_size: tuple[int, int] = Field(description="Grid size (nx, ny) for scanning")
    initial_height: float = Field(
        description="Initial adsorbate height above surface (Å)"
    )
    surface_area: float = Field(description="Surface area of the slab (Ų)")
    slab_thickness: float = Field(
        description="Thickness of the slab in z-direction (Å)"
    )
    total_sites_scanned: int = Field(
        description="Total number of adsorption sites scanned"
    )

    # Reference energies (for reuse in multi-adsorbate workflows)
    slab_energy: float = Field(description="Energy of clean slab (eV)")
    adsorbate_energy: float = Field(description="Energy of isolated adsorbate (eV)")

    # Best site information
    best_site_position: tuple[float, float] = Field(
        description="Position of best adsorption site (fractional coordinates)"
    )
    best_adsorption_energy: float = Field(
        description="Lowest adsorption energy found (eV)"
    )
    best_energy_per_area: float = Field(
        description="Energy per area at best site (eV/Ų)"
    )

    # Energy statistics
    mean_adsorption_energy: float = Field(
        description="Mean adsorption energy across all sites (eV)"
    )
    std_adsorption_energy: float = Field(
        description="Standard deviation of adsorption energies (eV)"
    )
    energy_range: float = Field(
        description="Energy range (max - min) across all sites (eV)"
    )

    # All site results
    site_results: list[AdsorptionSiteResult] = Field(
        description="Results for all scanned sites"
    )

    # Computational details
    input_parameters: dict[str, Any] | None = Field(
        None, description="SIESTA input parameters used"
    )

    @property
    def best_site(self) -> AdsorptionSiteResult:
        """Get the site with lowest adsorption energy."""
        return min(self.site_results, key=lambda x: x.adsorption_energy)

    @property
    def top_5_sites(self) -> list[AdsorptionSiteResult]:
        """Get the 5 sites with lowest adsorption energy."""
        return sorted(self.site_results, key=lambda x: x.adsorption_energy)[:5]


class AdsorptionOptimizationResult(BaseModel):
    """Results from adsorption geometry optimization."""

    initial_site: tuple[float, float] = Field(
        description="Initial adsorption site position (fractional)"
    )
    initial_adsorption_energy: float = Field(
        description="Initial adsorption energy before optimization (eV)"
    )
    final_adsorption_energy: float = Field(
        description="Final adsorption energy after optimization (eV)"
    )
    energy_improvement: float = Field(
        description="Energy lowering from optimization (eV)"
    )
    initial_total_energy: float = Field(description="Initial total energy (eV)")
    final_total_energy: float = Field(description="Final total energy (eV)")
    converged: bool = Field(description="Whether optimization converged")
    n_ionic_steps: int | None = Field(
        None, description="Number of ionic steps in optimization"
    )
