"""Output document of the Raman workflow."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pymatgen.core import Structure


class RamanModeData(BaseModel):
    """Per-mode results for the modes that were actually computed."""

    mode: int = Field(description="Mode index (0-based, sorted by frequency)")
    frequency_cm: float = Field(description="Frequency in cm^-1 (negative = imaginary)")
    irrep: str | None = Field(None, description="Irreducible representation label")
    raman_tensor: list[list[float]] = Field(
        description="Raman tensor V/(4pi) d(eps)/dQ (atomate-CRD normalisation)"
    )
    raman_activity: float = Field(description="45 alpha^2 + 7 beta^2")
    intensity: float = Field(
        description="Powder intensity as in atomate-CRD plot_raman.py (activity / 3)"
    )
    displacements: list[float] = Field(description="Displacements used (Angstrom)")
    epsilons: list[list[list[float]]] = Field(
        description="Dielectric tensors of the displaced structures"
    )
    task_uuids: list[str | None] = Field(description="UUIDs of the dielectric jobs")
    task_dirs: list[str | None] = Field(description="Directories of dielectric jobs")


class RamanDoc(BaseModel):
    """Raman tensors, activities and spectrum."""

    structure: Structure = Field(
        description="Primitive cell used for the normal modes and Raman runs"
    )
    formula_pretty: str = Field(description="Reduced formula")
    mp_id: str | None = Field(None, description="Materials Project id, if given")
    bandgap: float | None = Field(None, description="Band gap used for Rk (eV)")
    kpoints_length: int | None = Field(
        None, description="Rk of the k-mesh of the dielectric runs"
    )
    pointgroup: str | None = Field(None, description="Point group of the structure")
    step_size: float = Field(description="Displacement along each mode (Angstrom)")

    frequencies: list[float] = Field(description="All Gamma frequencies (cm^-1)")
    frequencies_thz: list[float] = Field(description="All Gamma frequencies (THz)")
    normalmodes: dict = Field(
        description="{'eigenvals': VASP-convention eigenvalues, 'eigenvecs': "
        "(3N, N, 3) mass-weighted eigenvectors}"
    )
    irrep_labels: list[str | None] = Field(description="Irrep label of each mode")
    raman_active: list[bool] = Field(description="Raman activity by symmetry")
    ir_active: list[bool] = Field(description="IR activity by symmetry")
    acoustic_modes: list[int] = Field(description="Indices treated as acoustic")
    computed_modes: list[int] = Field(description="Modes for which eps was computed")

    raman_tensor: dict[str, list[list[float]]] = Field(
        description="Raman tensor per computed mode"
    )
    modes: list[RamanModeData] = Field(description="Detailed per-mode data")
    unstable_modes: list[int] = Field(
        default_factory=list, description="Computed modes with imaginary frequency"
    )

    spectrum_frequencies: list[float] | None = Field(
        None, description="Frequency grid of the broadened spectrum (cm^-1)"
    )
    spectrum_intensities: list[float] | None = Field(
        None, description="Normalised broadened spectrum"
    )
    lorentzian_width: float | None = Field(None, description="Line width (cm^-1)")
    state: str = "successful"
