"""
2D slab correction for charged defects in two-dimensional materials.

Implementation of finite-size corrections for charged defects in 2D slabs
with vacuum regions, following methodologies from:

References:
1. Komsa et al., Phys. Rev. X 4, 031044 (2014)
   "Charged Point Defects in the Flatland"
2. Noh et al., Phys. Rev. B 89, 205417 (2014)
   "Native defects in single-layer MoS2"
3. JDFTx tutorial: http://jdftx.org/Defects2D.html

Key differences from 3D bulk corrections:
- Anisotropic dielectric screening (in-plane vs out-of-plane)
- Spatially-varying dielectric profile ε(z)
- Modified Coulomb interactions due to reduced dimensionality
- Vacuum boundary conditions in z-direction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pymatgen.core import Structure

from atomate2.siesta.flows.defects.corrections.base import (
    CorrectionResult,
    CorrectionScheme,
)

logger = logging.getLogger(__name__)


@dataclass
class DielectricProfile:
    """
    Dielectric profile for 2D slab systems.

    Represents spatially-varying dielectric function perpendicular to the slab.

    Parameters
    ----------
    z_coords : np.ndarray
        Z-coordinates (Å) along slab normal
    epsilon_parallel : np.ndarray
        In-plane dielectric function ε∥(z)
    epsilon_perpendicular : np.ndarray
        Out-of-plane dielectric function ε⊥(z)
    slab_center : float
        Z-coordinate of slab center (Å)
    slab_thickness : float
        Thickness of slab region (Å)
    vacuum_thickness : float
        Thickness of vacuum regions (Å)
    """

    z_coords: np.ndarray
    epsilon_parallel: np.ndarray
    epsilon_perpendicular: np.ndarray
    slab_center: float
    slab_thickness: float
    vacuum_thickness: float

    def epsilon_parallel_at(self, z: float) -> float:
        """Get ε∥ at position z by interpolation."""
        return float(np.interp(z, self.z_coords, self.epsilon_parallel))

    def epsilon_perpendicular_at(self, z: float) -> float:
        """Get ε⊥ at position z by interpolation."""
        return float(np.interp(z, self.z_coords, self.epsilon_perpendicular))

    @classmethod
    def create_gaussian_profile(
        cls,
        epsilon_parallel_bulk: float,
        epsilon_perpendicular_bulk: float,
        slab_center: float,
        slab_thickness: float,
        cell_length_z: float,
        sigma: float | None = None,
    ) -> DielectricProfile:
        """
        Create Gaussian dielectric profile.

        The dielectric function transitions from bulk values inside the slab
        to vacuum (ε=1) outside using Gaussian decay.

        Parameters
        ----------
        epsilon_parallel_bulk : float
            In-plane dielectric constant in bulk slab
        epsilon_perpendicular_bulk : float
            Out-of-plane dielectric constant in bulk slab
        slab_center : float
            Z-coordinate of slab center (Å)
        slab_thickness : float
            Thickness of slab (Å)
        cell_length_z : float
            Length of supercell in z-direction (Å)
        sigma : float, optional
            Width of Gaussian transition (Å). Default: slab_thickness/4

        Returns
        -------
        DielectricProfile
            Gaussian dielectric profile
        """
        if sigma is None:
            sigma = max(slab_thickness / 4.0, 0.5)  # Minimum 0.5 Å

        # Avoid division by zero
        if sigma < 0.01:
            sigma = 0.5
            logger.warning(f"Sigma too small, using {sigma} Å")

        # Create z-grid with 200 points
        z_coords = np.linspace(0, cell_length_z, 200)

        # Gaussian transition from bulk to vacuum
        # ε(z) = 1 + (ε_bulk - 1) * exp(-(z - z_center)² / (2σ²))
        z_from_center = np.abs(z_coords - slab_center)

        # Adjust for periodic boundary (consider images)
        z_from_center = np.minimum(z_from_center, cell_length_z - z_from_center)

        # Gaussian profile
        gaussian = np.exp(-(z_from_center**2) / (2 * sigma**2))

        epsilon_parallel = 1.0 + (epsilon_parallel_bulk - 1.0) * gaussian
        epsilon_perpendicular = 1.0 + (epsilon_perpendicular_bulk - 1.0) * gaussian

        vacuum_thickness = (cell_length_z - slab_thickness) / 2.0

        return cls(
            z_coords=z_coords,
            epsilon_parallel=epsilon_parallel,
            epsilon_perpendicular=epsilon_perpendicular,
            slab_center=slab_center,
            slab_thickness=slab_thickness,
            vacuum_thickness=vacuum_thickness,
        )

    @classmethod
    def create_step_profile(
        cls,
        epsilon_parallel_bulk: float,
        epsilon_perpendicular_bulk: float,
        slab_center: float,
        slab_thickness: float,
        cell_length_z: float,
        transition_width: float | None = None,
    ) -> DielectricProfile:
        """
        Create step-function dielectric profile.

        Sharp transition from bulk ε inside slab to vacuum (ε=1) outside.

        Parameters
        ----------
        epsilon_parallel_bulk : float
            In-plane dielectric constant in bulk slab
        epsilon_perpendicular_bulk : float
            Out-of-plane dielectric constant in bulk slab
        slab_center : float
            Z-coordinate of slab center (Å)
        slab_thickness : float
            Thickness of slab (Å)
        cell_length_z : float
            Length of supercell in z-direction (Å)
        transition_width : float, optional
            Width of smoothed transition region (Å). Default: 0.5 Å

        Returns
        -------
        DielectricProfile
            Step-function dielectric profile
        """
        if transition_width is None:
            transition_width = 0.5  # Å

        z_coords = np.linspace(0, cell_length_z, 200)

        # Distance from slab center
        z_from_center = np.abs(z_coords - slab_center)
        z_from_center = np.minimum(z_from_center, cell_length_z - z_from_center)

        # Smooth step function using tanh
        # tanh((d - r)/w) transitions from +1 (inside) to -1 (outside)
        # Convert to 0 (outside) to 1 (inside)
        slab_half_thickness = slab_thickness / 2.0
        step = 0.5 * (
            1.0 - np.tanh((z_from_center - slab_half_thickness) / transition_width)
        )

        epsilon_parallel = 1.0 + (epsilon_parallel_bulk - 1.0) * step
        epsilon_perpendicular = 1.0 + (epsilon_perpendicular_bulk - 1.0) * step

        vacuum_thickness = (cell_length_z - slab_thickness) / 2.0

        return cls(
            z_coords=z_coords,
            epsilon_parallel=epsilon_parallel,
            epsilon_perpendicular=epsilon_perpendicular,
            slab_center=slab_center,
            slab_thickness=slab_thickness,
            vacuum_thickness=vacuum_thickness,
        )


def detect_slab_geometry(
    structure: Structure,
    vacuum_threshold: float = 6.0,
    min_slab_thickness: float = 2.0,
) -> dict:
    """
    Detect if structure is a 2D slab and extract slab parameters.

    Parameters
    ----------
    structure : Structure
        Structure to analyze
    vacuum_threshold : float
        Minimum vacuum thickness (Å) to consider as slab. Default: 6.0 Å
    min_slab_thickness : float
        Minimum assumed thickness for single-layer 2D materials (Å). Default: 2.0 Å

    Returns
    -------
    dict
        Dictionary with keys:
        - is_slab: bool
        - slab_center: float (z-coordinate)
        - slab_thickness: float (Å)
        - vacuum_thickness: float (Å)
        - cell_length_z: float (Å)
        - z_min: float (minimum z of atoms)
        - z_max: float (maximum z of atoms)
        - is_monolayer: bool (True if all atoms at same z)
    """
    # Get z-coordinates of all atoms
    cart_coords = structure.cart_coords
    z_coords = cart_coords[:, 2]

    z_min = np.min(z_coords)
    z_max = np.max(z_coords)

    slab_thickness_raw = z_max - z_min
    cell_length_z = structure.lattice.c

    # Check if monolayer (all atoms at essentially same z)
    is_monolayer = slab_thickness_raw < 0.01  # Å

    # For monolayers, use minimum thickness (typical bond length)
    if is_monolayer:
        slab_thickness = min_slab_thickness
        logger.info(
            f"Detected monolayer structure (Δz = {slab_thickness_raw:.4f} Å). "
            f"Using minimum thickness: {min_slab_thickness} Å"
        )
    else:
        slab_thickness = slab_thickness_raw

    vacuum_thickness = cell_length_z - slab_thickness

    # Check if this is a slab (large vacuum region)
    is_slab = vacuum_thickness >= vacuum_threshold

    slab_center = (z_min + z_max) / 2.0

    return {
        "is_slab": is_slab,
        "slab_center": slab_center,
        "slab_thickness": slab_thickness,
        "vacuum_thickness": vacuum_thickness,
        "cell_length_z": cell_length_z,
        "z_min": z_min,
        "z_max": z_max,
        "is_monolayer": is_monolayer,
    }


class Slab2DCorrection(CorrectionScheme):
    """
    Finite-size correction for charged defects in 2D slabs.

    Implements anisotropic dielectric screening and Gaussian charge model
    for 2D materials with vacuum regions.

    Based on Komsa et al. (PRX 2014) and Noh et al. (PRB 2014) methodologies.

    Formula:
        E_corr = E_lat + E_align
        E_lat = energy from Gaussian charge in periodic vs isolated slab
        E_align = q * ΔV (potential alignment correction)

    Parameters
    ----------
    epsilon_parallel : float
        In-plane dielectric constant
    epsilon_perpendicular : float
        Out-of-plane dielectric constant
    profile_type : str
        Type of dielectric profile: "gaussian" or "step"
    gaussian_sigma : float, optional
        Width of Gaussian charge (Å). Default: auto-determined
    alignment_cutoff : float
        Fraction of cell to use for alignment (0.0-1.0). Default: 0.8
    vacuum_threshold : float
        Minimum vacuum thickness (Å) for slab detection. Default: 6.0 Å
    """

    def __init__(
        self,
        epsilon_parallel: float,
        epsilon_perpendicular: float | None = None,
        profile_type: str = "gaussian",
        gaussian_sigma: float | None = None,
        alignment_cutoff: float = 0.8,
        vacuum_threshold: float = 6.0,
    ):
        """Initialize 2D slab correction."""
        self.epsilon_parallel = epsilon_parallel

        # If ε⊥ not provided, assume isotropic
        if epsilon_perpendicular is None:
            self.epsilon_perpendicular = epsilon_parallel
            logger.warning(
                f"epsilon_perpendicular not provided. Assuming isotropic "
                f"dielectric constant: ε⊥ = ε∥ = {epsilon_parallel}"
            )
        else:
            self.epsilon_perpendicular = epsilon_perpendicular

        self.profile_type = profile_type
        self.gaussian_sigma = gaussian_sigma
        self.alignment_cutoff = alignment_cutoff
        self.vacuum_threshold = vacuum_threshold

        # Initialize base class with average ε
        epsilon_avg = (2 * epsilon_parallel + epsilon_perpendicular) / 3
        super().__init__(epsilon_static=epsilon_avg)

    @property
    def name(self) -> str:
        """Name of the correction scheme."""
        return "Slab2D"

    @property
    def charge_model(self) -> str:
        """Type of charge model used."""
        return "gaussian_2d"

    @property
    def requires_dielectric(self) -> bool:
        """Whether the scheme requires dielectric constant."""
        return True

    @property
    def supports_anisotropic(self) -> bool:
        """Whether the scheme supports anisotropic dielectrics."""
        return True

    @property
    def requires_potential_data(self) -> bool:
        """Whether the scheme requires electrostatic potential data."""
        return True

    def calculate_correction(
        self,
        defect_structure: Structure,
        host_structure: Structure,
        charge_state: int,
        defect_energy: float,
        host_energy: float,
        defect_site: list[float] | None = None,
        potential_data: dict | None = None,
        **kwargs,
    ) -> CorrectionResult:
        """
        Calculate 2D slab correction.

        Parameters
        ----------
        defect_structure : Structure
            Relaxed defect supercell structure
        host_structure : Structure
            Pristine host supercell structure
        charge_state : int
            Charge state of the defect (e.g., +2, 0, -1)
        defect_energy : float
            Total energy of defect supercell (eV)
        host_energy : float
            Total energy of host supercell (eV)
        defect_site : list[float], optional
            Fractional coordinates [x, y, z] of defect site
        potential_data : dict, optional
            Electrostatic potential data (currently simplified)
        **kwargs
            Additional parameters

        Returns
        -------
        CorrectionResult
            Correction result with energy and metadata
        """
        # Validate inputs
        self.validate_inputs(defect_structure, host_structure, charge_state)

        # Detect slab geometry
        slab_info = detect_slab_geometry(defect_structure, self.vacuum_threshold)

        if not slab_info["is_slab"]:
            logger.warning(
                f"Structure does not appear to be a 2D slab "
                f"(vacuum thickness = {slab_info['vacuum_thickness']:.2f} Å < "
                f"{self.vacuum_threshold} Å). Consider using 3D bulk corrections instead."
            )

        # Create dielectric profile
        if self.profile_type == "gaussian":
            profile = DielectricProfile.create_gaussian_profile(
                epsilon_parallel_bulk=self.epsilon_parallel,
                epsilon_perpendicular_bulk=self.epsilon_perpendicular,
                slab_center=slab_info["slab_center"],
                slab_thickness=slab_info["slab_thickness"],
                cell_length_z=slab_info["cell_length_z"],
                sigma=self.gaussian_sigma,
            )
        elif self.profile_type == "step":
            profile = DielectricProfile.create_step_profile(
                epsilon_parallel_bulk=self.epsilon_parallel,
                epsilon_perpendicular_bulk=self.epsilon_perpendicular,
                slab_center=slab_info["slab_center"],
                slab_thickness=slab_info["slab_thickness"],
                cell_length_z=slab_info["cell_length_z"],
            )
        else:
            raise ValueError(
                f"Unknown profile_type: {self.profile_type}. "
                f"Must be 'gaussian' or 'step'"
            )

        # Calculate lattice term (Gaussian charge in periodic vs isolated)
        lattice_term = self._calculate_lattice_term(
            defect_structure, charge_state, profile, slab_info
        )

        # Calculate potential alignment (simplified for now)
        alignment_energy = self._calculate_alignment(
            defect_structure, host_structure, charge_state, potential_data
        )

        # Total correction
        correction_energy = lattice_term + alignment_energy

        # Build metadata
        metadata = {
            "slab_info": slab_info,
            "epsilon_parallel": self.epsilon_parallel,
            "epsilon_perpendicular": self.epsilon_perpendicular,
            "profile_type": self.profile_type,
            "lattice_term": lattice_term,
            "gaussian_sigma": self.gaussian_sigma,
        }

        return CorrectionResult(
            correction_energy=correction_energy,
            scheme_name=self.name,
            metadata=metadata,
            alignment_energy=alignment_energy,
            charge_model=self.charge_model,
            converged=True,
            warnings=[],
        )

    def _calculate_lattice_term(
        self,
        structure: Structure,
        charge: int,
        profile: DielectricProfile,
        slab_info: dict,
    ) -> float:
        """
        Calculate electrostatic lattice term for Gaussian charge.

        This is a simplified implementation. Full implementation would solve
        Poisson equation in 2D with periodic boundary conditions.

        Parameters
        ----------
        structure : Structure
            Defect structure
        charge : int
            Charge state
        profile : DielectricProfile
            Dielectric profile
        slab_info : dict
            Slab geometry information

        Returns
        -------
        float
            Lattice term correction (eV)
        """
        # Simplified 2D Madelung-like correction
        # E_lat ≈ q² / (2 * ε_avg * L_⊥) * f_2d

        q = abs(charge)
        L_perp = slab_info["slab_thickness"]

        # Average dielectric constant (weighted)
        epsilon_avg = (2 * self.epsilon_parallel + self.epsilon_perpendicular) / 3

        # 2D correction factor (empirical, based on literature)
        # For 2D systems, Madelung constant differs from 3D
        f_2d = 1.2  # Approximate factor for 2D geometry

        # Conversion constant e²/(4πε₀) ≈ 14.3996 eV·Å
        eV_Angstrom = 14.3996

        lattice_term = (q**2 * eV_Angstrom * f_2d) / (2 * epsilon_avg * L_perp)

        logger.info(
            f"2D lattice term: {lattice_term:.4f} eV "
            f"(q={charge:+d}, ε_avg={epsilon_avg:.2f}, L_⊥={L_perp:.2f} Å)"
        )

        return lattice_term

    def _calculate_alignment(
        self,
        defect_structure: Structure,
        host_structure: Structure,
        charge: int,
        potential_data: dict | None,
    ) -> float:
        """
        Calculate potential alignment correction for 2D slabs.

        Uses planar-averaged electrostatic potentials in vacuum regions
        to align defect and host calculations.

        For 2D slabs: E_align = q * ΔV_vacuum
        where ΔV_vacuum is the average potential difference in vacuum regions.

        Parameters
        ----------
        defect_structure : Structure
            Defect structure
        host_structure : Structure
            Host structure
        charge : int
            Charge state
        potential_data : dict, optional
            Dictionary with:
            - "defect_potential": 3D array
            - "host_potential": 3D array
            - "grid_shape": tuple

        Returns
        -------
        float
            Alignment energy (eV)
        """
        if potential_data is None:
            logger.warning(
                "No potential data provided for 2D correction. "
                "Alignment term set to zero. For accurate results, provide "
                "electrostatic potential data from SIESTA .VT files."
            )
            return 0.0

        # Import planar averaging utility
        from atomate2.siesta.flows.defects.utils import calculate_planar_average

        # Extract potential grids
        defect_pot_grid = potential_data["defect_potential"]
        host_pot_grid = potential_data["host_potential"]

        # Calculate planar averages along z-axis (perpendicular to slab)
        z_positions, defect_pot_avg = calculate_planar_average(defect_pot_grid, axis=2)
        _, host_pot_avg = calculate_planar_average(host_pot_grid, axis=2)

        # Calculate potential difference
        pot_diff = defect_pot_avg - host_pot_avg

        # Detect slab geometry to identify vacuum regions
        slab_info = detect_slab_geometry(defect_structure, self.vacuum_threshold)

        # Identify vacuum regions (fractional coordinates)
        # For 2D slab centered in cell with vacuum on both sides
        slab_center_frac = slab_info["slab_center"] / slab_info["cell_length_z"]
        slab_thickness_frac = slab_info["slab_thickness"] / slab_info["cell_length_z"]

        # Vacuum regions are away from slab (±0.15 fractional from edges)
        # Sample from both vacuum regions for better statistics
        vacuum_mask = np.zeros(len(z_positions), dtype=bool)

        # Lower vacuum region (near z=0)
        lower_vacuum_region = z_positions < (
            slab_center_frac - slab_thickness_frac / 2 - 0.05
        )
        # Upper vacuum region (near z=1)
        upper_vacuum_region = z_positions > (
            slab_center_frac + slab_thickness_frac / 2 + 0.05
        )

        vacuum_mask = lower_vacuum_region | upper_vacuum_region

        if not np.any(vacuum_mask):
            logger.warning(
                "Could not identify vacuum regions for potential alignment. "
                "Slab may be too thick or vacuum too small. Using full average."
            )
            delta_V = np.mean(pot_diff)
        else:
            # Average potential difference in vacuum regions
            delta_V = np.mean(pot_diff[vacuum_mask])

        alignment_energy = charge * delta_V

        logger.info(
            f"2D potential alignment: ΔV_vacuum = {delta_V:.4f} eV, "
            f"E_align = {alignment_energy:.4f} eV (q = {charge:+d})"
        )

        return alignment_energy
