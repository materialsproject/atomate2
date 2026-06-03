"""
Makov-Payne finite-size correction scheme.

Implementation of the Makov-Payne correction with quadrupole term:
Makov, G., & Payne, M. C. (1995). Phys. Rev. B, 51, 4014.
DOI: 10.1103/PhysRevB.51.4014
"""

from __future__ import annotations

import logging

import numpy as np
from pymatgen.core import Structure

from atomate2.siesta.flows.defects.corrections.base import (
    CorrectionResult,
    CorrectionScheme,
)

logger = logging.getLogger(__name__)


class MakovPayneCorrection(CorrectionScheme):
    """
    Makov-Payne correction with quadrupole term.

    More sophisticated than Lany-Zunger, includes both monopole and
    quadrupole contributions to the finite-size error.

    Formula:
        E_corr = E_monopole + E_quadrupole
        E_monopole = (q^2 * α_M) / (2 * ε * L)
        E_quadrupole = (2π * q * Q) / (3 * ε * Ω)

    Where:
        - q = charge state
        - α_M = Madelung constant (~2.8373 for cubic)
        - ε = static dielectric constant
        - L = linear dimension of supercell
        - Q = quadrupole moment (estimated or from charge density)
        - Ω = volume of supercell

    Advantages:
        - More accurate than simple Lany-Zunger
        - Includes quadrupole effects
        - Good for validation and comparison

    Limitations:
        - Still assumes isotropic dielectric
        - Quadrupole moment needs to be estimated or calculated
        - No potential alignment
    """

    def __init__(
        self,
        epsilon_static: float,
        madelung_constant: float | None = None,
        quadrupole_moment: float | None = None,
        use_axis_average: bool = True,
    ):
        """
        Initialize Makov-Payne correction.

        Parameters
        ----------
        epsilon_static : float
            Static dielectric constant (scalar, isotropic)
        madelung_constant : float, optional
            Madelung constant for the lattice. If None, uses 2.8373 (cubic).
        quadrupole_moment : float, optional
            Quadrupole moment Q in eÅ². If None, will be estimated from
            defect size (conservative estimate: Q ≈ 0).
        use_axis_average : bool, optional
            If True, use average of lattice axes for L. If False, use
            cube root of volume. Default is True.
        """
        super().__init__(epsilon_static=epsilon_static)

        # Madelung constant (default for cubic lattice)
        self.madelung_constant = madelung_constant or 2.8373

        # Quadrupole moment (if None, will be set to 0 as conservative estimate)
        self.quadrupole_moment = quadrupole_moment

        # How to compute characteristic length
        self.use_axis_average = use_axis_average

    @property
    def name(self) -> str:
        """Name of the correction scheme."""
        return "Makov-Payne"

    @property
    def charge_model(self) -> str:
        """Type of charge model used."""
        return "point+quadrupole"

    @property
    def requires_dielectric(self) -> bool:
        """Whether the scheme requires dielectric constant."""
        return True

    @property
    def supports_anisotropic(self) -> bool:
        """Whether the scheme supports anisotropic dielectrics."""
        return False

    @property
    def requires_potential_data(self) -> bool:
        """Whether the scheme requires electrostatic potential data."""
        return False

    def calculate_correction(
        self,
        defect_structure: Structure,
        host_structure: Structure,
        charge_state: int,
        defect_energy: float,
        host_energy: float,
        defect_site: list[float] | None = None,
        quadrupole_moment: float | None = None,
        density_data: dict | None = None,
        **kwargs,
    ) -> CorrectionResult:
        """
        Calculate Makov-Payne correction.

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
            Fractional coordinates [x, y, z] of defect site.
            Used for quadrupole calculation from density.
        quadrupole_moment : float, optional
            Quadrupole moment Q in eÅ². If provided, overrides the
            value set during initialization and density calculation.
        density_data : dict, optional
            Dictionary with charge density grids:
            - "defect_density": 3D array (defect calculation)
            - "host_density": 3D array (host calculation)
            - "grid_shape": tuple
            If provided along with defect_site, will calculate
            quadrupole moment from density difference.
        **kwargs
            Additional parameters (not used)

        Returns
        -------
        CorrectionResult
            Correction result with energy and metadata
        """
        # Validate inputs
        self.validate_inputs(defect_structure, host_structure, charge_state)

        # Get quadrupole moment with priority:
        # 1. Explicit parameter quadrupole_moment
        # 2. Calculate from density_data (if available)
        # 3. Use instance value self.quadrupole_moment
        # 4. Default to 0.0

        Q_calculated_from_density = False

        if quadrupole_moment is not None:
            # Explicit parameter takes highest priority
            Q = quadrupole_moment
        elif density_data is not None:
            # Calculate from density if available
            try:
                Q = self.calculate_quadrupole_from_density(
                    defect_density=density_data["defect_density"],
                    host_density=density_data["host_density"],
                    cell=defect_structure.lattice.matrix,
                    defect_site_frac=defect_site,
                )
                Q_calculated_from_density = True
                logger.info(f"Using quadrupole moment from density: Q = {Q:.4f} eÅ²")
            except Exception as e:
                logger.warning(
                    f"Failed to calculate quadrupole from density: {e}. "
                    f"Using fallback value."
                )
                Q = (
                    self.quadrupole_moment
                    if self.quadrupole_moment is not None
                    else 0.0
                )
        else:
            # Use instance value or default
            Q = self.quadrupole_moment if self.quadrupole_moment is not None else 0.0

        # Get characteristic length L and volume Ω
        L = self._get_characteristic_length(defect_structure)
        Omega = defect_structure.volume

        # Calculate monopole term (same as Lany-Zunger)
        # E_monopole = (q^2 * α_M) / (2 * ε * L)
        q = abs(charge_state)
        alpha_M = self.madelung_constant
        epsilon = self.epsilon_static

        # Formula in eV (using e^2/(4πε_0) ≈ 14.3996 eV·Å)
        eV_Angstrom = 14.3996  # e^2/(4πε_0) in eV·Å
        monopole_term = (q**2 * alpha_M * eV_Angstrom) / (2 * epsilon * L)

        # Calculate quadrupole term
        # E_quadrupole = (2π * q * Q) / (3 * ε * Ω)
        # Note: Q is in units of eÅ², so we need eV·Å / Å³ = eV/Å²
        # Using e/(4πε_0) ≈ 14.3996 eV·Å
        import math

        quadrupole_term = (2 * math.pi * q * Q * eV_Angstrom) / (3 * epsilon * Omega)

        # Total correction
        correction_energy = monopole_term + quadrupole_term

        # Metadata
        metadata = {
            "madelung_constant": alpha_M,
            "characteristic_length_angstrom": L,
            "volume_angstrom3": Omega,
            "epsilon_static": epsilon,
            "charge_state": charge_state,
            "quadrupole_moment_eA2": Q,
            "quadrupole_calculated_from_density": Q_calculated_from_density,
            "monopole_term_eV": monopole_term,
            "quadrupole_term_eV": quadrupole_term,
            "use_axis_average": self.use_axis_average,
            "formula": "E_corr = (q^2*α)/(2*ε*L) + (2π*q*Q)/(3*ε*Ω)",
        }

        # Add warnings
        warnings = []
        if Q == 0.0:
            if not Q_calculated_from_density:
                warnings.append(
                    "Quadrupole moment Q=0 assumed (conservative estimate). "
                    "For accurate corrections, provide density_data from SIESTA .RHO files "
                    "to calculate Q from charge density."
                )
            else:
                warnings.append(
                    "Quadrupole moment calculated from density is Q=0. "
                    "This may indicate point-like defect or numerical cancellation."
                )

        return CorrectionResult(
            correction_energy=correction_energy,
            scheme_name=self.name,
            metadata=metadata,
            charge_model=self.charge_model,
            converged=True,
            warnings=warnings,
        )

    def _get_characteristic_length(self, structure: Structure) -> float:
        """
        Get characteristic length L of the supercell.

        Parameters
        ----------
        structure : Structure
            Supercell structure

        Returns
        -------
        float
            Characteristic length in Angstroms
        """
        if self.use_axis_average:
            # Use average of lattice parameters
            a, b, c = structure.lattice.abc
            L = (a + b + c) / 3.0
        else:
            # Use cube root of volume
            L = structure.volume ** (1.0 / 3.0)

        return L

    @staticmethod
    def calculate_quadrupole_from_density(
        defect_density: np.ndarray,
        host_density: np.ndarray,
        cell: np.ndarray,
        defect_site_frac: list[float] | None = None,
    ) -> float:
        """
        Calculate quadrupole moment from charge density difference.

        Uses the traceless quadrupole tensor component Q_zz:
        Q_zz = ∫ Δρ(r) * (3z² - r²) dr

        where Δρ(r) = ρ_defect(r) - ρ_host(r)

        Parameters
        ----------
        defect_density : np.ndarray
            3D charge density grid for defect calculation (electrons/Ų)
        host_density : np.ndarray
            3D charge density grid for host calculation (electrons/Ų)
        cell : np.ndarray
            3x3 lattice matrix (Angstroms)
        defect_site_frac : list[float], optional
            Fractional coordinates [x, y, z] of defect site.
            If None, assumes center of cell.

        Returns
        -------
        float
            Quadrupole moment Q in eÅ² (elementary charge × Angstrom²)

        Notes
        -----
        The quadrupole moment measures the spatial extent of the charge
        perturbation beyond the monopole term. It's important for
        accurate finite-size corrections, especially for extended defects.

        For a point charge, Q=0. For extended charge distributions,
        Q can be significant (typically 0-100 eÅ² for defects).
        """
        # Calculate density difference
        delta_rho = defect_density - host_density

        # Get grid shape
        nx, ny, nz = delta_rho.shape

        # Get cell volume
        volume = abs(np.linalg.det(cell))

        # Volume element
        dV = volume / (nx * ny * nz)

        # Set defect site (default to cell center)
        if defect_site_frac is None:
            defect_site_frac = [0.5, 0.5, 0.5]

        # Create grid of fractional coordinates relative to defect site
        x_frac = np.linspace(0, 1, nx, endpoint=False) - defect_site_frac[0]
        y_frac = np.linspace(0, 1, ny, endpoint=False) - defect_site_frac[1]
        z_frac = np.linspace(0, 1, nz, endpoint=False) - defect_site_frac[2]

        # Apply periodic boundary conditions (wrap to [-0.5, 0.5])
        x_frac = (x_frac + 0.5) % 1.0 - 0.5
        y_frac = (y_frac + 0.5) % 1.0 - 0.5
        z_frac = (z_frac + 0.5) % 1.0 - 0.5

        # Create 3D grids
        X_frac, Y_frac, Z_frac = np.meshgrid(x_frac, y_frac, z_frac, indexing="ij")

        # Convert to Cartesian coordinates (Angstroms)
        # r_cart = cell @ r_frac
        X_cart = cell[0, 0] * X_frac + cell[0, 1] * Y_frac + cell[0, 2] * Z_frac
        Y_cart = cell[1, 0] * X_frac + cell[1, 1] * Y_frac + cell[1, 2] * Z_frac
        Z_cart = cell[2, 0] * X_frac + cell[2, 1] * Y_frac + cell[2, 2] * Z_frac

        # Calculate r² = x² + y² + z²
        r_squared = X_cart**2 + Y_cart**2 + Z_cart**2

        # Calculate Q_zz = ∫ Δρ(r) * (3z² - r²) dr
        integrand = delta_rho * (3 * Z_cart**2 - r_squared)

        # Integrate (sum over grid and multiply by volume element)
        Q_zz = np.sum(integrand) * dV

        # For cubic/isotropic systems, use Q = |Q_zz|
        # (The sign depends on orientation; magnitude matters for correction)
        Q = abs(Q_zz)

        logger.info(
            f"Calculated quadrupole moment from density: Q = {Q:.4f} eÅ² "
            f"(Q_zz = {Q_zz:+.4f} eÅ²)"
        )

        return Q
