"""
Lany-Zunger finite-size correction scheme.

Implementation of the simple isotropic point-charge correction from:
Lany, S., & Zunger, A. (2008). Phys. Rev. B, 78, 235104.
DOI: 10.1103/PhysRevB.78.235104
"""

from __future__ import annotations

from pymatgen.core import Structure

from atomate2.siesta.flows.defects.corrections.base import (
    CorrectionResult,
    CorrectionScheme,
)


class LanyZungerCorrection(CorrectionScheme):
    """
    Lany-Zunger isotropic point-charge correction.

    Simple analytical correction based on a point charge in a periodic
    supercell. Uses Madelung constant and assumes isotropic dielectric.

    Formula:
        E_corr = (q^2 * α_M) / (2 * ε * L)

    Where:
        - q = charge state
        - α_M = Madelung constant (~2.8373 for cubic)
        - ε = static dielectric constant
        - L = linear dimension of supercell

    Advantages:
        - Very simple and fast
        - No potential data required
        - Good for initial estimates

    Limitations:
        - Assumes isotropic dielectric
        - Point charge model (less accurate than gaussian)
        - No potential alignment
    """

    def __init__(
        self,
        epsilon_static: float,
        madelung_constant: float | None = None,
        use_axis_average: bool = True,
    ):
        """
        Initialize Lany-Zunger correction.

        Parameters
        ----------
        epsilon_static : float
            Static dielectric constant (scalar, isotropic)
        madelung_constant : float, optional
            Madelung constant for the lattice. If None, uses 2.8373 (cubic).
        use_axis_average : bool, optional
            If True, use average of lattice axes for L. If False, use
            cube root of volume. Default is True.
        """
        super().__init__(epsilon_static=epsilon_static)

        # Madelung constant (default for cubic lattice)
        self.madelung_constant = madelung_constant or 2.8373

        # How to compute characteristic length
        self.use_axis_average = use_axis_average

    @property
    def name(self) -> str:
        """Name of the correction scheme."""
        return "Lany-Zunger"

    @property
    def charge_model(self) -> str:
        """Type of charge model used."""
        return "point"

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
        **kwargs,
    ) -> CorrectionResult:
        """
        Calculate Lany-Zunger correction.

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
            Fractional coordinates [x, y, z] of defect site (not used)
        **kwargs
            Additional parameters (not used)

        Returns
        -------
        CorrectionResult
            Correction result with energy and metadata
        """
        # Validate inputs
        self.validate_inputs(defect_structure, host_structure, charge_state)

        # Get characteristic length L
        L = self._get_characteristic_length(defect_structure)

        # Calculate correction energy
        # E_corr = (q^2 * α_M) / (2 * ε * L)
        # Note: Convert to eV using eV*Angstrom
        q = abs(charge_state)
        alpha_M = self.madelung_constant
        epsilon = self.epsilon_static

        # Formula in eV (using e^2/(4πε_0) ≈ 14.3996 eV·Å)
        eV_Angstrom = 14.3996  # e^2/(4πε_0) in eV·Å
        correction_energy = (q**2 * alpha_M * eV_Angstrom) / (2 * epsilon * L)

        # Metadata
        metadata = {
            "madelung_constant": alpha_M,
            "characteristic_length_angstrom": L,
            "epsilon_static": epsilon,
            "charge_state": charge_state,
            "use_axis_average": self.use_axis_average,
            "volume_angstrom3": defect_structure.volume,
            "formula": "E_corr = (q^2 * α_M) / (2 * ε * L)",
        }

        return CorrectionResult(
            correction_energy=correction_energy,
            scheme_name=self.name,
            metadata=metadata,
            charge_model=self.charge_model,
            converged=True,
            warnings=[],
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

    def estimate_convergence(
        self,
        structure: Structure,
        charge_state: int,
        target_accuracy_eV: float = 0.05,
    ) -> dict[str, float]:
        """
        Estimate supercell size needed for target accuracy.

        Parameters
        ----------
        structure : Structure
            Current supercell structure
        charge_state : int
            Charge state of defect
        target_accuracy_eV : float, optional
            Target accuracy in eV. Default is 0.05 eV.

        Returns
        -------
        dict
            Dictionary with current correction and recommended size
        """
        # Current correction
        L_current = self._get_characteristic_length(structure)
        current_corr = (
            abs(charge_state) ** 2
            * self.madelung_constant
            * 14.3996
            / (2 * self.epsilon_static * L_current)
        )

        # Required L for target accuracy
        # If we want correction < target_accuracy_eV:
        # L > (q^2 * α_M * 14.3996) / (2 * ε * target)
        L_required = (
            abs(charge_state) ** 2
            * self.madelung_constant
            * 14.3996
            / (2 * self.epsilon_static * target_accuracy_eV)
        )

        # Scale factor
        scale_factor = L_required / L_current

        return {
            "current_correction_eV": current_corr,
            "current_length_angstrom": L_current,
            "required_length_angstrom": L_required,
            "scale_factor": scale_factor,
            "is_converged": current_corr < target_accuracy_eV,
            "target_accuracy_eV": target_accuracy_eV,
        }
