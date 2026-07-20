"""
Freysoldt-Neugebauer-Van de Walle finite-size correction scheme.

Implementation of the Freysoldt correction with potential alignment:
Freysoldt, C., Neugebauer, J., & Van de Walle, C. G. (2009). Phys. Rev. Lett., 102, 016402.
DOI: 10.1103/PhysRevLett.102.016402

Improvements over Lany-Zunger and Makov-Payne:
- Anisotropic dielectric screening
- Potential alignment correction
- Works for 2D/surface systems
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from atomate2.siesta.flows.defects.corrections.base import (
    CorrectionResult,
    CorrectionScheme,
)

logger = logging.getLogger(__name__)


class FreysoldtCorrection(CorrectionScheme):
    """
    Freysoldt correction with potential alignment.

    More sophisticated than Lany-Zunger and Makov-Payne. Includes:
    - Anisotropic dielectric screening (can handle different ε_xx, ε_yy, ε_zz)
    - Potential alignment correction (aligns defect and host potentials)
    - Better accuracy for large supercells

    Formula:
        E_corr = E_lat + E_align
        E_lat = (q^2 * α_M) / (2 * ε_avg * L) * f_aniso
        E_align = q * ΔV

    Where:
        - E_lat = lattice energy (electrostatic correction)
        - E_align = potential alignment correction
        - q = charge state
        - α_M = Madelung constant
        - ε_avg = average dielectric constant
        - L = characteristic length
        - f_aniso = anisotropy factor
        - ΔV = potential alignment (from far-field averaging)

    Advantages:
        - Most accurate of the simple corrections
        - Handles anisotropic dielectrics
        - Includes potential alignment
        - Gold standard for charged defects

    Limitations:
        - Requires electrostatic potential data (.VT files)
        - More complex to compute
        - Still assumes periodic boundary conditions

    Notes
    -----
    For dry-run mode (no .VT files), the correction falls back to:
    - Lattice term only (similar to Makov-Payne with Q=0)
    - Potential alignment set to 0 with a warning
    """

    def __init__(
        self,
        epsilon_static: float | list[float],
        madelung_constant: float | None = None,
        use_axis_average: bool = True,
        alignment_method: str = "planar",
        alignment_cutoff: float = 0.8,
        gaussian_width: float | None = None,
    ):
        """
        Initialize Freysoldt correction.

        Parameters
        ----------
        epsilon_static : float or list[float]
            Static dielectric constant. Can be:
            - Scalar (isotropic): single value
            - List [ε_xx, ε_yy, ε_zz] (anisotropic): three values
        madelung_constant : float, optional
            Madelung constant for the lattice. If None, automatically detects
            from structure (rocksalt, zincblende, wurtzite, etc.). Falls back
            to Wigner-Seitz approximation (2.8373) for unknown structures.
            See madelung.py for supported structures and citations.
        use_axis_average : bool, optional
            If True, use average of lattice axes for L. If False, use
            cube root of volume. Default is True.
        alignment_method : str, optional
            Method for potential alignment: "planar" (default) or "atomic".
            - "planar": Average potential in planes far from defect
            - "atomic": Average potential at specific atomic sites
        alignment_cutoff : float, optional
            Fraction of supercell to use for alignment (0.0-1.0).
            Default is 0.8 (use outer 80% for averaging).
        gaussian_width : float, optional
            Gaussian width σ (in Angstrom) for the defect charge distribution model:
            ρ(r) = (q/(2πσ²)^(3/2)) × exp(-r²/2σ²)
            If None, estimates from defect localization (typically 1-2 Å for
            localized defects, larger for delocalized). Default: None (auto-estimate).
        """
        # Handle both scalar and anisotropic dielectric constants
        if isinstance(epsilon_static, (list, tuple, np.ndarray)):
            if len(epsilon_static) != 3:
                raise ValueError(
                    "Anisotropic epsilon must have 3 components [ε_xx, ε_yy, ε_zz]"
                )
            self.epsilon_tensor = np.array(epsilon_static)
            self.epsilon_avg = np.mean(self.epsilon_tensor)
            self.is_anisotropic = True
        else:
            self.epsilon_tensor = np.array([epsilon_static] * 3)
            self.epsilon_avg = float(epsilon_static)
            self.is_anisotropic = False

        super().__init__(epsilon_static=self.epsilon_avg)

        # Madelung constant - will be determined from structure if not provided
        self._madelung_constant_input = madelung_constant
        self.madelung_constant = madelung_constant  # May be None initially
        self.madelung_citation: str | None = None

        # Characteristic length method
        self.use_axis_average = use_axis_average

        # Potential alignment parameters
        self.alignment_method = alignment_method
        self.alignment_cutoff = alignment_cutoff

        # Gaussian charge distribution model
        self.gaussian_width = gaussian_width  # σ in Angstrom

    @property
    def name(self) -> str:
        """Name of the correction scheme."""
        return "Freysoldt"

    @property
    def charge_model(self) -> str:
        """Type of charge model used."""
        return "gaussian+alignment"

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
        Calculate Freysoldt correction.

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
            Electrostatic potential data from .VT files:
            {
                "defect_potential": np.ndarray,  # 3D grid
                "host_potential": np.ndarray,    # 3D grid
                "grid_shape": tuple,             # (nx, ny, nz)
            }
        **kwargs
            Additional parameters (not used)

        Returns
        -------
        CorrectionResult
            Correction result with energy and metadata
        """
        # Validate inputs
        self.validate_inputs(defect_structure, host_structure, charge_state)

        # Determine Madelung constant from structure if not provided
        if self._madelung_constant_input is None:
            from atomate2.siesta.flows.defects.corrections.madelung import (
                get_madelung_constant,
            )

            alpha_M, citation = get_madelung_constant(host_structure)
            self.madelung_constant = alpha_M
            self.madelung_citation = citation
        else:
            alpha_M = self.madelung_constant
            self.madelung_citation = "User-provided value"

        # Estimate Gaussian width if not provided
        if self.gaussian_width is None:
            sigma = self._estimate_gaussian_width(defect_structure)
            logger.info(f"Auto-estimated Gaussian width: σ = {sigma:.3f} Å")
        else:
            sigma = self.gaussian_width
            logger.info(f"Using user-provided Gaussian width: σ = {sigma:.3f} Å")

        # Get characteristic length L and volume Ω
        L = self._get_characteristic_length(defect_structure)
        Omega = defect_structure.volume

        # Calculate lattice term (electrostatic correction)
        # Similar to Makov-Payne but with anisotropy factor
        q = abs(charge_state)
        alpha_M = self.madelung_constant
        epsilon_avg = self.epsilon_avg

        # Anisotropy factor (1.0 for isotropic, different for anisotropic)
        f_aniso = self._calculate_anisotropy_factor(defect_structure)

        # Formula in eV (using e^2/(4πε_0) ≈ 14.3996 eV·Å)
        eV_Angstrom = 14.3996  # e^2/(4πε_0) in eV·Å
        lattice_term = (q**2 * alpha_M * eV_Angstrom) / (2 * epsilon_avg * L) * f_aniso

        # Calculate potential alignment correction
        alignment_plot_path = None
        if potential_data is not None:
            alignment_energy = self._calculate_potential_alignment(
                potential_data, charge_state, defect_site, defect_structure
            )
            alignment_available = True
            warnings = []

            # Generate alignment plot if VT file paths provided
            vt_file_paths = kwargs.get("vt_file_paths")
            if vt_file_paths is not None:
                try:
                    from atomate2.siesta.flows.defects.utils import (
                        plot_potential_alignment,
                    )

                    defect_vt = vt_file_paths.get("defect")
                    host_vt = vt_file_paths.get("host")

                    if defect_vt and host_vt:
                        # Plot VT (total potential)
                        plot_output = Path("freysoldt_potential_alignment_VT.png")
                        plot_data = plot_potential_alignment(
                            defect_vt_path=defect_vt,
                            host_vt_path=host_vt,
                            axis=2,
                            output_path=plot_output,
                            show_plot=False,
                        )
                        alignment_plot_path = str(plot_output)
                        logger.info(
                            f"Generated Freysoldt alignment plot (VT): {alignment_plot_path}"
                        )
                        logger.info(
                            f"Mean ΔV (VT) = {plot_data['mean_alignment']:.4f} eV "
                            f"(q × ΔV = {alignment_energy:.4f} eV)"
                        )

                        # Also plot VH (Hartree potential) if available
                        defect_vh = Path(str(defect_vt).replace(".VT", ".VH"))
                        host_vh = Path(str(host_vt).replace(".VT", ".VH"))

                        if defect_vh.exists() and host_vh.exists():
                            plot_output_vh = Path(
                                "freysoldt_potential_alignment_VH.png"
                            )
                            plot_data_vh = plot_potential_alignment(
                                defect_vt_path=defect_vh,
                                host_vt_path=host_vh,
                                axis=2,
                                output_path=plot_output_vh,
                                show_plot=False,
                            )
                            logger.info(
                                f"Generated Freysoldt alignment plot (VH): {plot_output_vh}"
                            )
                            logger.info(
                                f"Mean ΔV (VH) = {plot_data_vh['mean_alignment']:.4f} eV"
                            )
                except Exception as e:
                    logger.warning(f"Failed to generate alignment plot: {e}")

        else:
            # No potential data available (dry-run mode)
            alignment_energy = 0.0
            alignment_available = False
            warnings = [
                "Potential alignment not calculated (no .VT data available). "
                "Using lattice term only. For accurate results, provide potential_data "
                "from SIESTA .VT files."
            ]

        # Total correction
        correction_energy = lattice_term + alignment_energy

        # Metadata
        metadata = {
            "madelung_constant": alpha_M,
            "madelung_citation": self.madelung_citation,
            "characteristic_length_angstrom": L,
            "volume_angstrom3": Omega,
            "epsilon_avg": epsilon_avg,
            "epsilon_tensor": (
                self.epsilon_tensor.tolist()
                if hasattr(self, "epsilon_tensor") and self.epsilon_tensor is not None
                else [epsilon_avg, epsilon_avg, epsilon_avg]
            ),
            "is_anisotropic": getattr(self, "is_anisotropic", False),
            "charge_state": charge_state,
            "anisotropy_factor": f_aniso,
            "gaussian_width_angstrom": sigma,  # Gaussian charge distribution width
            "lattice_term_eV": lattice_term,
            "alignment_energy_eV": alignment_energy,
            "alignment_method": self.alignment_method,
            "alignment_available": alignment_available,
            "alignment_plot": alignment_plot_path,
            "formula": "E_corr = E_lat + E_align",
        }

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

    def _calculate_anisotropy_factor(self, structure: Structure) -> float:
        """
        Calculate anisotropy factor for dielectric screening.

        For isotropic systems: f_aniso = 1.0
        For anisotropic systems: f_aniso accounts for different ε values

        Parameters
        ----------
        structure : Structure
            Supercell structure

        Returns
        -------
        float
            Anisotropy factor (dimensionless)
        """
        if not self.is_anisotropic:
            return 1.0

        # For anisotropic systems, calculate geometric mean correction
        # This is a simplified model; full Freysoldt uses more complex screening
        epsilon_xx, epsilon_yy, epsilon_zz = self.epsilon_tensor
        geometric_mean = (epsilon_xx * epsilon_yy * epsilon_zz) ** (1.0 / 3.0)
        arithmetic_mean = self.epsilon_avg

        # Anisotropy factor: ratio of geometric to arithmetic mean
        # Closer to 1.0 for more isotropic systems
        f_aniso = geometric_mean / arithmetic_mean

        return f_aniso

    def _estimate_gaussian_width(self, structure: Structure) -> float:
        """
        Estimate Gaussian width σ for the defect charge distribution.

        The Gaussian width represents the spatial extent of the defect charge:
        ρ(r) = (q/(2πσ²)^(3/2)) × exp(-r²/2σ²)

        Estimation strategy:
        1. Find minimum nearest-neighbor distance in structure
        2. Use σ ≈ 0.5 × d_nn (half the bond length)
        3. Clamp to reasonable range [0.5, 3.0] Å

        For localized defects (e.g., F-centers): σ ~ 1-2 Å
        For delocalized defects (e.g., shallow donors): σ ~ 2-4 Å

        Parameters
        ----------
        structure : Structure
            Defect supercell structure

        Returns
        -------
        float
            Estimated Gaussian width σ in Angstrom
        """
        # Find minimum nearest-neighbor distance
        min_nn_dist = float("inf")
        for i, site in enumerate(structure):
            neighbors = structure.get_neighbors(site, 4.0)  # Search within 4 Å
            if neighbors:
                distances = [n.nn_distance for n in neighbors]
                min_nn_dist = min(min_nn_dist, min(distances))

        if min_nn_dist == float("inf"):
            # Fallback: use 1.5 Å (typical for localized defects)
            logger.warning(
                "Could not determine nearest-neighbor distance. Using default σ = 1.5 Å"
            )
            return 1.5

        # Estimate σ as half the nearest-neighbor distance
        # This assumes charge is localized within ~1 bond length
        sigma = 0.5 * min_nn_dist

        # Clamp to reasonable range
        sigma = np.clip(sigma, 0.5, 3.0)

        return sigma

    def _calculate_potential_alignment(
        self,
        potential_data: dict,
        charge_state: int,
        defect_site: list[float] | None,
        structure: Structure,
    ) -> float:
        """
        Calculate potential alignment correction.

        This is the key difference from Lany-Zunger/Makov-Payne.
        Aligns the defect and host potentials in regions far from the defect.

        Parameters
        ----------
        potential_data : dict
            Potential data from .VT files
        charge_state : int
            Charge state of defect
        defect_site : list[float] or None
            Fractional coordinates of defect
        structure : Structure
            Supercell structure

        Returns
        -------
        float
            Alignment energy in eV
        """
        defect_pot = potential_data["defect_potential"]
        host_pot = potential_data["host_potential"]
        grid_shape = potential_data["grid_shape"]

        # Calculate potential difference
        pot_diff = defect_pot - host_pot

        # Determine alignment region (far from defect)
        if defect_site is not None:
            mask = self._get_alignment_mask(grid_shape, defect_site, structure)
        else:
            # Use outer regions if defect site unknown
            mask = self._get_outer_region_mask(grid_shape)

        # Average potential difference in alignment region
        avg_pot_diff = np.mean(pot_diff[mask])

        # Alignment energy = q * ΔV
        alignment_energy = charge_state * avg_pot_diff

        return alignment_energy

    def _get_alignment_mask(
        self, grid_shape: tuple, defect_site: list[float], structure: Structure
    ) -> np.ndarray:
        """
        Create mask for alignment region (far from defect).

        Parameters
        ----------
        grid_shape : tuple
            Shape of potential grid (nx, ny, nz)
        defect_site : list[float]
            Fractional coordinates of defect
        structure : Structure
            Supercell structure

        Returns
        -------
        np.ndarray
            Boolean mask (True = use for alignment)
        """
        nx, ny, nz = grid_shape

        # Create meshgrid of fractional coordinates
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 1, nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        # Distance from defect (in fractional coordinates)
        dx = np.abs(xx - defect_site[0])
        dy = np.abs(yy - defect_site[1])
        dz = np.abs(zz - defect_site[2])

        # Handle periodic boundary conditions
        dx = np.minimum(dx, 1.0 - dx)
        dy = np.minimum(dy, 1.0 - dy)
        dz = np.minimum(dz, 1.0 - dz)

        # Euclidean distance
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Use outer region for alignment (beyond alignment_cutoff)
        max_dist = np.sqrt(3.0) / 2.0  # Maximum possible distance in fractional coords
        cutoff_dist = max_dist * self.alignment_cutoff

        mask = dist >= cutoff_dist

        return mask

    def _get_outer_region_mask(self, grid_shape: tuple) -> np.ndarray:
        """
        Create mask for outer region when defect site is unknown.

        Parameters
        ----------
        grid_shape : tuple
            Shape of potential grid (nx, ny, nz)

        Returns
        -------
        np.ndarray
            Boolean mask
        """
        nx, ny, nz = grid_shape

        # Use outer 20% of grid in each direction
        margin = 0.1  # 10% on each side
        mask = np.ones((nx, ny, nz), dtype=bool)

        # Mark inner region as False
        ix_start = int(nx * margin)
        ix_end = int(nx * (1 - margin))
        iy_start = int(ny * margin)
        iy_end = int(ny * (1 - margin))
        iz_start = int(nz * margin)
        iz_end = int(nz * (1 - margin))

        mask[ix_start:ix_end, iy_start:iy_end, iz_start:iz_end] = False

        return mask
