"""
Kumagai-Oba finite-size correction scheme.

Implementation of the Kumagai-Oba correction with atomic-site potential sampling:
Kumagai, Y., & Oba, F. (2014). Physical Review B, 89, 195205.
DOI: 10.1103/PhysRevB.89.195205

Improvements over Freysoldt:
- Atomic-site sampling instead of planar averaging
- Better accuracy for relaxed systems
- More robust for systems with significant ionic relaxation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from atomate2.siesta.flows.defects.corrections.base import (
    CorrectionResult,
    CorrectionScheme,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class KumagaiCorrection(CorrectionScheme):
    """
    Kumagai-Oba correction with atomic-site potential sampling.

    State-of-the-art correction for relaxed systems. Key innovation:
    - Samples electrostatic potential at atomic sites (not planar-averaged)
    - More accurate for systems with significant ionic relaxation
    - Better convergence with supercell size

    Formula:
        E_corr = E_lat + E_align
        E_lat = (q^2 * α_M) / (2 * ε * L)
        E_align = q * ΔV_atomic

    Where:
        - E_lat = lattice energy (electrostatic correction)
        - E_align = potential alignment from atomic-site sampling
        - q = charge state
        - α_M = Madelung constant
        - ε = dielectric constant
        - L = characteristic length
        - ΔV_atomic = average potential difference at sampling sites

    Advantages:
        - Most accurate for relaxed systems (SOTA)
        - Better than Freysoldt for ionic relaxation
        - Robust atomic-site sampling
        - Automatic outlier detection

    Limitations:
        - Requires electrostatic potential data (.VT files)
        - Needs sufficient sampling sites (>20 atoms)
        - Slightly more complex than simpler schemes

    Notes
    -----
    For dry-run mode (no .VT files), the correction falls back to:
    - Lattice term only (similar to Makov-Payne)
    - Potential alignment set to 0 with a warning

    Reference
    ---------
    Kumagai & Oba, PRB 89, 195205 (2014)
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.195205
    """  # noqa: RUF002

    def __init__(
        self,
        epsilon_static: float,
        madelung_constant: float | None = None,
        sampling_cutoff_fraction: float = 0.8,
        min_sampling_atoms: int = 20,
        outlier_threshold: float = 2.0,
    ) -> None:
        """
        Initialize Kumagai-Oba correction.

        Parameters
        ----------
        epsilon_static : float
            Static dielectric constant (scalar, isotropic)
        madelung_constant : float, optional
            Madelung constant for the lattice. If None, uses 2.8373 (cubic).
        sampling_cutoff_fraction : float, optional
            Fraction of half-supercell length for sampling cutoff.
            Default is 0.8 (atoms beyond 80% of L/2 are used).
        min_sampling_atoms : int, optional
            Minimum number of atoms required for reliable sampling.
            Default is 20. Warning issued if fewer atoms available.
        outlier_threshold : float, optional
            Number of standard deviations for outlier detection.
            Default is 2.0 (exclude atoms with ΔV > 2σ from mean).
        """  # noqa: RUF002
        super().__init__(epsilon_static=epsilon_static)

        # Madelung constant (default for cubic lattice)
        self.madelung_constant = madelung_constant or 2.8373

        # Sampling parameters
        self.sampling_cutoff_fraction = sampling_cutoff_fraction
        self.min_sampling_atoms = min_sampling_atoms
        self.outlier_threshold = outlier_threshold

    @property
    def name(self) -> str:
        """Name of the correction scheme."""
        return "Kumagai-Oba"

    @property
    def charge_model(self) -> str:
        """Type of charge model used."""
        return "point+atomic_sampling"

    @property
    def requires_dielectric(self) -> bool:
        """Whether the scheme requires dielectric constant."""
        return True

    @property
    def supports_anisotropic(self) -> bool:
        """Whether the scheme supports anisotropic dielectrics."""
        return False  # Kumagai-Oba uses isotropic epsilon

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
        Calculate Kumagai-Oba correction.

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
                "grid_coords": np.ndarray,       # Grid coordinates (optional)
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

        # Get characteristic length L
        L = self._get_characteristic_length(defect_structure)  # noqa: N806

        # Calculate lattice term (electrostatic correction)
        q = abs(charge_state)
        alpha_M = self.madelung_constant  # noqa: N806
        epsilon = self.epsilon_static

        # Formula in eV (using e^2/(4πε_0) ≈ 14.3996 eV·Å)
        eV_Angstrom = 14.3996  # e^2/(4πε_0) in eV·Å  # noqa: N806
        lattice_term = (q**2 * alpha_M * eV_Angstrom) / (2 * epsilon * L)

        # Calculate potential alignment correction using atomic-site sampling
        alignment_plot_path = None
        if potential_data is not None and defect_site is not None:
            alignment_result = self._calculate_atomic_site_alignment(
                potential_data,
                charge_state,
                defect_site,
                defect_structure,
                host_structure,
            )
            alignment_energy = alignment_result["alignment_energy"]
            alignment_available = True
            warnings = alignment_result["warnings"]
            alignment_metadata = alignment_result["metadata"]

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
                        plot_output = Path("kumagai_potential_alignment_VT.png")
                        plot_data = plot_potential_alignment(
                            defect_vt_path=defect_vt,
                            host_vt_path=host_vt,
                            axis=2,
                            output_path=plot_output,
                            show_plot=False,
                        )
                        alignment_plot_path = str(plot_output)
                        logger.info(
                            f"Generated Kumagai alignment plot (VT): {alignment_plot_path}"
                        )
                        logger.info(
                            f"Mean ΔV (VT) = {plot_data['mean_alignment']:.4f} eV "
                            f"(q × ΔV = {alignment_energy:.4f} eV)"  # noqa: RUF001
                        )

                        # Also plot VH (Hartree potential) if available
                        defect_vh = Path(str(defect_vt).replace(".VT", ".VH"))
                        host_vh = Path(str(host_vt).replace(".VT", ".VH"))

                        if defect_vh.exists() and host_vh.exists():
                            plot_output_vh = Path("kumagai_potential_alignment_VH.png")
                            plot_data_vh = plot_potential_alignment(
                                defect_vt_path=defect_vh,
                                host_vt_path=host_vh,
                                axis=2,
                                output_path=plot_output_vh,
                                show_plot=False,
                            )
                            logger.info(
                                f"Generated Kumagai alignment plot (VH): {plot_output_vh}"
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
            alignment_metadata = {}
            warnings = [
                "Potential alignment not calculated (no .VT data or defect site). "
                "Using lattice term only. For accurate results, provide potential_data "
                "from SIESTA .VT files and defect_site coordinates."
            ]

        # Total correction
        correction_energy = lattice_term + alignment_energy

        # Metadata
        metadata = {
            "madelung_constant": alpha_M,
            "characteristic_length_angstrom": L,
            "volume_angstrom3": defect_structure.volume,
            "epsilon_static": epsilon,
            "charge_state": charge_state,
            "lattice_term_eV": lattice_term,
            "alignment_energy_eV": alignment_energy,
            "sampling_method": "atomic_site",
            "alignment_available": alignment_available,
            "alignment_plot": alignment_plot_path,
            "formula": "E_corr = E_lat + E_align (atomic-site sampling)",
            **alignment_metadata,
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

        Uses average of lattice parameters (consistent with literature).

        Parameters
        ----------
        structure : Structure
            Supercell structure

        Returns
        -------
        float
            Characteristic length in Angstroms
        """
        a, b, c = structure.lattice.abc
        L = (a + b + c) / 3.0  # noqa: N806
        return L

    def _calculate_atomic_site_alignment(
        self,
        potential_data: dict,
        charge_state: int,
        defect_site: list[float],
        defect_structure: Structure,
        host_structure: Structure,
    ) -> dict:
        """
        Calculate potential alignment using atomic-site sampling.

        This is the key innovation of Kumagai-Oba: sample potential at
        atomic positions (not planar-averaged) to get better accuracy
        for relaxed systems.

        Parameters
        ----------
        potential_data : dict
            Potential data from .VT files
        charge_state : int
            Charge state of defect
        defect_site : list[float]
            Fractional coordinates of defect
        defect_structure : Structure
            Defect structure (relaxed)
        host_structure : Structure
            Host structure (pristine)

        Returns
        -------
        dict
            Dictionary with alignment_energy, warnings, and metadata
        """
        defect_pot = potential_data["defect_potential"]
        host_pot = potential_data["host_potential"]
        grid_shape = potential_data["grid_shape"]

        # Calculate potential difference
        pot_diff = defect_pot - host_pot

        # Get sampling atom indices (far from defect)
        sampling_indices = self._get_sampling_atoms(
            defect_site, defect_structure, host_structure
        )

        # Check if we have enough sampling atoms
        n_sampling = len(sampling_indices)
        warnings = []
        if n_sampling < self.min_sampling_atoms:
            warnings.append(
                f"Only {n_sampling} sampling atoms found (minimum recommended: "
                f"{self.min_sampling_atoms}). Alignment may be less reliable."
            )

        if n_sampling == 0:
            return {
                "alignment_energy": 0.0,
                "warnings": [
                    "No valid sampling atoms found! Cannot calculate alignment. "
                    "Supercell may be too small or defect site incorrect."
                ],
                "metadata": {
                    "n_sampling_atoms": 0,
                    "sampling_indices": [],
                },
            }

        # Sample potential at atomic sites
        potential_samples = []
        sampling_coords = []

        for idx in sampling_indices:
            # Use defect structure coords (relaxed positions)
            frac_coords = defect_structure[idx].frac_coords
            sampling_coords.append(frac_coords)

            # Interpolate potential at this atomic position
            pot_value = self._interpolate_potential_at_site(
                pot_diff, grid_shape, frac_coords
            )
            potential_samples.append(pot_value)

        potential_samples = np.array(potential_samples)
        sampling_coords = np.array(sampling_coords)

        # Remove outliers using robust statistics
        pot_clean, outlier_mask = self._remove_outliers(
            potential_samples, self.outlier_threshold
        )

        n_outliers = np.sum(outlier_mask)
        if n_outliers > 0:
            warnings.append(
                f"Removed {n_outliers} outlier atoms from alignment "
                f"(threshold: {self.outlier_threshold}σ)"  # noqa: RUF001
            )

        # Calculate average potential difference
        avg_pot_diff = np.mean(pot_clean)
        std_pot_diff = np.std(pot_clean)

        # Alignment energy = q * ΔV
        alignment_energy = charge_state * avg_pot_diff

        # Metadata
        metadata = {
            "n_sampling_atoms": n_sampling,
            "n_outliers_removed": n_outliers,
            "avg_potential_diff_eV": avg_pot_diff,
            "std_potential_diff_eV": std_pot_diff,
            "sampling_cutoff_fraction": self.sampling_cutoff_fraction,
            "outlier_threshold_sigma": self.outlier_threshold,
            "sampling_indices": sampling_indices.tolist(),
        }

        return {
            "alignment_energy": alignment_energy,
            "warnings": warnings,
            "metadata": metadata,
        }

    def _get_sampling_atoms(
        self,
        defect_site: list[float],
        defect_structure: Structure,
        host_structure: Structure,
    ) -> np.ndarray:
        """
        Get indices of atoms to use for potential sampling.

        Kumagai-Oba criterion:
        - Distance from defect > cutoff_fraction * L/2
        - L = minimum lattice parameter
        - Excludes atoms close to defect

        Parameters
        ----------
        defect_site : list[float]
            Fractional coordinates of defect
        defect_structure : Structure
            Defect structure (relaxed)
        host_structure : Structure
            Host structure (pristine)

        Returns
        -------
        np.ndarray
            Indices of atoms to use for sampling
        """
        # Cutoff distance: fraction of half the minimum lattice parameter
        min_lattice_param = min(defect_structure.lattice.abc)
        cutoff_distance = self.sampling_cutoff_fraction * (min_lattice_param / 2.0)

        # Convert defect site to Cartesian
        defect_cart = defect_structure.lattice.get_cartesian_coords(defect_site)

        # Find atoms beyond cutoff distance
        sampling_indices = []
        for i, site in enumerate(defect_structure):
            # Calculate distance from defect (using minimum image convention)
            dist = defect_structure.lattice.get_distance_and_image(
                defect_cart, site.coords
            )[0]

            if dist >= cutoff_distance:
                sampling_indices.append(i)

        return np.array(sampling_indices, dtype=int)

    def _interpolate_potential_at_site(
        self,
        potential_grid: np.ndarray,
        grid_shape: tuple,
        frac_coords: np.ndarray,
    ) -> float:
        """
        Interpolate potential value at atomic site using trilinear interpolation.

        Parameters
        ----------
        potential_grid : np.ndarray
            3D potential grid
        grid_shape : tuple
            Shape of grid (nx, ny, nz)
        frac_coords : np.ndarray
            Fractional coordinates [x, y, z] of site

        Returns
        -------
        float
            Interpolated potential value (eV)
        """
        nx, ny, nz = grid_shape

        # Convert fractional coords to grid indices (continuous)
        # Handle periodic boundary conditions
        ix = (frac_coords[0] % 1.0) * (nx - 1)
        iy = (frac_coords[1] % 1.0) * (ny - 1)
        iz = (frac_coords[2] % 1.0) * (nz - 1)

        # Get integer and fractional parts
        ix0 = int(np.floor(ix))
        iy0 = int(np.floor(iy))
        iz0 = int(np.floor(iz))

        ix1 = (ix0 + 1) % nx
        iy1 = (iy0 + 1) % ny
        iz1 = (iz0 + 1) % nz

        dx = ix - ix0
        dy = iy - iy0
        dz = iz - iz0

        # Trilinear interpolation
        c000 = potential_grid[ix0, iy0, iz0]
        c001 = potential_grid[ix0, iy0, iz1]
        c010 = potential_grid[ix0, iy1, iz0]
        c011 = potential_grid[ix0, iy1, iz1]
        c100 = potential_grid[ix1, iy0, iz0]
        c101 = potential_grid[ix1, iy0, iz1]
        c110 = potential_grid[ix1, iy1, iz0]
        c111 = potential_grid[ix1, iy1, iz1]

        # Interpolate in x
        c00 = c000 * (1 - dx) + c100 * dx
        c01 = c001 * (1 - dx) + c101 * dx
        c10 = c010 * (1 - dx) + c110 * dx
        c11 = c011 * (1 - dx) + c111 * dx

        # Interpolate in y
        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy

        # Interpolate in z
        value = c0 * (1 - dz) + c1 * dz

        return float(value)

    def _remove_outliers(
        self, data: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers using robust statistics.

        Uses median absolute deviation (MAD) for robust outlier detection.

        Parameters
        ----------
        data : np.ndarray
            Data array
        threshold : float
            Number of standard deviations for outlier detection

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (cleaned_data, outlier_mask)
            - cleaned_data: Data with outliers removed
            - outlier_mask: Boolean mask (True = outlier)
        """
        if len(data) == 0:
            return data, np.array([], dtype=bool)

        # Calculate median and MAD (more robust than mean/std)
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        # Convert MAD to equivalent standard deviation
        # For normal distribution: σ ≈ 1.4826 * MAD  # noqa: RUF003
        sigma_equivalent = 1.4826 * mad

        # Identify outliers
        if sigma_equivalent > 0:
            z_scores = np.abs(data - median) / sigma_equivalent
            outlier_mask = z_scores > threshold
        else:
            # All values identical - no outliers
            outlier_mask = np.zeros(len(data), dtype=bool)

        # Remove outliers
        cleaned_data = data[~outlier_mask]

        return cleaned_data, outlier_mask
