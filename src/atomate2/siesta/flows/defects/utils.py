"""
Utility functions for defect calculations.

Includes functions for reading SIESTA output files (potential, density)
for use in advanced finite-size correction schemes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def read_siesta_grid_file(
    file_path: str | Path,
    file_type: str = "VT",
) -> dict:
    """
    Read SIESTA grid file (.VT, .RHO, etc.) using sisl.

    Parameters
    ----------
    file_path : str or Path
        Path to SIESTA grid file (.VT for potential, .RHO for density)
    file_type : str, optional
        Type of file: "VT" (electrostatic potential), "RHO" (electron density),
        "VH" (Hartree potential), "DRHO" (density difference).
        Default is "VT".

    Returns
    -------
    dict
        Dictionary containing:
        - "data": np.ndarray, 3D grid data
        - "grid_shape": tuple, (nx, ny, nz)
        - "cell": np.ndarray, 3x3 lattice matrix in Angstroms
        - "origin": np.ndarray, origin of grid

    Examples
    --------
    >>> # Read electrostatic potential from defect calculation
    >>> defect_pot = read_siesta_grid_file("defect_run/SystemLabel.VT")
    >>> host_pot = read_siesta_grid_file("host_run/SystemLabel.VT")
    >>>
    >>> # Use in Freysoldt correction
    >>> potential_data = {
    ...     "defect_potential": defect_pot["data"],
    ...     "host_potential": host_pot["data"],
    ...     "grid_shape": defect_pot["grid_shape"],
    ... }

    Notes
    -----
    Requires sisl to be installed:
        pip install sisl
    """
    try:
        import sisl
    except ImportError:
        raise ImportError(
            "sisl is required to read SIESTA grid files. Install with: pip install sisl"
        )

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Grid file not found: {file_path}")

    # Handle compressed .gz files
    import gzip
    import tempfile

    temp_file = None

    if file_path.suffix == ".gz":
        # Decompress to temporary file
        logger.debug(f"Decompressing {file_path.name}...")
        with gzip.open(file_path, "rb") as f_in:
            # Create temp file with correct extension (.VT, .RHO, etc.)
            suffix = file_path.stem.split(".")[-1]  # Get .VT from siesta.VT.gz
            temp_file = tempfile.NamedTemporaryFile(
                mode="wb", suffix=f".{suffix}", delete=False
            )
            temp_file.write(f_in.read())
            temp_file.close()
            file_path = Path(temp_file.name)
            logger.debug(f"Decompressed to {file_path}")

    try:
        # Read grid using sisl
        if file_type.upper() in ["VT", "ELECTROSTATIC"]:
            # Electrostatic potential
            grid = sisl.get_sile(str(file_path)).read_grid()
        elif file_type.upper() in ["RHO", "DENSITY"]:
            # Electron density
            grid = sisl.get_sile(str(file_path)).read_grid()
        elif file_type.upper() in ["VH", "HARTREE"]:
            # Hartree potential
            grid = sisl.get_sile(str(file_path)).read_grid()
        elif file_type.upper() in ["DRHO", "DELTA_DENSITY"]:
            # Density difference
            grid = sisl.get_sile(str(file_path)).read_grid()
        else:
            # Generic grid
            grid = sisl.get_sile(str(file_path)).read_grid()
    finally:
        # Clean up temp file if created
        if temp_file is not None:
            try:
                Path(temp_file.name).unlink()
            except Exception:
                pass

    # Extract data
    data = grid.grid  # 3D numpy array
    grid_shape = data.shape
    cell = grid.cell  # Lattice matrix

    logger.info(
        f"Read {file_type} grid from {file_path.name}: "
        f"shape {grid_shape}, volume {np.linalg.det(cell):.2f} Ų"
    )

    return {
        "data": data,
        "grid_shape": grid_shape,
        "cell": cell,
        "origin": np.array([0.0, 0.0, 0.0]),  # SIESTA grids start at origin
    }


def prepare_freysoldt_potential_data(
    defect_vt_path: str | Path,
    host_vt_path: str | Path,
) -> dict:
    """
    Prepare potential data for Freysoldt correction from SIESTA .VT files.

    Parameters
    ----------
    defect_vt_path : str or Path
        Path to defect calculation .VT file
    host_vt_path : str or Path
        Path to host calculation .VT file

    Returns
    -------
    dict
        Potential data dictionary ready for FreysoldtCorrection.calculate_correction():
        {
            "defect_potential": np.ndarray,
            "host_potential": np.ndarray,
            "grid_shape": tuple,
        }

    Examples
    --------
    >>> # Prepare potential data
    >>> pot_data = prepare_freysoldt_potential_data(
    ...     "defect_run/siesta.VT", "host_run/siesta.VT"
    ... )
    >>>
    >>> # Use in Freysoldt correction
    >>> from atomate2.siesta.flows.defects.corrections import FreysoldtCorrection
    >>> correction = FreysoldtCorrection(epsilon_static=9.8)
    >>> result = correction.calculate_correction(
    ...     defect_structure=defect_struct,
    ...     host_structure=host_struct,
    ...     charge_state=2,
    ...     defect_energy=-100.0,
    ...     host_energy=-50.0,
    ...     potential_data=pot_data,  # Pass potential data here
    ... )

    Notes
    -----
    - Both .VT files must be on the same grid (same supercell)
    - Requires sisl: pip install sisl
    """
    # Read both potential files
    defect_data = read_siesta_grid_file(defect_vt_path, file_type="VT")
    host_data = read_siesta_grid_file(host_vt_path, file_type="VT")

    # Verify grids match
    if defect_data["grid_shape"] != host_data["grid_shape"]:
        raise ValueError(
            f"Grid shapes do not match: "
            f"defect {defect_data['grid_shape']} vs host {host_data['grid_shape']}"
        )

    # Check cells are similar
    cell_diff = np.abs(defect_data["cell"] - host_data["cell"]).max()
    if cell_diff > 1e-3:
        logger.warning(
            f"Defect and host cells differ by {cell_diff:.6f} Å. "
            f"This may indicate different supercells!"
        )

    return {
        "defect_potential": defect_data["data"],
        "host_potential": host_data["data"],
        "grid_shape": defect_data["grid_shape"],
    }


def read_siesta_density(
    rho_path: str | Path,
) -> dict:
    """
    Read SIESTA electron density file (.RHO).

    Parameters
    ----------
    rho_path : str or Path
        Path to .RHO file

    Returns
    -------
    dict
        Density data with keys: "data", "grid_shape", "cell"

    Examples
    --------
    >>> density = read_siesta_density("siesta.RHO")
    >>> print(f"Density grid: {density['grid_shape']}")
    >>> print(
    ...     f"Total electrons: {density['data'].sum() * volume / np.prod(density['grid_shape'])}"
    ... )

    Notes
    -----
    Requires sisl: pip install sisl
    """
    return read_siesta_grid_file(rho_path, file_type="RHO")


def prepare_density_data(
    defect_rho_path: str | Path,
    host_rho_path: str | Path,
) -> dict:
    """
    Prepare charge density data for defect corrections from SIESTA .RHO files.

    Parameters
    ----------
    defect_rho_path : str or Path
        Path to defect calculation .RHO file
    host_rho_path : str or Path
        Path to host calculation .RHO file

    Returns
    -------
    dict
        Density data dictionary ready for MakovPayneCorrection.calculate_correction():
        {
            "defect_density": np.ndarray,
            "host_density": np.ndarray,
            "grid_shape": tuple,
        }

    Examples
    --------
    >>> # Prepare density data
    >>> density_data = prepare_density_data(
    ...     "defect_run/siesta.RHO", "host_run/siesta.RHO"
    ... )
    >>>
    >>> # Use in Makov-Payne correction with automatic quadrupole calculation
    >>> from atomate2.siesta.flows.defects.corrections import MakovPayneCorrection
    >>> correction = MakovPayneCorrection(epsilon_static=9.8)
    >>> result = correction.calculate_correction(
    ...     defect_structure=defect_struct,
    ...     host_structure=host_struct,
    ...     charge_state=2,
    ...     defect_energy=-100.0,
    ...     host_energy=-50.0,
    ...     defect_site=[0.5, 0.5, 0.5],
    ...     density_data=density_data,  # Pass density data here
    ... )
    >>> print(f"Quadrupole moment: {result.metadata['quadrupole_moment_eA2']:.4f} eÅ²")

    Notes
    -----
    - Both .RHO files must be on the same grid (same supercell)
    - Requires sisl: pip install sisl
    - Used to calculate quadrupole moments for Makov-Payne correction
    """
    # Read both density files
    defect_data = read_siesta_density(defect_rho_path)
    host_data = read_siesta_density(host_rho_path)

    # Verify grids match
    if defect_data["grid_shape"] != host_data["grid_shape"]:
        raise ValueError(
            f"Grid shapes do not match: "
            f"defect {defect_data['grid_shape']} vs host {host_data['grid_shape']}"
        )

    # Check cells are similar
    cell_diff = np.abs(defect_data["cell"] - host_data["cell"]).max()
    if cell_diff > 1e-3:
        logger.warning(
            f"Defect and host cells differ by {cell_diff:.6f} Å. "
            f"This may indicate different supercells!"
        )

    return {
        "defect_density": defect_data["data"],
        "host_density": host_data["data"],
        "grid_shape": defect_data["grid_shape"],
    }


def calculate_planar_average(
    potential_grid: np.ndarray,
    axis: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate planar average of 3D potential grid along specified axis.

    Parameters
    ----------
    potential_grid : np.ndarray
        3D potential grid
    axis : int, optional
        Axis along which to average (0=x, 1=y, 2=z). Default is 2 (z-axis).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (positions, averaged_potential)
        - positions: fractional coordinates along axis
        - averaged_potential: planar-averaged potential values

    Examples
    --------
    >>> pot_data = read_siesta_grid_file("siesta.VT")
    >>> positions, avg_pot = calculate_planar_average(pot_data["data"], axis=2)
    >>> plt.plot(positions, avg_pot)
    """
    # Average over the two perpendicular directions
    axes_to_average = tuple(i for i in range(3) if i != axis)
    avg_potential = np.mean(potential_grid, axis=axes_to_average)

    # Create position array (fractional coordinates)
    n_points = avg_potential.shape[0]
    positions = np.linspace(0, 1, n_points)

    return positions, avg_potential


def plot_potential_alignment(
    defect_vt_path: str | Path,
    host_vt_path: str | Path,
    axis: int = 2,
    output_path: str | Path | None = None,
    show_plot: bool = True,
) -> dict:
    """
    Plot planar-averaged potentials for defect and host calculations.

    Creates a publication-quality plot showing:
    - Host potential (blue)
    - Defect potential (red)
    - Potential difference (green)
    - Alignment regions

    Parameters
    ----------
    defect_vt_path : str or Path
        Path to defect .VT file
    host_vt_path : str or Path
        Path to host .VT file
    axis : int, optional
        Axis for planar averaging (0=x, 1=y, 2=z). Default is 2.
    output_path : str or Path, optional
        Path to save plot. If None, plot is not saved.
    show_plot : bool, optional
        Whether to display plot interactively. Default is True.

    Returns
    -------
    dict
        Dictionary with plot data:
        {
            "positions": positions array,
            "host_potential": averaged host potential,
            "defect_potential": averaged defect potential,
            "potential_difference": difference,
            "axis": axis used,
        }

    Examples
    --------
    >>> from atomate2.siesta.flows.defects import plot_potential_alignment
    >>> plot_data = plot_potential_alignment(
    ...     "defect_run/siesta.VT",
    ...     "host_run/siesta.VT",
    ...     output_path="potential_alignment.png",
    ... )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    # Read potential files
    defect_data = read_siesta_grid_file(defect_vt_path, file_type="VT")
    host_data = read_siesta_grid_file(host_vt_path, file_type="VT")

    # Calculate planar averages
    pos_defect, pot_defect = calculate_planar_average(defect_data["data"], axis=axis)
    pos_host, pot_host = calculate_planar_average(host_data["data"], axis=axis)

    # Calculate difference
    pot_diff = pot_defect - pot_host

    # Create plot
    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Potentials
    ax1.plot(pos_host, pot_host, "b-", label="Host", linewidth=2)
    ax1.plot(pos_defect, pot_defect, "r-", label="Defect", linewidth=2, alpha=0.7)
    ax1.set_ylabel("Electrostatic Potential (eV)", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        "Planar-Averaged Electrostatic Potentials", fontsize=13, fontweight="bold"
    )

    # Plot 2: Difference
    ax2.plot(pos_defect, pot_diff, "g-", linewidth=2)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel(f"Fractional Coordinate ({'xyz'[axis]}-axis)", fontsize=12)
    ax2.set_ylabel("Potential Difference (eV)", fontsize=12)
    ax2.set_title("ΔV = V_defect - V_host", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Add alignment value annotation
    alignment_value = np.mean(pot_diff)
    ax2.text(
        0.02,
        0.98,
        f"Mean ΔV = {alignment_value:.4f} eV",
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save if requested
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved potential alignment plot to {output_path}")

    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    return {
        "positions": pos_host,
        "host_potential": pot_host,
        "defect_potential": pot_defect,
        "potential_difference": pot_diff,
        "axis": axis,
        "mean_alignment": alignment_value,
    }


def find_vt_files(
    calculation_dir: str | Path,
    system_label: str = "siesta",
) -> Path | None:
    """
    Find .VT file in a SIESTA calculation directory.

    Parameters
    ----------
    calculation_dir : str or Path
        Directory containing SIESTA calculation outputs
    system_label : str, optional
        SIESTA SystemLabel. Default is "siesta".

    Returns
    -------
    Path or None
        Path to .VT file if found, None otherwise

    Examples
    --------
    >>> vt_path = find_vt_files("job_2025-01-01-12-00-00-123456")
    >>> if vt_path:
    ...     print(f"Found: {vt_path}")
    """
    calc_dir = Path(calculation_dir)

    if not calc_dir.exists():
        logger.warning(f"Calculation directory not found: {calc_dir}")
        return None

    # Try common .VT file names (uncompressed)
    vt_candidates = [
        calc_dir / f"{system_label}.VT",
        calc_dir / "siesta.VT",
        calc_dir / "SystemLabel.VT",
    ]

    # Also search for any .VT file (uncompressed)
    vt_files = list(calc_dir.glob("*.VT"))

    for vt_path in vt_candidates + vt_files:
        if vt_path.exists():
            logger.info(f"Found .VT file: {vt_path}")
            return vt_path

    # Try compressed .VT.gz files in siesta_compressed subdirectory
    compressed_dir = calc_dir / "siesta_compressed"
    if compressed_dir.exists():
        vt_gz_candidates = [
            compressed_dir / f"{system_label}.VT.gz",
            compressed_dir / "siesta.VT.gz",
            compressed_dir / "SystemLabel.VT.gz",
        ]
        vt_gz_files = list(compressed_dir.glob("*.VT.gz"))

        for vt_gz_path in vt_gz_candidates + vt_gz_files:
            if vt_gz_path.exists():
                logger.info(f"Found compressed .VT file: {vt_gz_path}")
                return vt_gz_path

    logger.debug(f"No .VT or .VT.gz file found in {calc_dir}")
    return None
