"""Band structure calculation workflow for SIESTA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, job

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.jobs.core import BandStructureMaker, RelaxMaker, StaticMaker

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    import numpy as np
    from jobflow import Job
    from jobflow.core.reference import OutputReference
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def _get_attr(obj: Any, *attrs: str, default: Any = None) -> Any:
    """Get nested attribute from object or dict."""
    result = obj
    for attr in attrs:
        if result is None:
            return default
        if isinstance(result, dict):
            result = result.get(attr)
        else:
            result = getattr(result, attr, None)
    return result if result is not None else default


def _read_bands_file(bands_output: Any) -> dict | None:
    """
    Read SIESTA .bands file and return parsed data.

    Returns
    -------
    dict or None
        Dictionary with keys: k, energies, efermi, kmin, kmax, nband, nspin, nk
    """
    import gzip
    from pathlib import Path

    import numpy as np

    calc_dir = _get_attr(bands_output, "dir_name", default=".")
    calc_path = Path(calc_dir)

    # Find bands file
    search_dirs = [calc_path / "siesta_compressed", calc_path]
    patterns = ["siesta.bands.gz", "*.bands.gz", "siesta.bands", "*.bands"]

    bands_file = None
    is_gzipped = False

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            matches = list(search_dir.glob(pattern))
            if matches:
                bands_file = matches[0]
                is_gzipped = bands_file.suffix == ".gz"
                break
        if bands_file:
            break

    if bands_file is None or not bands_file.exists():
        return None

    # Read file
    f = gzip.open(bands_file, "rt") if is_gzipped else open(bands_file)  # noqa: SIM115

    try:
        ef_file = float(f.readline())
        kmin, kmax = map(float, f.readline().split())
        f.readline()  # skip energy range
        nband, nspin, nk = map(int, f.readline().split())

        k = np.zeros(nk)
        e = np.zeros((nband, nspin, nk))

        for ik in range(nk):
            values = []
            line = f.readline().split()
            k[ik] = float(line[0])
            values.extend([float(v) for v in line[1:]])

            while len(values) < nband * nspin:
                line = f.readline().split()
                values.extend([float(v) for v in line])

            for ispin in range(nspin):
                for iband in range(nband):
                    e[iband, ispin, ik] = values[ispin * nband + iband]
    finally:
        f.close()

    return {
        "k": k,
        "energies": e,
        "efermi": ef_file,
        "kmin": kmin,
        "kmax": kmax,
        "nband": nband,
        "nspin": nspin,
        "nk": nk,
    }


def _calculate_effective_mass(
    k: np.ndarray,
    energies: np.ndarray,
    band_idx: int,
    k_idx: int,
    efermi: float,  # noqa: ARG001
    is_cbm: bool = True,  # noqa: ARG001
) -> float | None:
    """
    Calculate effective mass at a band extremum using parabolic fitting.

    Parameters
    ----------
    k : array
        k-point positions
    energies : array
        Band energies (nband, nspin, nk)
    band_idx : int
        Band index
    k_idx : int
        k-point index of extremum
    efermi : float
        Fermi energy
    is_cbm : bool
        True for CBM (minimum), False for VBM (maximum)

    Returns
    -------
    float
        Effective mass in units of electron mass (m*/m_e)
    """
    import numpy as np
    from scipy.constants import electron_mass, eV, hbar

    # Get band energies around the extremum
    spin_idx = 0  # Use first spin channel
    band = energies[band_idx, spin_idx, :]

    # Use 5 points around extremum for fitting
    n_fit = 5
    k_start = max(0, k_idx - n_fit // 2)
    k_end = min(len(k), k_idx + n_fit // 2 + 1)

    k_fit = k[k_start:k_end]
    e_fit = band[k_start:k_end]

    if len(k_fit) < 3:
        return None

    # Fit parabola: E = a*k^2 + b*k + c
    try:
        # Shift k to be centered at extremum
        k_centered = k_fit - k_fit[len(k_fit) // 2]
        coeffs = np.polyfit(k_centered, e_fit, 2)
        a = coeffs[0]  # coefficient of k^2

        if abs(a) < 1e-10:
            return None

        # Convert to SI units
        # k is in 1/Angstrom, E is in eV
        # m* = hbar^2 / (d^2E/dk^2) = hbar^2 / (2a)
        # Need to convert: k: 1/Ang -> 1/m, E: eV -> J
        k_to_m = 1e10  # 1/Ang to 1/m
        a_si = a * eV / (k_to_m**2)  # eV/Ang^-2 to J/m^-2

        m_eff = (hbar**2) / (2 * a_si)
        m_eff_ratio = m_eff / electron_mass

        return abs(m_eff_ratio)
    except Exception:  # noqa: BLE001
        return None


def _analyze_band_gaps(k: np.ndarray, energies: np.ndarray, efermi: float) -> dict:
    """
    Analyze band gaps: direct vs indirect, locations, effective masses.

    Returns
    -------
    dict
        Analysis results including gap type, locations, effective masses
    """
    import numpy as np

    nband, _nspin, nk = energies.shape

    # Find VBM and CBM
    # Valence bands: below Fermi level, Conduction bands: above Fermi level
    vbm_energy = -np.inf
    vbm_k_idx = 0
    vbm_band_idx = 0
    cbm_energy = np.inf
    cbm_k_idx = 0
    cbm_band_idx = 0

    for iband in range(nband):
        for ik in range(nk):
            e = energies[iband, 0, ik]  # First spin channel
            if e <= efermi and e > vbm_energy:
                vbm_energy = e
                vbm_k_idx = ik
                vbm_band_idx = iband
            if e > efermi and e < cbm_energy:
                cbm_energy = e
                cbm_k_idx = ik
                cbm_band_idx = iband

    # Calculate band gap
    bandgap = cbm_energy - vbm_energy if cbm_energy > vbm_energy else 0.0

    # Direct or indirect?
    is_direct = abs(k[vbm_k_idx] - k[cbm_k_idx]) < 0.01 * (k[-1] - k[0])

    # Find direct gap at VBM location
    direct_gap_at_vbm = None
    if vbm_band_idx < nband - 1:
        # Find lowest conduction band at VBM k-point
        for iband in range(vbm_band_idx + 1, nband):
            e = energies[iband, 0, vbm_k_idx]
            if e > efermi:
                direct_gap_at_vbm = e - vbm_energy
                break

    # Calculate effective masses
    m_hole = _calculate_effective_mass(
        k, energies, vbm_band_idx, vbm_k_idx, efermi, is_cbm=False
    )
    m_electron = _calculate_effective_mass(
        k, energies, cbm_band_idx, cbm_k_idx, efermi, is_cbm=True
    )

    return {
        "bandgap_eV": bandgap,
        "is_direct": is_direct,
        "gap_type": "direct" if is_direct else "indirect",
        "vbm_eV": vbm_energy,
        "cbm_eV": cbm_energy,
        "vbm_k": k[vbm_k_idx],
        "cbm_k": k[cbm_k_idx],
        "vbm_k_idx": vbm_k_idx,
        "cbm_k_idx": cbm_k_idx,
        "direct_gap_at_vbm_eV": direct_gap_at_vbm,
        "effective_mass_hole": m_hole,
        "effective_mass_electron": m_electron,
    }


def _calculate_bandwidth(energies: np.ndarray, efermi: float) -> dict:
    """
    Calculate bandwidth of valence and conduction bands.

    Returns
    -------
    dict
        Bandwidth information
    """
    import numpy as np

    nband, _nspin, _nk = energies.shape

    valence_min = np.inf
    valence_max = -np.inf
    conduction_min = np.inf
    conduction_max = -np.inf

    for iband in range(nband):
        band_min = np.min(energies[iband, 0, :])
        band_max = np.max(energies[iband, 0, :])

        if band_max <= efermi:
            # Valence band
            valence_min = min(valence_min, band_min)
            valence_max = max(valence_max, band_max)
        elif band_min > efermi:
            # Conduction band
            conduction_min = min(conduction_min, band_min)
            conduction_max = max(conduction_max, band_max)

    return {
        "valence_bandwidth_eV": valence_max - valence_min
        if valence_max > valence_min
        else None,
        "conduction_bandwidth_eV": conduction_max - conduction_min
        if conduction_max > conduction_min
        else None,
        "valence_band_top_eV": valence_max if valence_max > -np.inf else None,
        "valence_band_bottom_eV": valence_min if valence_min < np.inf else None,
    }


def _calculate_band_velocities(k: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """
    Calculate group velocities for all bands: v = (1/hbar) * dE/dk.

    Returns
    -------
    array
        Velocities in m/s, shape (nband, nspin, nk)
    """
    import numpy as np
    from scipy.constants import eV, hbar

    nband, nspin, _nk = energies.shape
    velocities = np.zeros_like(energies)

    # Convert units: k in 1/Ang, E in eV
    # v = (1/hbar) * dE/dk
    # dE/dk in eV*Ang, need to convert to J*m
    k_to_m = 1e10  # 1/Ang to 1/m

    for iband in range(nband):
        for ispin in range(nspin):
            # Use central differences for derivative
            de_dk = np.gradient(energies[iband, ispin, :], k)
            # Convert to SI: eV*Ang -> J*m
            de_dk_si = de_dk * eV / k_to_m
            velocities[iband, ispin, :] = de_dk_si / hbar

    return velocities


def _plot_band_velocities(
    k: np.ndarray,
    velocities: np.ndarray,
    efermi: float,
    energies: np.ndarray,
    formula: str,
    kmin: float,
    kmax: float,
    structure: Structure | None = None,
) -> str:
    """Plot band velocities (group velocity) with high-symmetry k-point labels."""
    import matplotlib.pyplot as plt
    import numpy as np

    nband, nspin, _nk = velocities.shape

    _fig, ax = plt.subplots(figsize=(10, 6))

    # Plot velocities for bands near Fermi level
    for iband in range(nband):
        band_min = np.min(energies[iband, 0, :])
        band_max = np.max(energies[iband, 0, :])

        # Only plot bands within 5 eV of Fermi level
        if band_max < efermi - 5 or band_min > efermi + 5:
            continue

        for ispin in range(nspin):
            color = "b" if ispin == 0 else "r"
            # Convert to 10^6 m/s for readability
            v_plot = velocities[iband, ispin, :] / 1e6
            ax.plot(k, v_plot, color=color, linewidth=0.8, alpha=0.7)

    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)

    # Add high-symmetry labels (same as band structure plot)
    if structure is not None:
        labels, label_positions = _get_kpath_labels(structure)
        if labels and label_positions:
            tick_positions = [kmin + pos * (kmax - kmin) for pos in label_positions]
            for pos in tick_positions:
                ax.axvline(x=pos, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(labels, fontsize=12)
        else:
            ax.set_xlabel("k-path", fontsize=12)
    else:
        ax.set_xlabel("k-path", fontsize=12)

    ax.set_ylabel("Group Velocity (10⁶ m/s)", fontsize=12)
    ax.set_xlim(kmin, kmax)
    ax.set_title(f"Band Velocities: {formula}", fontsize=14)

    plt.tight_layout()
    plot_file = "band_velocities.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    return plot_file


@job
def analyze_band_structure(
    bands_output: Any,
    scf_output: Any,
    plot_bands: bool = True,
    energy_range: tuple[float, float] | None = None,
) -> dict:
    """
    Analyze band structure results and generate comprehensive plots.

    Parameters
    ----------
    bands_output : SiestaTaskDoc or dict
        Output from band structure calculation
    scf_output : SiestaTaskDoc or dict
        Output from SCF calculation (for Fermi level)
    plot_bands : bool
        Whether to generate band structure plots
    energy_range : tuple[float, float] | None
        Energy range for plot (relative to Fermi level). Default: (-5, 5) eV

    Returns
    -------
    dict
        Dictionary containing comprehensive band structure analysis:
        - Electronic properties (bandgap, VBM, CBM, Fermi level)
        - Gap analysis (direct/indirect, k-point locations)
        - Effective masses (hole and electron)
        - Bandwidth information
        - Generated plot files
    """
    from pathlib import Path

    import numpy as np

    # Extract basic information
    efermi = _get_attr(scf_output, "output", "efermi")
    formula = _get_attr(bands_output, "formula_pretty", default="unknown")

    # Read bands data for detailed analysis
    bands_data = _read_bands_file(bands_output)

    # Initialize summary
    summary = {
        "formula": formula,
        "efermi_eV": efermi,
        "plots": {},
    }

    if bands_data is not None:
        k = bands_data["k"]
        energies = bands_data["energies"]
        kmin = bands_data["kmin"]
        kmax = bands_data["kmax"]

        # Use file's Fermi level if not provided
        if efermi is None:
            efermi = bands_data["efermi"]
            summary["efermi_eV"] = efermi

        # Detailed gap analysis
        gap_analysis = _analyze_band_gaps(k, energies, efermi)
        summary.update(gap_analysis)

        # Bandwidth analysis
        bandwidth = _calculate_bandwidth(energies, efermi)
        summary.update(bandwidth)

        # Determine if metallic
        is_metallic = gap_analysis["bandgap_eV"] < 0.01
        summary["is_metallic"] = is_metallic

        # Calculate band velocities
        velocities = _calculate_band_velocities(k, energies)
        max_velocity = np.max(np.abs(velocities)) / 1e6  # 10^6 m/s
        summary["max_group_velocity_1e6_m_s"] = max_velocity

    else:
        # Fall back to basic info from TaskDoc
        bandgap = _get_attr(bands_output, "output", "bandgap")
        cbm = _get_attr(bands_output, "output", "cbm")
        vbm = _get_attr(bands_output, "output", "vbm")
        summary["bandgap_eV"] = bandgap
        summary["cbm_eV"] = cbm
        summary["vbm_eV"] = vbm
        summary["is_metallic"] = bandgap is not None and bandgap < 0.01
        summary["gap_type"] = "unknown"

    # Write comprehensive summary file
    summary_file = Path("band_structure_summary.txt")
    with open(summary_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Band Structure Analysis: {formula}\n")
        f.write("=" * 70 + "\n\n")

        f.write("ELECTRONIC PROPERTIES\n")
        f.write("-" * 70 + "\n")

        if summary.get("is_metallic"):
            f.write("  Material Type: METALLIC\n")
        else:
            f.write("  Material Type: SEMICONDUCTOR/INSULATOR\n")
            f.write(f"  Band Gap: {summary.get('bandgap_eV', 0):.4f} eV\n")
            f.write(f"  Gap Type: {summary.get('gap_type', 'unknown').upper()}\n")

            if summary.get("direct_gap_at_vbm_eV") is not None:
                f.write(
                    f"  Direct Gap at VBM: {summary['direct_gap_at_vbm_eV']:.4f} eV\n"
                )

        f.write(f"\n  Fermi Level: {summary.get('efermi_eV', 0):.4f} eV\n")

        if summary.get("vbm_eV") is not None:
            f.write(f"  VBM (Valence Band Maximum): {summary['vbm_eV']:.4f} eV")
            if summary.get("vbm_k") is not None:
                f.write(f"  at k = {summary['vbm_k']:.4f}")
            f.write("\n")

        if summary.get("cbm_eV") is not None:
            f.write(f"  CBM (Conduction Band Minimum): {summary['cbm_eV']:.4f} eV")
            if summary.get("cbm_k") is not None:
                f.write(f"  at k = {summary['cbm_k']:.4f}")
            f.write("\n")

        # Effective masses
        if (
            summary.get("effective_mass_hole") is not None
            or summary.get("effective_mass_electron") is not None
        ):
            f.write("\nEFFECTIVE MASSES\n")
            f.write("-" * 70 + "\n")
            if summary.get("effective_mass_hole") is not None:
                f.write(
                    f"  Hole effective mass (m_h*): "
                    f"{summary['effective_mass_hole']:.4f} m_e\n"
                )
            if summary.get("effective_mass_electron") is not None:
                f.write(
                    f"  Electron effective mass (m_e*): "
                    f"{summary['effective_mass_electron']:.4f} m_e\n"
                )

        # Bandwidth
        if summary.get("valence_bandwidth_eV") is not None:
            f.write("\nBANDWIDTH\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Valence band width: {summary['valence_bandwidth_eV']:.4f} eV\n")
            if summary.get("conduction_bandwidth_eV") is not None:
                f.write(
                    f"  Conduction band width: "
                    f"{summary['conduction_bandwidth_eV']:.4f} eV\n"
                )

        # Transport properties
        if summary.get("max_group_velocity_1e6_m_s") is not None:
            f.write("\nTRANSPORT PROPERTIES\n")
            f.write("-" * 70 + "\n")
            f.write(
                f"  Maximum group velocity: "
                f"{summary['max_group_velocity_1e6_m_s']:.2f} × 10⁶ m/s\n"  # noqa: RUF001
            )

        f.write("\n" + "=" * 70 + "\n")

    summary["summary_file"] = str(summary_file)

    # Generate plots if requested
    if plot_bands and bands_data is not None:
        try:
            # Main band structure plot
            plot_file = _plot_band_structure(
                bands_output, efermi, energy_range or (-5, 5)
            )
            if plot_file:
                summary["plots"]["band_structure"] = plot_file
                logger.info(f"Generated band structure plot: {plot_file}")

            # Band velocities plot
            structure = _get_attr(bands_output, "structure")
            velocity_plot = _plot_band_velocities(
                k, velocities, efermi, energies, formula, kmin, kmax, structure
            )
            if velocity_plot:
                summary["plots"]["band_velocities"] = velocity_plot
                logger.info(f"Generated band velocities plot: {velocity_plot}")

        except Exception as e:  # noqa: BLE001
            import traceback

            logger.warning(f"Error generating plots: {e}")
            logger.warning(traceback.format_exc())

    elif plot_bands:
        # Try basic plot even without full bands_data
        try:
            plot_file = _plot_band_structure(
                bands_output, efermi, energy_range or (-5, 5)
            )
            if plot_file:
                summary["plots"]["band_structure"] = plot_file
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not generate band structure plot: {e}")

    # For backward compatibility
    summary["plot_file"] = summary.get("plots", {}).get("band_structure")

    return summary


def _get_kpath_labels(structure: Structure) -> tuple[list[str], list[float]]:
    """
    Get high-symmetry k-path labels and positions from structure using pymatgen.

    Parameters
    ----------
    structure : Structure
        Crystal structure

    Returns
    -------
    tuple[list[str], list[float]]
        (labels, positions) where positions are normalized to [0, 1]
    """
    try:
        import numpy as np
        from pymatgen.symmetry.bandstructure import HighSymmKpath

        # Get high-symmetry k-path
        kpath = HighSymmKpath(structure)
        path = kpath.kpath["path"]
        kpts = kpath.kpath["kpoints"]

        # Calculate cumulative distances along path
        labels: list[str] = []
        positions: list[float] = []
        cumulative_dist = 0.0
        rec_lattice = structure.lattice.reciprocal_lattice

        for segment in path:
            for i, label in enumerate(segment):
                if i == 0 and labels and labels[-1] == label:
                    # Skip duplicate at segment boundaries
                    continue

                # Get k-point coordinates
                k_frac = kpts[label]
                k_cart = rec_lattice.get_cartesian_coords(k_frac)

                if labels:
                    # Calculate distance from previous point
                    prev_label = segment[i - 1] if i > 0 else labels[-1]
                    prev_k_frac = (
                        kpts[prev_label]
                        if prev_label in kpts
                        else kpts.get(labels[-1], [0, 0, 0])
                    )
                    prev_k_cart = rec_lattice.get_cartesian_coords(prev_k_frac)
                    dist = np.linalg.norm(k_cart - prev_k_cart)
                    cumulative_dist += dist

                # Convert label to proper symbol
                display_label = label.replace("\\Gamma", "Γ").replace("GAMMA", "Γ")
                if display_label == "G":
                    display_label = "Γ"

                labels.append(display_label)
                positions.append(cumulative_dist)

        # Normalize positions to [0, 1]
        if positions and positions[-1] > 0:
            positions = [p / positions[-1] for p in positions]

    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not generate k-path labels: {e}")
        return [], []
    else:
        return labels, positions


def _plot_band_structure(
    bands_output: dict,
    efermi: float | None,
    energy_range: tuple[float, float],
) -> str | None:
    """
    Generate band structure plot from SIESTA output.

    Parameters
    ----------
    bands_output : dict
        Band structure calculation output
    efermi : float | None
        Fermi energy for reference
    energy_range : tuple[float, float]
        Energy range relative to Fermi level (eV)

    Returns
    -------
    str | None
        Path to generated plot file, or None if failed
    """
    import gzip
    from pathlib import Path

    # Check for .bands file in the calculation directory
    calc_dir = _get_attr(bands_output, "dir_name", default=".")
    calc_path = Path(calc_dir)
    bands_file = None
    is_gzipped = False

    logger.info(f"Looking for .bands file in: {calc_path.absolute()}")

    # Look for bands file with various patterns in multiple locations
    # First check siesta_compressed/ subdirectory (where SIESTA stores compressed
    # outputs)
    search_dirs = [
        calc_path / "siesta_compressed",
        calc_path,
    ]

    # Patterns to search for (include gzipped versions)
    patterns = [
        "siesta.bands.gz",
        "*.bands.gz",
        "SystemLabel.bands.gz",
        "siesta.bands",
        "*.bands",
        "SystemLabel.bands",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            matches = list(search_dir.glob(pattern))
            if matches:
                bands_file = matches[0]
                is_gzipped = bands_file.suffix == ".gz"
                logger.info(f"Found bands file: {bands_file} (gzipped: {is_gzipped})")
                break
        if bands_file:
            break

    if bands_file is None or not bands_file.exists():
        # List directory contents for debugging
        if calc_path.exists():
            files = list(calc_path.iterdir())
            logger.warning(f"No .bands file found in {calc_dir}")
            logger.warning(
                f"Directory contains {len(files)} files: {[f.name for f in files[:10]]}"
            )
            # Also check siesta_compressed
            compressed_dir = calc_path / "siesta_compressed"
            if compressed_dir.exists():
                compressed_files = list(compressed_dir.iterdir())
                logger.warning(
                    f"siesta_compressed/ contains: "
                    f"{[f.name for f in compressed_files[:10]]}"
                )
        else:
            logger.warning(f"Directory does not exist: {calc_dir}")
        return None

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Read bands file (handle gzipped files)
        f = gzip.open(bands_file, "rt") if is_gzipped else open(bands_file)  # noqa: SIM115

        # Parse bands file
        try:
            ef_file = float(f.readline())
            kmin, kmax = map(float, f.readline().split())
            f.readline()  # skip energy range line
            nband, nspin, nk = map(int, f.readline().split())

            k = np.zeros(nk)
            e = np.zeros((nband, nspin, nk))

            # SIESTA bands file format: band energies can span multiple lines
            # Each k-point starts with k-value followed by band energies
            for ik in range(nk):
                # Read all values for this k-point (may span multiple lines)
                values = []
                # First line has k-value + some band energies
                line = f.readline().split()
                k[ik] = float(line[0])
                values.extend([float(v) for v in line[1:]])

                # Read continuation lines until we have all bands
                while len(values) < nband * nspin:
                    line = f.readline().split()
                    values.extend([float(v) for v in line])

                # Assign band energies (bands are outer, spin is inner in file)
                for ispin in range(nspin):
                    for iband in range(nband):
                        e[iband, ispin, ik] = values[ispin * nband + iband]
        finally:
            f.close()

        # Use Fermi level from file if not provided
        if efermi is None:
            efermi = ef_file

        # Shift to Fermi level
        e_shifted = e - efermi

        # Create plot
        _fig, ax = plt.subplots(figsize=(10, 6))

        # Plot bands
        for ispin in range(nspin):
            color = "b" if ispin == 0 else "r"
            label = f"Spin {'up' if ispin == 0 else 'down'}" if nspin > 1 else None
            for iband in range(nband):
                ax.plot(k, e_shifted[iband, ispin, :], color=color, linewidth=0.8)
            if label:
                ax.plot([], [], color=color, label=label)

        # Fermi level line
        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)

        # Get high-symmetry labels from structure
        structure = _get_attr(bands_output, "structure")
        if structure is not None:
            labels, label_positions = _get_kpath_labels(structure)
            if labels and label_positions:
                # Scale positions to actual k-range
                tick_positions = [kmin + pos * (kmax - kmin) for pos in label_positions]

                # Add vertical lines at high-symmetry points
                for pos in tick_positions:
                    ax.axvline(
                        x=pos, color="gray", linestyle="-", linewidth=0.5, alpha=0.5
                    )

                # Set x-ticks with labels
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(labels, fontsize=12)
            else:
                ax.set_xlabel("k-path", fontsize=12)
        else:
            ax.set_xlabel("k-path", fontsize=12)

        # Labels and formatting
        ax.set_ylabel("Energy (eV)", fontsize=12)
        ax.set_xlim(kmin, kmax)
        ax.set_ylim(energy_range[0], energy_range[1])

        formula = _get_attr(bands_output, "formula_pretty", default="")
        ax.set_title(f"Band Structure: {formula}", fontsize=14)

        if nspin > 1:
            ax.legend(loc="upper right")

        plt.tight_layout()

        # Save plot
        plot_file = "band_structure.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:  # noqa: BLE001
        import traceback

        logger.warning(f"Error generating band plot: {e}")
        logger.warning(traceback.format_exc())
        return None
    else:
        return plot_file


@dataclass
class BandStructureFlowMaker(BaseSiestaFlowMaker):
    """
    SIESTA Electronic Band Structure Workflow.

    A complete workflow for computing electronic band structure including:
    1. Optional structure relaxation
    2. Self-consistent field (SCF) calculation for ground state
    3. Non-self-consistent band structure calculation along high-symmetry k-path
    4. Band gap analysis and optional plotting

    The workflow automatically generates the high-symmetry k-path appropriate
    for the crystal system using seekpath/pymatgen conventions.

    Key Results:
    ------------
    - Band Structure: E(k) along high-symmetry path
    - Band Gap: Direct/indirect gap magnitude (eV)
    - VBM/CBM: Valence band maximum and conduction band minimum positions
    - Fermi Level: Chemical potential (eV)
    - Band Structure Plot: Publication-quality PNG (optional)

    Parameters
    ----------
    name : str
        Name of the workflow (default: "band structure")
    relax_maker : RelaxMaker | None
        Maker for optional structure relaxation. Set to None to skip relaxation.
        Default: Variable-cell relaxation with tight convergence.
    scf_maker : StaticMaker
        Maker for SCF calculation. Default uses DZP basis and 300 Ry cutoff.
    bands_maker : BandStructureMaker
        Maker for band structure calculation along k-path.
    plot_bands : bool
        Whether to generate band structure plot (default: True)
    energy_range : tuple[float, float] | None
        Energy range for plot relative to Fermi level (default: (-5, 5) eV)

    Examples
    --------
    >>> from atomate2.siesta.flows.bands import BandStructureFlowMaker
    >>> from pymatgen.core import Structure
    >>>
    >>> # Basic usage with relaxation
    >>> structure = Structure.from_file("Si.cif")
    >>> maker = BandStructureFlowMaker()
    >>> flow = maker.make(structure)
    >>>
    >>> # Skip relaxation (pre-relaxed structure)
    >>> maker = BandStructureFlowMaker(relax_maker=None)
    >>> flow = maker.make(structure)
    >>>
    >>> # Custom settings
    >>> from atomate2.siesta.jobs.core import StaticMaker, BandStructureMaker
    >>> maker = BandStructureFlowMaker(
    ...     scf_maker=StaticMaker(user_params={"PAO.BasisSize": "TZP"}),
    ...     energy_range=(-10, 10),
    ... )
    >>> flow = maker.make(structure)

    Notes
    -----
    - SCF calculation must complete before band structure calculation
    - Band structure uses same basis set and cutoff as SCF for consistency
    - K-path automatically determined from crystal symmetry
    - For spin-polarized systems, separate bands for each spin channel
    """

    name: str = "band structure"
    relax_maker: RelaxMaker | None = field(
        default_factory=lambda: RelaxMaker.variable_cell_relaxation(
            user_params={
                "MD.MaxForceTol": "0.02 eV/Ang",
                "MD.MaxStressTol": "0.5 GPa",
            }
        )
    )
    scf_maker: StaticMaker = field(
        default_factory=lambda: StaticMaker.scf(
            user_params={
                "PAO.BasisSize": "DZP",
                "Mesh.Cutoff": "300 Ry",
            }
        )
    )
    bands_maker: BandStructureMaker = field(
        default_factory=lambda: BandStructureMaker.bandstructure_calculation(
            user_params={
                "WriteBands": "true",  # Required to generate .bands file for plotting
            }
        )
    )
    plot_bands: bool = True
    energy_range: tuple[float, float] | None = None

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Create band structure calculation workflow.

        Parameters
        ----------
        structure : Structure
            Input structure (will be relaxed if relax_maker is provided)
        prev_dir : str | Path | None
            Previous calculation directory to copy files from

        Returns
        -------
        Flow
            Band structure calculation workflow
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        jobs: list[Job | Flow] = []
        current_structure: Structure | OutputReference = structure
        scf_prev_dir: str | Path | OutputReference | None = prev_dir

        # Step 1: Optional relaxation
        if self.relax_maker is not None:
            self.propagate_custodian_to_maker(self.relax_maker)
            relax_job = self.relax_maker.make(structure, prev_dir=prev_dir)
            relax_job.name = f"{self.name}_relax"
            jobs.append(relax_job)
            current_structure = relax_job.output.structure
            scf_prev_dir = relax_job.output.dir_name

        # Step 2: SCF calculation
        self.propagate_custodian_to_maker(self.scf_maker)
        scf_job = self.scf_maker.make(current_structure, prev_dir=scf_prev_dir)
        scf_job.name = f"{self.name}_scf"
        jobs.append(scf_job)

        # Step 3: Band structure calculation
        self.propagate_custodian_to_maker(self.bands_maker)
        bands_job = self.bands_maker.make(
            scf_job.output.structure,
            prev_dir=scf_job.output.dir_name,
        )
        bands_job.name = f"{self.name}_bands"
        jobs.append(bands_job)

        # Step 4: Analysis and plotting
        analysis_job = analyze_band_structure(
            bands_output=bands_job.output,
            scf_output=scf_job.output,
            plot_bands=self.plot_bands,
            energy_range=self.energy_range,
        )
        analysis_job.name = f"{self.name}_analysis"
        jobs.append(analysis_job)

        return Flow(jobs, output=analysis_job.output, name=self.name)
