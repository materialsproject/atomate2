"""Phonon plotting and analysis utilities for SIESTA phonon calculations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jobflow import job

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


@job
def plot_phonon_band_structure(
    phonon_doc: dict[str, Any],
    output_dir: str | Path = ".",
    filename: str = "phonon_bands.png",
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 300,
) -> dict[str, str]:
    """
    Plot phonon band structure along high-symmetry path.

    Parameters
    ----------
    phonon_doc : dict
        Phonon calculation results from PhononDocument
    output_dir : str | Path
        Directory to save plots
    filename : str
        Output filename for band structure plot
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Resolution of saved figure

    Returns
    -------
    dict
        Dictionary with paths to generated files
    """
    try:
        import matplotlib.pyplot as plt
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError as e:
        logger.error("matplotlib or phonopy not available for plotting")
        raise ImportError("Install matplotlib and phonopy for plotting") from e

    logger.info("Plotting phonon band structure")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract phonopy settings
    settings = phonon_doc.get("phonopy_settings", {})  # noqa: F841
    structure = phonon_doc["structure"]

    # Recreate phonopy object
    phonopy_structure = PhonopyAtoms(
        symbols=[str(s) for s in structure.species],
        cell=structure.lattice.matrix,
        scaled_positions=structure.frac_coords,
    )

    phonon = Phonopy(
        phonopy_structure,
        supercell_matrix=phonon_doc["supercell_matrix"],
        symprec=phonon_doc["symprec"],
    )

    # Set force constants
    force_constants = np.array(phonon_doc["force_constants"])
    phonon.force_constants = force_constants

    # Auto-generate band structure path using seekpath
    try:
        import seekpath

        cell = (
            structure.lattice.matrix,
            structure.frac_coords,
            [s.Z for s in structure.species],
        )
        path_data = seekpath.get_path(cell)

        # Extract path
        path = []
        labels = []
        for segment in path_data["path"]:
            for point_name in segment:
                if point_name not in labels:
                    labels.append(point_name)
                    path.append(path_data["point_coords"][point_name])

        # Create band structure
        bands = []
        distances = []
        special_points = [0]

        npoints = 51  # Points per segment
        current_distance = 0

        for i in range(len(path) - 1):
            q_start = np.array(path[i])
            q_end = np.array(path[i + 1])

            segment_qs = [
                q_start + t * (q_end - q_start) for t in np.linspace(0, 1, npoints)
            ]

            # Calculate frequencies for this segment
            freqs_segment = []
            for q in segment_qs:
                phonon.run_qpoints([q])
                freqs = phonon.get_qpoints_dict()["frequencies"][0]
                freqs_segment.append(freqs)

            # Calculate distances
            if i == 0:
                segment_distances = np.linspace(0, 1, npoints)
            else:
                segment_distances = np.linspace(0, 1, npoints) + current_distance

            bands.extend(freqs_segment)
            distances.extend(segment_distances)
            current_distance = segment_distances[-1]
            special_points.append(current_distance)

        # Plot band structure
        _fig, ax = plt.subplots(figsize=figsize)

        bands_array = np.array(bands)
        for band_idx in range(bands_array.shape[1]):
            ax.plot(distances, bands_array[:, band_idx], "b-", linewidth=1)

        # Add vertical lines at special points
        for sp in special_points:
            ax.axvline(x=sp, color="k", linewidth=0.5, linestyle="--", alpha=0.5)

        # Format plot
        ax.set_xlim(distances[0], distances[-1])
        ax.set_xticks(special_points)
        ax.set_xticklabels([label.replace("GAMMA", "Γ") for label in labels])
        ax.set_ylabel("Frequency (THz)", fontsize=12)
        ax.set_xlabel("Wave vector", fontsize=12)
        ax.set_title(
            f"Phonon Band Structure - {structure.composition.formula}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="r", linewidth=0.8, linestyle="-", alpha=0.7)

        plt.tight_layout()

        # Save figure
        band_file = output_dir / filename
        plt.savefig(band_file, dpi=dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Phonon band structure saved to {band_file}")
        return {"band_structure_plot": str(band_file)}

    except ImportError:
        logger.warning("seekpath not available, skipping band structure plot")
        return {"band_structure_plot": "seekpath_not_available"}


@job
def plot_phonon_dos(
    phonon_doc: dict[str, Any],
    output_dir: str | Path = ".",
    filename: str = "phonon_dos.png",
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 300,
) -> dict[str, str]:
    """
    Plot phonon density of states.

    Parameters
    ----------
    phonon_doc : dict
        Phonon calculation results
    output_dir : str | Path
        Directory to save plots
    filename : str
        Output filename
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Resolution

    Returns
    -------
    dict
        Paths to generated files
    """
    try:
        import matplotlib.pyplot as plt
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError as e:
        raise ImportError("Install matplotlib and phonopy for plotting") from e

    logger.info("Plotting phonon DOS")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    structure = phonon_doc["structure"]

    # Recreate phonopy object
    phonopy_structure = PhonopyAtoms(
        symbols=[str(s) for s in structure.species],
        cell=structure.lattice.matrix,
        scaled_positions=structure.frac_coords,
    )

    phonon = Phonopy(
        phonopy_structure,
        supercell_matrix=phonon_doc["supercell_matrix"],
        symprec=phonon_doc["symprec"],
    )

    force_constants = np.array(phonon_doc["force_constants"])
    phonon.force_constants = force_constants

    # Calculate DOS
    mesh = phonon_doc.get("mesh", (50, 50, 50))
    phonon.run_mesh(mesh)
    phonon.run_total_dos()

    dos_dict = phonon.get_total_dos_dict()
    frequencies = dos_dict["frequency_points"]
    dos = dos_dict["total_dos"]

    # Plot DOS
    _fig, ax = plt.subplots(figsize=figsize)

    ax.plot(frequencies, dos, "b-", linewidth=2)
    ax.fill_between(frequencies, 0, dos, alpha=0.3)

    ax.set_xlabel("Frequency (THz)", fontsize=12)
    ax.set_ylabel("Density of States", fontsize=12)
    ax.set_title(
        f"Phonon DOS - {structure.composition.formula}", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color="r", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlim(frequencies[0], frequencies[-1])

    plt.tight_layout()

    dos_file = output_dir / filename
    plt.savefig(dos_file, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Phonon DOS saved to {dos_file}")
    return {"dos_plot": str(dos_file)}


@job
def plot_thermal_properties(
    phonon_doc: dict[str, Any],
    output_dir: str | Path = ".",
    filename: str = "thermal_properties.png",
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 300,
) -> dict[str, str]:
    """
    Plot thermal properties (Cv, entropy, free energy) vs temperature.

    Parameters
    ----------
    phonon_doc : dict
        Phonon calculation results
    output_dir : str | Path
        Directory to save plots
    filename : str
        Output filename
    figsize : tuple
        Figure size
    dpi : int
        Resolution

    Returns
    -------
    dict
        Paths to generated files
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Install matplotlib for plotting") from e

    if "thermal_properties" not in phonon_doc:
        logger.warning("No thermal properties found in phonon_doc")
        return {"thermal_plot": "not_available"}

    logger.info("Plotting thermal properties")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thermal = phonon_doc["thermal_properties"]
    temps = thermal["temperatures"]
    cv = thermal["heat_capacity"]
    entropy = thermal["entropy"]
    free_energy = thermal["free_energy"]

    # Create 3-panel plot
    _fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Heat capacity
    axes[0].plot(temps, cv, "r-", linewidth=2)
    axes[0].set_ylabel("Cv (eV/K)", fontsize=11)
    axes[0].set_title(
        f"Thermal Properties - {phonon_doc['structure'].composition.formula}",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)

    # Entropy
    axes[1].plot(temps, entropy, "g-", linewidth=2)
    axes[1].set_ylabel("Entropy (eV/K)", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Free energy
    axes[2].plot(temps, free_energy, "b-", linewidth=2)
    axes[2].set_xlabel("Temperature (K)", fontsize=12)
    axes[2].set_ylabel("Free Energy (eV)", fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    thermal_file = output_dir / filename
    plt.savefig(thermal_file, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Thermal properties plot saved to {thermal_file}")
    return {"thermal_plot": str(thermal_file)}


@job
def write_phonon_summary(
    phonon_doc: dict[str, Any],
    output_dir: str | Path = ".",
    filename: str = "phonon_summary.txt",
) -> dict[str, str]:
    """
    Write comprehensive phonon results to text file.

    Parameters
    ----------
    phonon_doc : dict
        Phonon calculation results
    output_dir : str | Path
        Directory to save file
    filename : str
        Output filename

    Returns
    -------
    dict
        Path to generated file
    """
    from datetime import datetime

    logger.info("Writing phonon summary to text file")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    structure = phonon_doc["structure"]
    summary_file = output_dir / filename

    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PHONON CALCULATION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Structure information
        f.write("-" * 80 + "\n")
        f.write("STRUCTURE INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Formula: {structure.composition.formula}\n")
        f.write(f"Reduced Formula: {structure.composition.reduced_formula}\n")
        f.write(f"Number of atoms: {len(structure)}\n")
        f.write(f"Space group: {structure.get_space_group_info()}\n")
        f.write("Lattice parameters:\n")
        f.write(f"  a = {structure.lattice.a:.6f} Å\n")
        f.write(f"  b = {structure.lattice.b:.6f} Å\n")
        f.write(f"  c = {structure.lattice.c:.6f} Å\n")
        f.write(f"  α = {structure.lattice.alpha:.3f}°\n")  # noqa: RUF001
        f.write(f"  β = {structure.lattice.beta:.3f}°\n")
        f.write(f"  γ = {structure.lattice.gamma:.3f}°\n")  # noqa: RUF001
        f.write(f"  Volume = {structure.lattice.volume:.6f} Å³\n\n")

        # Calculation parameters
        f.write("-" * 80 + "\n")
        f.write("CALCULATION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Supercell matrix: {phonon_doc['supercell_matrix']}\n")
        f.write(f"Displacement: {phonon_doc['displacement']} Å\n")
        f.write(f"Symmetry precision: {phonon_doc['symprec']}\n")
        f.write(f"Number of displacements: {phonon_doc['n_displacements']}\n\n")

        # Phonon frequencies
        f.write("-" * 80 + "\n")
        f.write("PHONON FREQUENCIES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Minimum frequency: {phonon_doc['min_frequency']:.6f} THz\n")
        f.write(f"Maximum frequency: {phonon_doc['max_frequency']:.6f} THz\n")
        f.write(
            f"Frequency range: {phonon_doc['max_frequency'] - phonon_doc['min_frequency']:.6f} THz\n"
        )
        f.write(
            f"Imaginary frequencies: {'Yes' if phonon_doc['has_imaginary_frequencies'] else 'No'}\n\n"
        )

        if phonon_doc["has_imaginary_frequencies"]:
            f.write("⚠️  WARNING: Imaginary frequencies detected!\n")
            f.write("   This may indicate:\n")
            f.write("   - Structural instability\n")
            f.write("   - Incomplete structure relaxation\n")
            f.write("   - Insufficient force convergence\n\n")

        # Thermal properties
        if "thermal_properties" in phonon_doc:
            thermal = phonon_doc["thermal_properties"]
            f.write("-" * 80 + "\n")
            f.write("THERMAL PROPERTIES\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Temperature range: {min(thermal['temperatures']):.1f} - {max(thermal['temperatures']):.1f} K\n\n"
            )

            f.write(
                f"{'T (K)':>10} {'Cv (eV/K)':>15} {'S (eV/K)':>15} {'F (eV)':>15}\n"
            )
            f.write("-" * 60 + "\n")

            # Show every 10th point or key temperatures
            key_indices = []
            for target_t in [0, 100, 200, 300, 500, 1000]:
                idx = min(
                    range(len(thermal["temperatures"])),
                    key=lambda i: abs(thermal["temperatures"][i] - target_t),
                )
                if abs(thermal["temperatures"][idx] - target_t) < 20:
                    key_indices.append(idx)

            for idx in sorted(set(key_indices)):
                f.write(
                    f"{thermal['temperatures'][idx]:>10.1f} "
                    f"{thermal['heat_capacity'][idx]:>15.8f} "
                    f"{thermal['entropy'][idx]:>15.8f} "
                    f"{thermal['free_energy'][idx]:>15.6f}\n"
                )
            f.write("\n")

        # Convergence notes
        f.write("-" * 80 + "\n")
        f.write("CONVERGENCE RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        f.write("For production calculations, verify convergence of:\n\n")
        f.write("1. Supercell size:\n")
        f.write("   - Test different supercell sizes (2×2×2, 3×3×3, 4×4×4)\n")  # noqa: RUF001
        f.write("   - Criterion: Frequencies converged within 0.03 THz (1 cm⁻¹)\n\n")
        f.write("2. Displacement distance:\n")
        f.write("   - Test: 0.005, 0.01, 0.02 Å\n")
        f.write("   - Check linearity of force-displacement relation\n\n")
        f.write("3. SIESTA parameters:\n")
        f.write("   - K-points: Dense mesh (8×8×8 or higher)\n")  # noqa: RUF001
        f.write("   - Mesh cutoff: 400-500 Ry for accurate forces\n")
        f.write("   - Basis: DZP minimum, TZP for high accuracy\n\n")

        # Add standard footer
        from atomate2.siesta.utils.text_output import get_standard_footer

        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Analysis type": "Phonon calculation",
                    "Formula": structure.composition.reduced_formula,
                    "Number of atoms": str(len(structure)),
                },
            )
        )

    logger.info(f"Phonon summary written to {summary_file}")
    return {"summary_file": str(summary_file)}
