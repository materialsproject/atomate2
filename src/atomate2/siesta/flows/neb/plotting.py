"""Plotting utilities for NEB results."""

from __future__ import annotations

import logging

from jobflow import job

logger = logging.getLogger(__name__)


def _write_ase_neb_info(
    energies: list,
    energies_rel: list,
    reaction_coord: list,
    forces: list,
    max_forces: list,
    structures: list,
    iteration: int = 0,
    converged: bool = False,
    max_neb_force: float = None,
) -> None:
    """
    Write comprehensive NEB information to file.

    Parameters
    ----------
    energies : list
        Absolute energies for each image (eV).
    energies_rel : list
        Relative energies for each image (meV).
    reaction_coord : list
        Reaction coordinate values.
    forces : list
        Force arrays for each image.
    max_forces : list
        Maximum force magnitude for each image (eV/Å).
    structures : list
        Structures for each image.
    iteration : int
        Current iteration number.
    converged : bool
        Whether NEB optimization converged.
    max_neb_force : float
        Maximum NEB force at final iteration (eV/Å).
    """
    import numpy as np

    with open("ase_neb_info.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ASE NEB Calculation Summary (Iterative Optimization)\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("Overall NEB Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total iterations: {iteration + 1}\n")
        f.write(f"Number of images: {len(structures)}\n")
        f.write(f"Converged: {converged}\n")
        if max_neb_force is not None:
            f.write(f"Final max NEB force: {max_neb_force:.4f} eV/Å\n")
        f.write(
            f"Activation energy: {max(energies_rel):.2f} meV ({max(energies_rel) / 1000:.4f} eV)\n"
        )
        f.write(f"Transition state at image: {energies_rel.index(max(energies_rel))}\n")
        if any(mf is not None for mf in max_forces):
            valid_forces = [mf for mf in max_forces if mf is not None]
            if valid_forces:
                f.write(f"Maximum force (all images): {max(valid_forces):.4f} eV/Å\n")
        f.write("\n")

        # Per-image information
        f.write("Per-Image Information:\n")
        f.write("=" * 80 + "\n")
        for i, (rc, e, e_rel, mf) in enumerate(
            zip(reaction_coord, energies, energies_rel, max_forces, strict=False)
        ):
            f.write(f"\nImage {i}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Reaction coordinate: {rc:.6f}\n")
            f.write(f"  Energy (absolute):   {e:.6f} eV\n")
            f.write(f"  Energy (relative):   {e_rel:.2f} meV\n")
            if mf is not None:
                f.write(f"  Max force:           {mf:.4f} eV/Å\n")

            # Structure info
            struct = structures[i]
            f.write(f"  Composition:         {struct.composition}\n")
            f.write(f"  Number of atoms:     {len(struct)}\n")

            # Force components if available
            if forces[i] is not None:
                f.write("\n  Forces on atoms (eV/Å):\n")
                for j, force in enumerate(forces[i]):
                    force_mag = np.linalg.norm(force)
                    f.write(f"    Atom {j:3d} ({struct.species[j]:>2s}): ")
                    f.write(
                        f"Fx={force[0]:8.4f}  Fy={force[1]:8.4f}  Fz={force[2]:8.4f}  "
                    )
                    f.write(f"|F|={force_mag:8.4f}\n")

        f.write("\n")

        # Add standard footer
        from atomate2.siesta.utils.text_output import get_standard_footer

        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Analysis type": "NEB (Nudged Elastic Band)",
                    "Number of images": str(len(structures)),
                    "Converged": str(converged),
                    "Total iterations": str(iteration + 1),
                },
            )
        )

    logger.info("Wrote ASE NEB information to ase_neb_info.txt")


@job
def plot_ase_neb_results(neb_data: dict) -> dict:
    """
    Plot ASE NEB energy profile from optimization results.

    Parameters
    ----------
    neb_data : dict
        Dictionary containing reaction_coordinate, energies_relative_meV, etc.

    Returns
    -------
    dict
        Dictionary with activation energy and plot filename.
    """
    import matplotlib.pyplot as plt

    logger.info("plot_ase_neb_results()")

    # Extract data
    reaction_coord = neb_data["reaction_coordinate"]
    energies_rel = neb_data["energies_relative_meV"]
    activation_energy = neb_data["activation_energy_meV"]

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(reaction_coord, energies_rel, "o-", linewidth=2, markersize=8)
    plt.xlabel("Reaction Coordinate", fontsize=12)
    plt.ylabel("Energy (meV)", fontsize=12)
    plt.title("NEB Energy Profile", fontsize=14)
    plt.grid(True, alpha=0.3)  # noqa: FBT003

    # Add activation energy annotation
    plt.text(
        0.95,
        0.95,
        f"Ea = {activation_energy:.1f} meV",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Save plot
    plot_file = "neb_energy_profile.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved NEB energy profile to {plot_file}")
    logger.info(f"Activation energy: {activation_energy:.1f} meV")

    return {
        "activation_energy": activation_energy,
        "plot_file": plot_file,
        "num_images": neb_data["n_images"],
    }


def _write_neb_summary(
    all_iterations_df: list,
    final_df,
    barrier: float,
    ts_coord: float,
    ts_energy: float,
    ts_index: int,
    force_tol: float,
) -> None:
    """
    Write comprehensive NEB summary to text file.

    Parameters
    ----------
    all_iterations_df : list
        List of DataFrames, one per NEB iteration.
    final_df : DataFrame
        Final iteration DataFrame with all NEB data.
    barrier : float
        Activation energy barrier (meV).
    ts_coord : float
        Transition state reaction coordinate.
    ts_energy : float
        Transition state energy (meV).
    ts_index : int
        Index of transition state image.
    force_tol : float
        Force convergence tolerance (eV/Å).
    """
    import numpy as np

    from atomate2.siesta.utils.text_output import get_standard_footer

    summary_file = "neb_summary.txt"

    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("NEB (Nudged Elastic Band) Calculation Summary\n")
        f.write("=" * 80 + "\n\n")

        # ===== CONVERGENCE INFORMATION =====
        f.write("CONVERGENCE INFORMATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total iterations:           {len(all_iterations_df)}\n")
        f.write(f"Number of images:           {len(final_df)}\n")

        # Check convergence (all max forces below tolerance)
        max_force_final = final_df["F-max(atom)"].max()
        converged = max_force_final < force_tol
        f.write(f"Converged:                  {converged}\n")
        f.write(f"Force tolerance:            {force_tol:.4f} eV/Å\n")
        f.write(f"Final max force:            {max_force_final:.4f} eV/Å\n")
        f.write("\n")

        # ===== ENERGY BARRIER =====
        f.write("ENERGY BARRIER:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Activation energy:          {barrier:.2f} meV ({barrier / 1000:.6f} eV)\n"
        )
        f.write(f"Transition state image:     {ts_index}\n")
        f.write(f"TS reaction coordinate:     {ts_coord:.6f}\n")
        f.write(
            f"TS energy:                  {ts_energy:.2f} meV ({ts_energy / 1000:.6f} eV)\n"
        )

        # Get TS curvature if available
        if "Curvature" in final_df.columns:
            ts_curvature = final_df["Curvature"].iloc[ts_index]
            f.write(f"TS curvature:               {ts_curvature:.6e} eV/Å²\n")
        f.write("\n")

        # ===== CONVERGENCE EVOLUTION =====
        f.write("CONVERGENCE EVOLUTION:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Iter':<6} {'Barrier (meV)':<15} {'Max Force (eV/Å)':<20} {'Status':<15}\n"
        )
        f.write("-" * 80 + "\n")

        # Reference energy from first iteration
        initial_energy = all_iterations_df[0]["Energy"].values[0]

        for i, df in enumerate(all_iterations_df):
            # Calculate barrier for this iteration (use consistent reference)
            e_values = (df["Energy"].values - initial_energy) * 1000
            iter_barrier = e_values.max()
            iter_max_force = df["F-max(atom)"].max()
            status = "Converged" if iter_max_force < force_tol else "Running"

            f.write(
                f"{i + 1:<6} {iter_barrier:<15.2f} {iter_max_force:<20.4f} {status:<15}\n"
            )
        f.write("\n")

        # ===== PER-IMAGE DETAILS (Final Iteration) =====
        f.write("PER-IMAGE DETAILS (Final Iteration):\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'Image':<7} {'RC':<12} {'E (meV)':<12} {'E-diff (meV)':<15} "
            f"{'F-max (eV/Å)':<15} {'Curvature':<15}\n"
        )
        f.write("-" * 80 + "\n")

        reaction_coords = final_df["reaction-coordinate"].values
        energies = final_df["Energy"].values
        energies_rel = (energies - energies[0]) * 1000  # meV relative to first image
        e_diffs = (
            final_df["E-diff"].values * 1000 if "E-diff" in final_df.columns else None
        )
        max_forces = final_df["F-max(atom)"].values
        curvatures = (
            final_df["Curvature"].values if "Curvature" in final_df.columns else None
        )

        for i in range(len(final_df)):
            img_marker = " *TS*" if i == ts_index else ""
            f.write(f"{i:<7} {reaction_coords[i]:<12.6f} {energies_rel[i]:<12.2f} ")

            if e_diffs is not None:
                f.write(f"{e_diffs[i]:<15.2f} ")
            else:
                f.write(f"{'N/A':<15} ")

            f.write(f"{max_forces[i]:<15.4f} ")

            if curvatures is not None:
                f.write(f"{curvatures[i]:<15.6e}")
            else:
                f.write(f"{'N/A':<15}")

            f.write(f"{img_marker}\n")

        f.write("\n")

        # ===== STATISTICS =====
        f.write("STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Initial energy:             {energies[0]:.6f} eV\n")
        f.write(f"Final energy:               {energies[-1]:.6f} eV\n")
        f.write(
            f"Reaction energy:            {(energies[-1] - energies[0]) * 1000:.2f} meV\n"
        )
        f.write(
            f"Average spacing (RC):       {np.mean(np.diff(reaction_coords)):.6f}\n"
        )
        f.write(f"Min spacing (RC):           {np.min(np.diff(reaction_coords)):.6f}\n")
        f.write(f"Max spacing (RC):           {np.max(np.diff(reaction_coords)):.6f}\n")
        f.write("\n")

        # ===== CONVERGENCE ASSESSMENT =====
        f.write("CONVERGENCE ASSESSMENT:\n")
        f.write("-" * 80 + "\n")

        # Check individual image convergence
        converged_images = (max_forces < force_tol).sum()
        f.write(f"Converged images:           {converged_images}/{len(final_df)}\n")
        f.write(f"Images above tolerance:     {len(final_df) - converged_images}\n")

        # Find images with highest forces
        high_force_indices = np.argsort(max_forces)[-3:][::-1]
        f.write("\nImages with highest forces:\n")
        f.writelines(
            f"  Image {idx}: F-max = {max_forces[idx]:.4f} eV/Å, "
            f"RC = {reaction_coords[idx]:.6f}\n"
            for idx in high_force_indices
        )

        f.write("\n")

        # ===== BARRIER CONVERGENCE TREND =====
        if len(all_iterations_df) > 1:
            f.write("BARRIER CONVERGENCE TREND:\n")
            f.write("-" * 80 + "\n")

            barriers = []
            for df in all_iterations_df:
                e_values = (df["Energy"].values - initial_energy) * 1000
                barriers.append(e_values.max())

            # Calculate convergence metrics
            barrier_change = abs(barriers[-1] - barriers[-2])
            max_barrier_change = max(abs(np.diff(barriers)))
            avg_barrier_change = np.mean(abs(np.diff(barriers)))

            f.write(
                f"Last barrier change:        {barrier_change:.2f} meV "
                f"({barrier_change / barriers[-1] * 100:.2f}%)\n"
            )
            f.write(f"Max barrier change:         {max_barrier_change:.2f} meV\n")
            f.write(f"Avg barrier change:         {avg_barrier_change:.2f} meV\n")

            # Estimate if converged
            barrier_converged = barrier_change < 1.0  # 1 meV tolerance
            f.write(
                f"Barrier converged (<1meV):  {barrier_converged} "
                f"({barrier_change:.2f} meV)\n"
            )
            f.write("\n")

        # ===== OUTPUT FILES =====
        f.write("OUTPUT FILES:\n")
        f.write("-" * 80 + "\n")
        f.write(
            "  neb_energy_profile.png         - Energy profile with all iterations\n"
        )
        f.write("  neb_force_convergence.png      - Max forces across iterations\n")
        f.write("  neb_barrier_convergence.png    - Barrier height evolution\n")
        f.write("  neb_curvature_evolution.png    - Curvature evolution\n")
        f.write("  neb_summary.txt                - This file\n")
        f.write("\n")

        # Add standard footer
        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Analysis type": "NEB (Nudged Elastic Band)",
                    "Number of images": str(len(final_df)),
                    "Number of iterations": str(len(all_iterations_df)),
                    "Converged": str(converged),
                    "Activation energy": f"{barrier:.2f} meV",
                },
            )
        )

    logger.info(f"✓ NEB summary written to {summary_file}")


@job
def plot_neb_results(
    prev_dir: str,
    plot_force: bool = True,
    plot_barrier: bool = True,
    plot_curvature: bool = True,
    force_tol: float = 0.05,
    interp_points: int = 100,
) -> dict:
    """
    Plot NEB energy profile and convergence diagnostics from NEB.results file.

    Parameters
    ----------
    prev_dir : str
        Directory containing NEB.results file from NEB calculation.
    plot_force : bool
        Generate force convergence plot (default: True).
    plot_barrier : bool
        Generate barrier height convergence plot (default: True).
    plot_curvature : bool
        Generate curvature evolution plot (default: True).
    force_tol : float
        Force tolerance for convergence plot in eV/Å (default: 0.05).
    interp_points : int
        Number of points for cubic spline interpolation (default: 100).

    Returns
    -------
    dict
        Dictionary with activation energy, plot filenames, and convergence info.
    """
    import gzip
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.interpolate import CubicSpline

    logger.info("plot_neb_results()")
    prev_path = Path(prev_dir)

    # Check for NEB.results in both regular and compressed locations
    neb_results_file = prev_path / "NEB.results"
    neb_results_gz = prev_path / "siesta_compressed" / "NEB.results.gz"

    if neb_results_file.exists():
        file_to_read = neb_results_file
        use_gzip = False
    elif neb_results_gz.exists():
        file_to_read = neb_results_gz
        use_gzip = True
    else:
        logger.warning(
            f"NEB.results not found in {prev_dir} or {prev_dir}/siesta_compressed/"
        )
        return {"activation_energy": None, "plot_file": None}

    # Parse NEB.results file with full column data
    # Expected columns: Image, reaction-coordinate, Energy, E-diff, Curvature, F-max(atom)
    column_names = [
        "Image",
        "reaction-coordinate",
        "Energy",
        "E-diff",
        "Curvature",
        "F-max(atom)",
    ]

    all_iterations_df: list[pd.DataFrame] = []
    current_data: list[list[str]] = []

    # Read file (handle both plain and gzipped)
    if use_gzip:
        with gzip.open(file_to_read, "rt") as f:
            lines = f.readlines()
    else:
        with open(file_to_read) as f:
            lines = f.readlines()

    # Parse data blocks
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            continue  # Skip header lines
        if not line:  # Empty line marks end of iteration
            if current_data:
                try:
                    df = pd.DataFrame(current_data, columns=column_names)
                    df = df.apply(pd.to_numeric, errors="coerce").dropna()
                    if not df.empty:
                        all_iterations_df.append(df)
                except Exception as e:
                    logger.warning(f"Failed to parse iteration: {e}")
                current_data = []
        else:
            parts = line.split()
            if len(parts) >= 6:  # Full data line
                current_data.append(parts[:6])

    # Don't forget the last iteration
    if current_data:
        try:
            df = pd.DataFrame(current_data, columns=column_names)
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if not df.empty:
                all_iterations_df.append(df)
        except Exception as e:
            logger.warning(f"Failed to parse last iteration: {e}")

    if not all_iterations_df:
        logger.warning("Could not parse NEB.results data")
        return {"activation_energy": None, "plot_file": None}

    logger.info(
        f"Found {len(all_iterations_df)} NEB iterations with {len(all_iterations_df[-1])} images each"
    )

    # Get final iteration
    final_df = all_iterations_df[-1]
    reaction_coords = final_df["reaction-coordinate"].values
    energies = final_df["Energy"].values
    energies_rel = (energies - energies[0]) * 1000  # Convert to meV

    # Calculate barrier and transition state
    barrier = energies_rel.max()
    ts_index = energies_rel.argmax()
    ts_coord = reaction_coords[ts_index]
    ts_energy = energies_rel[ts_index]

    # 1. Create energy profile plot with cubic spline interpolation
    plt.figure(figsize=(10, 6))

    # Plot all iterations in gray (use same reference energy)
    initial_energy_ref = energies[0]
    for df in all_iterations_df[:-1]:
        rc = df["reaction-coordinate"].values
        e_iter = (df["Energy"].values - initial_energy_ref) * 1000
        plt.plot(rc, e_iter, "gray", alpha=0.3, linewidth=1)

    # Plot final iteration with interpolation
    cs = CubicSpline(reaction_coords, energies_rel)
    x_interp = np.linspace(reaction_coords.min(), reaction_coords.max(), interp_points)
    y_interp = cs(x_interp)

    plt.plot(
        x_interp, y_interp, "b-", linewidth=2, label="Final NEB Path (Interpolated)"
    )
    plt.plot(reaction_coords, energies_rel, "bo", markersize=6, label="NEB Images")
    plt.plot(
        ts_coord,
        ts_energy,
        "ro",
        markersize=10,
        label=f"Transition State (E = {ts_energy:.1f} meV)",
    )

    plt.xlabel("Reaction Coordinate", fontsize=12)
    plt.ylabel("Energy (meV)", fontsize=12)
    plt.title(f"NEB Energy Profile\nEnergy Barrier: {barrier:.1f} meV", fontsize=14)
    plt.grid(True, alpha=0.3)  # noqa: FBT003
    plt.legend()
    plt.tight_layout()

    plot_file = "neb_energy_profile.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Energy profile plot saved: {plot_file}")

    # Initialize results dictionary
    additional_plots: list[str] = []
    results = {
        "activation_energy": float(barrier),
        "ts_energy": float(ts_energy),
        "ts_coordinate": float(ts_coord),
        "num_images": len(energies),
        "num_iterations": len(all_iterations_df),
        "plot_file": plot_file,
        "additional_plots": additional_plots,
    }

    # 2. Force convergence plot
    if plot_force and "F-max(atom)" in final_df.columns:
        plt.figure(figsize=(10, 6))

        # Plot all iterations
        for df in all_iterations_df[:-1]:
            plt.plot(
                df["reaction-coordinate"],
                df["F-max(atom)"],
                "gray",
                alpha=0.3,
                linewidth=1,
            )

        # Plot final iteration
        plt.plot(
            final_df["reaction-coordinate"],
            final_df["F-max(atom)"],
            "b-o",
            linewidth=2,
            markersize=6,
            label="Final Iteration",
        )

        # Add force tolerance line
        plt.axhline(
            y=force_tol,
            color="red",
            linestyle="--",
            label=f"Force Tolerance ({force_tol} eV/Å)",
        )

        plt.xlabel("Reaction Coordinate", fontsize=12)
        plt.ylabel("Max Force (eV/Å)", fontsize=12)
        plt.title("Force Convergence Across Iterations", fontsize=14)
        plt.grid(True, alpha=0.3)  # noqa: FBT003
        plt.legend()
        plt.tight_layout()

        force_plot = "neb_force_convergence.png"
        plt.savefig(force_plot, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ Force convergence plot saved: {force_plot}")
        additional_plots.append(force_plot)

    # 3. Barrier convergence plot
    if plot_barrier:
        barriers = []
        initial_energy = energies[0]  # Reference energy for all iterations
        for df in all_iterations_df:
            e_values = (df["Energy"].values - initial_energy) * 1000
            barriers.append(float(e_values.max()))

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(barriers) + 1), barriers, "b-o", linewidth=2, markersize=6
        )
        plt.axhline(
            y=barrier,
            color="red",
            linestyle="--",
            label=f"Final Barrier ({barrier:.1f} meV)",
        )

        plt.xlabel("Iteration Number", fontsize=12)
        plt.ylabel("Energy Barrier (meV)", fontsize=12)
        plt.title("Barrier Height Convergence", fontsize=14)
        plt.grid(True, alpha=0.3)  # noqa: FBT003
        plt.legend()
        plt.tight_layout()

        barrier_plot = "neb_barrier_convergence.png"
        plt.savefig(barrier_plot, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ Barrier convergence plot saved: {barrier_plot}")
        additional_plots.append(barrier_plot)

    # 4. Curvature evolution plot
    if plot_curvature and "Curvature" in final_df.columns:
        plt.figure(figsize=(10, 6))

        # Plot all iterations
        for df in all_iterations_df[:-1]:
            plt.plot(
                df["reaction-coordinate"],
                df["Curvature"],
                "gray",
                alpha=0.3,
                linewidth=1,
            )

        # Plot final iteration
        plt.plot(
            final_df["reaction-coordinate"],
            final_df["Curvature"],
            "b-o",
            linewidth=2,
            markersize=6,
            label="Final Iteration",
        )

        # Mark transition state
        ts_curvature = final_df["Curvature"].iloc[ts_index]
        plt.plot(
            ts_coord,
            ts_curvature,
            "ro",
            markersize=10,
            label=f"TS Curvature ({ts_curvature:.2e})",
        )

        plt.xlabel("Reaction Coordinate", fontsize=12)
        plt.ylabel("Curvature (eV/Å²)", fontsize=12)
        plt.title("Curvature Evolution Across Iterations", fontsize=14)
        plt.grid(True, alpha=0.3)  # noqa: FBT003
        plt.legend()
        plt.tight_layout()

        curvature_plot = "neb_curvature_evolution.png"
        plt.savefig(curvature_plot, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ Curvature evolution plot saved: {curvature_plot}")
        additional_plots.append(curvature_plot)

    # Log summary
    logger.info("NEB Analysis Summary:")
    logger.info(f"  Activation energy: {barrier:.2f} meV")
    logger.info(f"  Transition state: RC = {ts_coord:.3f}, E = {ts_energy:.2f} meV")
    logger.info(f"  Number of images: {len(energies)}")
    logger.info(f"  Number of iterations: {len(all_iterations_df)}")

    # Write comprehensive text summary
    _write_neb_summary(
        all_iterations_df=all_iterations_df,
        final_df=final_df,
        barrier=barrier,
        ts_coord=ts_coord,
        ts_energy=ts_energy,
        ts_index=ts_index,
        force_tol=force_tol,
    )

    return results


@job
def print_dir(d):
    """Debug utility to log directory information."""
    logger.debug(f"Directory contents: {d}")
