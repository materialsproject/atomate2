"""Utility functions for convergence workflows."""

from __future__ import annotations

import logging
from typing import Any

from jobflow.core.job import job

from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@job
def collect_convergence_data(
    flow_results: dict[str, Any], job_metadata: list[dict], parameter_name: str
) -> dict[str, Any]:
    """
    Collect convergence data from flow results.

    Args:
        flow_results: Results dictionary from jobflow's run_locally
        job_metadata: List of dictionaries containing job names and UUIDs
        parameter_name: Name of the parameter being converged ('mesh_cutoff' or 'kpoints')

    Returns
    -------
        Dictionary with parameter values and corresponding energies
    """
    logger.info(f"Collecting {parameter_name} convergence data")
    data: dict[str, list] = {
        "parameters": [],
        "energies": [],
        "fermi_energies": [],
        "bandgaps": [],
        "max_forces": [],
        "mean_forces": [],
        "max_stress": [],
        "names": [],
    }

    for job_info in job_metadata:
        job_uuid = job_info["uuid"]
        job_name = job_info["name"]

        try:
            if job_uuid not in flow_results:
                logger.warning(
                    f"No results found for job {job_name} (UUID: {job_uuid})"
                )
                continue

            result = flow_results[job_uuid]

            # Handle both Response objects (normal mode) and dicts (dry_run mode)
            if isinstance(result, dict):
                # Dry_run mode returns dict, skip energy collection
                logger.debug(
                    f"Skipping {job_name} - dry_run mode (no energy available)"
                )
                continue
            # Normal mode: result is a Response object
            output = result.output
            energy = output.energy
            # Get Fermi energy if available (stored as 'efermi' in schema)
            fermi_energy = getattr(output, "efermi", None)
            bandgap = getattr(output, "bandgap", None)
            forces = getattr(output, "forces", None)
            stress = getattr(output, "stress", None)

            # Calculate force statistics
            max_force = None
            mean_force = None
            if forces is not None and len(forces) > 0:
                import numpy as np

                # forces is list of (x, y, z) tuples
                force_magnitudes = [
                    np.sqrt(f[0] ** 2 + f[1] ** 2 + f[2] ** 2) for f in forces
                ]
                max_force = max(force_magnitudes)
                mean_force = np.mean(force_magnitudes)

            # Calculate max stress component
            max_stress = None
            if stress is not None:
                import numpy as np

                # stress is 3x3 matrix
                stress_array = np.array(stress)
                max_stress = np.abs(stress_array).max()

            # Extract parameter value from job name
            # Handle both formats:
            # - "K-points Convergence-2x2x2" (from KpointsConvergenceFlowMaker)
            # - "Mesh-Kpoint Convergence - Stage 1 - Mesh 200Ry" (from MeshKpointConvergenceFlowMaker)
            param_value = job_name.split("-")[-1].strip()

            # If param_value starts with "Mesh " or "Kpoints ", extract just the value
            if param_value.startswith("Mesh "):
                param_value = param_value.replace("Mesh ", "")
            elif param_value.startswith("Kpoints "):
                param_value = param_value.replace("Kpoints ", "")

            data["names"].append(job_name)
            data["parameters"].append(param_value)
            data["energies"].append(energy)
            data["fermi_energies"].append(fermi_energy)
            data["bandgaps"].append(bandgap)
            data["max_forces"].append(max_force)
            data["mean_forces"].append(mean_force)
            data["max_stress"].append(max_stress)

            logger.debug(
                f"{job_name}: {param_value} -> E={energy} eV, Ef={fermi_energy} eV, "
                f"Gap={bandgap} eV, MaxF={max_force} eV/Å"
            )

        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.exception(f"Error processing job {job_name}: {e}")
            continue

    if not data["energies"]:
        logger.warning(f"No energies retrieved for {parameter_name} convergence")
    else:
        logger.info(
            f"Successfully collected {len(data['energies'])} data points for {parameter_name}"
        )

    return data


@job
def plot_convergence(
    convergence_data: dict[str, Any],
    parameter_name: str,
    output_file: str | None = None,
    verbosity: VerbosityLevel | int = VerbosityLevel.INFO,
) -> str | dict[str, Any]:
    """
    Plot convergence results.

    Creates individual plots for:
    - Total energy vs parameter
    - Energy differences (convergence)
    - Fermi energy vs parameter (if available)
    - Band gap vs parameter (if available)
    - Maximum forces vs parameter (if available)
    - Maximum stress vs parameter (if available)

    Args:
        convergence_data: Dictionary with all convergence data
        parameter_name: Name of parameter ('mesh_cutoff' or 'kpoints')
        output_file: Output filename base (default: auto-generated)
        verbosity: Verbosity level for console output

    Returns
    -------
        Dictionary with paths to all saved plot files
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Handle both VerbosityLevel enum and int
    verbosity_value = verbosity.value if hasattr(verbosity, "value") else verbosity

    if verbosity_value >= VerbosityLevel.INFO.value:
        console.print(f"[green]Plotting {parameter_name} convergence[/green]")

    parameters = convergence_data["parameters"]
    energies = np.array(convergence_data["energies"])

    # Check if we have any energies to plot (dry_run mode returns empty energies)
    if len(energies) == 0:
        if verbosity_value >= VerbosityLevel.WARNING.value:
            console.print(
                f"[yellow]No energies available for {parameter_name} convergence plot. "
                f"This is expected in dry_run mode.[/yellow]"
            )
        return {
            "convergence_data": convergence_data,
            "plot_file": None,
            "message": "No energies available (dry_run mode)",
        }

    # Convert parameters to numeric values for plotting
    if parameter_name == "mesh_cutoff":
        # Extract numeric value (e.g., "200Ry" -> 200)
        param_values = [
            float(p.replace("Ry", "").replace("eV", "")) for p in parameters
        ]
        xlabel = "Mesh Cutoff (Ry)"
    elif parameter_name == "kpoints":
        # For kpoints, extract first k-point value
        # Handles formats like "2x2x2", "[2, 2, 2]", or list objects
        param_values = []
        for p in parameters:
            if isinstance(p, str):
                # Handle "2x2x2" format
                if "x" in p:
                    kpts = [int(x) for x in p.replace("x", " ").split()]
                    param_values.append(kpts[0])
                # Handle "[2, 2, 2]" format
                elif "[" in p:
                    kpts = eval(p)
                    param_values.append(
                        kpts[0] if isinstance(kpts, (list, tuple)) else kpts
                    )
                else:
                    # Fallback: try to convert to int
                    param_values.append(int(p))
            elif isinstance(p, (list, tuple)):
                param_values.append(p[0])
            else:
                param_values.append(p)
        xlabel = "K-points (grid density)"
    else:
        param_values = list(range(len(parameters)))
        xlabel = parameter_name

    # Calculate energy differences from most converged (last point)
    energy_ref = energies[-1]
    energy_diff = (energies - energy_ref) * 1000  # Convert to meV

    # Get additional data arrays
    fermi_energies = convergence_data.get("fermi_energies", [])
    bandgaps = convergence_data.get("bandgaps", [])
    max_forces = convergence_data.get("max_forces", [])
    max_stress = convergence_data.get("max_stress", [])

    # Convert to numpy arrays (handling None values)
    fermi_energies_array = np.array(
        [ef if ef is not None else np.nan for ef in fermi_energies]
    )
    bandgaps_array = np.array([bg if bg is not None else np.nan for bg in bandgaps])
    max_forces_array = np.array([mf if mf is not None else np.nan for mf in max_forces])
    max_stress_array = np.array([ms if ms is not None else np.nan for ms in max_stress])

    # Check what data is available
    has_fermi_data = len(fermi_energies) > 0 and not np.all(
        np.isnan(fermi_energies_array)
    )
    has_bandgap_data = len(bandgaps) > 0 and not np.all(np.isnan(bandgaps_array))
    has_force_data = len(max_forces) > 0 and not np.all(np.isnan(max_forces_array))
    has_stress_data = len(max_stress) > 0 and not np.all(np.isnan(max_stress_array))

    # Determine base filename
    if output_file is None:
        base_name = f"convergence_{parameter_name}"
    else:
        base_name = output_file.replace(".png", "")

    plot_files = {}

    # Plot 1: Total Energy
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(param_values, energies, "o-", linewidth=2, markersize=8, color="#1f77b4")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Total Energy (eV)", fontsize=12)
    ax.set_title(
        f"Total Energy vs. {parameter_name.replace('_', ' ').title()}", fontsize=14
    )
    ax.grid(True, alpha=0.3)  # noqa: FBT003
    plt.tight_layout()
    energy_file = f"{base_name}_energy.png"
    plt.savefig(energy_file, dpi=150, bbox_inches="tight")
    plt.close()
    plot_files["energy"] = energy_file
    logger.info(f"Energy plot saved to {energy_file}")

    # Plot 2: Energy Differences (Convergence)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(param_values, energy_diff, "s-", linewidth=2, markersize=8, color="red")
    ax.axhline(y=1, color="green", linestyle="--", label="1 meV threshold", linewidth=2)
    ax.axhline(
        y=5, color="orange", linestyle="--", label="5 meV threshold", linewidth=2
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Energy Difference (meV)", fontsize=12)
    ax.set_title(
        f"Convergence: ΔE vs. {parameter_name.replace('_', ' ').title()}", fontsize=14
    )
    ax.grid(True, alpha=0.3)  # noqa: FBT003
    ax.legend(fontsize=11)
    plt.tight_layout()
    convergence_file = f"{base_name}_convergence.png"
    plt.savefig(convergence_file, dpi=150, bbox_inches="tight")
    plt.close()
    plot_files["convergence"] = convergence_file
    logger.info(f"Convergence plot saved to {convergence_file}")

    # Plot 3: Fermi Energy (if available)
    if has_fermi_data:
        valid_indices = ~np.isnan(fermi_energies_array)
        valid_params = [
            param_values[i] for i in range(len(param_values)) if valid_indices[i]
        ]
        valid_fermi = fermi_energies_array[valid_indices]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(
            valid_params, valid_fermi, "^-", linewidth=2, markersize=8, color="blue"
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Fermi Energy (eV)", fontsize=12)
        ax.set_title(
            f"Fermi Energy vs. {parameter_name.replace('_', ' ').title()}", fontsize=14
        )
        ax.grid(True, alpha=0.3)  # noqa: FBT003
        plt.tight_layout()
        fermi_file = f"{base_name}_fermi.png"
        plt.savefig(fermi_file, dpi=150, bbox_inches="tight")
        plt.close()
        plot_files["fermi"] = fermi_file
        logger.info(f"Fermi energy plot saved to {fermi_file}")

    # Plot 4: Band Gap (if available)
    if has_bandgap_data:
        valid_indices = ~np.isnan(bandgaps_array)
        valid_params = [
            param_values[i] for i in range(len(param_values)) if valid_indices[i]
        ]
        valid_bandgaps = bandgaps_array[valid_indices]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(
            valid_params,
            valid_bandgaps,
            "D-",
            linewidth=2,
            markersize=8,
            color="purple",
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Band Gap (eV)", fontsize=12)
        ax.set_title(
            f"Band Gap vs. {parameter_name.replace('_', ' ').title()}", fontsize=14
        )
        ax.grid(True, alpha=0.3)  # noqa: FBT003
        plt.tight_layout()
        bandgap_file = f"{base_name}_bandgap.png"
        plt.savefig(bandgap_file, dpi=150, bbox_inches="tight")
        plt.close()
        plot_files["bandgap"] = bandgap_file
        logger.info(f"Band gap plot saved to {bandgap_file}")

    # Plot 5: Maximum Forces (if available)
    if has_force_data:
        valid_indices = ~np.isnan(max_forces_array)
        valid_params = [
            param_values[i] for i in range(len(param_values)) if valid_indices[i]
        ]
        valid_forces = max_forces_array[valid_indices]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(
            valid_params,
            valid_forces,
            "v-",
            linewidth=2,
            markersize=8,
            color="darkgreen",
        )
        ax.axhline(
            y=0.01,
            color="green",
            linestyle="--",
            label="0.01 eV/Å (tight)",
            linewidth=2,
        )
        ax.axhline(
            y=0.05,
            color="orange",
            linestyle="--",
            label="0.05 eV/Å (loose)",
            linewidth=2,
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Max Force (eV/Å)", fontsize=12)
        ax.set_title(
            f"Max Forces vs. {parameter_name.replace('_', ' ').title()}", fontsize=14
        )
        ax.grid(True, alpha=0.3)  # noqa: FBT003
        ax.legend(fontsize=11)
        plt.tight_layout()
        force_file = f"{base_name}_forces.png"
        plt.savefig(force_file, dpi=150, bbox_inches="tight")
        plt.close()
        plot_files["forces"] = force_file
        logger.info(f"Forces plot saved to {force_file}")

    # Plot 6: Maximum Stress (if available)
    if has_stress_data:
        valid_indices = ~np.isnan(max_stress_array)
        valid_params = [
            param_values[i] for i in range(len(param_values)) if valid_indices[i]
        ]
        valid_stress = max_stress_array[valid_indices]

        _fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(
            valid_params, valid_stress, "h-", linewidth=2, markersize=8, color="brown"
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Max Stress Component (eV/Å³)", fontsize=12)
        ax.set_title(
            f"Max Stress vs. {parameter_name.replace('_', ' ').title()}", fontsize=14
        )
        ax.grid(True, alpha=0.3)  # noqa: FBT003
        plt.tight_layout()
        stress_file = f"{base_name}_stress.png"
        plt.savefig(stress_file, dpi=150, bbox_inches="tight")
        plt.close()
        plot_files["stress"] = stress_file
        logger.info(f"Stress plot saved to {stress_file}")

    # Print summary of plots created
    if verbosity_value >= VerbosityLevel.INFO.value:
        console.print(f"[green]Created {len(plot_files)} convergence plots:[/green]")
        for plot_type, plot_path in plot_files.items():
            console.print(f"  - {plot_type}: {plot_path}")

    # Save convergence data to text file
    txt_file = f"{base_name}.txt"
    with open(txt_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"CONVERGENCE ANALYSIS: {parameter_name.upper()}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Parameter: {parameter_name}\n")
        f.write(f"Number of calculations: {len(parameters)}\n")
        f.write(f"Reference energy (most converged): {energy_ref:.6f} eV\n\n")

        # Build header based on available data
        f.write("-" * 120 + "\n")
        header = f"{'Parameter':<15} {'Energy':<12} {'ΔE':<10}"
        if has_fermi_data:
            header += f" {'Fermi':<10}"
        if has_bandgap_data:
            header += f" {'Gap':<10}"
        if has_force_data:
            header += f" {'MaxF':<10} {'MeanF':<10}"
        if has_stress_data:
            header += f" {'MaxStress':<12}"
        f.write(header + "\n")

        # Units row
        units = f"{'':15} {'(eV)':<12} {'(meV)':<10}"
        if has_fermi_data:
            units += f" {'(eV)':<10}"
        if has_bandgap_data:
            units += f" {'(eV)':<10}"
        if has_force_data:
            units += f" {'(eV/Å)':<10} {'(eV/Å)':<10}"
        if has_stress_data:
            units += f" {'(eV/Å³)':<12}"
        f.write(units + "\n")
        f.write("-" * 120 + "\n")

        # Data rows
        for i in range(len(parameters)):
            param = parameters[i]
            energy = energies[i]
            delta_e = energy_diff[i]

            row = f"{param!s:<15} {energy:<12.6f} {delta_e:<10.3f}"

            if has_fermi_data:
                ef = fermi_energies[i] if i < len(fermi_energies) else None
                row += (
                    f" {ef if ef is not None else 'N/A':<10.6f}"
                    if ef is not None
                    else f" {'N/A':<10}"
                )

            if has_bandgap_data:
                bg = bandgaps[i] if i < len(bandgaps) else None
                row += (
                    f" {bg if bg is not None else 'N/A':<10.6f}"
                    if bg is not None
                    else f" {'N/A':<10}"
                )

            if has_force_data:
                mf = max_forces[i] if i < len(max_forces) else None
                meanf = (
                    convergence_data.get("mean_forces", [])[i]
                    if i < len(convergence_data.get("mean_forces", []))
                    else None
                )
                row += (
                    f" {mf if mf is not None else 'N/A':<10.6f}"
                    if mf is not None
                    else f" {'N/A':<10}"
                )
                row += (
                    f" {meanf if meanf is not None else 'N/A':<10.6f}"
                    if meanf is not None
                    else f" {'N/A':<10}"
                )

            if has_stress_data:
                ms = max_stress[i] if i < len(max_stress) else None
                row += (
                    f" {ms if ms is not None else 'N/A':<12.6f}"
                    if ms is not None
                    else f" {'N/A':<12}"
                )

            f.write(row + "\n")

        f.write("-" * 120 + "\n\n")

        # Convergence analysis
        f.write("CONVERGENCE ANALYSIS:\n")
        f.write("-" * 80 + "\n")

        # Find first point converged to 1 meV
        converged_1mev_idx = None
        for i, delta_e in enumerate(energy_diff):
            if abs(delta_e) < 1.0:
                converged_1mev_idx = i
                break

        # Find first point converged to 5 meV
        converged_5mev_idx = None
        for i, delta_e in enumerate(energy_diff):
            if abs(delta_e) < 5.0:
                converged_5mev_idx = i
                break

        if converged_1mev_idx is not None:
            f.write(
                f"✓ Converged to 1 meV at: {parameters[converged_1mev_idx]} "
                f"(ΔE = {energy_diff[converged_1mev_idx]:.3f} meV)\n"
            )
        else:
            f.write("✗ NOT converged to 1 meV with tested parameters\n")

        if converged_5mev_idx is not None:
            f.write(
                f"✓ Converged to 5 meV at: {parameters[converged_5mev_idx]} "
                f"(ΔE = {energy_diff[converged_5mev_idx]:.3f} meV)\n"
            )
        else:
            f.write("✗ NOT converged to 5 meV with tested parameters\n")

        f.write("\n")

        # Energy statistics
        energy_range = energies.max() - energies.min()
        f.write("\nEnergy Statistics:\n")
        f.write(f"  Range: {energy_range * 1000:.3f} meV\n")
        f.write(f"  Max ΔE: {abs(energy_diff).max():.3f} meV\n")
        f.write(f"  Min ΔE: {abs(energy_diff).min():.3f} meV\n")

        if has_fermi_data:
            valid_fermi_energies = fermi_energies_array[~np.isnan(fermi_energies_array)]
            if len(valid_fermi_energies) > 0:
                fermi_range = valid_fermi_energies.max() - valid_fermi_energies.min()
                f.write("\nFermi Energy Statistics:\n")
                f.write(f"  Range: {fermi_range:.6f} eV\n")
                f.write(f"  Mean: {valid_fermi_energies.mean():.6f} eV\n")
                f.write(f"  Min: {valid_fermi_energies.min():.6f} eV\n")
                f.write(f"  Max: {valid_fermi_energies.max():.6f} eV\n")

        if has_bandgap_data:
            valid_bandgaps = bandgaps_array[~np.isnan(bandgaps_array)]
            if len(valid_bandgaps) > 0:
                f.write("\nBand Gap Statistics:\n")
                f.write(f"  Mean: {valid_bandgaps.mean():.6f} eV\n")
                f.write(f"  Min: {valid_bandgaps.min():.6f} eV\n")
                f.write(f"  Max: {valid_bandgaps.max():.6f} eV\n")
                f.write(
                    f"  Range: {valid_bandgaps.max() - valid_bandgaps.min():.6f} eV\n"
                )

        if has_force_data:
            valid_forces = max_forces_array[~np.isnan(max_forces_array)]
            if len(valid_forces) > 0:
                f.write("\nForce Statistics:\n")
                f.write(f"  Max Force: {valid_forces.max():.6f} eV/Å\n")
                f.write(f"  Min Force: {valid_forces.min():.6f} eV/Å\n")
                f.write(f"  Mean: {valid_forces.mean():.6f} eV/Å\n")
                if valid_forces.max() < 0.01:
                    f.write("  ✓ All forces converged (< 0.01 eV/Å)\n")
                elif valid_forces.max() < 0.05:
                    f.write("  ✓ Forces reasonably converged (< 0.05 eV/Å)\n")
                else:
                    f.write("  ✗ Forces NOT converged (> 0.05 eV/Å)\n")

        if has_stress_data:
            valid_stress = max_stress_array[~np.isnan(max_stress_array)]
            if len(valid_stress) > 0:
                f.write("\nStress Statistics:\n")
                f.write(f"  Max Component: {valid_stress.max():.6f} eV/Å³\n")
                f.write(f"  Min Component: {valid_stress.min():.6f} eV/Å³\n")
                f.write(f"  Mean: {valid_stress.mean():.6f} eV/Å³\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("=" * 80 + "\n")

        if converged_1mev_idx is not None:
            f.write(
                f"For high accuracy (< 1 meV): use {parameters[converged_1mev_idx]} or higher\n"
            )
        else:
            f.write(
                "For high accuracy (< 1 meV): increase parameter beyond tested range\n"
            )

        if converged_5mev_idx is not None:
            f.write(
                f"For standard accuracy (< 5 meV): use {parameters[converged_5mev_idx]} or higher\n"
            )
        else:
            f.write(
                "For standard accuracy (< 5 meV): increase parameter beyond tested range\n"
            )

        f.write("\n")

        # Add standard footer
        from atomate2.siesta.utils.text_output import get_standard_footer

        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Analysis type": f"{parameter_name} convergence",
                    "Number of calculations": str(len(parameters)),
                },
            )
        )

    logger.info(f"Convergence data saved to {txt_file}")
    if verbosity_value >= VerbosityLevel.INFO.value:
        console.print(f"[green]Convergence data saved to: {txt_file}[/green]")

    # Return dictionary with all generated files
    plot_files["txt_file"] = txt_file
    return plot_files
