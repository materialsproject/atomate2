"""
Debugging Workflows - Analyzing Intermediate Outputs
====================================================

This tutorial shows how to analyze intermediate outputs from workflows:
- Reading SIESTA output files programmatically
- Extracting energies, forces, and stress
- Plotting convergence history
- Comparing expected vs actual results

Category: troubleshooting
Difficulty: Intermediate
Time: 25 minutes
"""

import re
from pathlib import Path


# Optional visualization
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping plots")


# =============================================================================
# Step 1: Understanding SIESTA Output Files
# =============================================================================

print("=" * 70)
print("Step 1: Understanding SIESTA Output Files")
print("=" * 70)

output_files = """
SIESTA generates several output files during calculation:

1. siesta.out     - Main output file (all information)
2. siesta.XV      - Final atomic positions and velocities
3. siesta.STRUCT_OUT - Final structure in SIESTA format
4. siesta.EIG     - Eigenvalues (band energies)
5. siesta.DM      - Density matrix (restart file)
6. siesta.bands   - Band structure data
7. siesta.DOS     - Density of states data

For debugging, focus on:
- siesta.out for errors and convergence
- siesta.XV for final structure
- siesta.DM for restart capabilities
"""
print(output_files)


# =============================================================================
# Step 2: Parsing SCF Convergence
# =============================================================================

print("\n" + "=" * 70)
print("Step 2: Parsing SCF Convergence")
print("=" * 70)


def parse_scf_convergence(output_file: str | Path) -> dict:
    """Parse SCF convergence history from SIESTA output.

    Parameters
    ----------
    output_file : str or Path
        Path to siesta.out file

    Returns
    -------
    dict
        Dictionary with iterations, energies, dE, Harris energies
    """
    output_file = Path(output_file)

    if not output_file.exists():
        print(f"File not found: {output_file}")
        return {}

    iterations = []
    energies = []
    delta_e = []
    harris_energies = []

    scf_pattern = re.compile(r"scf:\s+(\d+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)")

    with open(output_file) as f:
        for line in f:
            match = scf_pattern.search(line)
            if match:
                iterations.append(int(match.group(1)))
                energies.append(float(match.group(2)))
                delta_e.append(float(match.group(3)))
                harris_energies.append(float(match.group(4)))

    return {
        "iterations": iterations,
        "energies": energies,
        "delta_e": delta_e,
        "harris_energies": harris_energies,
    }


# Example usage (with mock data for demonstration)
print(
    """
Usage example:
--------------
scf_data = parse_scf_convergence("job_*/siesta.out")

# Check if converged
if scf_data["delta_e"]:
    final_dE = abs(scf_data["delta_e"][-1])
    if final_dE < 1e-4:
        print(f"Converged! Final dE = {final_dE:.2e} eV")
    else:
        print(f"NOT converged. Final dE = {final_dE:.2e} eV")

# Check for oscillations
energies = scf_data["energies"]
if len(energies) > 5:
    energy_diff = np.diff(energies)
    oscillations = (energy_diff[:-1] * energy_diff[1:] < 0).sum()
    if oscillations > len(energies) // 3:
        print("Warning: Oscillating convergence detected!")
        print("Suggestion: Reduce SCF.Mixer.Weight to 0.01-0.05")
"""
)


# =============================================================================
# Step 3: Parsing Geometry Optimization
# =============================================================================

print("\n" + "=" * 70)
print("Step 3: Parsing Geometry Optimization")
print("=" * 70)


def parse_geometry_optimization(output_file: str | Path) -> dict:
    """Parse geometry optimization history from SIESTA output.

    Parameters
    ----------
    output_file : str or Path
        Path to siesta.out file

    Returns
    -------
    dict
        Dictionary with steps, energies, max_forces, max_stress
    """
    output_file = Path(output_file)

    if not output_file.exists():
        return {}

    steps = []
    energies = []
    max_forces = []

    # Pattern for energy per step
    energy_pattern = re.compile(r"siesta:\s+E_KS\(eV\)\s+=\s+([-\d.E+]+)")
    # Pattern for forces
    force_pattern = re.compile(r"siesta:\s+Max\s+Force\s+=\s+([\d.E+]+)")

    step = 0
    with open(output_file) as f:
        for line in f:
            energy_match = energy_pattern.search(line)
            if energy_match:
                step += 1
                steps.append(step)
                energies.append(float(energy_match.group(1)))

            force_match = force_pattern.search(line)
            if force_match:
                max_forces.append(float(force_match.group(1)))

    return {
        "steps": steps,
        "energies": energies,
        "max_forces": max_forces[: len(steps)] if max_forces else [],
    }


print(
    """
Usage example:
--------------
geo_data = parse_geometry_optimization("job_*/siesta.out")

# Check energy convergence
if geo_data["energies"]:
    energies = geo_data["energies"]
    energy_change = abs(energies[-1] - energies[-2]) if len(energies) > 1 else float('inf')
    print(f"Energy change in last step: {energy_change:.6f} eV")

# Check force convergence
if geo_data["max_forces"]:
    final_force = geo_data["max_forces"][-1]
    if final_force < 0.04:  # Default tolerance
        print(f"Forces converged: {final_force:.4f} eV/A")
    else:
        print(f"Forces NOT converged: {final_force:.4f} eV/A")
        print("Suggestion: Increase MD.NumCGsteps or reduce MD.MaxForceTol")
"""
)


# =============================================================================
# Step 4: Plotting Convergence (if matplotlib available)
# =============================================================================

print("\n" + "=" * 70)
print("Step 4: Plotting Convergence")
print("=" * 70)


def plot_scf_convergence(scf_data: dict, output_file: str = "scf_convergence.png"):
    """Plot SCF convergence history."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - skipping plot")
        return

    if not scf_data.get("energies"):
        print("No SCF data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Energy vs iteration
    ax1 = axes[0]
    ax1.plot(scf_data["iterations"], scf_data["energies"], "b-o", markersize=3)
    ax1.set_xlabel("SCF Iteration")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title("SCF Energy Convergence")
    ax1.grid(True, alpha=0.3)

    # Delta E vs iteration (log scale)
    ax2 = axes[1]
    delta_e_abs = [abs(x) for x in scf_data["delta_e"]]
    ax2.semilogy(scf_data["iterations"], delta_e_abs, "r-o", markersize=3)
    ax2.axhline(y=1e-4, color="g", linestyle="--", label="Typical tolerance (1e-4)")
    ax2.set_xlabel("SCF Iteration")
    ax2.set_ylabel("|Delta E| (eV)")
    ax2.set_title("SCF Energy Change")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def plot_geometry_optimization(geo_data: dict, output_file: str = "geo_opt.png"):
    """Plot geometry optimization history."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - skipping plot")
        return

    if not geo_data.get("energies"):
        print("No geometry optimization data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Energy vs step
    ax1 = axes[0]
    ax1.plot(geo_data["steps"], geo_data["energies"], "b-o")
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title("Geometry Optimization Energy")
    ax1.grid(True, alpha=0.3)

    # Max force vs step
    if geo_data.get("max_forces"):
        ax2 = axes[1]
        ax2.semilogy(
            range(1, len(geo_data["max_forces"]) + 1), geo_data["max_forces"], "r-o"
        )
        ax2.axhline(
            y=0.04, color="g", linestyle="--", label="Default tolerance (0.04 eV/A)"
        )
        ax2.set_xlabel("Optimization Step")
        ax2.set_ylabel("Max Force (eV/A)")
        ax2.set_title("Force Convergence")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


print(
    """
Plotting functions available:
- plot_scf_convergence(scf_data, "scf_convergence.png")
- plot_geometry_optimization(geo_data, "geo_opt.png")

Example usage:
--------------
scf_data = parse_scf_convergence("job_*/siesta.out")
plot_scf_convergence(scf_data)

geo_data = parse_geometry_optimization("job_*/siesta.out")
plot_geometry_optimization(geo_data)
"""
)


# =============================================================================
# Step 5: Comparing Expected vs Actual Results
# =============================================================================

print("\n" + "=" * 70)
print("Step 5: Comparing Expected vs Actual Results")
print("=" * 70)


def compare_results(calculated: dict, reference: dict, tolerances: dict = None):
    """Compare calculated results with reference values.

    Parameters
    ----------
    calculated : dict
        Dictionary with calculated properties (energy, volume, etc.)
    reference : dict
        Dictionary with reference values
    tolerances : dict, optional
        Tolerances for each property

    Returns
    -------
    dict
        Comparison results with pass/fail status
    """
    if tolerances is None:
        tolerances = {
            "energy": 0.01,  # eV/atom
            "volume": 1.0,  # A^3
            "lattice_a": 0.05,  # A
            "band_gap": 0.1,  # eV
        }

    results = {}
    for prop, calc_value in calculated.items():
        if prop in reference:
            ref_value = reference[prop]
            diff = abs(calc_value - ref_value)
            tol = tolerances.get(prop, 0.01)
            passed = diff <= tol

            results[prop] = {
                "calculated": calc_value,
                "reference": ref_value,
                "difference": diff,
                "tolerance": tol,
                "passed": passed,
            }

    return results


# Example usage
print(
    """
Example comparison:
-------------------
calculated = {
    "energy": -215.432,  # eV
    "volume": 40.05,     # A^3
    "lattice_a": 5.43,   # A
    "band_gap": 1.12,    # eV
}

reference = {
    "energy": -215.45,   # Literature value
    "volume": 40.0,      # Experimental
    "lattice_a": 5.431,  # Experimental
    "band_gap": 1.17,    # Experimental (Si)
}

comparison = compare_results(calculated, reference)

for prop, data in comparison.items():
    status = "PASS" if data["passed"] else "FAIL"
    print(f"{prop}: {data['calculated']:.3f} vs {data['reference']:.3f} "
          f"(diff={data['difference']:.3f}) [{status}]")
"""
)

# Run example
calculated = {
    "energy": -215.432,
    "volume": 40.05,
    "lattice_a": 5.43,
    "band_gap": 1.12,
}

reference = {
    "energy": -215.45,
    "volume": 40.0,
    "lattice_a": 5.431,
    "band_gap": 1.17,
}

comparison = compare_results(calculated, reference)

print("\nExample output:")
for prop, data in comparison.items():
    status = "PASS" if data["passed"] else "FAIL"
    print(
        f"  {prop}: {data['calculated']:.3f} vs {data['reference']:.3f} "
        f"(diff={data['difference']:.3f}) [{status}]"
    )


# =============================================================================
# Step 6: Reading Final Structure
# =============================================================================

print("\n" + "=" * 70)
print("Step 6: Reading Final Structure")
print("=" * 70)

print(
    """
Reading final structure from SIESTA output:
-------------------------------------------

# Method 1: From siesta.XV (positions after optimization)
from pymatgen.io.siesta import SiestaInput
structure = SiestaInput.from_file("job_*/siesta.XV").structure

# Method 2: From STRUCT_OUT (if available)
from pymatgen.core import Structure
# STRUCT_OUT is similar format to STRUCT_IN

# Method 3: Using atomate2siesta OutputDoc
from atomate2.siesta.schemas.task import OutputDoc
# OutputDoc contains final_structure automatically

# Check structure quality
print(f"Volume: {structure.volume:.2f} A^3")
print(f"Density: {structure.density:.3f} g/cm^3")

# Check bond lengths
from pymatgen.analysis.structure_analyzer import oxide_proximity_analysis
# or manual check
for i, site_i in enumerate(structure):
    for j, site_j in enumerate(structure):
        if i < j:
            dist = site_i.distance(site_j)
            if dist < 3.0:  # Reasonable bonding distance
                print(f"{site_i.species_string}-{site_j.species_string}: {dist:.3f} A")
"""
)


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(
    """
Key functions for analyzing SIESTA output:

1. parse_scf_convergence(output_file)
   - Returns iterations, energies, delta_E
   - Use to identify oscillations and convergence issues

2. parse_geometry_optimization(output_file)
   - Returns steps, energies, forces
   - Use to check if optimization converged

3. plot_scf_convergence(data) / plot_geometry_optimization(data)
   - Visual inspection of convergence
   - Helps identify systematic issues

4. compare_results(calculated, reference)
   - Validate results against literature/experiment
   - Automatic pass/fail checking

Pro tips:
- Always plot convergence for failed calculations
- Compare with simpler systems that work
- Check literature for expected values
- Save working parameter sets as references

Next tutorial:
- ../performance_optimization/ - Making calculations faster
"""
)
