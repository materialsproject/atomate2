"""Equation of state workflow for SIESTA. Based on the common EOS workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

# from atomate2.aims.flows.core import DoubleRelaxMaker
from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.common.flows.eos import CommonEosMaker
from atomate2.siesta.powerups import update_user_siesta_settings
from jobflow import Flow, Maker, job

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from atomate2.common.jobs.eos import EOSPostProcessor  # noqa: F401

import logging

logger = logging.getLogger(__name__)

# Rich console for pretty output
try:
    from rich.console import Console

    console = Console()
except ImportError:
    console = None


@dataclass
class SiestaEosFlowMaker(BaseSiestaFlowMaker, CommonEosMaker):
    """
    SIESTA Equation of State (EOS) Workflow.

    This workflow determines the relationship between a material's energy and volume,
    which is fundamental for calculating mechanical properties and understanding
    high-pressure behavior.

    Workflow Steps
    --------------

    1. Initial Relaxation (optional):

       - Fully relax the structure to equilibrium (zero pressure, minimum forces)
       - Determines the reference state V₀, E₀

    2. Structure Deformation:

       - Apply uniform volume strains (default: -5% to +5%)
       - Generate multiple structures at different volumes
       - Maintains crystal symmetry during deformation

    3. Energy Calculations:

       - Relax atomic positions at each fixed volume (or run static calculations)
       - Calculate total energy E(V) for each volume point using SIESTA
       - Optionally calculate stress tensors

    4. EOS Fitting:

       - Fit E(V) data to analytical EOS models:
         • Murnaghan: Simple linear B(P) assumption
         • Birch-Murnaghan: Finite strain theory (2nd/3rd order)
         • Vinet: Universal EOS based on interatomic potentials
         • Pourier-Tarantola: Logarithmic strain formulation
       - Extract equilibrium properties: E₀, V₀, B₀ (bulk modulus), B₁

    5. Analysis & Output:

       - Generate E(V) plots with fitted curves
       - Export results to summary file with theoretical background
       - Compare different EOS model predictions

    Key Results
    -----------

    • E₀: Equilibrium energy (eV)
    • V₀: Equilibrium volume (Ų)
    • B₀: Bulk modulus (GPa) - material's resistance to compression
    • B₁: Pressure derivative of bulk modulus (dimensionless)

    Applications
    ------------

    • Determine bulk modulus (material stiffness/compressibility)
    • Predict high-pressure behavior and phase transitions
    • Validate DFT accuracy against experimental data
    • Calculate thermal expansion (when combined with phonon calculations)
    • Study pressure-induced phenomena in materials

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    initial_relax_maker : .Maker | None
        Maker to relax the input structure, defaults to variable-cell relaxation.
    eos_relax_maker : .Maker
        Maker to relax deformed structures for the EOS fit (default: fixed-cell).
    static_maker : .Maker | None
        Maker to generate statics after each relaxation, defaults to None.
    linear_strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to +5%.
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6.
    postprocessor : .atomate2.common.jobs.EOSPostProcessor
        Optional postprocessing step, defaults to
        `atomate2.common.jobs.PostProcessEosEnergy`.
    _store_transformation_information : .bool = False
        Whether to store the information about transformations. Unfortunately
        needed at present to handle issues with emmet and pydantic validation

    Example
    -------
    >>> from atomate2.siesta.flows.eos import SiestaEosFlowMaker
    >>> from pymatgen.core import Structure
    >>> structure = Structure.from_file("Si.cif")
    >>> maker = SiestaEosFlowMaker(
    ...     linear_strain=(-0.05, 0.05),
    ...     number_of_frames=7
    ... )
    >>> flow = maker.make(structure)
    """

    name: str = "siesta eos"
    initial_relax_maker: Maker | None = field(
        default_factory=lambda: RelaxMaker.variable_cell_relaxation({})
    )
    eos_relax_maker: Maker | None = field(
        default_factory=lambda: RelaxMaker.fixed_cell_relaxation()
    )
    # number_of_frames: int = 7

    # Dry-run support
    dry_run: bool = False
    dry_run_output_dir: str = "dry_run_output"
    dry_run_format: str = "cif"

    # Internal batch-workflow controls. Declared here so siesta does not depend
    # on a customized common.flows.eos: the common EOS maker may read these
    # attributes (to suppress the docstring panel and prefix job names in batch
    # runs) but is not required to define them.
    _suppress_print: bool = False
    _global_counter: tuple[int, int, int, int] | None = None

    def __post_init__(self):
        """Propagate settings (dry_run, custodian, tier, manager_config) to child makers."""
        # Call parent to handle dry_run, use_custodian, tier, manager_config propagation
        super().__post_init__()

    @classmethod
    # def from_parameters(cls, parameters: dict[str, Any], **kwargs) -> SiestaEosFlowMaker:
    def from_parameters(
        cls, parameters: dict[str, Any], **kwargs
    ) -> SiestaEosFlowMaker:
        """Creation of SiestaEosFlowMaker from parameters.

        Parameters
        ----------
        parameters : dict
            Dictionary of common parameters for both makers. The one exception is
            `species_dir` which can be either a string or a dict with keys [`initial`,
            `eos`]. If a string is given, it will be interpreted as the `species_dir`
            for the `eos` Maker; the initial double relaxation will be done then with
            the default `light` and `tight` species' defaults.
        kwargs
            Keyword arguments passed to `CommonEosMaker`.
        """
        # species_dir = parameters.setdefault("species_dir", "tight")
        initial_params = parameters.copy()
        eos_params = parameters.copy()
        # if isinstance(species_dir, dict):
        #    initial_params["species_dir"] = species_dir.get("initial")
        #    eos_params["species_dir"] = species_dir.get("eos", "tight")
        return cls(
            initial_relax_maker=RelaxMaker.fixed_cell_relaxation(initial_params),
            eos_relax_maker=RelaxMaker.fixed_cell_relaxation(user_params=eos_params),
            **kwargs,
        )


# Alias for easier import
EOSFlowMaker = SiestaEosFlowMaker


# =============================================================================
# EOS Parameter Convergence Workflow
# =============================================================================


@dataclass
class EOSFullBasisConvergenceFlowMaker(BaseSiestaFlowMaker):
    """
    Full basis convergence testing with EOS: tests all combinations of basis sizes
    and PAO parameters (EnergyShift, SplitNorm).

    This comprehensive workflow runs EOS calculations for every combination of:
    - Basis sizes (SZ, DZ, DZP, TZP, etc.)
    - PAO.EnergyShift values (orbital confinement)
    - PAO.SplitNorm values (multiple-zeta splitting)

    Use this for complete parameter optimization to determine the best basis settings
    for accurate bulk properties (V₀, E₀, B₀).

    For simple basis comparison with fixed parameters, use EOSBasisConvergenceMaker instead.

    Inherits from BaseSiestaFlowMaker, so dry_run=True automatically propagates
    to child makers (initial_relax_maker, eos_relax_maker, static_maker).

    The workflow:
    1. Creates EOS calculations for each basis size × EnergyShift × SplitNorm combination
    2. Collects EOS results (equilibrium volume, energy, bulk modulus)
    3. Analyzes convergence for each basis size
    4. Determines optimal PAO parameters for each basis
    5. Compares basis sizes with their optimal parameters
    6. Generates comprehensive summary and recommendations

    Parameters
    ----------
    name : str
        Name of the workflow
    basis_sizes : list[str]
        List of basis sizes to test (e.g., ["DZ", "DZP", "TZP"])
    energy_shifts : list[float]
        List of PAO.EnergyShift values to test (in Ry)
    split_norms : list[float]
        List of PAO.SplitNorm values to test
    a2s_kpts : list[int] | None
        K-points grid [nk1, nk2, nk3]
    linear_strain : tuple[float, float]
        Strain range for EOS (e.g., (-0.05, 0.05) for ±5%)
    number_of_frames : int
        Number of strain points for EOS fit
    initial_relax_maker : Maker | None
        Maker for initial variable-cell relaxation
    eos_relax_maker : Maker | None
        Maker for fixed-cell relaxation at each strain
    static_maker : Maker | None
        Maker for static calculations after relaxation (optional)
    dry_run : bool
        If True, skip SIESTA calculations and only save structures (inherited).
    dry_run_output_dir : str
        Directory to save dry-run structures (inherited).
    dry_run_format : str
        Output format for dry-run structures (inherited).

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> structure = Structure.from_file("POSCAR")
    >>> # Test 3 basis × 4 energy shifts × 3 split norms = 36 EOS calculations
    >>> maker = EOSFullBasisConvergenceFlowMaker(
    ...     basis_sizes=["DZ", "DZP", "TZP"],
    ...     energy_shifts=[0.005, 0.010, 0.015, 0.020],
    ...     split_norms=[0.15, 0.20, 0.25],
    ...     a2s_kpts=[4, 4, 4],
    ...     linear_strain=(-0.05, 0.05),
    ...     number_of_frames=7
    ... )
    >>> flow = maker.make(structure)
    """

    name: str = "EOS Full Basis Convergence"
    basis_sizes: list[str] = None
    energy_shifts: list[float] = None
    split_norms: list[float] = None
    a2s_kpts: list[int] | None = None
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 7
    initial_relax_maker: Maker | None = None
    eos_relax_maker: Maker | None = None
    static_maker: Maker | None = None

    def __post_init__(self):
        """Set defaults for lists if not provided."""
        if self.basis_sizes is None:
            self.basis_sizes = ["DZ", "DZP", "TZP"]
        if self.energy_shifts is None:
            self.energy_shifts = [0.005, 0.010, 0.015, 0.020]
        if self.split_norms is None:
            self.split_norms = [0.15, 0.20, 0.25]
        if self.initial_relax_maker is None:
            self.initial_relax_maker = RelaxMaker.variable_cell_relaxation()
        if self.eos_relax_maker is None:
            self.eos_relax_maker = RelaxMaker.fixed_cell_relaxation()

        # Call parent __post_init__ to propagate dry_run after creating makers
        super().__post_init__()

    def make(
        self,
        structure: Structure,
        prev_dir: str | None = None,
    ) -> Flow:
        """
        Create EOS parameter convergence workflow.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object
        prev_dir : str | None
            Previous directory (not used here)

        Returns
        -------
        Flow
            A Flow containing all EOS jobs and analysis
        """
        # Print workflow description box
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        logger.info(
            f"{self.name}.make() called for {structure.composition.reduced_formula}"
        )

        n_total = (
            len(self.basis_sizes) * len(self.energy_shifts) * len(self.split_norms)
        )

        if console:
            console.print(
                "[bold cyan]Creating EOS Parameter Convergence Workflow[/bold cyan]"
            )
            console.print(f"  Basis sizes: {', '.join(self.basis_sizes)}")
            console.print(
                f"  Testing {len(self.basis_sizes)} × {len(self.energy_shifts)} × {len(self.split_norms)} = "
                f"{n_total} EOS workflows"
            )
            console.print(
                f"  {n_total} × {self.number_of_frames} = {n_total * self.number_of_frames} total calculations"
            )

        # Create EOS jobs for each basis size × parameter combination
        eos_jobs = []
        job_metadata = []

        # Calculate total jobs across all EOS workflows
        jobs_per_eos = self.number_of_frames
        if self.initial_relax_maker:
            jobs_per_eos += 1  # Add initial relaxation

        # If static_maker is used, double the number of jobs (one static per relax)
        if self.static_maker:
            jobs_per_eos *= 2

        total_jobs = n_total * jobs_per_eos

        eos_counter = 0
        absolute_job_counter = 0  # Absolute counter across all jobs

        for basis_size in self.basis_sizes:
            for energy_shift in self.energy_shifts:
                for split_norm in self.split_norms:
                    eos_counter += 1
                    # Create label for this combination
                    label = f"{basis_size}-ES{energy_shift:.3f}-SN{split_norm:.2f}"

                    # Calculate absolute job range for this EOS workflow
                    job_start = absolute_job_counter + 1
                    job_end = absolute_job_counter + jobs_per_eos
                    absolute_job_counter = job_end

                    # Create EOS maker (suppress docstring print for batch workflows)
                    eos_maker = SiestaEosFlowMaker(
                        name=f"{self.name}-{label}",
                        initial_relax_maker=self.initial_relax_maker,
                        eos_relax_maker=self.eos_relax_maker,
                        static_maker=self.static_maker,
                        linear_strain=self.linear_strain,
                        number_of_frames=self.number_of_frames,
                        _suppress_print=True,  # Don't print docstring for each EOS
                        _global_counter=(
                            eos_counter,
                            n_total,
                            job_start,
                            total_jobs,
                        ),  # Pass global counters
                    )

                    # Create EOS flow
                    eos_flow = eos_maker.make(structure)

                    # Update with basis parameters
                    basis_params = {
                        "PAO.BasisSize": basis_size,
                        "PAO.BasisType": "split",
                        "PAO.EnergyShift": f"{energy_shift} Ry",
                        "PAO.SplitNorm": split_norm,
                    }
                    if self.a2s_kpts is not None:
                        basis_params["a2s_kpts"] = self.a2s_kpts

                    eos_flow = update_user_siesta_settings(eos_flow, basis_params)
                    eos_flow.name = f"{self.name}-{label}"

                    # Rename the plot and summary jobs to include parameters in filename
                    for eos_job in eos_flow.jobs:
                        if "plot" in eos_job.name.lower():
                            # Update the output_file parameter for plot
                            eos_job.function_kwargs[
                                "output_file"
                            ] = f"eos_fit_{label}.png"
                        elif "summary" in eos_job.name.lower():
                            # Update the output_file parameter for summary
                            eos_job.function_kwargs[
                                "output_file"
                            ] = f"eos_summary_{label}.txt"

                    eos_jobs.append(eos_flow)
                    job_metadata.append(
                        {
                            "basis_size": basis_size,
                            "energy_shift": energy_shift,
                            "split_norm": split_norm,
                            "label": label,
                        }
                    )

        # Create collection and analysis job
        collect_job = collect_eos_parameter_data(
            eos_outputs=[job.output for job in eos_jobs],
            job_metadata=job_metadata,
            basis_sizes=self.basis_sizes,
        )
        collect_job.name = f"{self.name}-collect"

        # Create unified output job that generates all plots and summaries in ONE directory
        unified_output_job = generate_eos_full_basis_outputs(
            data=collect_job.output,
        )
        unified_output_job.name = f"{self.name}-outputs"

        # Combine all jobs into workflow
        all_jobs = eos_jobs + [collect_job, unified_output_job]
        flow = Flow(all_jobs, output=unified_output_job.output, name=self.name)

        logger.info(
            f"Created EOS parameter convergence flow with {len(eos_jobs)} EOS calculations"
        )

        return flow


@job
def collect_eos_parameter_data(
    eos_outputs: list[Any], job_metadata: list[dict[str, Any]], basis_sizes: list[str]
) -> dict[str, Any]:
    """
    Collect EOS results from multiple basis sizes and parameter combinations.

    Parameters
    ----------
    eos_outputs : list[Any]
        List of EOS job outputs
    job_metadata : list[dict]
        Metadata for each job (basis_size, energy_shift, split_norm, label)
    basis_sizes : list[str]
        List of basis sizes being tested

    Returns
    -------
    dict
        Dictionary containing collected EOS results
    """
    import numpy as np

    logger.info(f"Collecting EOS results for {len(basis_sizes)} basis sizes")

    if console:
        console.print("\n[bold cyan]Collecting EOS Results[/bold cyan]")
        console.print(f"  Basis sizes: {', '.join(basis_sizes)}")
        console.print(f"  Received {len(eos_outputs)} EOS outputs")
        console.print(
            f"  Output types: {[type(out).__name__ for out in eos_outputs[:3]]}"
        )

    basis_size_list = []
    energy_shifts = []
    split_norms = []
    v0_values = []
    e0_values = []
    b0_values = []
    lattice_a_values = []
    lattice_b_values = []
    lattice_c_values = []
    lattice_alpha_values = []
    lattice_beta_values = []
    lattice_gamma_values = []
    labels = []
    all_volumes = []
    all_energies = []
    run_times = []  # Total wall time for each EOS workflow

    for idx, (output, metadata) in enumerate(zip(eos_outputs, job_metadata)):
        try:
            # Debug: print what we received
            if console:
                console.print(
                    f"  Processing {metadata['label']}: type={type(output).__name__}"
                )

            # Extract EOS data from CommonEosMaker flow output
            # The output is a dict with structure:
            # {
            #   "relax": {"energy": [...], "volume": [...], "EOS": {...}},
            #   "initial_relax": {"E0": ..., "V0": ...}
            # }

            if not isinstance(output, dict):
                logger.warning(
                    f"Output is not a dict for {metadata['label']}: {type(output).__name__}"
                )
                if console:
                    console.print("    → [yellow]Output is not a dict[/yellow]")
                continue

            # Prefer 'static' job results if available, otherwise 'relax'
            job_type = None
            if "static" in output and "EOS" in output.get("static", {}):
                job_type = "static"
            elif "relax" in output and "EOS" in output.get("relax", {}):
                job_type = "relax"
            else:
                logger.warning(f"No EOS data found in output for {metadata['label']}")
                if console:
                    console.print(
                        f"    → [yellow]No EOS data found. Available keys: {list(output.keys())}[/yellow]"
                    )
                continue

            if console:
                console.print(f"    → Using {job_type} EOS data")

            # Extract volumes and energies
            volumes = output[job_type].get("volume", [])
            energies = output[job_type].get("energy", [])

            # Get EOS fit parameters (use birch_murnaghan as default, or first available)
            eos_models = output[job_type]["EOS"]

            # Prefer birch_murnaghan, otherwise use first available model
            eos_fit_data = None
            for model_name in [
                "birch_murnaghan",
                "vinet",
                "murnaghan",
                "birch",
                "pourier_tarantola",
            ]:
                if (
                    model_name in eos_models
                    and "exception" not in eos_models[model_name]
                ):
                    eos_fit_data = eos_models[model_name]
                    break

            if eos_fit_data is None:
                logger.warning(f"No valid EOS fit found for {metadata['label']}")
                if console:
                    console.print("    → [yellow]No valid EOS fit found[/yellow]")
                continue

            # Extract V0, E0, B0 from fit
            v0 = eos_fit_data.get("v0")
            e0 = eos_fit_data.get("e0")
            b0_gpa = eos_fit_data.get("b0 GPa")

            if v0 is None or e0 is None or b0_gpa is None:
                logger.warning(f"Missing EOS parameters for {metadata['label']}")
                continue

            # Extract equilibrium lattice parameters by scaling reference structure to V₀
            # The EOS applies isotropic strain, so lattice parameters scale as V^(1/3)
            # We take a reference structure and scale it to the equilibrium volume V₀
            structures = output[job_type].get("structure", [])
            a, b, c = None, None, None
            alpha, beta, gamma = 90.0, 90.0, 90.0

            if structures and volumes:
                # Find the structure closest to v0 as reference
                vol_array = np.array(volumes)
                closest_idx = np.argmin(np.abs(vol_array - v0))
                ref_structure = structures[closest_idx]
                v_ref = volumes[closest_idx]

                try:
                    ref_lattice = ref_structure.lattice
                    # Scale factor: since V scales as (length)^3, length scales as V^(1/3)
                    # a₀/a_ref = (V₀/V_ref)^(1/3)
                    scale = (v0 / v_ref) ** (1.0 / 3.0)

                    # Scale lattice parameters to equilibrium volume
                    a = ref_lattice.a * scale
                    b = ref_lattice.b * scale
                    c = ref_lattice.c * scale
                    # Angles are preserved under isotropic scaling
                    alpha, beta, gamma = (
                        ref_lattice.alpha,
                        ref_lattice.beta,
                        ref_lattice.gamma,
                    )

                    logger.info(
                        f"Scaled lattice from V_ref={v_ref:.3f} Å³ to V₀={v0:.3f} Å³ "
                        f"(scale={scale:.6f}): a={a:.4f}, b={b:.4f}, c={c:.4f} Å"
                    )
                except AttributeError:
                    logger.warning(
                        f"Could not extract lattice from structure for {metadata['label']}, using cubic approximation"
                    )
                    a = v0 ** (1.0 / 3.0)
                    b, c = a, a
                    alpha, beta, gamma = 90.0, 90.0, 90.0
            else:
                # Fallback: calculate assuming cubic if no structures available
                logger.warning(
                    f"No structures available for {metadata['label']}, using cubic approximation"
                )
                a = v0 ** (1.0 / 3.0)
                b, c = a, a
                alpha, beta, gamma = 90.0, 90.0, 90.0

            # Extract EOS parameters
            basis_size_list.append(metadata["basis_size"])
            energy_shifts.append(metadata["energy_shift"])
            split_norms.append(metadata["split_norm"])
            labels.append(metadata["label"])

            v0_values.append(v0)
            e0_values.append(e0)
            b0_values.append(b0_gpa)
            lattice_a_values.append(a)
            lattice_b_values.append(b)
            lattice_c_values.append(c)
            lattice_alpha_values.append(alpha)
            lattice_beta_values.append(beta)
            lattice_gamma_values.append(gamma)

            all_volumes.append(volumes)
            all_energies.append(energies)

            # Extract timing data if available
            # run_time is now a list of actual wall times (extracted in schema)
            timing_data = output[job_type].get("run_time", [])
            total_time = None
            if timing_data:
                # Sum all run times for this EOS workflow
                valid_times = [
                    t
                    for t in timing_data
                    if t is not None and isinstance(t, (int, float))
                ]
                if valid_times:
                    total_time = sum(valid_times)
                    logger.info(
                        f"Total run time for {metadata['label']}: {total_time:.1f}s ({len(valid_times)} calcs)"
                    )

            run_times.append(total_time)

            if console:
                console.print(
                    f"  ✓ {metadata['label']}: V₀={v0:.3f} Ų, "
                    f"E₀={e0:.6f} eV, B₀={b0_gpa:.2f} GPa"
                )

        except Exception as e:
            logger.error(f"Error extracting EOS for {metadata['label']}: {e}")
            if console:
                console.print(f"  [red]✗ {metadata['label']}: Error - {e}[/red]")

    if console:
        console.print(f"\n[green]Collected {len(v0_values)} EOS results[/green]")

    return {
        "basis_sizes": np.array(basis_size_list).tolist(),
        "energy_shifts": np.array(energy_shifts).tolist(),
        "split_norms": np.array(split_norms).tolist(),
        "labels": labels,
        "v0_values": np.array(v0_values).tolist(),
        "e0_values": np.array(e0_values).tolist(),
        "b0_values": np.array(b0_values).tolist(),
        "lattice_a": np.array(lattice_a_values).tolist(),
        "lattice_b": np.array(lattice_b_values).tolist(),
        "lattice_c": np.array(lattice_c_values).tolist(),
        "lattice_alpha": np.array(lattice_alpha_values).tolist(),
        "lattice_beta": np.array(lattice_beta_values).tolist(),
        "lattice_gamma": np.array(lattice_gamma_values).tolist(),
        "all_volumes": all_volumes,  # List of lists
        "all_energies": all_energies,  # List of lists
        "run_times": run_times,  # Total time per EOS workflow
    }


@job
def plot_eos_parameter_fits_from_data(
    data: dict[str, Any], output_file: str = "eos_parameter_fits_combined.png"
) -> dict[str, str]:
    """
    Plot all EOS fits on a single figure using collected data.

    Parameters
    ----------
    data : dict
        Collected EOS data from collect_eos_parameter_data
        Must contain: basis_sizes, labels, all_volumes, all_energies
    output_file : str
        Output filename for plot

    Returns
    -------
    dict
        Dictionary with plot file path
    """
    import numpy as np
    import matplotlib.pyplot as plt

    logger.info("Creating combined EOS plot")

    if console:
        console.print("\n[bold cyan]Creating Combined EOS Plot[/bold cyan]")

    # Get basis sizes from data
    basis_sizes_array = np.array(data.get("basis_sizes", []))
    if len(basis_sizes_array) == 0:
        logger.error("No basis size data in collected results")
        # Create empty plot with error message
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No data collected\n\nEOS calculations may have failed",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color="red",
            fontweight="bold",
        )
        ax.set_xlabel("Volume (Ų)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Energy (eV)", fontsize=12, fontweight="bold")
        ax.set_title(
            "EOS Parameter Convergence: All Fits Combined",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        return {"plot": output_file}

    # Organize data by basis size
    unique_basis = sorted(
        set(basis_sizes_array),
        key=lambda x: ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"].index(x)
        if x in ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"]
        else 999,
    )

    # Create colors and markers for different basis/parameter combinations
    colors = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    markers = ["o", "s", "^", "v", "D", "p", "*", "h", "<", ">"]
    linestyles = ["-", "--", "-.", ":"]

    # Create color map for basis sizes
    basis_colors = {}
    for i, basis in enumerate(unique_basis):
        basis_colors[basis] = colors[i % len(colors)]

    # Get volumes and energies from collected data
    all_volumes = data.get("all_volumes", [])
    all_energies = data.get("all_energies", [])
    basis_sizes_list = data.get("basis_sizes", [])
    labels_list = data.get("labels", [])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9))

    # Get V0 and E0 values for fitting
    v0_values = data.get("v0_values", [])
    e0_values = data.get("e0_values", [])
    energy_shifts = data.get("energy_shifts", [])
    split_norms = data.get("split_norms", [])

    # Track which basis sizes have been plotted for legend grouping
    plotted_basis_data = {}  # basis -> count of parameters
    for basis in unique_basis:
        plotted_basis_data[basis] = 0

    # Plot each EOS curve with fitted line
    n_plotted = 0

    for i, (volumes, energies, basis, label, v0, e0, es, sn) in enumerate(
        zip(
            all_volumes,
            all_energies,
            basis_sizes_list,
            labels_list,
            v0_values,
            e0_values,
            energy_shifts,
            split_norms,
        )
    ):
        try:
            volumes = np.array(volumes)
            energies = np.array(energies)

            if len(volumes) == 0 or len(energies) == 0:
                if console:
                    console.print(f"  ⚠ {label}: No volume/energy data")
                continue

            # Sort by volume for cleaner lines
            sort_idx = np.argsort(volumes)
            volumes = volumes[sort_idx]
            energies = energies[sort_idx]

            # Get color and marker for this entry
            color = basis_colors[basis]
            param_idx = plotted_basis_data[basis]
            marker = markers[param_idx % len(markers)]
            linestyle = linestyles[param_idx % len(linestyles)]

            # Create descriptive legend label
            data_label = f"{basis} ES={es:.3f} SN={sn:.2f} (data)"
            fit_label = f"{basis} ES={es:.3f} SN={sn:.2f} (B-M)"

            # Plot raw data points with distinct markers
            ax.scatter(
                volumes,
                energies,
                s=80,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidths=1,
                zorder=5,
                label=data_label,
                alpha=0.8,
            )

            # Fit and plot EOS curve
            from pymatgen.analysis.eos import EOS

            try:
                eos = EOS(eos_name="birch_murnaghan")
                eos_fit = eos.fit(volumes, energies)

                # Generate smooth curve
                v_fit = np.linspace(volumes.min() * 0.95, volumes.max() * 1.05, 100)
                e_fit = eos_fit.func(v_fit)

                # Plot fitted curve with linestyle variation
                ax.plot(
                    v_fit,
                    e_fit,
                    linestyle=linestyle,
                    color=color,
                    alpha=0.75,
                    linewidth=2.5,
                    label=fit_label,
                )

                # Mark equilibrium point with large X
                ax.plot(
                    v0,
                    e0,
                    "X",
                    color=color,
                    markersize=12,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    zorder=10,
                )

            except Exception as fit_error:
                logger.warning(f"Could not fit EOS for {label}: {fit_error}")
                # Just plot line connecting points
                ax.plot(
                    volumes,
                    energies,
                    linestyle=linestyle,
                    color=color,
                    alpha=0.6,
                    linewidth=1.5,
                    label=fit_label,
                )

            plotted_basis_data[basis] += 1
            n_plotted += 1

            if console:
                console.print(f"  ✓ Plotted {label}: {len(volumes)} points")

        except Exception as e:
            logger.error(f"Error plotting curve {i}: {e}")
            if console:
                console.print(f"  [red]✗ Curve {i}: {e}[/red]")

    # Formatting
    ax.set_xlabel("Volume (Ų)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Energy (eV)", fontsize=14, fontweight="bold")
    ax.set_title(
        "EOS Comparison: Basis Sets + PAO Parameters", fontsize=16, fontweight="bold"
    )

    # Check if any data was plotted
    if n_plotted == 0:
        # No data was plotted - add error message to plot
        ax.text(
            0.5,
            0.5,
            "No EOS data could be extracted\n\nCheck individual job outputs",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color="red",
            fontweight="bold",
        )
        logger.error("No EOS data was successfully extracted for plotting")
        if console:
            console.print("[red]✗ No EOS curves plotted[/red]")
    else:
        # Create legend - place outside if many entries
        n_entries = n_plotted * 2  # Each has data + fit entry
        if n_entries > 12:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=8,
                ncol=2,
                framealpha=0.95,
            )
        else:
            ax.legend(loc="best", fontsize=9, ncol=2, framealpha=0.95)

        n_basis = len(unique_basis)
        logger.info(
            f"Successfully plotted {n_plotted} EOS curves ({n_basis} basis sizes)"
        )
        if console:
            console.print(
                f"[green]✓ Successfully plotted {n_plotted} curves ({n_basis} basis sizes)[/green]"
            )

    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Combined EOS plot saved to {output_file}")
    if console:
        console.print(f"[green]Combined EOS plot saved to: {output_file}[/green]")

    return {"plot": output_file}


@job
def plot_eos_parameter_timing(
    data: dict[str, Any], output_file: str = "eos_parameter_timing.png"
) -> dict[str, str]:
    """
    Plot timing analysis for EOS parameter convergence study.

    Parameters
    ----------
    data : dict
        Collected EOS data from collect_eos_parameter_data
        Must contain: basis_sizes, energy_shifts, split_norms, run_times, labels
    output_file : str
        Output filename for timing plot

    Returns
    -------
    dict
        Dictionary with output file path
    """
    import matplotlib.pyplot as plt
    import numpy as np

    basis_sizes = np.array(data["basis_sizes"])
    energy_shifts = np.array(data["energy_shifts"])
    split_norms = np.array(data["split_norms"])
    run_times = np.array(data.get("run_times", []))
    labels = data["labels"]

    # Check if timing data is available
    if len(run_times) == 0 or all(t is None for t in run_times):
        logger.info("No timing data available - skipping timing plot")
        return {"timing_plot": None}

    # Filter out None values
    valid_mask = np.array([t is not None for t in run_times])
    if not np.any(valid_mask):
        return {"timing_plot": None}

    basis_sizes_valid = basis_sizes[valid_mask]
    energy_shifts_valid = energy_shifts[valid_mask]
    _split_norms_valid = split_norms[valid_mask]
    run_times_valid = run_times[valid_mask]
    _labels_valid = [label for label, v in zip(labels, valid_mask) if v]

    unique_basis = sorted(
        set(basis_sizes_valid),
        key=lambda x: ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"].index(x)
        if x in ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"]
        else 999,
    )

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Color map for basis sizes
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_basis)))
    basis_colors = {basis: colors[i] for i, basis in enumerate(unique_basis)}

    # Plot 1: Timing vs EnergyShift (grouped by basis)
    for basis in unique_basis:
        mask = basis_sizes_valid == basis
        if np.sum(mask) > 0:
            es_vals = energy_shifts_valid[mask]
            times = run_times_valid[mask]
            ax1.scatter(
                es_vals, times, color=basis_colors[basis], label=basis, s=80, alpha=0.7
            )

    ax1.set_xlabel("PAO.EnergyShift (Ry)", fontsize=11)
    ax1.set_ylabel("Wall Time (seconds)", fontsize=11)
    ax1.set_title("EOS Timing vs EnergyShift", fontsize=12, fontweight="bold")
    ax1.legend(title="Basis", fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Plot 2: Timing comparison by basis (box plot or bar)
    basis_time_data = []
    basis_labels = []
    for basis in unique_basis:
        mask = basis_sizes_valid == basis
        if np.sum(mask) > 0:
            basis_time_data.append(run_times_valid[mask])
            basis_labels.append(basis)

    positions = np.arange(len(basis_labels))
    bp = ax2.boxplot(
        basis_time_data, positions=positions, labels=basis_labels, patch_artist=True
    )

    # Color boxes
    for patch, basis in zip(bp["boxes"], basis_labels):
        patch.set_facecolor(basis_colors[basis])
        patch.set_alpha(0.7)

    ax2.set_xlabel("Basis Size", fontsize=11)
    ax2.set_ylabel("Wall Time (seconds)", fontsize=11)
    ax2.set_title("Timing Distribution by Basis", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Timing analysis plot saved to {output_file}")
    if console:
        console.print(f"[green]Timing analysis plot saved to: {output_file}[/green]")

    return {"timing_plot": output_file}


@job
def write_eos_parameter_summary(
    data: dict[str, Any], output_file: str | None = None
) -> dict[str, str]:
    """
    Write summary of EOS parameter convergence study.

    Parameters
    ----------
    data : dict
        Collected EOS data
    output_file : str | None
        Output filename (auto-generated if None)

    Returns
    -------
    dict
        Dictionary with summary file path
    """
    import numpy as np

    if output_file is None:
        output_file = "eos_parameter_convergence_summary.txt"

    logger.info(f"Writing EOS parameter summary to {output_file}")

    basis_sizes = np.array(data["basis_sizes"])
    energy_shifts = np.array(data["energy_shifts"])
    split_norms = np.array(data["split_norms"])
    v0_values = np.array(data["v0_values"])
    e0_values = np.array(data["e0_values"])
    b0_values = np.array(data["b0_values"])
    lattice_a = np.array(data.get("lattice_a", []))
    lattice_b = np.array(data.get("lattice_b", []))
    lattice_c = np.array(data.get("lattice_c", []))
    lattice_alpha = np.array(data.get("lattice_alpha", []))
    lattice_beta = np.array(data.get("lattice_beta", []))
    lattice_gamma = np.array(data.get("lattice_gamma", []))
    _labels = data["labels"]
    run_times = np.array(data.get("run_times", [None] * len(v0_values)))

    if len(v0_values) == 0:
        logger.error("No EOS data collected - cannot write summary")
        with open(output_file, "w") as f:
            f.write("=" * 90 + "\n")
            f.write("EOS PARAMETER CONVERGENCE STUDY\n")
            f.write("=" * 90 + "\n\n")
            f.write("ERROR: No EOS data was collected.\n")
            f.write("Check individual job directories for calculation outputs.\n")
        return {"summary": output_file}

    unique_basis = sorted(
        set(basis_sizes),
        key=lambda x: ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"].index(x)
        if x in ["SZ", "DZ", "DZP", "SZP", "DZDP", "TZ", "TZP", "TZDP"]
        else 999,
    )

    with open(output_file, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("EOS PARAMETER CONVERGENCE STUDY\n")
        f.write("Basis Sizes + PAO Parameters (EnergyShift & SplitNorm)\n")
        f.write("=" * 90 + "\n\n")

        f.write(
            f"Tested {len(unique_basis)} basis sizes with {len(np.unique(energy_shifts))} "
            f"EnergyShift × {len(np.unique(split_norms))} SplitNorm values\n"
        )
        f.write(f"Total EOS calculations: {len(v0_values)}\n\n")

        # GLOBAL COMPARISON (Global table first)
        f.write("=" * 90 + "\n")
        f.write("OVERALL COMPARISON\n")
        f.write("=" * 90 + "\n\n")

        f.write("Optimal EOS Parameters for Each Basis:\n")
        f.write("-" * 110 + "\n")
        f.write(
            f"{'Basis':<10} {'V₀ (Ų)':<14} {'E₀ (eV)':<16} {'B₀ (GPa)':<12} {'Opt ES':<12} {'Opt SN':<10} {'Time (s)':<12}\n"
        )
        f.write("-" * 110 + "\n")

        for basis in unique_basis:
            mask = basis_sizes == basis
            if np.sum(mask) > 0:
                basis_v0 = v0_values[mask]
                basis_e0 = e0_values[mask]
                basis_b0 = b0_values[mask]
                basis_es = energy_shifts[mask]
                basis_sn = split_norms[mask]
                basis_time = run_times[mask]

                opt_idx = np.argmin(basis_e0)
                time_str = (
                    f"{basis_time[opt_idx]:<12.1f}"
                    if basis_time[opt_idx] is not None
                    else f"{'N/A':<12}"
                )
                f.write(
                    f"{basis:<10} {basis_v0[opt_idx]:<14.4f} {basis_e0[opt_idx]:<16.6f} "
                    f"{basis_b0[opt_idx]:<12.2f} {basis_es[opt_idx]:<12.3f} {basis_sn[opt_idx]:<10.2f} {time_str}\n"
                )

        f.write("\n")

        # Find global minimum
        global_min_idx = np.argmin(e0_values)
        f.write("Global Optimum:\n")
        f.write(f"  Basis Size      = {basis_sizes[global_min_idx]}\n")
        f.write(f"  PAO.EnergyShift = {energy_shifts[global_min_idx]:.3f} Ry\n")
        f.write(f"  PAO.SplitNorm   = {split_norms[global_min_idx]:.2f}\n")
        f.write(f"  V₀              = {v0_values[global_min_idx]:.4f} Ų\n")
        f.write(f"  E₀              = {e0_values[global_min_idx]:.6f} eV\n")
        f.write(f"  B₀              = {b0_values[global_min_idx]:.2f} GPa\n\n")

        # ============================================================================
        # EQUILIBRIUM PROPERTIES TABLE - Using optimal parameters for each basis
        # ============================================================================
        f.write("=" * 90 + "\n")
        f.write("EOS PARAMETERS FOR DIFFERENT BASIS SETS\n")
        f.write("=" * 90 + "\n\n")

        # Check if lattice data is available
        has_lattice_data = len(lattice_a) > 0

        if has_lattice_data:
            f.write("EQUILIBRIUM PROPERTIES (with lattice parameters):\n")
            f.write("-" * 120 + "\n")
            f.write(
                f"{'Basis Set':<12} {'V₀ (Ų)':<12} {'E₀ (eV)':<15} {'B₀ (GPa)':<12} "
                f"{'a (Å)':<10} {'b (Å)':<10} {'c (Å)':<10} {'α (°)':<8} {'β (°)':<8} {'γ (°)':<8}\n"
            )
            f.write("-" * 120 + "\n")
        else:
            f.write("EQUILIBRIUM PROPERTIES:\n")
            f.write("-" * 90 + "\n")
            f.write(
                f"{'Basis Set':<12} {'V₀ (Ų)':<14} {'E₀ (eV)':<18} {'B₀ (GPa)':<14} {'B₁':<12}\n"
            )
            f.write("-" * 90 + "\n")

        # Store optimal values for each basis for convergence analysis
        opt_v0_list = []
        opt_e0_list = []
        opt_b0_list = []
        opt_a_list = []
        opt_b_list = []
        opt_c_list = []

        for basis in unique_basis:
            mask = basis_sizes == basis
            if np.sum(mask) > 0:
                basis_v0 = v0_values[mask]
                basis_e0 = e0_values[mask]
                basis_b0 = b0_values[mask]

                opt_idx = np.argmin(basis_e0)
                v0_opt = basis_v0[opt_idx]
                e0_opt = basis_e0[opt_idx]
                b0_opt = basis_b0[opt_idx]

                # Store for convergence analysis
                opt_v0_list.append(v0_opt)
                opt_e0_list.append(e0_opt)
                opt_b0_list.append(b0_opt)

                if has_lattice_data:
                    # Use actual lattice parameters
                    basis_a = lattice_a[mask]
                    basis_b = lattice_b[mask]
                    basis_c = lattice_c[mask]
                    basis_alpha = lattice_alpha[mask]
                    basis_beta = lattice_beta[mask]
                    basis_gamma = lattice_gamma[mask]

                    a_opt = basis_a[opt_idx]
                    b_opt = basis_b[opt_idx]
                    c_opt = basis_c[opt_idx]
                    alpha_opt = basis_alpha[opt_idx]
                    beta_opt = basis_beta[opt_idx]
                    gamma_opt = basis_gamma[opt_idx]

                    opt_a_list.append(a_opt)
                    opt_b_list.append(b_opt)
                    opt_c_list.append(c_opt)

                    f.write(
                        f"{basis:<12} {v0_opt:<12.6f} {e0_opt:<15.6f} {b0_opt:<12.4f} "
                        f"{a_opt:<10.6f} {b_opt:<10.6f} {c_opt:<10.6f} "
                        f"{alpha_opt:<8.2f} {beta_opt:<8.2f} {gamma_opt:<8.2f}\n"
                    )
                else:
                    # Fallback: calculate assuming cubic (a³ = V₀)
                    a_opt = v0_opt ** (1 / 3)
                    opt_a_list.append(a_opt)
                    opt_b_list.append(a_opt)
                    opt_c_list.append(a_opt)

                    # B₁ is typically around 4 for most materials (assume 4.0 as placeholder)
                    b1_opt = 4.0

                    f.write(
                        f"{basis:<12} {v0_opt:<14.6f} {e0_opt:<18.6f} {b0_opt:<14.4f} {b1_opt:<12.1f}\n"
                    )

        f.write("\n")

        # ============================================================================
        # CONVERGENCE ANALYSIS
        # ============================================================================
        f.write("=" * 90 + "\n")
        f.write("CONVERGENCE ANALYSIS\n")
        f.write("=" * 90 + "\n\n")

        opt_v0_arr = np.array(opt_v0_list)
        _opt_e0_arr = np.array(opt_e0_list)
        opt_b0_arr = np.array(opt_b0_list)
        opt_a_arr = np.array(opt_a_list)

        v0_range = opt_v0_arr.max() - opt_v0_arr.min()
        v0_pct = (v0_range / opt_v0_arr.mean()) * 100
        b0_range = opt_b0_arr.max() - opt_b0_arr.min()
        b0_pct = (b0_range / opt_b0_arr.mean()) * 100
        a_range = opt_a_arr.max() - opt_a_arr.min()
        a_pct = (a_range / opt_a_arr.mean()) * 100

        f.write("Bulk Properties:\n")
        f.write(f"  V₀ range: {v0_range:.6f} Ų ({v0_pct:.2f}%)\n")
        f.write(f"  B₀ range: {b0_range:.4f} GPa ({b0_pct:.2f}%)\n\n")

        f.write("Lattice Parameters:\n")
        f.write(f"  a range: {a_range:.6f} Å ({a_pct:.2f}%)\n")

        if has_lattice_data and len(opt_b_list) > 0:
            # Check if system is cubic (b ≈ a and c ≈ a)
            opt_b_arr = np.array(opt_b_list)
            opt_c_arr = np.array(opt_c_list)
            b_range = opt_b_arr.max() - opt_b_arr.min()
            c_range = opt_c_arr.max() - opt_c_arr.min()
            b_pct = (b_range / opt_b_arr.mean()) * 100 if opt_b_arr.mean() > 0 else 0
            c_pct = (c_range / opt_c_arr.mean()) * 100 if opt_c_arr.mean() > 0 else 0

            is_cubic = np.allclose(opt_a_arr, opt_b_arr, rtol=0.01) and np.allclose(
                opt_a_arr, opt_c_arr, rtol=0.01
            )
            if is_cubic:
                f.write("  (Cubic system: a ≈ b ≈ c)\n")
            else:
                f.write(f"  b range: {b_range:.6f} Å ({b_pct:.2f}%)\n")
                f.write(f"  c range: {c_range:.6f} Å ({c_pct:.2f}%)\n")
        else:
            f.write("  (Cubic approximation from V₀)\n")

        f.write(
            f"  Lattice constant 'a' converges from {opt_a_arr.min():.6f} Å to {opt_a_arr.max():.6f} Å\n\n"
        )

        f.write("Recommendation: Choose basis set where parameters have converged\n")
        f.write("(minimal change with further increase in basis quality)\n\n")

        f.write("Convergence Criteria:\n")
        f.write("  Excellent: < 0.5% variation\n")
        f.write("  Good:      < 1.0% variation\n")
        f.write("  Fair:      < 2.0% variation\n")
        f.write("  Poor:      > 2.0% variation (need higher basis quality)\n\n")

        # Assessment
        def assess_convergence(pct):
            if pct < 0.5:
                return "Excellent ✓"
            elif pct < 1.0:
                return "Good ✓"
            elif pct < 2.0:
                return "Fair ⚠"
            else:
                return "Poor ✗"

        f.write("Convergence Status:\n")
        f.write(f"  V₀: {assess_convergence(v0_pct)}\n")
        f.write(f"  B₀: {assess_convergence(b0_pct)}\n")
        f.write(f"  a:  {assess_convergence(a_pct)}\n\n")

        f.write("=" * 90 + "\n\n")

        # Per-basis detailed sections
        for basis in unique_basis:
            mask = basis_sizes == basis
            n_calcs = np.sum(mask)

            f.write("=" * 90 + "\n")
            f.write(f"BASIS SIZE: {basis} ({n_calcs} EOS calculations)\n")
            f.write("=" * 90 + "\n\n")

            basis_v0 = v0_values[mask]
            basis_e0 = e0_values[mask]
            basis_b0 = b0_values[mask]
            basis_es = energy_shifts[mask]
            basis_sn = split_norms[mask]

            # Find optimal for this basis
            optimal_idx = np.argmin(basis_e0)

            f.write(f"Optimal Parameters for {basis}:\n")
            f.write(f"  PAO.EnergyShift = {basis_es[optimal_idx]:.3f} Ry\n")
            f.write(f"  PAO.SplitNorm   = {basis_sn[optimal_idx]:.2f}\n")
            f.write(f"  V₀              = {basis_v0[optimal_idx]:.4f} Ų\n")
            f.write(f"  E₀              = {basis_e0[optimal_idx]:.6f} eV\n")
            f.write(f"  B₀              = {basis_b0[optimal_idx]:.2f} GPa\n\n")

            # Parameter variation
            v0_var = (basis_v0.max() - basis_v0.min()) / basis_v0.mean() * 100
            e0_var = (basis_e0.max() - basis_e0.min()) * 1000  # meV
            b0_var = (basis_b0.max() - basis_b0.min()) / basis_b0.mean() * 100

            f.write(f"Parameter Variation for {basis}:\n")
            f.write(
                f"  V₀ range:   {basis_v0.min():.4f} - {basis_v0.max():.4f} Ų ({v0_var:.2f}%)\n"
            )
            f.write(
                f"  E₀ range:   {basis_e0.min():.6f} - {basis_e0.max():.6f} eV ({e0_var:.2f} meV)\n"
            )
            f.write(
                f"  B₀ range:   {basis_b0.min():.2f} - {basis_b0.max():.2f} GPa ({b0_var:.2f}%)\n\n"
            )

            # Convergence assessment
            v0_threshold = 0.5
            e0_threshold = 5.0
            b0_threshold = 2.0

            v0_status = (
                "✓"
                if v0_var < v0_threshold
                else "⚠"
                if v0_var < v0_threshold * 2
                else "✗"
            )
            e0_status = (
                "✓"
                if e0_var < e0_threshold
                else "⚠"
                if e0_var < e0_threshold * 2
                else "✗"
            )
            b0_status = (
                "✓"
                if b0_var < b0_threshold
                else "⚠"
                if b0_var < b0_threshold * 2
                else "✗"
            )

            f.write("Convergence Assessment:\n")
            f.write(
                f"  {v0_status} V₀: {v0_var:.2f}% variation (threshold: {v0_threshold}%)\n"
            )
            f.write(
                f"  {e0_status} E₀: {e0_var:.2f} meV variation (threshold: {e0_threshold} meV)\n"
            )
            f.write(
                f"  {b0_status} B₀: {b0_var:.2f}% variation (threshold: {b0_threshold}%)\n\n"
            )

            # Detailed results table for this basis
            f.write(f"Detailed Results for {basis} Basis:\n")
            f.write("-" * 90 + "\n")
            f.write(
                f"{'ES (Ry)':<12} {'SN':<10} {'V₀ (Ų)':<14} {'E₀ (eV)':<16} {'B₀ (GPa)':<12}\n"
            )
            f.write("-" * 90 + "\n")

            sort_idx = np.argsort(basis_e0)
            for idx in sort_idx:
                marker = " ★" if idx == optimal_idx else ""
                f.write(
                    f"{basis_es[idx]:<12.3f} {basis_sn[idx]:<10.2f} "
                    f"{basis_v0[idx]:<14.4f} {basis_e0[idx]:<16.6f} "
                    f"{basis_b0[idx]:<12.2f}{marker}\n"
                )

            f.write("-" * 90 + "\n")
            f.write("★ = Optimal parameters for this basis size\n")
            f.write("\n")

        # Final recommendations
        f.write("=" * 90 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 90 + "\n\n")

        global_min_e0 = e0_values.min()

        # Find smallest converged basis
        for basis in unique_basis:
            mask = basis_sizes == basis
            if np.sum(mask) > 0:
                min_e0 = np.min(e0_values[mask])
                e_diff = (min_e0 - global_min_e0) * 1000  # meV

                if e_diff < 5.0:
                    f.write(f"✓ Recommended basis: {basis}\n")
                    f.write(f"  Energy difference from best: {e_diff:.4f} meV\n\n")

                    # Get optimal parameters
                    basis_e0 = e0_values[mask]
                    basis_es = energy_shifts[mask]
                    basis_sn = split_norms[mask]
                    opt_idx = np.argmin(basis_e0)

                    f.write("  Optimal settings for production:\n")
                    f.write(f"    PAO.BasisSize:   {basis}\n")
                    f.write(f"    PAO.EnergyShift: {basis_es[opt_idx]:.3f} Ry\n")
                    f.write(f"    PAO.SplitNorm:   {basis_sn[opt_idx]:.2f}\n")
                    break

        f.write("\n")
        f.write("Balance between accuracy and computational cost:\n")
        f.write("  → For production: Use smallest basis converged within 5 meV\n")
        f.write(
            "  → For high accuracy: Use largest tested basis with optimal parameters\n"
        )
        f.write("  → Consider EOS fitting quality and computational resources\n")

    logger.info(f"EOS parameter summary written to {output_file}")
    if console:
        console.print(f"[green]Summary written to: {output_file}[/green]")

    return {"summary": output_file}


@job
def generate_eos_full_basis_outputs(
    data: dict[str, Any],
    output_dir: str | Path = ".",
) -> dict[str, Any]:
    """
    Generate unified EOS full basis convergence outputs in ONE directory.

    Creates all plots and summaries with proper naming in a single location:
    - Combined EOS fits plot: eos_parameter_fits_combined.png
    - Timing analysis plot: eos_timing.png
    - Summary file: eos_parameter_convergence_summary.txt

    Parameters
    ----------
    data : dict
        Collected EOS data from collect_eos_parameter_data
    output_dir : str | Path
        Output directory (default: current directory)

    Returns
    -------
    dict
        Dictionary with paths to all generated files
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_files: dict[str, Any] = {
        "combined_plot": None,
        "timing_plot": None,
        "summary": None,
    }

    logger.info(f"Generating unified EOS full basis outputs in {output_path}")
    if console:
        console.print("\n[bold cyan]Generating EOS Full Basis Outputs[/bold cyan]")
        console.print(f"  Output directory: {output_path}")

    # Generate combined EOS plot
    plot_file = output_path / "eos_parameter_fits_combined.png"
    result = plot_eos_parameter_fits_from_data.__wrapped__(
        data=data, output_file=str(plot_file)
    )
    output_files["combined_plot"] = result.get("plot")
    logger.info(f"Generated combined plot: {plot_file}")

    # Generate timing plot
    timing_file = output_path / "eos_timing.png"
    result = plot_eos_parameter_timing.__wrapped__(
        data=data, output_file=str(timing_file)
    )
    output_files["timing_plot"] = result.get("plot")
    logger.info(f"Generated timing plot: {timing_file}")

    # Generate summary
    summary_file = output_path / "eos_parameter_convergence_summary.txt"
    result = write_eos_parameter_summary.__wrapped__(
        data=data, output_file=str(summary_file)
    )
    output_files["summary"] = result.get("summary")
    logger.info(f"Generated summary: {summary_file}")

    if console:
        console.print(f"\n[green]✓ All outputs generated in: {output_path}[/green]")
        console.print("  - Combined plot: eos_parameter_fits_combined.png")
        console.print("  - Timing plot: eos_timing.png")
        console.print("  - Summary: eos_parameter_convergence_summary.txt")

    return output_files
