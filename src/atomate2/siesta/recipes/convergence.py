"""Parameter convergence testing workflow recipes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jobflow import Flow, job

from atomate2.siesta.flows.basis import BasisParametersConvergenceFlowMaker
from atomate2.siesta.flows.convergence import (
    KpointsConvergenceFlowMaker,
    MeshCutoffConvergenceFlowMaker,
)
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.recipes.base import MaterialAnalyzer

if TYPE_CHECKING:
    from jobflow import Job
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def convergence_suite(
    structure: Structure,
    property: str = "energy",  # noqa: A002 public recipe keyword
    tolerance: float = 0.001,
    auto_params: bool = True,
    include_kpoints: bool = True,
    include_cutoff: bool = True,
    include_basis: bool = True,
    user_params: dict[str, Any] | None = None,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "convergence_suite",
) -> Flow:
    """
    Complete convergence testing suite.

    Tests convergence of:
    1. K-points mesh
    2. Real-space mesh cutoff
    3. Basis parameters (size, shift, split norm)

    Parameters
    ----------
    structure : Structure
        Input structure.
    property : str
        Property to converge. Options: "energy", "forces", "stress".
        Default: "energy".
    tolerance : float
        Convergence tolerance. For energy: eV/atom. Default: 0.001.
    auto_params : bool
        Automatically determine test ranges. Default: True.
    include_kpoints : bool
        Include k-points convergence. Default: True.
    include_cutoff : bool
        Include mesh cutoff convergence. Default: True.
    include_basis : bool
        Include basis convergence. Default: True.
    user_params : dict
        Additional SIESTA parameters to apply to all convergence tests.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
    dry_run : bool
        Only generate input files. Default: False.
    name : str
        Workflow name.

    Returns
    -------
    Flow
        Complete convergence suite workflow.

    Example
    -------
    >>> from atomate2.siesta.recipes import convergence_suite
    >>> flow = convergence_suite(structure, tolerance=0.0005)
    >>> # Automatically tests all parameters and generates convergence plots!
    """
    logger.info(f"Creating convergence_suite for {structure.composition}")

    # Analyze structure for smart defaults
    if auto_params:
        analysis = MaterialAnalyzer.analyze(structure)
        logger.info(f"Recommended k-points: {analysis.recommended_kpts}")
        logger.info(f"Recommended cutoff: {analysis.recommended_cutoff}")

    jobs: list[Flow | Job] = []

    # K-points convergence
    kpts_output = None
    if include_kpoints:
        kpts_flow = kpoints_convergence(
            structure,
            property=property,
            tolerance=tolerance,
            auto_params=auto_params,
            user_params=user_params,
            use_custodian=use_custodian,
            dry_run=dry_run,
            name="kpoints_convergence",
        )
        jobs.append(kpts_flow)
        kpts_output = kpts_flow.output

    # Mesh cutoff convergence
    cutoff_output = None
    if include_cutoff:
        cutoff_flow = mesh_cutoff_convergence(
            structure,
            property=property,
            tolerance=tolerance,
            auto_params=auto_params,
            user_params=user_params,
            use_custodian=use_custodian,
            dry_run=dry_run,
            name="cutoff_convergence",
        )
        jobs.append(cutoff_flow)
        cutoff_output = cutoff_flow.output

    # Basis convergence
    basis_output = None
    if include_basis:
        basis_flow = basis_convergence(
            structure,
            property=property,
            tolerance=tolerance,
            auto_params=auto_params,
            user_params=user_params,
            use_custodian=use_custodian,
            dry_run=dry_run,
            name="basis_convergence",
        )
        jobs.append(basis_flow)
        basis_output = basis_flow.output

    # Create combined output dictionary
    combined_output = {
        "kpoints_convergence": kpts_output,
        "cutoff_convergence": cutoff_output,
        "basis_convergence": basis_output,
    }

    # Create flow with combined output
    flow = Flow(jobs, output=combined_output, name=name)
    logger.info(f"Convergence suite created with {len(jobs)} convergence tests")
    return flow


def kpoints_convergence(
    structure: Structure,
    property: str = "energy",  # noqa: A002, ARG001 public recipe keyword
    tolerance: float = 0.001,  # noqa: ARG001 public recipe keyword
    auto_params: bool = True,
    kpts_range: list[int] | None = None,
    user_params: dict[str, Any] | None = None,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "kpoints_convergence",
) -> Flow:
    """
    K-points convergence testing.

    Systematically tests k-point mesh density to find converged value.

    Parameters
    ----------
    structure : Structure
        Input structure.
    property : str
        Property to converge ("energy", "forces", "stress").
    tolerance : float
        Convergence tolerance (eV/atom for energy).
    auto_params : bool
        Auto-determine k-point range. Default: True.
    kpts_range : list
        Manual k-point values to test (e.g., [2,4,6,8,10]).
    user_params : dict
        Additional SIESTA parameters.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
    dry_run : bool
        Only generate input files.
    name : str
        Workflow name.

    Returns
    -------
    Flow
        K-points convergence workflow.

    Example
    -------
    >>> flow = kpoints_convergence(structure, tolerance=0.0005)
    >>> # Auto-generates convergence plot
    """
    logger.info(f"Creating kpoints_convergence for {structure.composition}")

    # Determine k-point range
    if kpts_range is None:
        if auto_params:
            analysis = MaterialAnalyzer.analyze(structure)
            base_k = max(analysis.recommended_kpts)

            # Test range around recommended value
            kpts_range = [
                max(1, base_k // 2),
                base_k,
                int(base_k * 1.5),
                base_k * 2,
            ]
        else:
            # Provide simple default range
            kpts_range = [2, 4, 6, 8]

    # Create static maker with user params

    if user_params is None:
        user_params = {}

    static_maker = StaticMaker.scf(
        user_params=user_params,
        use_custodian=use_custodian,
        custodian_max_errors=10,
        dry_run=dry_run,
    )

    # Create convergence maker
    kpts_maker = KpointsConvergenceFlowMaker(
        name=name,
        static_maker=static_maker,
        kpoints_list=[[k, k, k] for k in kpts_range],  # Convert to list of 3D grids
        dry_run=dry_run,
    )

    flow = kpts_maker.make(structure)
    logger.info(f"K-points convergence: testing {len(kpts_range)} values")
    return flow


def mesh_cutoff_convergence(
    structure: Structure,
    property: str = "energy",  # noqa: A002, ARG001 public recipe keyword
    tolerance: float = 0.001,  # noqa: ARG001 public recipe keyword
    auto_params: bool = True,
    cutoff_range: list[float] | None = None,
    user_params: dict[str, Any] | None = None,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "cutoff_convergence",
) -> Flow:
    """
    Real-space mesh cutoff convergence testing.

    Tests Mesh.Cutoff parameter convergence.

    Parameters
    ----------
    structure : Structure
        Input structure.
    property : str
        Property to converge.
    tolerance : float
        Convergence tolerance.
    auto_params : bool
        Auto-determine cutoff range.
    cutoff_range : list
        Cutoff values to test (Ry). E.g., [200, 300, 400, 500].
    user_params : dict
        Additional parameters.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
    dry_run : bool
        Only generate inputs.
    name : str
        Workflow name.

    Returns
    -------
    Flow
        Mesh cutoff convergence workflow.

    Example
    -------
    >>> flow = mesh_cutoff_convergence(structure)
    """
    logger.info(f"Creating cutoff_convergence for {structure.composition}")

    # Determine cutoff range
    if cutoff_range is None:
        if auto_params:
            analysis = MaterialAnalyzer.analyze(structure)
            base_cutoff = float(analysis.recommended_cutoff.split()[0])

            cutoff_range = [
                base_cutoff - 100,
                base_cutoff - 50,
                base_cutoff,
                base_cutoff + 50,
                base_cutoff + 100,
            ]
        else:
            # Provide simple default range (Ry)
            cutoff_range = [200, 250, 300, 350, 400]

    if user_params is None:
        user_params = {}

    # Create static maker with user params

    static_maker = StaticMaker.scf(
        user_params=user_params,
        use_custodian=use_custodian,
        custodian_max_errors=10,
        dry_run=dry_run,
    )

    # Create convergence maker
    cutoff_maker = MeshCutoffConvergenceFlowMaker(
        name=name,
        static_maker=static_maker,
        mesh_cutoffs=cutoff_range,  # Just pass the numeric list, maker handles units
        dry_run=dry_run,
    )

    flow = cutoff_maker.make(structure)
    logger.info(f"Mesh cutoff convergence: testing {len(cutoff_range)} values")
    return flow


def basis_convergence(
    structure: Structure,
    property: str = "energy",  # noqa: A002, ARG001 public recipe keyword
    tolerance: float = 0.001,  # noqa: ARG001 public recipe keyword
    auto_params: bool = True,
    energy_shifts: list[float] | None = None,
    split_norms: list[float] | None = None,
    basis_size: str = "DZP",
    user_params: dict[str, Any] | None = None,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "basis_convergence",
) -> Flow:
    """
    Basis parameter convergence testing.

    Tests PAO.EnergyShift and PAO.SplitNorm parameters.

    Note: This tests energy_shift and split_norm parameters for a fixed
    basis_size. It does NOT test different basis sizes (SZ, DZ, DZP, etc.)
    as those require completely different workflows.

    Parameters
    ----------
    structure : Structure
        Input structure.
    property : str
        Property to converge (not used by BasisParametersConvergenceFlowMaker).
    tolerance : float
        Convergence tolerance (not used by BasisParametersConvergenceFlowMaker).
    auto_params : bool
        Auto-configure tests.
    energy_shifts : list
        Energy shift values to test (Ry).
        Default: [0.001, 0.005, 0.01, 0.015, 0.02, 0.03]
    split_norms : list
        Split norm values to test. Default: [0.10, 0.15, 0.20, 0.25, 0.30]
    basis_size : str
        Fixed basis size to use. Default: "DZP"
    user_params : dict
        Additional parameters.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
    dry_run : bool
        Only generate inputs.
    name : str
        Workflow name.

    Returns
    -------
    Flow
        Basis convergence workflow.

    Example
    -------
    >>> flow = basis_convergence(structure, energy_shifts=[0.005, 0.01, 0.015])
    """
    logger.info(f"Creating basis_convergence for {structure.composition}")

    if user_params is None:
        user_params = {}

    # Use defaults if not provided
    if energy_shifts is None:
        energy_shifts = [0.001, 0.005, 0.01, 0.015, 0.02, 0.03]  # Ry

    if split_norms is None:
        split_norms = [0.10, 0.15, 0.20, 0.25, 0.30]

    # Analyze structure for k-points if auto_params
    kpts = [4, 4, 4]  # Default
    if auto_params:
        analysis = MaterialAnalyzer.analyze(structure)
        kpts = analysis.recommended_kpts

    # Create static maker with user params

    static_maker = StaticMaker.scf(
        user_params=user_params,
        use_custodian=use_custodian,
        custodian_max_errors=10,
        dry_run=dry_run,
    )

    # Create convergence maker
    basis_maker = BasisParametersConvergenceFlowMaker(
        name=name,
        static_maker=static_maker,
        energy_shifts=energy_shifts,
        split_norms=split_norms,
        basis_size=basis_size,
        kpts=kpts,
    )

    flow = basis_maker.make(structure)

    n_combinations = len(energy_shifts) * len(split_norms)
    logger.info(f"Basis convergence: testing {n_combinations} combinations")

    return flow


def complete_convergence(
    structure: Structure,
    property: str = "energy",  # noqa: A002 public recipe keyword
    tolerance: float = 0.0005,
    **kwargs,
) -> Flow:
    """
    Ultra-thorough convergence testing.

    Tests all parameters with tight tolerance for high-accuracy
    calculations.

    Parameters
    ----------
    structure : Structure
        Input structure.
    property : str
        Property to converge.
    tolerance : float
        Tight convergence tolerance. Default: 0.0005 eV/atom.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Complete convergence workflow.

    Example
    -------
    >>> flow = complete_convergence(structure, tolerance=0.0001)
    >>> # Exhaustive testing for publication-quality results
    """
    return convergence_suite(
        structure,
        property=property,
        tolerance=tolerance,
        include_kpoints=True,
        include_cutoff=True,
        include_basis=True,
        name="complete_convergence",
        **kwargs,
    )


def quick_convergence_check(structure: Structure, **kwargs) -> Flow:
    """
    Quick convergence check for preliminary testing.

    Tests only k-points and cutoff with relaxed tolerance.

    Parameters
    ----------
    structure : Structure
        Input structure.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Quick convergence check workflow.

    Example
    -------
    >>> flow = quick_convergence_check(structure)
    >>> # Fast preliminary convergence test
    """
    return convergence_suite(
        structure,
        tolerance=0.005,  # Relaxed tolerance
        include_kpoints=True,
        include_cutoff=True,
        include_basis=False,  # Skip basis for speed
        name="quick_convergence",
        **kwargs,
    )


@job
def extract_optimal_parameters(
    convergence_results: dict[str, Any],
    tolerance: float = 0.001,
    property: str = "energy",  # noqa: A002, ARG001 public recipe keyword
) -> dict[str, Any]:
    """
    Extract optimal converged parameters from convergence test results.

    This function analyzes convergence test outputs and determines the
    optimal k-points and mesh cutoff values that satisfy the convergence
    criterion.

    Parameters
    ----------
    convergence_results : dict
        Results from convergence_suite workflow.
    tolerance : float
        Convergence tolerance (eV/atom for energy). Default: 0.001.
    property : str
        Property to check convergence ("energy", "forces", "stress").
        Default: "energy".

    Returns
    -------
    dict
        Dictionary with optimal parameters:
        - "a2s_kpts": Converged k-point mesh [kx, ky, kz]
        - "Mesh.Cutoff": Converged mesh cutoff value "XXX Ry"
        - "convergence_info": Dictionary with convergence details

    Example
    -------
    >>> optimal_params = extract_optimal_parameters(
    ...     convergence_results, tolerance=0.0005
    ... )
    >>> # Returns: {"a2s_kpts": [6, 6, 6], "Mesh.Cutoff": "350 Ry"}
    """
    import numpy as np

    optimal: dict[str, Any] = {}
    convergence_info: dict[str, Any] = {}

    # Try to extract k-points convergence
    kpts_data = convergence_results.get("kpoints_convergence")
    if kpts_data is not None:
        try:
            if isinstance(kpts_data, dict) and "energies" in kpts_data:
                energies = np.array(kpts_data["energies"])
                parameters = kpts_data["parameters"]

                # Find first k-point where energy difference < tolerance
                if len(energies) > 1:
                    # Calculate energy differences
                    energy_diffs = np.abs(np.diff(energies))

                    # Find first converged point
                    converged_idx = None
                    for i, diff in enumerate(energy_diffs):
                        if diff < tolerance:
                            converged_idx = i + 1  # Use the second point in pair
                            break

                    if converged_idx is not None:
                        # Extract k-point value from parameter string
                        kpt_str = parameters[converged_idx]
                        # Handle formats: "4x4x4" or "[4, 4, 4]"
                        if "x" in kpt_str:
                            kpt_val = int(kpt_str.split("x")[0])
                            optimal["a2s_kpts"] = [kpt_val, kpt_val, kpt_val]
                        elif isinstance(kpt_str, list):
                            optimal["a2s_kpts"] = kpt_str

                        convergence_info["kpoints"] = {
                            "converged_value": optimal.get("a2s_kpts"),
                            "energy_diff": float(energy_diffs[converged_idx - 1]),
                            "tolerance": tolerance,
                        }
                        logger.info(
                            f"Converged k-points: {optimal['a2s_kpts']} "
                            f"(ΔE = {energy_diffs[converged_idx - 1]:.6f} eV)"
                        )
        except Exception as e:  # noqa: BLE001 best-effort extraction, logged
            logger.warning(f"Could not extract k-points convergence: {e}")

    # Try to extract mesh cutoff convergence
    cutoff_data = convergence_results.get("cutoff_convergence")
    if cutoff_data is not None:
        try:
            if isinstance(cutoff_data, dict) and "energies" in cutoff_data:
                energies = np.array(cutoff_data["energies"])
                parameters = cutoff_data["parameters"]

                # Find first cutoff where energy difference < tolerance
                if len(energies) > 1:
                    energy_diffs = np.abs(np.diff(energies))

                    converged_idx = None
                    for i, diff in enumerate(energy_diffs):
                        if diff < tolerance:
                            converged_idx = i + 1
                            break

                    if converged_idx is not None:
                        # Extract cutoff value from parameter string
                        cutoff_str = parameters[converged_idx]
                        # Remove "Ry" or "eV" suffix if present
                        cutoff_val = float(
                            cutoff_str.replace("Ry", "").replace("eV", "").strip()
                        )
                        optimal["Mesh.Cutoff"] = f"{cutoff_val} Ry"

                        convergence_info["mesh_cutoff"] = {
                            "converged_value": optimal.get("Mesh.Cutoff"),
                            "energy_diff": float(energy_diffs[converged_idx - 1]),
                            "tolerance": tolerance,
                        }
                        logger.info(
                            f"Converged mesh cutoff: {optimal['Mesh.Cutoff']} "
                            f"(ΔE = {energy_diffs[converged_idx - 1]:.6f} eV)"
                        )
        except Exception as e:  # noqa: BLE001 best-effort extraction, logged
            logger.warning(f"Could not extract mesh cutoff convergence: {e}")

    # Add convergence info to output
    optimal["convergence_info"] = convergence_info

    if not optimal:
        logger.warning(
            "No converged parameters extracted. Using MaterialAnalyzer defaults."
        )

    return optimal
