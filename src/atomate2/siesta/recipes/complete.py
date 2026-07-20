"""Complete material characterization workflow recipes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jobflow import Flow, Response, job

from atomate2.siesta.recipes.convergence import (
    convergence_suite,
    extract_optimal_parameters,
)
from atomate2.siesta.recipes.electronic import electronic_properties
from atomate2.siesta.recipes.mechanical import mechanical_properties
from atomate2.siesta.recipes.thermal import thermal_properties

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def complete_material_study(
    structure: Structure,
    properties: list[str] | None = None,
    auto_params: bool = True,
    user_params: dict[str, Any] | None = None,
    phonon_user_params: dict[str, Any] | None = None,
    tier: str | None = None,
    preset: str | None = None,
    supercell_matrix: tuple[int, int, int] | None = None,
    temperature_range: tuple[float, float, float] = (0, 1000, 10),
    ignore_imaginary_modes: bool = False,
    test_convergence: bool = False,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "complete_material_study",
) -> Flow:
    """
    Complete material characterization workflow.

    The ultimate one-liner for comprehensive material study!

    Performs a complete characterization based on material type:
    - Electronic properties (bands, DOS, optical)
    - Mechanical properties (elastic constants, bulk modulus)
    - Thermal properties (phonons, QHA, thermal expansion)
    - Convergence testing (optional)

    Parameters
    ----------
    structure : Structure
        Input structure.
    properties : list of str
        Properties to calculate. Options: "electronic", "mechanical", "thermal", "all".
        Default: Auto-detect based on material type.
    auto_params : bool
        Automatically determine optimal parameters. Default: True.
    user_params : dict
        Override specific SIESTA parameters.
    phonon_user_params : dict
        Override specific SIESTA parameters for phonon force calculations.
        If None, uses user_params with automatic k-point scaling for supercells.
    tier : str
        Tier level. Auto-detected if None.
    preset : str
        Preset name. Auto-detected if None.
    supercell_matrix : tuple
        Supercell size for phonon calculation (only used if "thermal" in properties).
        Auto-detected if None.
    temperature_range : tuple
        (T_min, T_max, T_step) in Kelvin for QHA (only used if "thermal" in properties).
        Default: (0, 1000, 10).
    ignore_imaginary_modes : bool
        Use all volumes even if they have imaginary frequencies (only used if
        "thermal" in properties).
        Default: False.
    test_convergence : bool
        Perform convergence testing first. Default: False.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
    dry_run : bool
        Only generate input files. Default: False.
    name : str
        Workflow name.

    Returns
    -------
    Flow
        Complete characterization workflow.

    Example
    -------
    >>> from atomate2.siesta.recipes import RecipeBook
    >>> from pymatgen.core import Structure
    >>>
    >>> structure = Structure.from_file("POSCAR")
    >>>
    >>> # Complete characterization in ONE LINE!
    >>> flow = RecipeBook.complete_material_study(structure)
    >>>
    >>> # Or be specific about properties
    >>> flow = RecipeBook.complete_material_study(
    ...     structure, properties=["electronic", "mechanical"]
    ... )
    >>>
    >>> # With convergence testing
    >>> flow = RecipeBook.complete_material_study(structure, test_convergence=True)
    """
    logger.info(f"Creating complete_material_study for {structure.composition}")

    # Auto-determine properties to calculate
    if properties is None or "all" in properties:
        properties = ["electronic", "mechanical", "thermal"]
    elif isinstance(properties, str):
        properties = [properties]

    jobs = []

    # Convergence testing (optional, runs first)
    if test_convergence:
        logger.info("Adding convergence testing with automatic parameter extraction")
        conv_flow = convergence_suite(
            structure,
            auto_params=auto_params,
            use_custodian=use_custodian,
            dry_run=dry_run,
            name="convergence",
        )
        jobs.append(conv_flow)

        # Extract optimal parameters from convergence results
        extract_job = extract_optimal_parameters(
            convergence_results=conv_flow.output, tolerance=0.001, property="energy"
        )
        extract_job.name = "extract_converged_params"
        jobs.append(extract_job)

        # Create a wrapper job that builds property workflows with converged parameters
        @job
        def create_property_workflows_with_convergence(
            conv_params: dict,
            structure: Structure,
            properties: list[str],
            user_params: dict | None,
            phonon_user_params: dict | None,
            tier: str | None,
            preset: str | None,
            supercell_matrix: tuple[int, int, int] | None,
            temperature_range: tuple[float, float, float],
            ignore_imaginary_modes: bool,
            dry_run: bool,
        ) -> Response:
            """Create property workflows using converged parameters."""
            # Merge converged params with user params (user takes precedence)
            merged_params = conv_params.copy()
            merged_params.pop("convergence_info", None)
            if user_params:
                merged_params.update(user_params)

            logger.info(f"Using converged parameters: {merged_params}")

            property_jobs = []

            # Electronic properties
            if "electronic" in properties:
                elec_flow = electronic_properties(
                    structure,
                    relax=True,
                    auto_params=False,  # Don't auto-detect, use converged params
                    user_params=merged_params,
                    tier=tier,
                    preset=preset,
                    use_custodian=use_custodian,
                    dry_run=dry_run,
                    name="electronic",
                )
                property_jobs.append(elec_flow)

            # Mechanical properties
            if "mechanical" in properties:
                mech_flow = mechanical_properties(
                    structure,
                    auto_params=False,
                    user_params=merged_params,
                    tier=tier,
                    preset=preset,
                    include_eos=True,
                    include_elastic=True,
                    use_custodian=use_custodian,
                    dry_run=dry_run,
                    name="mechanical",
                )
                property_jobs.append(mech_flow)

            # Thermal properties
            if "thermal" in properties:
                therm_flow = thermal_properties(
                    structure,
                    auto_params=False,
                    user_params=merged_params,
                    phonon_user_params=phonon_user_params,
                    tier=tier,
                    preset=preset,
                    supercell_matrix=supercell_matrix,
                    temperature_range=temperature_range,
                    ignore_imaginary_modes=ignore_imaginary_modes,
                    include_phonons=True,
                    include_gruneisen=True,
                    include_qha=True,
                    use_custodian=use_custodian,
                    dry_run=dry_run,
                    name="thermal",
                )
                property_jobs.append(therm_flow)

            # Use Response to replace this job with the property workflow jobs
            return Response(
                replace=property_jobs,
                output=property_jobs[-1].output if property_jobs else None,
            )

        # Create the wrapper job that depends on convergence results
        property_flow_job = create_property_workflows_with_convergence(
            conv_params=extract_job.output,
            structure=structure,
            properties=properties,
            user_params=user_params,
            phonon_user_params=phonon_user_params,
            tier=tier,
            preset=preset,
            supercell_matrix=supercell_matrix,
            temperature_range=temperature_range,
            ignore_imaginary_modes=ignore_imaginary_modes,
            dry_run=dry_run,
        )
        property_flow_job.name = "create_property_workflows"
        jobs.append(property_flow_job)

    else:
        # No convergence testing - create property workflows directly
        # Electronic properties
        if "electronic" in properties:
            logger.info("Adding electronic properties workflow")
            elec_flow = electronic_properties(
                structure,
                relax=True,
                auto_params=auto_params,
                user_params=user_params,
                tier=tier,
                preset=preset,
                use_custodian=use_custodian,
                dry_run=dry_run,
                name="electronic",
            )
            jobs.append(elec_flow)

        # Mechanical properties
        if "mechanical" in properties:
            logger.info("Adding mechanical properties workflow")
            mech_flow = mechanical_properties(
                structure,
                auto_params=auto_params,
                user_params=user_params,
                tier=tier,
                preset=preset,
                include_eos=True,
                include_elastic=True,
                use_custodian=use_custodian,
                dry_run=dry_run,
                name="mechanical",
            )
            jobs.append(mech_flow)

        # Thermal properties
        if "thermal" in properties:
            logger.info("Adding thermal properties workflow")
            therm_flow = thermal_properties(
                structure,
                auto_params=auto_params,
                user_params=user_params,
                phonon_user_params=phonon_user_params,
                tier=tier,
                preset=preset,
                supercell_matrix=supercell_matrix,
                temperature_range=temperature_range,
                ignore_imaginary_modes=ignore_imaginary_modes,
                include_phonons=True,
                include_gruneisen=True,
                include_qha=True,
                use_custodian=use_custodian,
                dry_run=dry_run,
                name="thermal",
            )
            jobs.append(therm_flow)

    # Create master flow
    flow = Flow(jobs, output=jobs[-1].output if jobs else None, name=name)

    total_jobs = sum(1 for _ in flow)
    logger.info(
        f"Complete material study created: {len(properties)} property categories, "
        f"~{total_jobs} total jobs"
    )

    return flow


def quick_characterization(structure: Structure, **kwargs) -> Flow:
    """
    Quick material characterization.

    Fast preliminary study with essential properties only:
    - Structure relaxation
    - Band structure
    - Bulk modulus

    Parameters
    ----------
    structure : Structure
        Input structure.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Quick characterization workflow.

    Example
    -------
    >>> flow = quick_characterization(structure)
    >>> # Fast preliminary study (~1-2 hours)
    """
    logger.info("Creating quick_characterization")

    return complete_material_study(
        structure,
        properties=["electronic"],
        test_convergence=False,
        name="quick_characterization",
        **kwargs,
    )


def battery_cathode_screening(structure: Structure, **kwargs) -> Flow:
    """
    Battery cathode material screening workflow.

    Optimized for Li-ion cathode characterization:
    - Electronic structure (band gap, DOS)
    - Structural stability (elastic constants)
    - Volume changes (EOS)
    - Ionic mobility (NEB - future)

    Parameters
    ----------
    structure : Structure
        Cathode structure.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Battery cathode screening workflow.

    Example
    -------
    >>> flow = battery_cathode_screening(licoo2_structure)
    """
    logger.info("Creating battery_cathode_screening")

    return complete_material_study(
        structure,
        properties=["electronic", "mechanical"],
        preset="relax_standard",
        name="battery_cathode_screening",
        **kwargs,
    )


def thermoelectric_analysis(
    structure: Structure,
    supercell_matrix: tuple[int, int, int] | None = None,
    **kwargs,
) -> Flow:
    """
    Thermoelectric material characterization.

    Calculates properties relevant for thermoelectrics:
    - Electronic structure (band structure, DOS)
    - Thermal properties (phonons, thermal conductivity)
    - Mechanical stability (elastic constants)

    Parameters
    ----------
    structure : Structure
        Input structure.
    supercell_matrix : tuple
        Supercell size for phonon calculation. Auto-detected if None.
    **kwargs
        Additional parameters (phonon_user_params, temperature_range,
        ignore_imaginary_modes, etc.).

    Returns
    -------
    Flow
        Thermoelectric analysis workflow.

    Example
    -------
    >>> flow = thermoelectric_analysis(pbs_structure, supercell_matrix=(2, 2, 2))
    """
    logger.info("Creating thermoelectric_analysis")

    return complete_material_study(
        structure,
        properties=["electronic", "thermal", "mechanical"],
        supercell_matrix=supercell_matrix,
        name="thermoelectric_analysis",
        **kwargs,
    )


def high_temperature_ceramic(
    structure: Structure,
    max_temperature: float = 2000.0,
    supercell_matrix: tuple[int, int, int] | None = None,
    **kwargs,
) -> Flow:
    """
    High-temperature ceramic characterization.

    Focus on mechanical and thermal properties:
    - Elastic constants
    - Thermal expansion
    - High-temperature stability
    - Bulk modulus

    Parameters
    ----------
    structure : Structure
        Ceramic structure.
    max_temperature : float
        Maximum temperature in K. Default: 2000.
    supercell_matrix : tuple
        Supercell size for phonon calculation. Auto-detected if None.
    **kwargs
        Additional parameters (phonon_user_params, ignore_imaginary_modes, etc.).

    Returns
    -------
    Flow
        High-temperature ceramic workflow.

    Example
    -------
    >>> flow = high_temperature_ceramic(
    ...     alumina_structure, max_temperature=2500, supercell_matrix=(2, 2, 2)
    ... )
    """
    logger.info("Creating high_temperature_ceramic workflow")

    # Override thermal parameters for high temperature
    if "user_params" not in kwargs:
        kwargs["user_params"] = {}

    return complete_material_study(
        structure,
        properties=["mechanical", "thermal"],
        supercell_matrix=supercell_matrix,
        temperature_range=(0, max_temperature, 20),
        name="high_temperature_ceramic",
        **kwargs,
    )


def magnetic_material_study(structure: Structure, **kwargs) -> Flow:
    """
    Magnetic material characterization.

    Optimized for magnetic systems:
    - Spin-polarized electronic structure
    - Magnetic moments
    - Exchange coupling
    - Mechanical properties

    Parameters
    ----------
    structure : Structure
        Magnetic structure.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Magnetic material study workflow.

    Example
    -------
    >>> flow = magnetic_material_study(fe_structure)
    """
    logger.info("Creating magnetic_material_study")

    # Force spin-polarized calculation
    if "user_params" not in kwargs:
        kwargs["user_params"] = {}
    kwargs["user_params"]["Spin"] = "polarized"
    kwargs["preset"] = "magnetic_correlated"

    return complete_material_study(
        structure,
        properties=["electronic", "mechanical"],
        name="magnetic_material_study",
        **kwargs,
    )


def semiconductor_device_study(structure: Structure, **kwargs) -> Flow:
    """
    Semiconductor device characterization.

    Focus on electronic and optical properties:
    - Accurate band gap
    - Band structure
    - Optical absorption
    - Effective masses

    Parameters
    ----------
    structure : Structure
        Semiconductor structure.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Semiconductor device study workflow.

    Example
    -------
    >>> flow = semiconductor_device_study(si_structure)
    """
    logger.info("Creating semiconductor_device_study")

    kwargs["preset"] = "relax_standard"

    return complete_material_study(
        structure,
        properties=["electronic"],
        name="semiconductor_device_study",
        **kwargs,
    )


def structural_phase_transition(
    structure: Structure,
    supercell_matrix: tuple[int, int, int] | None = None,
    **kwargs,
) -> Flow:
    """
    Phase transition characterization.

    Studies structural stability:
    - Phonon stability check
    - Pressure-dependent structure
    - Elastic constants
    - Free energy vs temperature

    Parameters
    ----------
    structure : Structure
        Input structure.
    supercell_matrix : tuple
        Supercell size for phonon calculation. Auto-detected if None.
    **kwargs
        Additional parameters (phonon_user_params, temperature_range,
        ignore_imaginary_modes, etc.).

    Returns
    -------
    Flow
        Phase transition study workflow.

    Example
    -------
    >>> flow = structural_phase_transition(structure, supercell_matrix=(2, 2, 2))
    """
    logger.info("Creating structural_phase_transition study")

    return complete_material_study(
        structure,
        properties=["mechanical", "thermal"],
        supercell_matrix=supercell_matrix,
        name="phase_transition_study",
        **kwargs,
    )
