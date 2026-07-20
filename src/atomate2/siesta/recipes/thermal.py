"""Thermal property workflow recipes."""
# ruff: noqa: RUF002 Greek alpha in docstrings denotes thermal expansion coefficient

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from jobflow import Flow

from atomate2.siesta.flows.phonon import (
    SiestaGruneisenFlowMaker,
    SiestaPhononFlowMaker,
    SiestaQhaFlowMaker,
)
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.recipes.base import MaterialAnalyzer
from atomate2.siesta.sets.tiers import apply_tier_preset

if TYPE_CHECKING:
    from jobflow import Job
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def thermal_properties(
    structure: Structure,
    auto_params: bool = True,
    user_params: dict[str, Any] | None = None,
    phonon_user_params: dict[str, Any] | None = None,
    tier: str | None = None,
    preset: str | None = None,
    include_phonons: bool = True,
    include_gruneisen: bool = True,
    include_qha: bool = True,
    supercell_matrix: tuple[int, int, int] | None = None,
    temperature_range: tuple[float, float, float] = (0, 1000, 10),
    ignore_imaginary_modes: bool = False,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "thermal_properties",
) -> Flow:
    """
    Complete thermal properties workflow.

    Calculates:
    1. Phonon dispersion and DOS
    2. Grüneisen parameters
    3. Quasi-harmonic approximation (QHA)
    4. Thermal expansion α(T)
    5. Heat capacity Cv(T), Cp(T)
    6. Entropy S(T)
    7. Gibbs free energy G(T,P)

    Parameters
    ----------
    structure : Structure
        Input structure.
    auto_params : bool
        Automatically determine optimal parameters. Default: True.
    user_params : dict
        Override specific SIESTA parameters for relaxation.
    phonon_user_params : dict
        Override specific SIESTA parameters for phonon force calculations.
        If None, uses user_params with automatic k-point scaling for supercells.
        If provided, disables automatic k-point scaling.
    tier : str
        Tier level. Auto-detected if None.
    preset : str
        Preset name. Auto-detected if None.
    include_phonons : bool
        Include phonon calculation. Default: True.
    include_gruneisen : bool
        Include Grüneisen parameters. Default: True.
    include_qha : bool
        Include QHA calculation. Default: True.
    supercell_matrix : tuple
        Supercell size for phonon calculation. Auto-detected if None.
    temperature_range : tuple
        (T_min, T_max, T_step) in Kelvin. Default: (0, 1000, 10).
    ignore_imaginary_modes : bool
        Use all volumes even if they have imaginary frequencies. Default: False.
    use_custodian : bool
        Enable custodian for automatic error recovery. Default: True.
    dry_run : bool
        Only generate input files. Default: False.
    name : str
        Workflow name.

    Returns
    -------
    Flow
        Thermal properties workflow.

    Example
    -------
    >>> from atomate2.siesta.recipes import thermal_properties
    >>> flow = thermal_properties(structure)
    >>> # Complete thermal characterization including phonons, Grüneisen, QHA!
    """
    logger.info(f"Creating thermal_properties workflow for {structure.composition}")

    # Analyze structure
    if auto_params:
        analysis = MaterialAnalyzer.analyze(structure)
        if tier is None:
            tier = "intermediate"  # Phonons need good accuracy
        if preset is None:
            preset = "phonon_high_accuracy"

        auto_user_params = {
            "a2s_kpts": analysis.recommended_kpts,
            "Mesh.Cutoff": analysis.recommended_cutoff,
            "PAO.BasisSize": "DZP",  # Always use DZP for phonons
            "SCF.MustConverge": True,
            "SCF.H.Tolerance": 1e-5,  # Tight convergence for forces (in eV)
        }

        if analysis.is_metal:
            auto_user_params.update(
                {
                    "OccupationFunction": "MP",
                    "ElectronicTemperature": "300 K",
                    "SCF.Mixer.Weight": 0.005,
                }
            )

        if user_params:
            auto_user_params.update(user_params)
        user_params = auto_user_params

        # Auto-determine supercell if not provided
        if supercell_matrix is None:
            # Target ~100-200 atoms for phonon calculation
            natoms = analysis.num_atoms
            if natoms <= 10:
                supercell_matrix = (3, 3, 3)
            elif natoms <= 20:
                supercell_matrix = (2, 2, 2)
            else:
                supercell_matrix = (1, 1, 1)

    jobs: list[Flow | Job] = []

    # Create relax maker for each workflow - use class method
    relax_maker = RelaxMaker.fixed_cell_relaxation(
        user_params=user_params,
        use_custodian=use_custodian,
        custodian_max_errors=10,
        dry_run=dry_run,
    )
    if preset:
        relax_maker = apply_tier_preset(
            relax_maker, preset, override_params=user_params
        )
    elif tier:
        relax_maker.input_set_generator.tier = tier

    # In dry_run mode, only create a simple relaxation job to generate input files
    # Complex phonon workflows with job dependencies don't work in dry_run mode
    if dry_run:
        logger.info("Dry-run mode: Creating simple relaxation job only")
        relax_job = relax_maker.make(structure)
        relax_job.name = "thermal_relax_dry_run"
        # Dry-run returns a single relaxation Job (no multi-step flow is built).
        return cast("Flow", relax_job)

    # Create static maker for phonon calculations
    # If phonon_user_params provided, use it directly (no automatic scaling)
    # Otherwise, use user_params with automatic k-point scaling for supercells
    if phonon_user_params is not None:
        # User explicitly provided phonon parameters - use as-is
        final_phonon_params = phonon_user_params.copy()
        logger.info("Using explicit phonon_user_params (no automatic k-point scaling)")
    else:
        # Automatic k-point scaling for supercells
        # IMPORTANT: Larger supercell → smaller BZ → fewer k-points needed
        final_phonon_params = user_params.copy() if user_params else {}
        if supercell_matrix is not None and "a2s_kpts" in final_phonon_params:
            original_kpts = final_phonon_params["a2s_kpts"]
            scaled_kpts = [
                max(1, int(k // supercell_matrix[i]))
                for i, k in enumerate(original_kpts)
            ]
            final_phonon_params["a2s_kpts"] = scaled_kpts
            logger.info(
                f"Scaled k-points for phonon supercell {supercell_matrix}: "
                f"{original_kpts} → {scaled_kpts}"
            )

    static_maker = StaticMaker.scf(
        user_params=final_phonon_params,
        use_custodian=use_custodian,
        custodian_max_errors=10,
        dry_run=dry_run,
    )
    static_maker.name = "phonon_static"

    if preset:
        static_maker = apply_tier_preset(
            static_maker, preset, override_params=final_phonon_params
        )

    # Phonon calculation
    if include_phonons:
        phonon_maker = SiestaPhononFlowMaker(
            name="phonons",
            relax_maker=relax_maker,
            static_maker=static_maker,
            min_length=6.0,
            dry_run=dry_run,
        )
        # Convert tuple to list format if supercell_matrix provided
        phonon_supercell_matrix = None
        if supercell_matrix is not None:
            phonon_supercell_matrix = [
                [supercell_matrix[0], 0, 0],
                [0, supercell_matrix[1], 0],
                [0, 0, supercell_matrix[2]],
            ]
        phonon_job = phonon_maker.make(
            structure, supercell_matrix=phonon_supercell_matrix
        )
        jobs.append(phonon_job)

    # Grüneisen parameters
    if include_gruneisen:
        # Create phonon maker for Grüneisen
        # Convert tuple to list format if supercell_matrix provided
        gruneisen_supercell_matrix = None
        if supercell_matrix is not None:
            gruneisen_supercell_matrix = [
                [supercell_matrix[0], 0, 0],
                [0, supercell_matrix[1], 0],
                [0, 0, supercell_matrix[2]],
            ]
        gruneisen_phonon_maker = SiestaPhononFlowMaker(
            relax_maker=relax_maker,
            static_maker=static_maker,
            supercell_matrix=gruneisen_supercell_matrix,
            min_length=6.0 if supercell_matrix is None else None,
            dry_run=dry_run,
        )
        logger.info(
            "Gruneisen phonon maker created with "
            f"supercell_matrix={gruneisen_supercell_matrix}, "
            f"min_length={6.0 if supercell_matrix is None else None}"
        )
        gruneisen_maker = SiestaGruneisenFlowMaker(
            name="gruneisen",
            structure_optimizer=relax_maker,
            phonon_maker=gruneisen_phonon_maker,
            perc_vol=0.01,  # ±1% volume change
            dry_run=dry_run,
        )
        # NOTE: Gruneisen maker doesn't take supercell_matrix in .make()
        # It only uses what's in phonon_maker.supercell_matrix
        gruneisen_job = gruneisen_maker.make(structure)
        jobs.append(gruneisen_job)

    # QHA calculation
    if include_qha:
        # Create phonon maker for QHA
        # Convert tuple to list format if supercell_matrix provided
        qha_supercell_matrix = None
        if supercell_matrix is not None:
            qha_supercell_matrix = [
                [supercell_matrix[0], 0, 0],
                [0, supercell_matrix[1], 0],
                [0, 0, supercell_matrix[2]],
            ]
        qha_phonon_maker = SiestaPhononFlowMaker(
            relax_maker=relax_maker,
            static_maker=static_maker,
            supercell_matrix=qha_supercell_matrix,
            min_length=6.0 if supercell_matrix is None else None,
            dry_run=dry_run,
        )
        logger.info(
            f"QHA phonon maker created with supercell_matrix={qha_supercell_matrix}, "
            f"min_length={6.0 if supercell_matrix is None else None}"
        )
        # Generate temperature list from range
        t_min, t_max, t_step = temperature_range
        temperature_list = list(range(int(t_min), int(t_max) + 1, int(t_step)))

        qha_maker = SiestaQhaFlowMaker(
            name="qha",
            structure_optimizer=relax_maker,
            phonon_maker=qha_phonon_maker,
            temperature=cast("list[float]", temperature_list),
            ignore_imaginary_modes=ignore_imaginary_modes,
            dry_run=dry_run,
        )
        # NOTE: QHA automatically extracts supercell_matrix from
        # phonon_maker.supercell_matrix (see SiestaQhaFlowMaker.make() lines 296-313)
        qha_job = qha_maker.make(structure)
        jobs.append(qha_job)

    # Create flow
    flow = Flow(jobs, output=jobs[-1].output if jobs else None, name=name)
    logger.info(f"Thermal properties workflow created with {len(jobs)} jobs")
    return flow


def phonon_workflow(
    structure: Structure, supercell_matrix: tuple[int, int, int] | None = None, **kwargs
) -> Flow:
    """
    Phonon calculation workflow.

    Simplified wrapper for phonon dispersion and DOS calculation.

    Parameters
    ----------
    structure : Structure
        Input structure.
    supercell_matrix : tuple
        Supercell size. Auto-detected if None.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Phonon workflow.

    Example
    -------
    >>> flow = phonon_workflow(structure, supercell_matrix=(2, 2, 2))
    """
    # Extract name from kwargs if provided, otherwise use default
    name = kwargs.pop("name", "phonon_workflow")

    return thermal_properties(
        structure,
        include_phonons=True,
        include_gruneisen=False,
        include_qha=False,
        supercell_matrix=supercell_matrix,
        name=name,
        **kwargs,
    )


def gruneisen_workflow(
    structure: Structure,
    supercell_matrix: tuple[int, int, int] | None = None,
    volume_change: float = 0.01,  # noqa: ARG001 documented API param, reserved
    **kwargs,
) -> Flow:
    """
    Grüneisen parameter calculation workflow.

    Calculates mode-dependent Grüneisen parameters for thermal
    expansion analysis.

    Parameters
    ----------
    structure : Structure
        Input structure.
    supercell_matrix : tuple
        Supercell size. Auto-detected if None.
    volume_change : float
        Fractional volume change (default: 0.01 = ±1%).
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Grüneisen workflow.

    Example
    -------
    >>> flow = gruneisen_workflow(structure)
    """
    # Extract name from kwargs if provided, otherwise use default
    name = kwargs.pop("name", "gruneisen_workflow")

    return thermal_properties(
        structure,
        include_phonons=False,
        include_gruneisen=True,
        include_qha=False,
        supercell_matrix=supercell_matrix,
        name=name,
        **kwargs,
    )


def qha_workflow(
    structure: Structure,
    supercell_matrix: tuple[int, int, int] | None = None,
    temperature_range: tuple[float, float, float] = (0, 1000, 10),
    **kwargs,
) -> Flow:
    """
    Quasi-harmonic approximation workflow.

    Calculates temperature-dependent thermodynamic properties:
    - Thermal expansion α(T)
    - Heat capacity Cv(T), Cp(T)
    - Entropy S(T)
    - Gibbs free energy G(T,P)

    Parameters
    ----------
    structure : Structure
        Input structure.
    supercell_matrix : tuple
        Supercell size. Auto-detected if None.
    temperature_range : tuple
        (T_min, T_max, T_step) in Kelvin.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        QHA workflow.

    Example
    -------
    >>> flow = qha_workflow(structure, temperature_range=(0, 1500, 20))
    """
    # Extract name from kwargs if provided, otherwise use default
    name = kwargs.pop("name", "qha_workflow")

    return thermal_properties(
        structure,
        include_phonons=False,
        include_gruneisen=False,
        include_qha=True,
        supercell_matrix=supercell_matrix,
        temperature_range=temperature_range,
        name=name,
        **kwargs,
    )


def thermal_expansion_workflow(structure: Structure, **kwargs) -> Flow:
    """
    Thermal expansion coefficient calculation.

    Combines Grüneisen parameters and QHA for accurate α(T).

    Parameters
    ----------
    structure : Structure
        Input structure.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Thermal expansion workflow.

    Example
    -------
    >>> flow = thermal_expansion_workflow(structure)
    """
    # Extract name from kwargs if provided, otherwise use default
    name = kwargs.pop("name", "thermal_expansion")

    return thermal_properties(
        structure,
        include_phonons=True,
        include_gruneisen=True,
        include_qha=True,
        name=name,
        **kwargs,
    )


def high_temperature_properties(
    structure: Structure, max_temperature: float = 2000.0, **kwargs
) -> Flow:
    """
    High-temperature thermodynamic properties.

    Extended temperature range for high-T materials.

    Parameters
    ----------
    structure : Structure
        Input structure.
    max_temperature : float
        Maximum temperature in Kelvin. Default: 2000 K.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        High-temperature properties workflow.

    Example
    -------
    >>> flow = high_temperature_properties(ceramic_structure, max_temperature=3000)
    """
    # Extract name from kwargs if provided, otherwise use default
    name = kwargs.pop("name", "high_temperature_properties")

    return thermal_properties(
        structure,
        temperature_range=(0, max_temperature, 20),
        name=name,
        **kwargs,
    )


def vibrational_stability_check(structure: Structure, **kwargs) -> Flow:
    """
    Check vibrational stability via phonon calculation.

    Identifies imaginary modes that indicate structural instability.

    Parameters
    ----------
    structure : Structure
        Input structure.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Stability check workflow.

    Example
    -------
    >>> flow = vibrational_stability_check(structure)
    >>> # Check for imaginary frequencies in results
    """
    return phonon_workflow(structure, name="stability_check", **kwargs)
