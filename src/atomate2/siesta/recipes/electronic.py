"""Electronic property workflow recipes."""

from __future__ import annotations

import logging
from typing import Any

from jobflow import Flow
from pymatgen.core import Structure

from atomate2.siesta.jobs.core import BandStructureMaker, DOSMaker, RelaxMaker
from atomate2.siesta.recipes.base import MaterialAnalyzer
from atomate2.siesta.sets.tiers import apply_tier_preset

logger = logging.getLogger(__name__)


def electronic_properties(
    structure: Structure,
    relax: bool = True,
    auto_params: bool = True,
    user_params: dict[str, Any] | None = None,
    tier: str | None = None,
    preset: str | None = None,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "electronic_properties",
) -> Flow:
    """
    Complete electronic structure workflow: relaxation + bands + DOS.

    This recipe performs a full electronic structure characterization:
    1. Structure relaxation (optional)
    2. Self-consistent calculation
    3. Band structure along high-symmetry path
    4. Density of states

    Parameters
    ----------
    structure : Structure
        Input structure.
    relax : bool
        If True, perform structure relaxation first. Default: True.
    auto_params : bool
        If True, automatically determine optimal parameters. Default: True.
    user_params : dict
        Override specific SIESTA parameters.
    tier : str
        Tier level ("basic", "intermediate", "advanced", "expert").
        Auto-detected if None.
    preset : str
        Preset name for tier system. Auto-detected if None.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
    dry_run : bool
        If True, only generate input files. Default: False.
    name : str
        Workflow name.

    Returns
    -------
    Flow
        Electronic properties workflow.

    Example
    -------
    >>> from atomate2.siesta.recipes import electronic_properties
    >>> from pymatgen.core import Structure
    >>>
    >>> structure = Structure.from_file("POSCAR")
    >>> flow = electronic_properties(structure)
    >>> # That's it! Automatic parameter selection, relaxation, and band structure
    """
    logger.info(f"Creating electronic_properties workflow for {structure.composition}")

    # Analyze structure if auto_params enabled
    if auto_params:
        analysis = MaterialAnalyzer.analyze(structure)
        logger.info(
            f"Material analysis: {analysis.formula}, "
            f"Metal: {analysis.is_metal}, "
            f"Atoms: {analysis.num_atoms}"
        )

        # Use analyzed parameters if not explicitly provided
        if tier is None:
            tier = analysis.recommended_tier
        if preset is None:
            preset = analysis.recommended_preset

        # Build recommended params
        auto_user_params = {
            "a2s_kpts": analysis.recommended_kpts,
            "Mesh.Cutoff": analysis.recommended_cutoff,
            "PAO.BasisSize": analysis.recommended_basis,
        }

        # Add metal-specific parameters
        if analysis.is_metal:
            auto_user_params.update(
                {
                    "OccupationFunction": "MP",
                    "ElectronicTemperature": "300 K",
                    "SCF.Mixer.Weight": 0.005,
                }
            )

        # Merge with user params (user params take precedence)
        if user_params:
            auto_user_params.update(user_params)
        user_params = auto_user_params

    # Create jobs
    jobs = []

    if relax:
        # Relaxation job - use class method that accepts user_params
        relax_maker = RelaxMaker.fixed_cell_relaxation(
            user_params=user_params,
            use_custodian=use_custodian,
            custodian_max_errors=10,
            dry_run=dry_run,
        )
        relax_maker.name = "relax"

        if preset:
            relax_maker = apply_tier_preset(
                relax_maker, preset, override_params=user_params
            )
        elif tier:
            relax_maker.input_set_generator.tier = tier

        relax_job = relax_maker.make(structure)
        jobs.append(relax_job)

        # Band structure uses relaxed structure
        band_maker = BandStructureMaker.bandstructure_calculation(
            user_params=user_params,
            use_custodian=use_custodian,
            custodian_max_errors=10,
            dry_run=dry_run,
        )
        band_maker.name = "band_structure"

        if preset:
            band_maker = apply_tier_preset(
                band_maker, preset, override_params=user_params
            )
        elif tier:
            band_maker.input_set_generator.tier = tier

        band_job = band_maker.make(
            relax_job.output.structure, prev_dir=relax_job.output.dir_name
        )
        jobs.append(band_job)

        # DOS calculation uses relaxed structure
        dos_maker = DOSMaker.dos_calculation(
            user_params=user_params,
            use_custodian=use_custodian,
            custodian_max_errors=10,
            dry_run=dry_run,
        )
        dos_maker.name = "dos"

        if preset:
            dos_maker = apply_tier_preset(
                dos_maker, preset, override_params=user_params
            )
        elif tier:
            dos_maker.input_set_generator.tier = tier

        dos_job = dos_maker.make(
            relax_job.output.structure, prev_dir=relax_job.output.dir_name
        )
        jobs.append(dos_job)

    else:
        # Direct band structure without relaxation
        band_maker = BandStructureMaker.bandstructure_calculation(
            user_params=user_params,
            use_custodian=use_custodian,
            custodian_max_errors=10,
            dry_run=dry_run,
        )
        band_maker.name = "band_structure"

        if preset:
            band_maker = apply_tier_preset(
                band_maker, preset, override_params=user_params
            )
        elif tier:
            band_maker.input_set_generator.tier = tier

        band_job = band_maker.make(structure)
        jobs.append(band_job)

        # DOS calculation without relaxation
        dos_maker = DOSMaker.dos_calculation(
            user_params=user_params,
            use_custodian=use_custodian,
            custodian_max_errors=10,
            dry_run=dry_run,
        )
        dos_maker.name = "dos"

        if preset:
            dos_maker = apply_tier_preset(
                dos_maker, preset, override_params=user_params
            )
        elif tier:
            dos_maker.input_set_generator.tier = tier

        dos_job = dos_maker.make(structure)
        jobs.append(dos_job)

    # Create flow
    flow = Flow(jobs, output=jobs[-1].output, name=name)

    logger.info(f"Electronic properties workflow created with {len(jobs)} jobs")
    return flow


def band_structure_workflow(
    structure: Structure, relax_first: bool = True, **kwargs
) -> Flow:
    """
    Band structure calculation workflow.

    Simplified wrapper for band structure calculation with automatic
    parameter detection.

    Parameters
    ----------
    structure : Structure
        Input structure.
    relax_first : bool
        Relax structure before band calculation. Default: True.
    **kwargs
        Additional parameters passed to electronic_properties().

    Returns
    -------
    Flow
        Band structure workflow.

    Example
    -------
    >>> flow = band_structure_workflow(structure)
    """
    return electronic_properties(
        structure, relax=relax_first, name="band_structure_workflow", **kwargs
    )


def dos_workflow(
    structure: Structure,
    relax_first: bool = True,
    dos_kpts_density: float = 1.5,
    auto_params: bool = True,
    user_params: dict[str, Any] | None = None,
    tier: str | None = None,
    preset: str | None = None,
    use_custodian: bool = True,
    dry_run: bool = False,
) -> Flow:
    """
    Density of states calculation workflow.

    Parameters
    ----------
    structure : Structure
        Input structure.
    relax_first : bool
        Relax structure first. Default: True.
    dos_kpts_density : float
        K-point density multiplier for DOS. Default: 1.5 (50% denser than relax).
    auto_params : bool
        If True, automatically determine optimal parameters. Default: True.
    user_params : dict
        Override specific SIESTA parameters.
    tier : str
        Tier level ("basic", "intermediate", "advanced", "expert").
        Auto-detected if None.
    preset : str
        Preset name for tier system. Auto-detected if None.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
    dry_run : bool
        If True, only generate input files. Default: False.

    Returns
    -------
    Flow
        DOS workflow.

    Example
    -------
    >>> flow = dos_workflow(structure, dos_kpts_density=2.0)
    """
    logger.info(f"Creating DOS workflow for {structure.composition}")

    # Analyze structure if auto_params enabled
    if auto_params:
        analysis = MaterialAnalyzer.analyze(structure)
        logger.info(
            f"Material analysis: {analysis.formula}, "
            f"Metal: {analysis.is_metal}, "
            f"Atoms: {analysis.num_atoms}"
        )

        # Use analyzed parameters if not explicitly provided
        if tier is None:
            tier = analysis.recommended_tier
        if preset is None:
            preset = analysis.recommended_preset

        # Calculate DOS k-points (denser mesh than relaxation)
        dos_kpts = [int(k * dos_kpts_density) for k in analysis.recommended_kpts]

        # Build recommended params
        auto_user_params = {
            "a2s_kpts": dos_kpts,
            "Mesh.Cutoff": analysis.recommended_cutoff,
            "PAO.BasisSize": analysis.recommended_basis,
        }

        # Add metal-specific parameters
        if analysis.is_metal:
            auto_user_params.update(
                {
                    "OccupationFunction": "MP",
                    "ElectronicTemperature": "300 K",
                    "SCF.Mixer.Weight": 0.005,
                }
            )

        # Merge with user params (user params take precedence)
        if user_params:
            auto_user_params.update(user_params)
        user_params = auto_user_params

    # Create jobs
    jobs = []

    if relax_first:
        # Relaxation job
        relax_maker = RelaxMaker.fixed_cell_relaxation(
            user_params=user_params,
            use_custodian=use_custodian,
            custodian_max_errors=10,
            dry_run=dry_run,
        )
        relax_maker.name = "relax"

        if preset:
            relax_maker = apply_tier_preset(
                relax_maker, preset, override_params=user_params
            )
        elif tier:
            relax_maker.input_set_generator.tier = tier

        relax_job = relax_maker.make(structure)
        jobs.append(relax_job)

        # DOS uses relaxed structure
        dos_maker = DOSMaker.dos_calculation(
            user_params=user_params,
            use_custodian=use_custodian,
            custodian_max_errors=10,
            dry_run=dry_run,
        )
        dos_maker.name = "dos"

        if preset:
            dos_maker = apply_tier_preset(
                dos_maker, preset, override_params=user_params
            )
        elif tier:
            dos_maker.input_set_generator.tier = tier

        dos_job = dos_maker.make(
            relax_job.output.structure, prev_dir=relax_job.output.dir_name
        )
        jobs.append(dos_job)

    else:
        # Direct DOS without relaxation
        dos_maker = DOSMaker.dos_calculation(
            user_params=user_params,
            use_custodian=use_custodian,
            custodian_max_errors=10,
            dry_run=dry_run,
        )
        dos_maker.name = "dos"

        if preset:
            dos_maker = apply_tier_preset(
                dos_maker, preset, override_params=user_params
            )
        elif tier:
            dos_maker.input_set_generator.tier = tier

        dos_job = dos_maker.make(structure)
        jobs.append(dos_job)

    # Create flow
    flow = Flow(jobs, output=jobs[-1].output, name="dos_workflow")

    logger.info(f"DOS workflow created with {len(jobs)} jobs")
    return flow


def optical_properties(
    structure: Structure,
    relax_first: bool = True,
    energy_range: tuple[float, float] = (0.0, 10.0),
    **kwargs,
) -> Flow:
    """
    Optical properties calculation workflow.

    Calculates dielectric function, absorption, reflectivity.

    Parameters
    ----------
    structure : Structure
        Input structure.
    relax_first : bool
        Relax structure first. Default: True.
    energy_range : tuple
        Energy range for optical properties (eV). Default: (0.0, 10.0).
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Optical properties workflow.

    Example
    -------
    >>> flow = optical_properties(structure, energy_range=(0, 15))
    """
    # Add optical calculation parameters
    user_params = kwargs.get("user_params", {})
    user_params.update(
        {
            "OpticalCalculation": True,
            "Optical.Energy.Minimum": f"{energy_range[0]} eV",
            "Optical.Energy.Maximum": f"{energy_range[1]} eV",
            "Optical.Broaden": "0.1 eV",
            "Optical.Mesh": "[100, 100, 100]",
        }
    )
    kwargs["user_params"] = user_params

    return electronic_properties(
        structure, relax=relax_first, name="optical_properties", **kwargs
    )


def metal_properties(structure: Structure, relax_first: bool = True, **kwargs) -> Flow:
    """
    Workflow optimized for metallic systems.

    Uses appropriate occupation function, electronic temperature,
    and mixer settings for metals.

    Parameters
    ----------
    structure : Structure
        Input structure.
    relax_first : bool
        Relax structure first. Default: True.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Metal properties workflow.

    Example
    -------
    >>> flow = metal_properties(al_structure)
    """
    # Force metal-specific parameters
    user_params = kwargs.get("user_params", {})
    user_params.update(
        {
            "OccupationFunction": "MP",
            "ElectronicTemperature": "300 K",
            "SCF.Mixer.Weight": 0.005,
            "SCF.Mixer.History": 10,
        }
    )
    kwargs["user_params"] = user_params
    kwargs["preset"] = "surface_metal"  # Good preset for metals

    return electronic_properties(
        structure, relax=relax_first, name="metal_properties", **kwargs
    )


def semiconductor_properties(
    structure: Structure, relax_first: bool = True, **kwargs
) -> Flow:
    """
    Workflow optimized for semiconductors.

    Uses appropriate parameters for band gap calculation.

    Parameters
    ----------
    structure : Structure
        Input structure.
    relax_first : bool
        Relax structure first. Default: True.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Semiconductor properties workflow.

    Example
    -------
    >>> flow = semiconductor_properties(si_structure)
    """
    # Semiconductor-specific parameters
    user_params = kwargs.get("user_params", {})
    user_params.update(
        {
            "OccupationFunction": "FD",
            "ElectronicTemperature": "25 K",  # Room temperature
            "SCF.MustConverge": True,
        }
    )
    kwargs["user_params"] = user_params
    kwargs["preset"] = "relax_standard"

    return electronic_properties(
        structure, relax=relax_first, name="semiconductor_properties", **kwargs
    )
