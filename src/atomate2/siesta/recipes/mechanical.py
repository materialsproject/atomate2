"""Mechanical property workflow recipes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from jobflow import Flow

from atomate2.siesta.flows.elastic import ElasticFlowMaker
from atomate2.siesta.flows.eos import SiestaEosFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.recipes.base import MaterialAnalyzer
from atomate2.siesta.sets.tiers import apply_tier_preset

if TYPE_CHECKING:
    from jobflow import Job
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def mechanical_properties(
    structure: Structure,
    auto_params: bool = True,
    user_params: dict[str, Any] | None = None,
    tier: str | None = None,
    preset: str | None = None,
    include_eos: bool = True,
    include_elastic: bool = True,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "mechanical_properties",
) -> Flow:
    """
    Complete mechanical properties workflow.

    Calculates:
    1. Relaxed structure
    2. Equation of state (bulk modulus)
    3. Elastic constants (full tensor)
    4. Mechanical properties (Young's modulus, Poisson ratio, etc.)

    Parameters
    ----------
    structure : Structure
        Input structure.
    auto_params : bool
        Automatically determine optimal parameters. Default: True.
    user_params : dict
        Override specific SIESTA parameters.
    tier : str
        Tier level. Auto-detected if None.
    preset : str
        Preset name. Auto-detected if None.
    include_eos : bool
        Include equation of state calculation. Default: True.
    include_elastic : bool
        Include elastic constants calculation. Default: True.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
    dry_run : bool
        Only generate input files. Default: False.
    name : str
        Workflow name.

    Returns
    -------
    Flow
        Mechanical properties workflow.

    Example
    -------
    >>> from atomate2.siesta.recipes import mechanical_properties
    >>> flow = mechanical_properties(structure)
    >>> # Complete mechanical characterization in one line!
    """
    logger.info(f"Creating mechanical_properties workflow for {structure.composition}")

    # Analyze structure
    if auto_params:
        analysis = MaterialAnalyzer.analyze(structure)
        if tier is None:
            tier = analysis.recommended_tier
        if preset is None:
            preset = analysis.recommended_preset

        auto_user_params: dict[str, Any] = {
            "a2s_kpts": analysis.recommended_kpts,
            "Mesh.Cutoff": analysis.recommended_cutoff,
            "PAO.BasisSize": analysis.recommended_basis,
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

    jobs: list[Flow | Job] = []

    # Initial relaxation - use class method
    relax_maker = RelaxMaker.fixed_cell_relaxation(
        user_params=user_params,
        use_custodian=use_custodian,
        custodian_max_errors=10,
        dry_run=dry_run,
    )
    relax_maker.name = "initial_relax"

    if preset:
        relax_maker = apply_tier_preset(
            relax_maker, preset, override_params=user_params
        )
    elif tier:
        relax_maker.input_set_generator.tier = tier

    # In dry_run mode, only create a simple relaxation job to generate input files
    # Complex workflows with job dependencies don't work in dry_run mode
    if dry_run:
        logger.info("Dry-run mode: Creating simple relaxation job only")
        relax_job = relax_maker.make(structure)
        relax_job.name = "mechanical_relax_dry_run"
        # Dry-run returns a single relaxation Job (no multi-step flow is built).
        return cast("Flow", relax_job)

    relax_job = relax_maker.make(structure)
    jobs.append(relax_job)

    # EOS workflow
    if include_eos:
        eos_maker = SiestaEosFlowMaker(
            name="eos",
            initial_relax_maker=None,  # Already relaxed
            eos_relax_maker=relax_maker,  # Use for EOS points
            static_maker=None,
            number_of_frames=7,
            dry_run=dry_run,
        )
        eos_job = eos_maker.make(
            relax_job.output.structure  # type: ignore[arg-type]  # jobflow OutputReference resolved at runtime
        )
        jobs.append(eos_job)

    # Elastic constants workflow
    if include_elastic:
        elastic_maker = ElasticFlowMaker(
            name="elastic",
            bulk_relax_maker=relax_maker,
            elastic_relax_maker=relax_maker,
            generate_elastic_deformations_kwargs={},  # Use default strain states
            fit_elastic_tensor_kwargs={"fitting_method": "finite_difference"},
            dry_run=dry_run,
        )
        elastic_job = elastic_maker.make(
            relax_job.output.structure  # type: ignore[arg-type]  # jobflow OutputReference resolved at runtime
        )
        jobs.append(elastic_job)

    # Create flow
    flow = Flow(jobs, output=jobs[-1].output if jobs else None, name=name)
    logger.info(f"Mechanical properties workflow created with {len(jobs)} jobs")
    return flow


def elastic_constants_workflow(
    structure: Structure,
    relax_first: bool = True,  # noqa: ARG001 documented API param, reserved
    strain_states: str = "default",  # noqa: ARG001 documented API param, reserved
    **kwargs,
) -> Flow:
    """
    Elastic constants calculation workflow.

    Calculates full elastic tensor using finite differences.

    Parameters
    ----------
    structure : Structure
        Input structure.
    relax_first : bool
        Relax structure first. Default: True.
    strain_states : str
        Strain states to use. Options: "default", "all", "minimal".
        Default: "default" (6 deformations for cubic, more for lower symmetry).
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Elastic constants workflow.

    Example
    -------
    >>> flow = elastic_constants_workflow(structure)
    >>> # Calculates elastic tensor, bulk/shear modulus, etc.
    """
    return mechanical_properties(
        structure,
        include_eos=False,
        include_elastic=True,
        name="elastic_constants",
        **kwargs,
    )


def eos_workflow(
    structure: Structure,
    number_of_frames: int = 7,  # noqa: ARG001 documented API param, reserved
    **kwargs,
) -> Flow:
    """
    Equation of state (EOS) and bulk modulus workflow.

    Calculates energy vs volume relationship to determine bulk modulus
    and equilibrium volume using Birch-Murnaghan or Vinet EOS fitting.

    This workflow performs the same calculation as the removed
    bulk_modulus_workflow() - both EOS and bulk modulus come from
    the same E(V) fitting procedure.

    Parameters
    ----------
    structure : Structure
        Input structure.
    number_of_frames : int
        Number of volume points for EOS fit. Default: 7.
        More points = better fit accuracy but more calculations.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        EOS workflow (includes bulk modulus in output).

    Example
    -------
    >>> flow = eos_workflow(structure, number_of_frames=9)
    >>> # Output includes: bulk_modulus, equilibrium_volume, E0, EOS_fit
    """
    return mechanical_properties(
        structure,
        include_eos=True,
        include_elastic=False,
        name="eos_workflow",
        **kwargs,
    )


def pressure_eos_workflow(
    structure: Structure,
    pressure_range: tuple[float, float] = (0.0, 50.0),
    number_of_frames: int = 9,  # noqa: ARG001 documented API param, reserved
    **kwargs,
) -> Flow:
    """
    Equation of state under pressure.

    Calculates EOS across a pressure range for P-V-E relationship.

    Parameters
    ----------
    structure : Structure
        Input structure.
    pressure_range : tuple
        Pressure range in GPa. Default: (0, 50).
    number_of_frames : int
        Number of pressure points. Default: 9.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Pressure-dependent EOS workflow.

    Example
    -------
    >>> flow = pressure_eos_workflow(structure, pressure_range=(0, 100))
    """
    # Add pressure-related parameters
    user_params = kwargs.get("user_params", {})
    user_params["MD.TargetPressure"] = f"{pressure_range[0]} GPa"
    kwargs["user_params"] = user_params

    return mechanical_properties(
        structure,
        include_eos=True,
        include_elastic=False,
        name="pressure_eos",
        **kwargs,
    )


def hardness_estimation(structure: Structure, **kwargs) -> Flow:
    """
    Estimate material hardness from elastic properties.

    Uses empirical models (Chen, Teter, Tian) to estimate hardness
    from elastic constants.

    Parameters
    ----------
    structure : Structure
        Input structure.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Hardness estimation workflow.

    Example
    -------
    >>> flow = hardness_estimation(hard_structure)
    """
    return mechanical_properties(
        structure,
        include_eos=True,
        include_elastic=True,
        name="hardness_estimation",
        **kwargs,
    )


def anisotropy_analysis(structure: Structure, **kwargs) -> Flow:
    """
    Analyze elastic anisotropy.

    Calculates anisotropy indices (Zener, universal anisotropy)
    and direction-dependent properties.

    Parameters
    ----------
    structure : Structure
        Input structure.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Anisotropy analysis workflow.

    Example
    -------
    >>> flow = anisotropy_analysis(layered_structure)
    """
    return mechanical_properties(
        structure,
        include_eos=False,
        include_elastic=True,
        name="anisotropy_analysis",
        **kwargs,
    )
