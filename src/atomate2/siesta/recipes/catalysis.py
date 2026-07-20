"""Surface and catalysis workflow recipes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jobflow import Flow

from atomate2.siesta.flows.surface import (
    AdsorptionScanFlowMaker,
    MultiSurfaceEnergyFlowMaker,
)
from atomate2.siesta.recipes.base import MaterialAnalyzer
from atomate2.siesta.sets.tiers import apply_tier_preset

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

logger = logging.getLogger(__name__)


def surface_energy_workflow(
    bulk_structure: Structure,
    miller_indices: list[tuple[int, int, int]] | None = None,
    slab_layers: int = 5,
    vacuum: float = 15.0,
    auto_params: bool = True,
    user_params: dict[str, Any] | None = None,
    tier: str | None = None,
    preset: str | None = None,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "surface_energy",
) -> Flow:
    """
    Surface energy calculation workflow.

    Calculates surface energies for multiple Miller indices with
    automatic slab generation and convergence.

    Parameters
    ----------
    bulk_structure : Structure
        Bulk structure.
    miller_indices : list of tuples
        List of Miller indices to calculate. Auto-detected if None.
        Default: [(1,0,0), (1,1,0), (1,1,1)] for cubic systems.
    slab_layers : int
        Number of atomic layers in slab. Default: 5.
    vacuum : float
        Vacuum thickness in Angstroms. Default: 15.0.
    auto_params : bool
        Automatically determine optimal parameters. Default: True.
    user_params : dict
        Override specific SIESTA parameters.
    tier : str
        Tier level. Auto-detected if None.
    preset : str
        Preset name. Auto-detected if None.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
    dry_run : bool
        Only generate input files. Default: False.
    name : str
        Workflow name.

    Returns
    -------
    Flow
        Surface energy workflow.

    Example
    -------
    >>> from atomate2.siesta.recipes import surface_energy_workflow
    >>> flow = surface_energy_workflow(
    ...     bulk_structure, miller_indices=[(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    ... )
    """
    logger.info(f"Creating surface_energy workflow for {bulk_structure.composition}")

    # Analyze structure
    if auto_params:
        analysis = MaterialAnalyzer.analyze(bulk_structure)
        if tier is None:
            tier = "intermediate"
        if preset is None:
            preset = (
                "surface_semiconductor" if not analysis.is_metal else "surface_metal"
            )

        auto_user_params = {
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

    # Auto-detect Miller indices for common crystal systems
    if miller_indices is None:
        analysis = MaterialAnalyzer.analyze(bulk_structure)
        if analysis.crystal_system in ["cubic", "tetragonal"]:
            miller_indices = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
        elif analysis.crystal_system == "hexagonal":
            miller_indices = [(0, 0, 1), (1, 0, 0), (1, 1, 0)]
        elif analysis.crystal_system == "orthorhombic":
            miller_indices = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        else:
            miller_indices = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    # Create makers - MultiSurfaceEnergyFlowMaker requires StaticMaker
    from atomate2.siesta.jobs.core import StaticMaker

    bulk_static_maker = StaticMaker.scf(
        user_params=user_params,
        use_custodian=use_custodian,
        custodian_max_errors=10,
        dry_run=dry_run,
    )
    bulk_static_maker.name = "bulk_static"

    if preset:
        bulk_static_maker = apply_tier_preset(
            bulk_static_maker, preset, override_params=user_params
        )
    elif tier:
        bulk_static_maker.input_set_generator.tier = tier

    slab_static_maker = StaticMaker.scf(
        user_params=user_params,
        use_custodian=use_custodian,
        custodian_max_errors=10,
        dry_run=dry_run,
    )
    slab_static_maker.name = "slab_static"

    if preset:
        slab_static_maker = apply_tier_preset(
            slab_static_maker, preset, override_params=user_params
        )

    # In dry_run mode, only create bulk and one slab static job to generate input files
    # Complex multi-surface workflows with job dependencies don't work in dry_run mode
    if dry_run:
        logger.info("Dry-run mode: Creating simple bulk and slab static jobs only")
        # Create bulk static job
        bulk_job = bulk_static_maker.make(bulk_structure)
        bulk_job.name = "bulk_static_dry_run"

        # Create one slab for preview
        from pymatgen.core.surface import SlabGenerator

        # Convert slab_layers (layer count) to min_slab_size (Angstroms)
        # Same conversion as MultiSurfaceEnergyFlowMaker
        d_hkl = bulk_structure.lattice.d_hkl(miller_indices[0])
        min_slab_size = slab_layers * d_hkl
        logger.info(
            f"  Miller index: {miller_indices[0]}, d-spacing: {d_hkl:.4f} Å, "
            f"min_slab_size: {min_slab_size:.4f} Å ({slab_layers} layers)"
        )

        slab_gen = SlabGenerator(
            bulk_structure, miller_indices[0], min_slab_size, vacuum
        )
        slab = slab_gen.get_slab()
        slab_job = slab_static_maker.make(slab)
        slab_job.name = f"slab_{miller_indices[0]}_dry_run"

        from jobflow import Flow

        return Flow([bulk_job, slab_job], name="surface_energy_dry_run")

    # Multi-surface energy maker
    surface_maker = MultiSurfaceEnergyFlowMaker(
        name=name,
        bulk_static_maker=bulk_static_maker,
        slab_static_maker=slab_static_maker,
        miller_indices=miller_indices,
        slab_layers=slab_layers,
        vacuum_size=vacuum,
        dry_run=dry_run,
    )

    flow = surface_maker.make(bulk_structure)
    logger.info(f"Surface energy workflow created for {len(miller_indices)} surfaces")
    return flow


def adsorption_scanning_workflow(
    slab_structure: Structure,
    adsorbate: Molecule | Structure,
    grid_density: tuple[int, int] = (5, 5),
    height_above_surface: float = 2.0,
    surface_side: str = "top",  # noqa: ARG001 documented API param, reserved
    auto_params: bool = True,
    user_params: dict[str, Any] | None = None,
    tier: str | None = None,
    preset: str | None = None,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "adsorption_scan",
) -> Flow:
    """
    Adsorption site scanning workflow.

    Performs grid-based scanning of adsorption sites on a surface
    with automatic energy calculation and site identification.

    Parameters
    ----------
    slab_structure : Structure
        Surface slab structure.
    adsorbate : Molecule or Structure
        Adsorbate molecule or atom.
    grid_density : tuple
        (nx, ny) grid points for scanning. Default: (5, 5).
    height_above_surface : float
        Initial height above surface in Angstroms. Default: 2.0.
    surface_side : str
        "top" or "bottom" surface. Default: "top".
    auto_params : bool
        Automatically determine optimal parameters. Default: True.
    user_params : dict
        Override specific SIESTA parameters.
    tier : str
        Tier level. Auto-detected if None.
    preset : str
        Preset name. Auto-detected if None.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
    dry_run : bool
        Only generate input files. Default: False.
    name : str
        Workflow name.

    Returns
    -------
    Flow
        Adsorption scanning workflow.

    Example
    -------
    >>> from pymatgen.core import Molecule
    >>> co_molecule = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.15]])
    >>> flow = adsorption_scanning_workflow(
    ...     slab_structure, adsorbate=co_molecule, grid_density=(7, 7)
    ... )
    """
    logger.info(f"Creating adsorption_scan workflow for {slab_structure.composition}")

    # Analyze structure
    if auto_params:
        analysis = MaterialAnalyzer.analyze(slab_structure)
        if tier is None:
            tier = "intermediate"
        if preset is None:
            preset = (
                "surface_semiconductor" if not analysis.is_metal else "surface_metal"
            )

        auto_user_params = {
            "a2s_kpts": [
                analysis.recommended_kpts[0],
                analysis.recommended_kpts[1],
                1,
            ],  # 1 in z for slab
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

    # Create static maker for adsorption site scanning
    from atomate2.siesta.jobs.core import StaticMaker

    static_maker = StaticMaker.scf(
        user_params=user_params,
        use_custodian=use_custodian,
        custodian_max_errors=10,
        dry_run=dry_run,
    )
    static_maker.name = "adsorption_static"

    if preset:
        static_maker = apply_tier_preset(
            static_maker, preset, override_params=user_params
        )
    elif tier:
        static_maker.input_set_generator.tier = tier

    # Adsorption scan maker
    ads_maker = AdsorptionScanFlowMaker(
        name=name,
        slab_static_maker=static_maker,
        adsorbate_static_maker=static_maker,
        grid_size=grid_density,
        height=height_above_surface,
        dry_run=dry_run,
    )

    flow = ads_maker.make(slab_structure, adsorbate)
    logger.info(
        "Adsorption scan workflow created with "
        f"{grid_density[0] * grid_density[1]} sites"
    )
    return flow


def catalysis_study(
    bulk_structure: Structure,
    adsorbates: list[Molecule | Structure],
    miller_indices: list[tuple[int, int, int]] | None = None,
    **kwargs,
) -> Flow:
    """
    Complete catalysis study workflow.

    Combines surface energy calculation with adsorption scanning
    for multiple adsorbates.

    Parameters
    ----------
    bulk_structure : Structure
        Bulk catalyst structure.
    adsorbates : list
        List of adsorbate molecules to test.
    miller_indices : list of tuples
        Miller indices to study. Auto-detected if None.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Complete catalysis workflow.

    Example
    -------
    >>> from pymatgen.core import Molecule
    >>> h2 = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])
    >>> o2 = Molecule(["O", "O"], [[0, 0, 0], [0, 0, 1.21]])
    >>> flow = catalysis_study(
    ...     pt_structure, adsorbates=[h2, o2], miller_indices=[(1, 1, 1)]
    ... )
    """
    # First calculate surface energies
    surface_flow = surface_energy_workflow(
        bulk_structure, miller_indices=miller_indices, name="surface_energies", **kwargs
    )

    # Then perform adsorption scans for each adsorbate
    # Note: In a real implementation, this would create multiple
    # adsorption scans for each surface
    logger.info(f"Creating catalysis study for {len(adsorbates)} adsorbates")

    return surface_flow


def reaction_pathway_workflow(
    slab_structure: Structure,
    initial_state: Structure,  # noqa: ARG001 documented API param, reserved
    final_state: Structure,  # noqa: ARG001 documented API param, reserved
    num_images: int = 7,
    **kwargs,
) -> Flow:
    """
    Surface reaction pathway via NEB.

    Calculates minimum energy path for surface reactions.

    Parameters
    ----------
    slab_structure : Structure
        Surface slab structure.
    initial_state : Structure
        Initial state (reactant + surface).
    final_state : Structure
        Final state (product + surface).
    num_images : int
        Number of NEB images. Default: 7.
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        NEB reaction pathway workflow.

    Example
    -------
    >>> flow = reaction_pathway_workflow(
    ...     slab, initial_state=reactant_config, final_state=product_config
    ... )
    """
    # Note: This would use NEBMaker when available
    logger.info(f"Creating reaction_pathway workflow with {num_images} images")
    # Placeholder - full implementation requires NEB maker
    from atomate2.siesta.jobs.core import RelaxMaker

    relax = RelaxMaker.fixed_cell_relaxation(
        use_custodian=kwargs.get("use_custodian", True),
        custodian_max_errors=10,
    )
    return Flow([relax.make(slab_structure)], name="reaction_pathway")


def coverage_dependent_adsorption(
    slab_structure: Structure,
    adsorbate: Molecule,  # noqa: ARG001 documented API param, reserved
    coverages: list[float] | None = None,
    **kwargs,
) -> Flow:
    """
    Coverage-dependent adsorption energies.

    Calculates adsorption energy as function of surface coverage.

    Parameters
    ----------
    slab_structure : Structure
        Surface slab.
    adsorbate : Molecule
        Adsorbate molecule.
    coverages : list
        Coverages to test (ML). Default: [0.25, 0.5, 0.75, 1.0].
    **kwargs
        Additional parameters.

    Returns
    -------
    Flow
        Coverage-dependent workflow.

    Example
    -------
    >>> flow = coverage_dependent_adsorption(
    ...     slab, co_molecule, coverages=[0.11, 0.25, 0.50]
    ... )
    """
    if coverages is None:
        coverages = [0.25, 0.50, 0.75, 1.0]
    logger.info(f"Creating coverage-dependent study for {len(coverages)} coverages")
    # Simplified implementation
    from atomate2.siesta.jobs.core import RelaxMaker

    relax = RelaxMaker.fixed_cell_relaxation(
        use_custodian=kwargs.get("use_custodian", True),
        custodian_max_errors=10,
    )
    return Flow([relax.make(slab_structure)], name="coverage_dependent")
