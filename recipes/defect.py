"""Defect workflow recipes."""

from __future__ import annotations

import logging

from jobflow import Flow
from pymatgen.core import Structure

from atomate2.siesta.flows.defects import DefectFlowMaker
from atomate2.siesta.recipes.base import MaterialAnalyzer

logger = logging.getLogger(__name__)


def complete_defect_study(
    structure: Structure,
    supercell_matrix: list[list[int]] | None = None,
    charge_states: list[int] | None = None,
    epsilon_static: float | None = None,
    dopants: list[str] | None = None,
    interstitial_species: list[str] | None = None,
    auto_params: bool = True,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "complete_defect_study",
) -> list[Flow]:
    """
    Complete defect study: all vacancies, antisites, and interstitials.

    This recipe generates flows for all defect types in a material:
    - All symmetry-unique vacancies
    - All antisite defects (atom swaps)
    - Interstitial defects (if species specified)

    Parameters
    ----------
    structure : Structure
        Pristine host structure (unit cell or primitive cell).
    supercell_matrix : list[list[int]]
        Supercell transformation matrix. If None, uses [[2,0,0],[0,2,0],[0,0,2]].
    charge_states : list[int]
        Charge states to consider. If None, uses [0].
    epsilon_static : float
        Static dielectric constant for finite-size corrections. If None,
        auto-estimated from material type.
    dopants : list[str]
        Dopant elements for substitutional defects. If None, only generates antisites.
    interstitial_species : list[str]
        Species for interstitial defects. If None, skips interstitials.
    auto_params : bool
        If True, automatically determine optimal parameters. Default: True.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
        Note: DefectFlowMaker may not support custodian yet.
    dry_run : bool
        If True, only generate input files. Default: False.
    name : str
        Base name for workflows.

    Returns
    -------
    list[Flow]
        List of defect flows (one per defect type × charge state).

    Example
    -------
    >>> from atomate2.siesta.recipes import RecipeBook
    >>> from pymatgen.core import Structure
    >>>
    >>> structure = Structure.from_file("POSCAR")
    >>> flows = RecipeBook.complete_defect_study(structure)
    >>> # Generates all vacancy and antisite defects automatically!
    >>>
    >>> # With interstitials
    >>> flows = RecipeBook.complete_defect_study(
    ...     structure,
    ...     interstitial_species=["Li", "H"],
    ...     charge_states=[0, +1],
    ... )
    """
    logger.info(f"Creating complete_defect_study for {structure.composition}")

    # Default supercell
    if supercell_matrix is None:
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    # Default charge states
    if charge_states is None:
        charge_states = [0]

    # Auto-determine epsilon if not provided
    if epsilon_static is None and auto_params:
        analysis = MaterialAnalyzer.analyze(structure)
        # Rough estimates: metals ~10, semiconductors ~10-20, insulators ~5-10
        if analysis.is_metal:
            epsilon_static = 10.0
        elif analysis.max_z <= 18:  # Light elements
            epsilon_static = 8.0
        else:
            epsilon_static = 12.0
        logger.info(f"Auto-estimated epsilon_static = {epsilon_static}")

    all_flows = []

    # Generate all vacancies
    logger.info("Generating vacancy defects...")
    vacancy_flows = DefectFlowMaker.from_pristine_structure(
        structure,
        defect_type="vacancy",
        supercell_matrix=supercell_matrix,
        charge_states=charge_states,
        epsilon_static=epsilon_static,
        dry_run=dry_run,
    )
    all_flows.extend(vacancy_flows)
    logger.info(f"  → Generated {len(vacancy_flows)} vacancy flows")

    # Generate antisite defects (or dopant substitutions)
    logger.info("Generating substitutional defects...")
    substitution_flows = DefectFlowMaker.from_pristine_structure(
        structure,
        defect_type="substitution",
        dopants=dopants,  # None = antisites, list = dopants
        supercell_matrix=supercell_matrix,
        charge_states=charge_states,
        epsilon_static=epsilon_static,
        dry_run=dry_run,
    )
    all_flows.extend(substitution_flows)
    logger.info(f"  → Generated {len(substitution_flows)} substitution flows")

    # Generate interstitial defects (if species provided)
    if interstitial_species:
        logger.info("Generating interstitial defects...")
        for species in interstitial_species:
            interstitial_flows = DefectFlowMaker.from_pristine_structure(
                structure,
                defect_type="interstitial",
                species=species,
                supercell_matrix=supercell_matrix,
                charge_states=charge_states,
                epsilon_static=epsilon_static,
                dry_run=dry_run,
            )
            all_flows.extend(interstitial_flows)
            logger.info(
                f"  → Generated {len(interstitial_flows)} {species} interstitial flows"
            )

    logger.info(f"Total defect flows: {len(all_flows)}")
    return all_flows


def vacancy_study(
    structure: Structure,
    supercell_matrix: list[list[int]] | None = None,
    charge_states: list[int] | None = None,
    epsilon_static: float | None = None,
    auto_params: bool = True,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "vacancy_study",
) -> list[Flow]:
    """
    Generate all symmetry-unique vacancy defects.

    This recipe generates flows for all unique vacancy sites in the structure,
    automatically detecting symmetry-equivalent positions.

    Parameters
    ----------
    structure : Structure
        Pristine host structure (unit cell or primitive cell).
    supercell_matrix : list[list[int]]
        Supercell transformation matrix. If None, uses [[2,0,0],[0,2,0],[0,0,2]].
    charge_states : list[int]
        Charge states to consider. If None, uses [0].
    epsilon_static : float
        Static dielectric constant for finite-size corrections. If None,
        auto-estimated from material type.
    auto_params : bool
        If True, automatically determine optimal parameters. Default: True.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
        Note: DefectFlowMaker may not support custodian yet.
    dry_run : bool
        If True, only generate input files. Default: False.
    name : str
        Base name for workflows.

    Returns
    -------
    list[Flow]
        List of vacancy flows.

    Example
    -------
    >>> from atomate2.siesta.recipes import RecipeBook
    >>> flows = RecipeBook.vacancy_study(structure, charge_states=[0, +2])
    """
    logger.info(f"Creating vacancy_study for {structure.composition}")

    # Default values
    if supercell_matrix is None:
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    if charge_states is None:
        charge_states = [0]

    # Auto-determine epsilon
    if epsilon_static is None and auto_params:
        analysis = MaterialAnalyzer.analyze(structure)
        epsilon_static = 10.0 if analysis.is_metal else 8.0
        logger.info(f"Auto-estimated epsilon_static = {epsilon_static}")

    flows = DefectFlowMaker.from_pristine_structure(
        structure,
        defect_type="vacancy",
        supercell_matrix=supercell_matrix,
        charge_states=charge_states,
        epsilon_static=epsilon_static,
        dry_run=dry_run,
    )

    logger.info(f"Generated {len(flows)} vacancy flows")
    return flows


def substitution_study(
    structure: Structure,
    dopants: list[str] | str,
    species: str | None = None,
    supercell_matrix: list[list[int]] | None = None,
    charge_states: list[int] | None = None,
    epsilon_static: float | None = None,
    auto_params: bool = True,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "substitution_study",
) -> list[Flow]:
    """
    Generate substitutional dopant defects.

    This recipe generates flows for dopant atoms substituting host atoms.
    For example, Li doping on Mg sites in MgO.

    Parameters
    ----------
    structure : Structure
        Pristine host structure.
    dopants : list[str] or str
        Dopant element(s) to substitute. Can be single element or list.
    species : str
        Host species to replace. If None, tries all species.
    supercell_matrix : list[list[int]]
        Supercell transformation matrix. If None, uses [[2,0,0],[0,2,0],[0,0,2]].
    charge_states : list[int]
        Charge states to consider. If None, uses [0].
    epsilon_static : float
        Static dielectric constant. If None, auto-estimated.
    auto_params : bool
        If True, automatically determine optimal parameters. Default: True.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
        Note: DefectFlowMaker may not support custodian yet.
    dry_run : bool
        If True, only generate input files. Default: False.
    name : str
        Base name for workflows.

    Returns
    -------
    list[Flow]
        List of substitutional defect flows.

    Example
    -------
    >>> from atomate2.siesta.recipes import RecipeBook
    >>>
    >>> # Li dopant on Mg sites
    >>> flows = RecipeBook.substitution_study(
    ...     structure,
    ...     dopants="Li",
    ...     species="Mg",
    ...     charge_states=[-1, 0],
    ... )
    >>>
    >>> # Multiple dopants
    >>> flows = RecipeBook.substitution_study(
    ...     structure,
    ...     dopants=["Li", "Na", "K"],
    ...     species="Mg",
    ... )
    """
    logger.info(f"Creating substitution_study for {structure.composition}")

    # Ensure dopants is a list
    if isinstance(dopants, str):
        dopants = [dopants]

    # Default values
    if supercell_matrix is None:
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    if charge_states is None:
        charge_states = [0]

    # Auto-determine epsilon
    if epsilon_static is None and auto_params:
        analysis = MaterialAnalyzer.analyze(structure)
        epsilon_static = 10.0 if analysis.is_metal else 8.0
        logger.info(f"Auto-estimated epsilon_static = {epsilon_static}")

    all_flows = []
    for dopant in dopants:
        flows = DefectFlowMaker.from_pristine_structure(
            structure,
            defect_type="substitution",
            species=species,
            dopants=dopant,
            supercell_matrix=supercell_matrix,
            charge_states=charge_states,
            epsilon_static=epsilon_static,
            dry_run=dry_run,
        )
        all_flows.extend(flows)
        logger.info(f"Generated {len(flows)} {dopant} substitution flows")

    logger.info(f"Total substitution flows: {len(all_flows)}")
    return all_flows


def antisite_study(
    structure: Structure,
    supercell_matrix: list[list[int]] | None = None,
    charge_states: list[int] | None = None,
    epsilon_static: float | None = None,
    auto_params: bool = True,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "antisite_study",
) -> list[Flow]:
    """
    Generate all antisite defects (atom swapping).

    This recipe generates flows for all antisite pairs (A_B and B_A) where
    atoms swap positions. For example, Mg_O and O_Mg in MgO.

    Parameters
    ----------
    structure : Structure
        Pristine host structure.
    supercell_matrix : list[list[int]]
        Supercell transformation matrix. If None, uses [[2,0,0],[0,2,0],[0,0,2]].
    charge_states : list[int]
        Charge states to consider. If None, uses [0].
    epsilon_static : float
        Static dielectric constant. If None, auto-estimated.
    auto_params : bool
        If True, automatically determine optimal parameters. Default: True.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
        Note: DefectFlowMaker may not support custodian yet.
    dry_run : bool
        If True, only generate input files. Default: False.
    name : str
        Base name for workflows.

    Returns
    -------
    list[Flow]
        List of antisite defect flows.

    Example
    -------
    >>> from atomate2.siesta.recipes import RecipeBook
    >>> flows = RecipeBook.antisite_study(structure, charge_states=[0])
    """
    logger.info(f"Creating antisite_study for {structure.composition}")

    # Default values
    if supercell_matrix is None:
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    if charge_states is None:
        charge_states = [0]

    # Auto-determine epsilon
    if epsilon_static is None and auto_params:
        analysis = MaterialAnalyzer.analyze(structure)
        epsilon_static = 10.0 if analysis.is_metal else 8.0
        logger.info(f"Auto-estimated epsilon_static = {epsilon_static}")

    # dopants=None triggers antisite generation
    flows = DefectFlowMaker.from_pristine_structure(
        structure,
        defect_type="substitution",
        dopants=None,  # None = generate antisites
        supercell_matrix=supercell_matrix,
        charge_states=charge_states,
        epsilon_static=epsilon_static,
        dry_run=dry_run,
    )

    logger.info(f"Generated {len(flows)} antisite flows")
    return flows


def interstitial_study(
    structure: Structure,
    species: str | list[str],
    supercell_matrix: list[list[int]] | None = None,
    charge_states: list[int] | None = None,
    epsilon_static: float | None = None,
    auto_params: bool = True,
    use_custodian: bool = True,
    dry_run: bool = False,
    name: str = "interstitial_study",
) -> list[Flow]:
    """
    Generate interstitial defects at high-symmetry sites.

    This recipe generates flows for interstitial atoms at automatically
    detected high-symmetry positions (body centers, face centers, etc.).

    Parameters
    ----------
    structure : Structure
        Pristine host structure.
    species : str or list[str]
        Interstitial species. Can be single element or list.
    supercell_matrix : list[list[int]]
        Supercell transformation matrix. If None, uses [[2,0,0],[0,2,0],[0,0,2]].
    charge_states : list[int]
        Charge states to consider. If None, uses [0].
    epsilon_static : float
        Static dielectric constant. If None, auto-estimated.
    auto_params : bool
        If True, automatically determine optimal parameters. Default: True.
    use_custodian : bool
        If True, use custodian for automatic error handling. Default: True.
        Note: DefectFlowMaker may not support custodian yet.
    dry_run : bool
        If True, only generate input files. Default: False.
    name : str
        Base name for workflows.

    Returns
    -------
    list[Flow]
        List of interstitial defect flows.

    Example
    -------
    >>> from atomate2.siesta.recipes import RecipeBook
    >>>
    >>> # Li interstitials
    >>> flows = RecipeBook.interstitial_study(
    ...     structure,
    ...     species="Li",
    ...     charge_states=[0, +1],
    ... )
    >>>
    >>> # Multiple interstitial species
    >>> flows = RecipeBook.interstitial_study(
    ...     structure,
    ...     species=["Li", "H"],
    ... )
    """
    logger.info(f"Creating interstitial_study for {structure.composition}")

    # Ensure species is a list
    if isinstance(species, str):
        species = [species]

    # Default values
    if supercell_matrix is None:
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    if charge_states is None:
        charge_states = [0]

    # Auto-determine epsilon
    if epsilon_static is None and auto_params:
        analysis = MaterialAnalyzer.analyze(structure)
        epsilon_static = 10.0 if analysis.is_metal else 8.0
        logger.info(f"Auto-estimated epsilon_static = {epsilon_static}")

    all_flows = []
    for spec in species:
        flows = DefectFlowMaker.from_pristine_structure(
            structure,
            defect_type="interstitial",
            species=spec,
            supercell_matrix=supercell_matrix,
            charge_states=charge_states,
            epsilon_static=epsilon_static,
            dry_run=dry_run,
        )
        all_flows.extend(flows)
        logger.info(f"Generated {len(flows)} {spec} interstitial flows")

    logger.info(f"Total interstitial flows: {len(all_flows)}")
    return all_flows
