"""Module registry for automatic dataclass initialization with tier-based filtering.

This module provides a registration system for SIESTA input parameter dataclasses,
enabling automatic initialization based on calculation tiers and material-specific presets.

Tier System: Two Separate Concepts
-----------------------------------
**1. Module Tier Classification** (for module organization - 4 core tiers):
   Each module is assigned to one of 4 core tiers based on complexity:
   - **basic**: 6 essential modules (pseudopotentials, basis_sets, general_system, xc_functional, kpoints, mesh_cutoff)
   - **intermediate**: 7 additional modules (chemical_analysis, constraints, electronic_structure, lua_scripting, md_relaxation, scf_loop, spin)
   - **advanced**: 9 additional modules (auxiliary_force_field, charge_dipole, denchar, dftu, dos_bands, grids_advanced, optical, phonons, wannier90)
   - **expert**: 6 additional modules (efficiency, hamiltonian_overlap, netcdf, parallel, rttddft, solvers)

**2. User-Facing Tier Levels** (for workflow selection - 7 tier names):
   Users specify tier names that map to module sets and parameter presets:
   - **dirty**: Minimal quality (SZ/[1,1,1]/50 Ry) → activates basic modules (6 total)
   - **basic**: Fast quality (DZP/[3,3,3]/150 Ry) → activates basic modules (6 total)
   - **intermediate**: Standard quality (DZP/[6,6,6]/200 Ry) → activates basic+intermediate modules (13 total)
   - **advanced**: High quality (TZP/[6,6,6]/300 Ry) → activates basic+intermediate+advanced modules (22 total)
   - **expert**: Publication quality (TZP/[8,8,8]/400 Ry) → activates all modules (28 total)
   - **ultra**: Benchmark quality (TZDP/[10,10,10]/800 Ry) → activates all modules (28 total)
   - **all**: No parameter defaults → activates all modules (28 total)

The registry solves the problem of manual module initialization in base.py by:
1. Automatically discovering all available parameter modules
2. Organizing them by tier and category
3. Enabling dynamic initialization based on user needs
4. Supporting material-specific parameter presets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataclassModule:
    """Metadata for a SIESTA input parameter dataclass module.

    Attributes
    ----------
    name : str
        Unique identifier for the module (e.g., "basis_sets", "scf_loop")
    module_path : str
        Full Python import path to the module
    class_name : str
        Name of the dataclass in the module
    setup_method : str
        Name of the classmethod used to initialize the module from user_params
    fdf_attribute : str
        Name of the attribute containing FDF arguments after setup
    instance_attribute : str
        Name of the attribute where the module instance is stored (e.g., "md_relaxation")
        Defaults to name if not specified. This enables reuse of updated instances.
    tier : str
        Module tier classification: "basic", "intermediate", "advanced", or "expert"
        Note: This is NOT the same as user-facing tier levels (which also include
        "dirty", "ultra", "all"). This field categorizes which tier the module
        belongs to, not which tier the user selects.
    category : str
        Functional category: "electronic", "structural", "convergence", etc.
    priority : int
        Initialization order (lower = earlier). Modules with dependencies
        should have higher priority values
    description : str
        Brief description of the module's purpose
    """

    name: str
    module_path: str
    class_name: str
    setup_method: str
    fdf_attribute: str
    instance_attribute: str = ""  # Auto-set to name in __post_init__ if not provided
    tier: str = "intermediate"
    category: str = "general"
    priority: int = 50
    description: str = ""

    def __post_init__(self):
        """Validate tier and category values, auto-set instance_attribute."""
        # Module tier classification (not user-facing tier levels)
        # Modules are categorized into 4 core tiers for organization
        # User-facing tiers (dirty, ultra, all) map to these core tiers
        valid_module_tiers = {"basic", "intermediate", "advanced", "expert"}
        if self.tier not in valid_module_tiers:
            raise ValueError(
                f"Invalid module tier '{self.tier}'. Must be one of {valid_module_tiers}. "
                f"Note: This is the module's tier classification, not the user-facing tier level."
            )

        # Auto-set instance_attribute to name if not provided
        if not self.instance_attribute:
            self.instance_attribute = self.name


# Global module registry
MODULE_REGISTRY: dict[str, DataclassModule] = {}


def register_module(
    name: str,
    module_path: str,
    class_name: str,
    setup_method: str,
    fdf_attribute: str,
    tier: str = "intermediate",
    category: str = "general",
    priority: int = 50,
    description: str = "",
) -> None:
    """Register a dataclass module for automatic initialization.

    Parameters
    ----------
    name : str
        Unique module identifier
    module_path : str
        Full import path (e.g., "atomate2.siesta.dataclass.basis_sets_and_projectors")
    class_name : str
        Dataclass name (e.g., "BasisSetsAndProjectors")
    setup_method : str
        Setup classmethod name (e.g., "setup_basis_sets_and_projectors")
    fdf_attribute : str
        Attribute containing FDF args (e.g., "basis_set_fdf_arguments")
    tier : str, optional
        Calculation tier (default: "intermediate")
    category : str, optional
        Functional category (default: "general")
    priority : int, optional
        Init priority, lower=earlier (default: 50)
    description : str, optional
        Brief module description

    Raises
    ------
    ValueError
        If module with same name already registered
    """
    if name in MODULE_REGISTRY:
        raise ValueError(
            f"Module '{name}' already registered. "
            f"Existing: {MODULE_REGISTRY[name].module_path}, "
            f"New: {module_path}"
        )

    MODULE_REGISTRY[name] = DataclassModule(
        name=name,
        module_path=module_path,
        class_name=class_name,
        setup_method=setup_method,
        fdf_attribute=fdf_attribute,
        tier=tier,
        category=category,
        priority=priority,
        description=description,
    )

    logger.debug(f"Registered module: {name} (tier={tier}, priority={priority})")


def get_modules_for_tier(tier: str) -> dict[str, DataclassModule]:
    """Get all modules for a given tier level.

    Tiers are hierarchical:
    - basic: Only basic modules
    - intermediate: basic + intermediate modules
    - advanced: basic + intermediate + advanced modules
    - expert: All modules (basic + intermediate + advanced + expert)

    Parameters
    ----------
    tier : str
        Tier level: "basic", "intermediate", "advanced", "expert", or "all"
        Also supports TIER_DEFAULTS entries: "dirty", "ultra"

    Returns
    -------
    dict[str, DataclassModule]
        Dictionary mapping module names to DataclassModule objects
        for the requested tier

    Raises
    ------
    ValueError
        If tier is invalid

    Notes
    -----
    The tier parameter serves two purposes:
    1. Module activation (controlled by tier_hierarchy below)
    2. Default parameters (loaded from TIER_DEFAULTS in defaults.py)

    TIER_DEFAULTS entries that match core tier names ("basic", "intermediate",
    "advanced", "expert") use those tier levels for module activation.

    Extended tier names ("dirty", "ultra") are mapped to appropriate core levels:
    - "dirty": Uses "basic" module activation (minimal modules)
    - "ultra": Uses "expert" module activation (all modules)
    """
    tier_hierarchy = {
        # Core tier levels (module activation hierarchy)
        "basic": ["basic"],
        "intermediate": ["basic", "intermediate"],
        "advanced": ["basic", "intermediate", "advanced"],
        "expert": ["basic", "intermediate", "advanced", "expert"],
        "all": ["basic", "intermediate", "advanced", "expert"],
        # Extended tier levels (map to core for module activation)
        # These correspond to TIER_DEFAULTS entries in tiers/defaults.py
        "dirty": ["basic"],  # Minimal settings → basic modules only
        "ultra": [
            "basic",
            "intermediate",
            "advanced",
            "expert",
        ],  # Ultra high quality → all modules
    }

    if tier not in tier_hierarchy:
        raise ValueError(
            f"Invalid tier '{tier}'. Must be one of {list(tier_hierarchy.keys())}"
        )

    allowed_tiers = tier_hierarchy[tier]

    return {
        name: module
        for name, module in MODULE_REGISTRY.items()
        if module.tier in allowed_tiers
    }


def get_modules_by_category(category: str) -> dict[str, DataclassModule]:
    """Get all modules in a specific category.

    Parameters
    ----------
    category : str
        Category name (e.g., "electronic", "structural", "convergence")

    Returns
    -------
    dict[str, DataclassModule]
        Dictionary mapping module names to DataclassModule objects
        in the requested category
    """
    return {
        name: module
        for name, module in MODULE_REGISTRY.items()
        if module.category == category
    }


def get_sorted_modules(modules: dict[str, DataclassModule]) -> list[DataclassModule]:
    """Sort modules by initialization priority.

    Parameters
    ----------
    modules : dict[str, DataclassModule]
        Modules to sort

    Returns
    -------
    list[DataclassModule]
        Modules sorted by priority (lower priority = initialized first)
    """
    return sorted(modules.values(), key=lambda m: m.priority)


# ==============================================================================
# Module Registration - All 28 SIESTA Parameter Modules
# ==============================================================================


def register_all_modules():
    """Register all SIESTA input parameter dataclass modules.

    Modules are organized into 4 tiers:
    - BASIC: Essential parameters for any calculation (~6 modules)
    - INTERMEDIATE: Common advanced parameters (~10 modules)
    - ADVANCED: Specialized calculations (~7 modules)
    - EXPERT: Performance tuning & rare use cases (~5 modules)

    Priority ordering ensures dependencies are initialized first:
    - 1-19: Core modules (pseudos, basis, XC, mesh, kpoints)
    - 20-39: Electronic structure (spin, SCF, occupation)
    - 40-59: Structural (MD, relaxation, constraints)
    - 60-79: Specialized (phonons, optical, DOS/bands)
    - 80-99: Performance & advanced (parallel, solvers, efficiency)
    """
    # =========================================================================
    # TIER 1: BASIC - Essential parameters for any calculation
    # =========================================================================

    register_module(
        name="pseudopotentials",
        module_path="atomate2.siesta.dataclass.pseudopotentials",
        class_name="Pseudopotentials",
        setup_method="setup_pseudos",
        fdf_attribute="pseudo_path",  # Special: returns path, not fdf_arguments
        tier="basic",
        category="electronic",
        priority=5,
        description="Pseudopotential file paths and species definitions",
    )

    register_module(
        name="basis_sets",
        module_path="atomate2.siesta.dataclass.basis_sets_and_projectors",
        class_name="BasisSetsAndProjectors",
        setup_method="setup_basis_sets_and_projectors",
        fdf_attribute="basis_set_fdf_arguments",
        tier="basic",
        category="electronic",
        priority=10,
        description="PAO basis set size, energy shift, split norm parameters",
    )

    register_module(
        name="xc_functional",
        module_path="atomate2.siesta.dataclass.exchange_correlation_functionals",
        class_name="ExchangeCorrelationFunctionals",
        setup_method="setup_xc_settings",
        fdf_attribute="xc_fdf_arguments",
        tier="basic",
        category="electronic",
        priority=15,
        description="Exchange-correlation functional (LDA, GGA, etc.)",
    )

    register_module(
        name="kpoints",
        module_path="atomate2.siesta.dataclass.kpoint_sampling",
        class_name="KPointSampling",
        setup_method="setup_kpoint_settings",
        fdf_attribute="kpoint_fdf_arguments",
        tier="basic",
        category="electronic",
        priority=20,
        description="K-point grid and Brillouin zone sampling",
    )

    register_module(
        name="mesh_cutoff",
        module_path="atomate2.siesta.dataclass.real_space_grid_parameters",
        class_name="RealSpaceGridParameters",
        setup_method="setup_grid_settings",
        fdf_attribute="grid_fdf_arguments",
        tier="basic",
        category="numerical",
        priority=25,
        description="Real-space grid mesh cutoff energy",
    )

    register_module(
        name="general_system",
        module_path="atomate2.siesta.dataclass.general_system_descriptors",
        class_name="GeneralSystemDescriptors",
        setup_method="setup_system_descriptors",  # Need to implement
        fdf_attribute="system_fdf_arguments",
        tier="basic",
        category="general",
        priority=1,
        description="System label, name, and general descriptors",
    )

    # =========================================================================
    # TIER 2: INTERMEDIATE - Common advanced parameters
    # =========================================================================

    register_module(
        name="spin",
        module_path="atomate2.siesta.dataclass.spin_settings",
        class_name="SpinSettings",
        setup_method="setup_spin_settings",
        fdf_attribute="spin_fdf_arguments",
        tier="intermediate",
        category="electronic",
        priority=18,
        description="Spin polarization and magnetic properties",
    )

    register_module(
        name="scf_loop",
        module_path="atomate2.siesta.dataclass.scf_loop_parameters",
        class_name="SCFLoopParameters",
        setup_method="setup_scf_settings",
        fdf_attribute="scf_fdf_arguments",
        tier="intermediate",
        category="convergence",
        priority=30,
        description="SCF convergence mixer parameters and tolerances",
    )

    register_module(
        name="electronic_structure",
        module_path="atomate2.siesta.dataclass.electronic_structure_calculation_options",
        class_name="ElectronicStructureCalculationOptions",
        setup_method="setup_electronic_structure_settings",
        fdf_attribute="electronic_structure_fdf_arguments",
        tier="intermediate",
        category="electronic",
        priority=35,
        description="Occupation functions and electronic temperature",
    )

    register_module(
        name="md_relaxation",
        module_path="atomate2.siesta.dataclass.molecular_dynamics_and_relaxation",
        class_name="MolecularDynamicsAndRelaxation",
        setup_method="setup_md_relax_settings",  # Need to implement
        fdf_attribute="relaxation_fdf_arguments",
        tier="intermediate",
        category="structural",
        priority=40,
        description="MD and geometry optimization parameters",
    )

    register_module(
        name="constraints",
        module_path="atomate2.siesta.dataclass.general_constraints",
        class_name="GeneralConstraints",
        setup_method="setup_constraints",  # Need to implement
        fdf_attribute="constraints_fdf_arguments",
        tier="intermediate",
        category="structural",
        priority=45,
        description="Atomic position and cell constraints",
    )

    register_module(
        name="lua_scripting",
        module_path="atomate2.siesta.dataclass.external_control_and_scripting",
        class_name="ExternalControlAndScripting",
        setup_method="setup_lua_settings",
        fdf_attribute="lua_fdf_arguments",
        tier="intermediate",
        category="advanced",
        priority=55,
        description="Lua scripting for custom algorithms (FLOS)",
    )

    register_module(
        name="chemical_analysis",
        module_path="atomate2.siesta.dataclass.chemical_analysis",
        class_name="ChemicalAnalysis",
        setup_method="setup_chemical_analysis",
        fdf_attribute="chemical_analysis_fdf_arguments",
        tier="intermediate",
        category="analysis",
        priority=50,
        description="Mulliken populations, COOP, bond analysis",
    )

    # =========================================================================
    # TIER 3: ADVANCED - Specialized calculations
    # =========================================================================

    register_module(
        name="phonons",
        module_path="atomate2.siesta.dataclass.phonon_calculations",
        class_name="PhononCalculations",
        setup_method="setup_phonon_settings",  # Need to implement
        fdf_attribute="phonon_fdf_arguments",
        tier="advanced",
        category="phonons",
        priority=60,
        description="Force constant and vibrational property calculations",
    )

    register_module(
        name="optical",
        module_path="atomate2.siesta.dataclass.optical_properties",
        class_name="OpticalProperties",
        setup_method="setup_optical_settings",  # Need to implement
        fdf_attribute="optical_fdf_arguments",
        tier="advanced",
        category="optics",
        priority=65,
        description="Optical absorption and dielectric properties",
    )

    register_module(
        name="wannier90",
        module_path="atomate2.siesta.dataclass.wannier90",
        class_name="Wannier90",
        setup_method="setup_wannier90",
        fdf_attribute="wannier90_fdf_arguments",
        tier="advanced",
        category="analysis",
        priority=68,
        description="Wannier90 interface for maximally localized Wannier functions",
    )

    register_module(
        name="dos_bands",
        module_path="atomate2.siesta.dataclass.density_of_states_and_band_structure",
        class_name="DensityOfStatesAndBandStructure",
        setup_method="setup_dos_bands_settings",  # Need to implement
        fdf_attribute="bands_fdf_arguments",
        tier="advanced",
        category="electronic",
        priority=62,
        description="DOS and band structure k-point paths",
    )

    register_module(
        name="dftu",
        module_path="atomate2.siesta.dataclass.dftu",
        class_name="DFTU",
        setup_method="setup_dftu_settings",  # Need to implement
        fdf_attribute="dftu_fdf_arguments",
        tier="advanced",
        category="electronic",
        priority=70,
        description="DFT+U for correlated electron systems",
    )

    register_module(
        name="charge_dipole",
        module_path="atomate2.siesta.dataclass.charge_dipole_electric_field",
        class_name="ChargeDipoleElectricField",
        setup_method="setup_charge_dipole_settings",
        fdf_attribute="charge_dipole_fdf_arguments",
        tier="advanced",
        category="electronic",
        priority=72,
        description="External electric fields and dipole corrections",
    )

    register_module(
        name="grids_advanced",
        module_path="atomate2.siesta.dataclass.grids",
        class_name="Grids",
        setup_method="setup_advanced_grids",
        fdf_attribute="grids_fdf_arguments",
        tier="advanced",
        category="numerical",
        priority=74,
        description="Advanced grid settings (save density, etc.)",
    )

    register_module(
        name="auxiliary_force_field",
        module_path="atomate2.siesta.dataclass.auxiliary_force_field",
        class_name="AuxiliaryForceField",
        setup_method="setup_auxiliary_force_field",
        fdf_attribute="auxiliary_fdf_arguments",
        tier="advanced",
        category="advanced",
        priority=78,
        description="Molecular mechanics force fields (MM.* parameters)",
    )

    register_module(
        name="denchar",
        module_path="atomate2.siesta.dataclass.denchar",
        class_name="Denchar",
        setup_method="setup_denchar",  # Need to implement
        fdf_attribute="denchar_fdf_arguments",
        tier="advanced",
        category="analysis",
        priority=76,
        description="Charge density plotting (denchar utility)",
    )

    # =========================================================================
    # TIER 4: EXPERT - Performance tuning & rare use cases
    # =========================================================================

    register_module(
        name="parallel",
        module_path="atomate2.siesta.dataclass.parallel_options",
        class_name="ParallelOptions",
        setup_method="setup_parallel_settings",  # Need to implement
        fdf_attribute="parallel_fdf_arguments",
        tier="expert",
        category="performance",
        priority=80,
        description="MPI parallelization and domain decomposition",
    )

    register_module(
        name="solvers",
        module_path="atomate2.siesta.dataclass.solvers_and_performance_options",
        class_name="SolversAndPerformanceOptions",
        setup_method="setup_solver_settings",  # Need to implement
        fdf_attribute="solver_fdf_arguments",
        tier="expert",
        category="performance",
        priority=85,
        description="Diagonalization methods and solver tuning",
    )

    register_module(
        name="efficiency",
        module_path="atomate2.siesta.dataclass.efficiency_options",
        class_name="EfficiencyOptions",
        setup_method="setup_efficiency_settings",  # Need to implement
        fdf_attribute="efficiency_fdf_arguments",
        tier="expert",
        category="performance",
        priority=90,
        description="Memory and I/O optimization settings",
    )

    register_module(
        name="hamiltonian_overlap",
        module_path="atomate2.siesta.dataclass.hamiltonian_and_overlap_parameters",
        class_name="HamiltonianAndOverlapParameters",
        setup_method="setup_hamiltonian_settings",
        fdf_attribute="hamiltonian_fdf_arguments",
        tier="expert",
        category="advanced",
        priority=92,
        description="Hamiltonian matrix cutoffs and sparsity",
    )

    register_module(
        name="netcdf",
        module_path="atomate2.siesta.dataclass.netcdf_options",
        class_name="NetcdfOptions",
        setup_method="setup_netcdf_settings",  # Need to implement
        fdf_attribute="netcdf_fdf_arguments",
        tier="expert",
        category="io",
        priority=94,
        description="NetCDF output format options",
    )

    register_module(
        name="rttddft",
        module_path="atomate2.siesta.dataclass.rttddft",
        class_name="RTTDDFT",
        setup_method="setup_rttddft",
        fdf_attribute="rttddft_fdf_arguments",
        tier="expert",
        category="advanced",
        priority=96,
        description="Real-time time-dependent DFT calculations",
    )

    # =========================================================================
    # Remaining specialized modules (tier assignments TBD)
    # =========================================================================

    # NOTE: The following modules exist but may need setup_*() methods:
    # - auxiliary_force_field.py
    # - chemical_analysis.py
    # - rttddft.py
    # - structural_information.py
    # - wannier90.py
    #
    # These can be added once their setup methods are implemented or
    # when specific use cases require them.

    logger.info(f"Registered {len(MODULE_REGISTRY)} dataclass modules")


# Auto-register all modules on import
register_all_modules()
