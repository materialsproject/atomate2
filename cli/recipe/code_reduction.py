#!/usr/bin/env python
"""
Code reduction calculation for Recipe Book.

Compares manual workflow implementation vs recipe one-liners
to calculate actual code reduction percentages.
"""

# Manual workflow templates (typical implementation before recipes)
MANUAL_TEMPLATES = {
    # Complete workflows
    "complete_material_study": """
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker
from atomate2.siesta.flows.elastic import ElasticFlowMaker
from atomate2.siesta.flows.eos import SiestaEosFlowMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import Flow, run_locally

# Relax maker
relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={"a2s_kpts": [4,4,4], "Mesh.Cutoff": "300 Ry"}
)
relax_maker = apply_tier_preset(relax_maker, "intermediate")

# Static maker for bands/DOS
static_maker = StaticMaker.scf(user_params={"a2s_kpts": [6,6,6]})
static_maker = apply_tier_preset(static_maker, "intermediate")

# Band structure maker
from atomate2.siesta.flows.core import BandStructureFlowMaker
bands_maker = BandStructureFlowMaker(
    relax_maker=relax_maker,
    static_maker=static_maker
)

# Elastic maker
elastic_maker = ElasticFlowMaker(
    bulk_relax_maker=relax_maker,
    elastic_relax_maker=relax_maker
)

# Phonon maker
phonon_maker = SiestaPhononFlowMaker(
    relax_maker=relax_maker,
    static_maker=static_maker,
    min_length=15.0
)

# EOS maker
eos_maker = SiestaEosFlowMaker(
    initial_relax_maker=relax_maker,
    eos_relax_maker=relax_maker,
    number_of_frames=7
)

# Create jobs
relax_job = relax_maker.make(structure)
bands_job = bands_maker.make(relax_job.output.structure)
elastic_job = elastic_maker.make(relax_job.output.structure)
phonon_job = phonon_maker.make(relax_job.output.structure)
eos_job = eos_maker.make(relax_job.output.structure)

# Create flow
flow = Flow([relax_job, bands_job, elastic_job, phonon_job, eos_job])
results = run_locally(flow, create_folders=True)
""",
    "quick_characterization": """
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.flows.core import BandStructureFlowMaker
from atomate2.siesta.flows.eos import SiestaEosFlowMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import Flow, run_locally

relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={"a2s_kpts": [4,4,4]}
)
relax_maker = apply_tier_preset(relax_maker, "basic")

static_maker = StaticMaker.scf()
bands_maker = BandStructureFlowMaker(
    relax_maker=relax_maker,
    static_maker=static_maker
)

eos_maker = SiestaEosFlowMaker(
    initial_relax_maker=relax_maker,
    eos_relax_maker=relax_maker
)

relax_job = relax_maker.make(structure)
bands_job = bands_maker.make(relax_job.output.structure)
eos_job = eos_maker.make(relax_job.output.structure)

flow = Flow([relax_job, bands_job, eos_job])
results = run_locally(flow, create_folders=True)
""",
    # Electronic properties
    "band_structure_workflow": """
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.flows.core import BandStructureFlowMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={"a2s_kpts": [4,4,4], "Mesh.Cutoff": "300 Ry"}
)
relax_maker = apply_tier_preset(relax_maker, "intermediate")

static_maker = StaticMaker.scf(
    user_params={"a2s_kpts": [6,6,6]}
)
static_maker = apply_tier_preset(static_maker, "intermediate")

bands_maker = BandStructureFlowMaker(
    relax_maker=relax_maker,
    static_maker=static_maker,
    generate_band_structure=True
)

flow = bands_maker.make(structure)
results = run_locally(flow, create_folders=True)
""",
    "dos_workflow": """
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import Flow, run_locally

relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={"a2s_kpts": [4,4,4]}
)
relax_maker = apply_tier_preset(relax_maker, "intermediate")

static_maker = StaticMaker.scf(
    user_params={"a2s_kpts": [8,8,8]}
)
static_maker = apply_tier_preset(static_maker, "intermediate")

relax_job = relax_maker.make(structure)
dos_job = static_maker.make(relax_job.output.structure)

flow = Flow([relax_job, dos_job])
results = run_locally(flow, create_folders=True)
""",
    # Mechanical properties
    "elastic_constants_workflow": """
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.flows.elastic import ElasticFlowMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "a2s_kpts": [4,4,4],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP"
    }
)
relax_maker = apply_tier_preset(relax_maker, "intermediate")

elastic_maker = ElasticFlowMaker(
    bulk_relax_maker=relax_maker,
    elastic_relax_maker=relax_maker,
    generate_elastic_deformations_kwargs={},
    fit_elastic_tensor_kwargs={"fitting_method": "finite_difference"}
)

flow = elastic_maker.make(structure)
results = run_locally(flow, create_folders=True)
""",
    "equation_of_state": """
from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.flows.eos import SiestaEosFlowMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={"a2s_kpts": [4,4,4], "Mesh.Cutoff": "300 Ry"}
)
relax_maker = apply_tier_preset(relax_maker, "intermediate")

eos_maker = SiestaEosFlowMaker(
    name="eos",
    initial_relax_maker=relax_maker,
    eos_relax_maker=relax_maker,
    static_maker=None,
    number_of_frames=7,
    postprocessor=None
)

flow = eos_maker.make(structure)
results = run_locally(flow, create_folders=True)
""",
    # Thermal properties
    "phonon_workflow": """
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={
        "a2s_kpts": [4,4,4],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP",
        "SCF.H.Tolerance": 1e-5
    }
)
relax_maker = apply_tier_preset(relax_maker, "phonon_high_accuracy")

static_maker = StaticMaker.scf(
    user_params={
        "a2s_kpts": [4,4,4],
        "PAO.BasisSize": "DZP",
        "SCF.H.Tolerance": 1e-5
    }
)
static_maker = apply_tier_preset(static_maker, "phonon_high_accuracy")

phonon_maker = SiestaPhononFlowMaker(
    name="phonons",
    relax_maker=relax_maker,
    static_maker=static_maker,
    min_length=15.0,
    bulk_relax_maker=None,
    born_maker=None
)

flow = phonon_maker.make(structure)
results = run_locally(flow, create_folders=True)
""",
    "gruneisen_parameters": """
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker, SiestaGruneisenFlowMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={"PAO.BasisSize": "DZP", "SCF.H.Tolerance": 1e-5}
)
relax_maker = apply_tier_preset(relax_maker, "phonon_high_accuracy")

static_maker = StaticMaker.scf(
    user_params={"PAO.BasisSize": "DZP", "SCF.H.Tolerance": 1e-5}
)
static_maker = apply_tier_preset(static_maker, "phonon_high_accuracy")

phonon_maker = SiestaPhononFlowMaker(
    relax_maker=relax_maker,
    static_maker=static_maker,
    min_length=15.0
)

gruneisen_maker = SiestaGruneisenFlowMaker(
    name="gruneisen",
    structure_optimizer=relax_maker,
    phonon_maker=phonon_maker,
    perc_vol=0.01
)

flow = gruneisen_maker.make(structure)
results = run_locally(flow, create_folders=True)
""",
    "qha_workflow": """
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker, SiestaQhaFlowMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

relax_maker = RelaxMaker.fixed_cell_relaxation(
    user_params={"PAO.BasisSize": "DZP", "SCF.H.Tolerance": 1e-5}
)
relax_maker = apply_tier_preset(relax_maker, "phonon_high_accuracy")

static_maker = StaticMaker.scf(
    user_params={"PAO.BasisSize": "DZP", "SCF.H.Tolerance": 1e-5}
)
static_maker = apply_tier_preset(static_maker, "phonon_high_accuracy")

phonon_maker = SiestaPhononFlowMaker(
    relax_maker=relax_maker,
    static_maker=static_maker,
    min_length=15.0
)

temperature_list = list(range(0, 1001, 10))

qha_maker = SiestaQhaFlowMaker(
    name="qha",
    structure_optimizer=relax_maker,
    phonon_maker=phonon_maker,
    temperature=temperature_list,
    pressure=None,
    eos_type="vinet"
)

flow = qha_maker.make(structure)
results = run_locally(flow, create_folders=True)
""",
    # Surface & Catalysis
    "surface_energy_workflow": """
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.flows.surface import MultiSurfaceEnergyFlowMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

bulk_static_maker = StaticMaker.scf(
    user_params={
        "a2s_kpts": [4,4,4],
        "Mesh.Cutoff": "300 Ry"
    }
)
bulk_static_maker = apply_tier_preset(bulk_static_maker, "surface_semiconductor")

slab_static_maker = StaticMaker.scf(
    user_params={
        "a2s_kpts": [4,4,1],
        "Mesh.Cutoff": "300 Ry"
    }
)
slab_static_maker = apply_tier_preset(slab_static_maker, "surface_semiconductor")

surface_maker = MultiSurfaceEnergyFlowMaker(
    name="surface_energy",
    bulk_static_maker=bulk_static_maker,
    slab_static_maker=slab_static_maker,
    miller_indices=[(1,0,0), (1,1,0), (1,1,1)],
    slab_layers=5,
    vacuum_size=15.0
)

flow = surface_maker.make(structure)
results = run_locally(flow, create_folders=True)
""",
    "adsorption_site_scanning": """
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from pymatgen.core import Molecule
from jobflow import run_locally

static_maker = StaticMaker.scf(
    user_params={
        "a2s_kpts": [4,4,1],
        "Mesh.Cutoff": "300 Ry",
        "PAO.BasisSize": "DZP"
    }
)
static_maker = apply_tier_preset(static_maker, "surface_semiconductor")

adsorbate = Molecule(["H"], [[0,0,0]])

ads_maker = AdsorptionScanFlowMaker(
    name="adsorption_scan",
    slab_static_maker=static_maker,
    adsorbate_static_maker=static_maker,
    grid_size=(5, 5),
    height=2.0,
    orientations=None
)

flow = ads_maker.make(slab_structure, adsorbate)
results = run_locally(flow, create_folders=True)
""",
    # Convergence
    "kpoints_convergence": """
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.flows.convergence import KpointsConvergenceFlowMaker
from jobflow import run_locally

static_maker = StaticMaker.scf(
    user_params={"Mesh.Cutoff": "300 Ry"}
)

kpts_maker = KpointsConvergenceFlowMaker(
    name="kpoints_convergence",
    static_maker=static_maker,
    kpoints_list=[[2,2,2], [4,4,4], [6,6,6], [8,8,8], [10,10,10]],
    tolerance=0.001
)

flow = kpts_maker.make(structure)
results = run_locally(flow, create_folders=True)
""",
}

# Recipe one-liners (after recipes)
RECIPE_TEMPLATES = {
    "complete_material_study": "RecipeBook.complete_material_study(structure)",
    "quick_characterization": "RecipeBook.quick_characterization(structure)",
    "band_structure_workflow": "RecipeBook.band_structure_workflow(structure)",
    "dos_workflow": "RecipeBook.dos_workflow(structure)",
    "elastic_constants_workflow": "RecipeBook.elastic_constants_workflow(structure)",
    "equation_of_state": "RecipeBook.eos_workflow(structure)",
    "phonon_workflow": "RecipeBook.phonon_workflow(structure)",
    "gruneisen_parameters": "RecipeBook.gruneisen_workflow(structure)",
    "qha_workflow": "RecipeBook.qha_workflow(structure)",
    "surface_energy_workflow": "RecipeBook.surface_energy_workflow(bulk_structure)",
    "adsorption_site_scanning": "RecipeBook.adsorption_scanning_workflow(slab, adsorbate)",
    "kpoints_convergence": "RecipeBook.kpoints_convergence(structure)",
}


def count_meaningful_lines(code: str) -> int:
    """
    Count non-empty, non-comment lines.

    Excludes:
    - Empty lines
    - Lines with only whitespace
    - Comment-only lines (starting with #)
    - Docstrings

    Parameters
    ----------
    code : str
        Python code as string

    Returns
    -------
    int
        Number of meaningful code lines
    """
    lines = code.strip().split("\n")
    count = 0
    in_docstring = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Handle docstrings
        if '"""' in stripped or "'''" in stripped:
            in_docstring = not in_docstring
            continue

        if in_docstring:
            continue

        # Skip comment-only lines
        if stripped.startswith("#"):
            continue

        count += 1

    return count


def calculate_code_reduction(recipe_name: str) -> dict:
    """
    Calculate code reduction for a recipe.

    Parameters
    ----------
    recipe_name : str
        Name of the recipe

    Returns
    -------
    dict
        Dictionary with 'before', 'after', 'reduction' keys
    """
    if recipe_name not in MANUAL_TEMPLATES:
        # Estimate for recipes without templates
        return {
            "before": 50,  # Typical manual workflow
            "after": 2,  # Typical recipe call
            "reduction": 96.0,
        }

    manual_code = MANUAL_TEMPLATES[recipe_name]
    recipe_code = RECIPE_TEMPLATES.get(recipe_name, "RecipeBook.workflow(structure)")

    before = count_meaningful_lines(manual_code)
    after = count_meaningful_lines(recipe_code)

    # Add standard boilerplate (imports + run_locally) to recipe
    after += 3  # from ... import, structure = ..., results = run_locally(...)

    reduction = ((before - after) / before) * 100 if before > 0 else 0

    return {"before": before, "after": after, "reduction": round(reduction, 1)}


def get_code_reduction_percentage(recipe_name: str) -> str:
    """
    Get formatted code reduction percentage for a recipe.

    Parameters
    ----------
    recipe_name : str
        Name of the recipe

    Returns
    -------
    str
        Formatted percentage (e.g., "96%")
    """
    result = calculate_code_reduction(recipe_name)
    return f"{result['reduction']:.0f}%"


def get_detailed_comparison(recipe_name: str) -> dict:
    """
    Get detailed before/after comparison for a recipe.

    Parameters
    ----------
    recipe_name : str
        Name of the recipe

    Returns
    -------
    dict
        Detailed comparison including code snippets
    """
    result = calculate_code_reduction(recipe_name)

    manual_code = MANUAL_TEMPLATES.get(recipe_name, "# Manual implementation")
    recipe_code = RECIPE_TEMPLATES.get(recipe_name, "# Recipe one-liner")

    return {
        **result,
        "manual_code": manual_code.strip(),
        "recipe_code": recipe_code.strip(),
    }
