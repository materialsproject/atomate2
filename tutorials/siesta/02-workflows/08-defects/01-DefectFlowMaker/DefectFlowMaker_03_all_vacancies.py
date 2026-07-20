"""
Generate vacancy defects using from_pristine_structure() with species filtering.

Demonstrates:
  - Example 1: All vacancies (Mg + O) with shared chemical potentials
  - Example 2: Only Mg vacancies (species="Mg")
  - Example 3: Only O vacancies (species="O")
  - Example 4a: Band structure + PDOS with default parameters
  - Example 4b: Custom PDOS energy range and k-grid
  - Example 4c: Custom band path (skips auto-generation)

Key optimization: When auto_calculate_chemical_potentials=True, reference
calculations are shared across all defects. For MgO with 8 vacancy sites,
this creates only 2 reference jobs (Mg bulk + O2 molecule) instead of 16.
"""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.flows.defects import DefectFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.tiers import apply_tier_preset
from jobflow import run_locally

# Create MgO unit cell (rock salt structure)
lattice = Lattice.cubic(4.212)
mgo = Structure(
    lattice,
    ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"],
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ],
)

# Create makers with tier preset
defect_relax_maker = apply_tier_preset(
    RelaxMaker.fixed_cell_relaxation(use_custodian=True), "defect_dirty"
)
host_static_maker = apply_tier_preset(StaticMaker(use_custodian=True), "defect_dirty")

# Common kwargs for all examples
common_kwargs = dict(
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    charge_states=[0],
    epsilon_static=9.8,
    use_symmetry=False,
    use_custodian=True,
    defect_relax_maker=defect_relax_maker,
    host_static_maker=host_static_maker,
    dry_run=True,
    skip_relax=True,
    auto_calculate_chemical_potentials=True,
)

# ============================================================
# Example 1: All vacancies (both Mg and O)
# - Creates 8 defect flows (4 Mg + 4 O vacancies)
# - Only 2 shared reference jobs (Mg bulk + O2 molecule)
# - 1 shared host calculation
# ============================================================
print("=" * 60)
print("Example 1: All vacancies (Mg + O)")
print("=" * 60)

flows_all = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    **common_kwargs,
)

n_jobs_all = len(flows_all.jobs) if hasattr(flows_all, "jobs") else len(flows_all)
print(f"  Total jobs in flow: {n_jobs_all}")
print(
    "  Expected: 1 host + 4 ref jobs (2 ref calc + 2 mu extract) + 8 defect flows + 1 summary"
)

# run_locally(flows_all, create_folders=True, ensure_success=True)

# ============================================================
# Example 2: Only Mg vacancies (species="Mg")
# - Creates 4 defect flows (Mg vacancies only)
# - Only 1 shared reference job (Mg bulk)
# - 1 shared host calculation
# ============================================================
print("\n" + "=" * 60)
print("Example 2: Only Mg vacancies (species='Mg')")
print("=" * 60)

flows_mg = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    species="Mg",
    **common_kwargs,
)

n_jobs_mg = len(flows_mg.jobs) if hasattr(flows_mg, "jobs") else len(flows_mg)
print(f"  Total jobs in flow: {n_jobs_mg}")
print(
    "  Expected: 1 host + 2 ref jobs (1 ref calc + 1 mu extract) + 4 defect flows + 1 summary"
)

# run_locally(flows_mg, create_folders=True, ensure_success=True)

# ============================================================
# Example 3: Only O vacancies (species="O")
# - Creates 4 defect flows (O vacancies only)
# - Only 1 shared reference job (O2 molecule)
# - 1 shared host calculation
# ============================================================
print("\n" + "=" * 60)
print("Example 3: Only O vacancies (species='O')")
print("=" * 60)

flows_o = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    species="O",
    **common_kwargs,
)

n_jobs_o = len(flows_o.jobs) if hasattr(flows_o, "jobs") else len(flows_o)
print(f"  Total jobs in flow: {n_jobs_o}")
print(
    "  Expected: 1 host + 2 ref jobs (1 ref calc + 1 mu extract) + 4 defect flows + 1 summary"
)

# run_locally(flows_o, create_folders=True, ensure_success=True)

# ============================================================
# Example 4a: Mg vacancy with default bands + PDOS
# - Enables band structure (auto-generated k-path) and PDOS
# - Produces siesta.bands, siesta.PDOS, siesta.PDOS.xml
# - No extra jobs: runs within the same SIESTA calculation
# ============================================================
print("\n" + "=" * 60)
print("Example 4a: Mg vacancy with bands + PDOS (defaults)")
print("=" * 60)

flows_bands = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    species="Mg",
    include_bandstructure=True,
    include_pdos=True,
    **common_kwargs,
)

n_jobs_bands = (
    len(flows_bands.jobs) if hasattr(flows_bands, "jobs") else len(flows_bands)
)
print(f"  Total jobs in flow: {n_jobs_bands}")
print("  Band structure: auto-generated k-path, 20 interpolation points")
print("  PDOS: default range EF [-15.0, 15.0] eV (relative to Fermi level)")


run_locally(flows_bands, create_folders=True, ensure_success=True)
# ============================================================
# Example 4b: Custom PDOS energy range and k-grid
# - Wider energy range and finer resolution
# - Custom PDOS k-grid for better sampling
# ============================================================
print("\n" + "=" * 60)
print("Example 4b: O vacancy with custom PDOS parameters")
print("=" * 60)

flows_custom_pdos = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    species="O",
    include_pdos=True,
    include_bandstructure=True,
    pdos_fdf_params={
        "%block ProjectedDensityOfStates": ["EF -15.0 10.0 0.02 500 eV"],
        "%block PDOS.kgrid.MonkhorstPack": [
            "2 0 0 0.0",
            "0 2 0 0.0",
            "0 0 2 0.0",
        ],
    },
    **common_kwargs,
)

n_jobs_custom_pdos = (
    len(flows_custom_pdos.jobs)
    if hasattr(flows_custom_pdos, "jobs")
    else len(flows_custom_pdos)
)
print(f"  Total jobs in flow: {n_jobs_custom_pdos}")
print("  PDOS: custom range EF [-15.0, 10.0] eV, 0.02 eV broadening, 500 points")

# run_locally(flows_custom_pdos, create_folders=True, ensure_success=True)

# ============================================================
# Example 4c: Custom band path (skips auto-generation)
# - User-defined BandLines in pi/a units
# - Useful when default symmetry path is not desired
# ============================================================
print("\n" + "=" * 60)
print("Example 4c: Mg vacancy with custom band path")
print("=" * 60)

flows_custom_bands = DefectFlowMaker.from_pristine_structure(
    mgo,
    defect_type="vacancy",
    species="Mg",
    include_bandstructure=True,
    bands_fdf_params={
        "BandLinesScale": "pi/a",
        "%block BandLines": [
            "1 0.0 0.0 0.0  # Gamma",
            "40 0.5 0.0 0.0  # X",
            "40 0.5 0.5 0.0  # M",
            "40 0.0 0.0 0.0  # Gamma",
        ],
    },
    **common_kwargs,
)

n_jobs_custom_bands = (
    len(flows_custom_bands.jobs)
    if hasattr(flows_custom_bands, "jobs")
    else len(flows_custom_bands)
)
print(f"  Total jobs in flow: {n_jobs_custom_bands}")
print("  Band path: custom Gamma-X-M-Gamma in pi/a units")

# run_locally(flows_custom_bands, create_folders=True, ensure_success=True)
# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"  All vacancies:  {n_jobs_all} jobs (shared Mg + O reference calculations)")
print(f"  Mg only:        {n_jobs_mg} jobs (shared Mg reference only)")
print(f"  O only:         {n_jobs_o} jobs (shared O reference only)")
print(f"  Bands + PDOS:   {n_jobs_bands} jobs (with band structure & PDOS)")
print(f"  Custom PDOS:    {n_jobs_custom_pdos} jobs (custom PDOS parameters)")
print(f"  Custom bands:   {n_jobs_custom_bands} jobs (custom band path)")
print("\nOutput files:")
print("  - Individual summaries: job*/defect_summary_q=*.txt")
print("  - Combined summary: all_defects_summary.txt")
print("  - Band structure (when enabled): siesta.bands")
print("  - PDOS (when enabled): siesta.DOS, siesta.PDOS, siesta.PDOS.xml")
