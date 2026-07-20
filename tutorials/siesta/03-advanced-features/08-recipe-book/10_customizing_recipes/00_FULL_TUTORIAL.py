#!/usr/bin/env python
"""Tutorial: Customizing Recipe Book Workflows.

This tutorial demonstrates how to customize SIESTA FDF parameters
when using Recipe Book one-liners.

Key customization methods:
1. user_params - Direct FDF parameter overrides
2. tier - Computational tier level (basic → expert)
3. preset - Material-specific parameter sets
4. Combined customization
"""

from pymatgen.core import Lattice, Structure

# Create test structure (diamond cubic silicon)
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

print("=" * 80)
print("CUSTOMIZING RECIPE BOOK WORKFLOWS")
print("=" * 80)
print()

# ==============================================================================
# Method 1: Using user_params (Direct FDF Parameter Customization)
# ==============================================================================
print("Method 1: Direct FDF Parameter Customization (user_params)")
print("-" * 80)
print()

print("Example 1a: Basic parameter override")
print("-" * 40)
code1a = """
flow = RecipeBook.phonon_workflow(
    silicon,
    user_params={
        'PAO.BasisSize': 'TZP',           # Triple-zeta polarized
        'Mesh.Cutoff': '400 Ry',          # Higher cutoff
        'a2s_kpts': [6, 6, 6],            # Denser k-point mesh
        'SCF.H.Tolerance': 1e-5,          # Tighter convergence
    }
)
"""
print(code1a)
print("✅ Overrides default parameters while keeping recipe simplicity")
print()

print("Example 1b: Magnetic calculation")
print("-" * 40)
code1b = """
flow = RecipeBook.band_structure_workflow(
    structure,
    user_params={
        'Spin': 'polarized',              # Enable spin polarization
        'a2s_magnetic_ordering': 'FM',    # Ferromagnetic ordering
        'OccupationFunction': 'MP',       # Methfessel-Paxton smearing
        'ElectronicTemperature': '300 K', # Electronic temperature
    }
)
"""
print(code1b)
print("✅ Recipe handles magnetic systems with custom parameters")
print()

print("Example 1c: DFT+U calculation")
print("-" * 40)
code1c = """
flow = RecipeBook.elastic_constants_workflow(
    structure,
    user_params={
        'xc': 'PBE',
        'DFTU.ProjectorGenerationMethod': 2,  # 1=Hydrogenic, 2=Bessel, 3=Filtered
        '%block DFTU.Proj': [
            'Cu 1              # element, number of l-shells',
            'n=3 2             # n=3 (3d), l=2 (d-shell)',
            '7.0 0.0           # U=7 eV, J=0',
            '0.0 0.0           # rc, omega (use defaults)',
        ],
    }
)
"""
print(code1c)
print("✅ Recipe supports DFT+U with SIESTA parameters")
print()

# ==============================================================================
# Method 2: Using tier parameter (Computational Level)
# ==============================================================================
print("\n" + "=" * 80)
print("Method 2: Using Tier Levels (Computational Rigor)")
print("-" * 80)
print()

print("Tier levels available:")
print("  • basic_dirty  - Ultra-fast (testing, 1 min)")
print("  • basic        - Fast (rough screening, 5-10 min)")
print("  • intermediate - Balanced (default, 15-30 min)")
print("  • advanced     - High accuracy (publication, 1-2 hours)")
print("  • expert       - Maximum precision (benchmarks, 4-8 hours)")
print()

print("Example 2a: Quick testing with basic tier")
print("-" * 40)
code2a = """
flow = RecipeBook.complete_material_study(
    silicon,
    tier='basic'  # Fast parameters for initial screening
)
"""
print(code2a)
print("✅ Uses coarse k-points, smaller basis, looser convergence")
print()

print("Example 2b: Publication-quality with advanced tier")
print("-" * 40)
code2b = """
flow = RecipeBook.phonon_workflow(
    silicon,
    tier='advanced'  # High-accuracy parameters
)
"""
print(code2b)
print("✅ Dense k-points, large basis, tight convergence")
print()

# ==============================================================================
# Method 3: Using preset parameter (Material-Specific Presets)
# ==============================================================================
print("\n" + "=" * 80)
print("Method 3: Using Presets (Material-Specific Parameter Sets)")
print("-" * 80)
print()

print("Available presets (26 total, organized in 10 categories):")
print("  • relax_standard, quick_relax, high_accuracy_relax")
print("  • phonon_high_accuracy, phonon_production")
print("  • surface_metal, surface_semiconductor, surface_ionic")
print("  • 2d_material, 2d_vdw_bilayer")
print("  • magnetic_afm, magnetic_fm")
print("  • ... and 16 more")
print()
print("View all: atomate2siesta-presets list")
print()

print("Example 3a: Using surface preset")
print("-" * 40)
code3a = """
flow = RecipeBook.surface_energy_workflow(
    bulk_structure,
    preset='surface_metal'  # Optimized for metal surfaces
)
"""
print(code3a)
print("✅ Applies preset: dense k-mesh, smearing, dipole correction")
print()

print("Example 3b: Using phonon preset")
print("-" * 40)
code3b = """
flow = RecipeBook.phonon_workflow(
    silicon,
    preset='phonon_high_accuracy'  # Optimized for phonons
)
"""
print(code3b)
print("✅ Applies preset: tight convergence, separate force parameters")
print()

# ==============================================================================
# Method 4: Combining Customization Methods
# ==============================================================================
print("\n" + "=" * 80)
print("Method 4: Combining Preset + User Parameters")
print("-" * 80)
print()

print("Example 4a: Preset + parameter override")
print("-" * 40)
code4a = """
flow = RecipeBook.surface_energy_workflow(
    bulk_structure,
    preset='surface_metal',       # Start with preset
    user_params={                  # Override specific params
        'a2s_kpts': [8, 8, 1],    # Custom k-mesh for surface
        'ElectronicTemperature': '500 K',  # Higher smearing
    }
)
"""
print(code4a)
print("✅ Preset provides base configuration, user_params override specifics")
print()

print("Example 4b: Tier + user parameters")
print("-" * 40)
code4b = """
flow = RecipeBook.qha_workflow(
    silicon,
    tier='advanced',              # High-accuracy tier
    user_params={                  # Additional customization
        'MD.NumCGsteps': 200,     # More relaxation steps
        'MD.MaxForceTol': 0.01,   # Tighter force tolerance
    }
)
"""
print(code4b)
print("✅ Tier sets computational level, user_params fine-tune details")
print()

# ==============================================================================
# Method 5: Complex Real-World Examples
# ==============================================================================
print("\n" + "=" * 80)
print("Method 5: Real-World Complex Customization")
print("-" * 80)
print()

print("Example 5a: Transition metal oxide with DFT+U and magnetism")
print("-" * 40)
code5a = """
from pymatgen.core import Lattice, Structure

# Create NiO structure
nio = Structure.from_spacegroup(
    "Fm-3m",
    Lattice.cubic(4.177),
    ["Ni", "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5]]
)

flow = RecipeBook.band_structure_workflow(
    nio,
    preset='magnetic_correlated',  # Magnetic + correlated system preset
    user_params={
        'DFTU.ProjectorGenerationMethod': 2,  # Bessel projectors
        '%block DFTU.Proj': [
            'Ni 1              # element, number of l-shells',
            'n=3 2             # n=3 (3d), l=2 (d-shell)',
            '5.3 0.0           # U=5.3 eV, J=0',
            '0.0 0.0           # defaults',
        ],
        'xc': 'PBE',
        'a2s_kpts': [6, 6, 6],
        'Mesh.Cutoff': '500 Ry',
    }
)
"""
print(code5a)
print("✅ Combines: magnetic preset + DFT+U + custom convergence")
print()

print("Example 5b: 2D material with van der Waals")
print("-" * 40)
code5b = """
# Graphene structure
from pymatgen.core import Lattice, Structure
import numpy as np

a = 2.46
lattice = [
    [a, 0, 0],
    [a * np.cos(np.radians(120)), a * np.sin(np.radians(120)), 0],
    [0, 0, 20.0]  # 20 Å vacuum
]
graphene = Structure(
    lattice,
    ["C", "C"],
    [[0, 0, 0.5], [1/3, 2/3, 0.5]]
)

flow = RecipeBook.phonon_workflow(
    graphene,
    preset='2d_material',          # 2D material preset
    user_params={
        'vdw': 'DRSLL',            # van der Waals functional
        'a2s_kpts': [12, 12, 1],   # Dense in-plane, 1 out-of-plane
        'PAO.BasisSize': 'TZP',    # Triple-zeta for accuracy
    }
)
"""
print(code5b)
print("✅ Combines: 2D preset + vdW functional + custom k-mesh")
print()

print("Example 5c: Surface adsorption with custom parameters")
print("-" * 40)
code5c = """
from pymatgen.core import Molecule

# CO molecule
co = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.15]])

flow = RecipeBook.adsorption_scanning_workflow(
    slab_structure,
    adsorbate=co,
    grid_density=(7, 7),           # 7x7 grid scan
    preset='surface_metal',        # Metal surface preset
    user_params={
        'a2s_kpts': [5, 5, 1],     # Surface k-mesh
        'Mesh.Cutoff': '400 Ry',   # High cutoff
        'SCF.Mixer.Weight': 0.005, # Slow mixing for metals
        'OccupationFunction': 'MP',
        'ElectronicTemperature': '300 K',
    }
)
"""
print(code5c)
print("✅ Combines: surface preset + grid scanning + metal parameters")
print()

# ==============================================================================
# Method 6: Recipe-Specific Parameters
# ==============================================================================
print("\n" + "=" * 80)
print("Method 6: Recipe-Specific Workflow Parameters")
print("-" * 80)
print()

print("Different recipes accept different workflow-specific parameters:")
print()

print("Example 6a: Phonon workflow parameters")
print("-" * 40)
code6a = """
flow = RecipeBook.phonon_workflow(
    silicon,
    supercell_matrix=(2, 2, 2),    # Explicit supercell
    # OR use min_length instead:
    # min_length=15.0,              # Auto supercell from length
    user_params={
        'PAO.BasisSize': 'DZP',
        'a2s_kpts': [4, 4, 4],
    }
)
"""
print(code6a)
print("✅ Workflow parameters (supercell) + FDF parameters (user_params)")
print()

print("Example 6b: Surface energy workflow parameters")
print("-" * 40)
code6b = """
flow = RecipeBook.surface_energy_workflow(
    bulk_structure,
    miller_indices=[(1,0,0), (1,1,0), (1,1,1)],  # Which surfaces
    slab_layers=5,                  # Slab thickness (layers)
    vacuum=15.0,                    # Vacuum spacing (Å)
    user_params={
        'a2s_kpts': [6, 6, 1],      # Surface k-mesh
        'Mesh.Cutoff': '350 Ry',
    }
)
"""
print(code6b)
print("✅ Surface parameters (Miller, layers, vacuum) + FDF parameters")
print()

print("Example 6c: EOS workflow parameters")
print("-" * 40)
code6c = """
flow = RecipeBook.eos_workflow(
    silicon,
    number_of_frames=9,             # 9 volume points
    user_params={
        'PAO.BasisSize': 'TZP',
        'a2s_kpts': [8, 8, 8],
        'Mesh.Cutoff': '400 Ry',
    }
)
"""
print(code6c)
print("✅ EOS parameter (frames) + FDF parameters")
print()

# ==============================================================================
# Summary Table
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Customization Methods")
print("=" * 80)
print()

print("┌────────────────┬──────────────────────────┬────────────────────────────┐")
print("│ Method         │ When to Use              │ Example                    │")
print("├────────────────┼──────────────────────────┼────────────────────────────┤")
print("│ user_params    │ Override specific FDF    │ {'PAO.BasisSize': 'TZP'}   │")
print("│                │ parameters               │                            │")
print("├────────────────┼──────────────────────────┼────────────────────────────┤")
print("│ tier           │ Set computational level  │ tier='advanced'            │")
print("│                │ (fast → accurate)        │                            │")
print("├────────────────┼──────────────────────────┼────────────────────────────┤")
print("│ preset         │ Material-specific params │ preset='surface_metal'     │")
print("│                │ (26 presets available)   │                            │")
print("├────────────────┼──────────────────────────┼────────────────────────────┤")
print("│ preset +       │ Base config + fine-tune  │ preset='2d_material',      │")
print("│ user_params    │                          │ user_params={...}          │")
print("├────────────────┼──────────────────────────┼────────────────────────────┤")
print("│ Workflow       │ Recipe-specific options  │ miller_indices=[(1,0,0)],  │")
print("│ parameters     │ (surfaces, supercells)   │ slab_layers=5              │")
print("└────────────────┴──────────────────────────┴────────────────────────────┘")
print()

print("Best Practices:")
print("  1. Start with preset if available for your material type")
print("  2. Use tier to set computational level (basic → expert)")
print("  3. Override specific parameters with user_params")
print("  4. Combine methods for complex calculations")
print()

print("View available presets:")
print("  atomate2siesta-presets list")
print()

print("View preset details:")
print("  atomate2siesta-presets show surface_metal")
print()

print("Search presets by category:")
print("  atomate2siesta-presets category 2d")
print("  atomate2siesta-presets category magnetic")
print()

# ==============================================================================
# Quick Reference
# ==============================================================================
print("=" * 80)
print("QUICK REFERENCE")
print("=" * 80)
print()

print("All recipes accept these parameters:")
print()
print("  RecipeBook.<recipe_name>(")
print("      structure,               # Required")
print("      user_params={...},       # FDF parameter overrides")
print("      tier='intermediate',     # Computational level")
print("      preset='preset_name',    # Material-specific preset")
print("      dry_run=False,           # Generate inputs only")
print("      name='workflow_name',    # Custom workflow name")
print("      # ... recipe-specific parameters")
print("  )")
print()

print("Priority order (later overrides earlier):")
print("  1. Tier defaults        (basic/intermediate/advanced/expert)")
print("  2. Preset parameters    (material-specific)")
print("  3. user_params          (your explicit overrides)")
print()

print("✅ Tutorial complete!")
print()
print("Next: Try customizing a recipe with your own parameters")
print("      atomate2siesta-recipe list  # See all available recipes")
print()
