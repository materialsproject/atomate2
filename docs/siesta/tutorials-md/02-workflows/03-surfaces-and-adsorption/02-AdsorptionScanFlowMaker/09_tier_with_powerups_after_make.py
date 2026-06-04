#!/usr/bin/env python
"""Adsorption scanning with tier preset + powerups applied AFTER make().

This tutorial demonstrates an alternative workflow where powerups are applied
to the flow AFTER calling .make(). This is now possible with the fixed tier
propagation system that preserves existing user_params.

Approach Comparison:
-------------------
08_tier_with_powerups.py: Apply powerups to MAKERS before flow creation
09_tier_with_powerups_after_make.py: Apply powerups to FLOW after .make() ← THIS FILE

Both approaches now work correctly!

Best Practice Pattern:
----------------------
tier preset → provides foundation (k-points, basis, cutoff)
powerups    → customize specific physics (mixing, occupation, vdW)
custodian   → handle convergence issues automatically

This tutorial shows that you can now use update_user_siesta_settings() on flows
created from tier-based FlowMakers, and the user params will be preserved!
"""

from jobflow import run_locally
from pymatgen.core import Lattice, Molecule, Structure

from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.powerups import update_user_siesta_settings

# Create MgO(100) slab
lattice = Lattice.from_parameters(a=4.2, b=4.2, c=19.6, alpha=90, beta=90, gamma=90)
species = ["Mg", "Mg", "O", "O", "Mg", "Mg", "O", "O"]
coords = [
    [0.0, 0.0, 0.32],
    [0.5, 0.5, 0.32],  # Mg layer
    [0.5, 0.0, 0.36],
    [0.0, 0.5, 0.36],  # O layer
    [0.0, 0.0, 0.43],
    [0.5, 0.5, 0.43],  # Mg layer
    [0.5, 0.0, 0.47],
    [0.0, 0.5, 0.47],  # O layer
]
slab = Structure(lattice, species, coords)

# Create CO molecule
molecule = Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])

# Step 1: Create workflow with tier='basic' for fast calculations
# ================================================================
# tier='basic' provides:
#   - PAO.BasisSize: DZP
#   - a2s_kpts: [3, 3, 3]
#   - Mesh.Cutoff: 150 Ry
#   - Fast SCF convergence settings

flow_maker = AdsorptionScanFlowMaker(
    grid_size=(3, 3),  # 3×3 grid of adsorption sites
    height=2.0,  # Adsorbate height above surface (Å)
    use_custodian=True,  # Enable automatic error handling
    custodian_max_errors=10,  # Allow up to 10 error corrections
    tier="basic",  # Start with basic tier defaults
)

# Create the workflow
workflow = flow_maker.make(slab, molecule)

# Step 2: Use powerups to customize specific parameters AFTER make()
# ===================================================================
# This now works correctly! The tier propagation preserves user_params,
# so powerups applied after .make() will be merged with tier defaults.

workflow = update_user_siesta_settings(
    workflow,
    {
        # SCF Convergence (tighter for surface calculations)
        "SCF.Mixer.Weight": 0.1,  # Slower mixing (more stable for surfaces)
        "SCF.Mixer.History": 8,  # More Pulay history (better convergence)
        "SCF.DM.Tolerance": 1e-5,  # Tighter convergence criterion
        # Occupation (Methfessel-Paxton for metals/surfaces)
        "OccupationFunction": "MP",  # Better for metallic/surface systems
        "OccupationMPOrder": 1,  # First-order MP
        "ElectronicTemperature": "25 meV",  # Small smearing
        # Output options
        "WriteCoorStep": True,  # Write coordinates at each step
        "WriteMullikenPop": 1,  # Mulliken population analysis
        "a2s_kpts": [1, 1, 1],
    },
)

# Step 3: Run the workflow
# =========================
print("=" * 70)
print("ADSORPTION SCANNING: Tier + Powerups (After Make)")
print("=" * 70)
print("\n📊 Workflow Configuration:")
print("  • Grid size: 3×3 (9 adsorption sites)")
print("  • Tier: basic (DZP, [3,3,3], 150 Ry baseline)")
print("  • Powerups: Applied AFTER .make() using update_user_siesta_settings()")
print("  • Custodian: Enabled (automatic error recovery)")
print("\n⚙️  Base tier='basic' would provide:")
print("  • PAO.BasisSize: DZP")
print("  • K-points: [3, 3, 3]")
print("  • Mesh.Cutoff: 150 Ry")
print("  • SCF.Mixer.Weight: 0.25 (default)")
print("\n🔧 Powerups override/add (applied after .make()):")
print("  • SCF.Mixer.Weight: 0.1 (slower, more stable)")
print("  • SCF.Mixer.History: 8 (more Pulay history)")
print("  • SCF.DM.Tolerance: 1e-5 (tighter convergence)")
print("  • OccupationFunction: MP (better for surfaces)")
print("  • WriteCoorStep: True (enhanced output)")
print("\n🛡️  Custodian handles:")
print("  • SCF convergence failures")
print("  • Geometry optimization issues")
print("  • Up to 10 automatic corrections")
print("\n" + "=" * 70)

results = run_locally(workflow, create_folders=True)

print("\n✓ Adsorption scan complete!")
print("\n📈 Results Analysis:")
print("  Check job directories for:")
print("  • siesta.fdf - Verify tier + powerup parameters applied")
print("  • custodian.json - Error handling log (if errors occurred)")
print("  • adsorption_scan_results.json - Adsorption energies")
print("  • adsorption_sites.png - Site visualization")
print("\n💡 Key Takeaway:")
print("  Approach 2 (this file): Powerups AFTER .make()")
print("  1. Create FlowMaker with tier='basic'")
print("  2. Call .make() to generate workflow")
print("  3. Use update_user_siesta_settings() to add powerups")
print("  4. User params are preserved and merged with tier defaults")
print("  = Production-ready workflow! 🎯")
