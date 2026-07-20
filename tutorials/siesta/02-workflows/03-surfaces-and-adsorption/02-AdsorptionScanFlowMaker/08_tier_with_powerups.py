#!/usr/bin/env python
"""Adsorption scanning with tier preset + powerup customization.

This tutorial demonstrates the recommended workflow for production calculations:
1. Start with a tier preset (e.g., 'basic') for sensible defaults
2. Use powerups to customize specific parameters for your system
3. Enable custodian for automatic error handling

This approach combines:
- Speed and efficiency from tier='basic'
- Fine-tuned accuracy for critical parameters via powerups
- Robustness from automatic error recovery

Best Practice Pattern:
----------------------
tier preset → provides foundation (k-points, basis, cutoff)
powerups    → customize specific physics (mixing, occupation, vdW)
custodian   → handle convergence issues automatically
"""

from jobflow import run_locally
from pymatgen.core import Lattice, Molecule, Structure

from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.core import StaticSetGenerator

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

# Step 1: Customize child makers with powerups BEFORE creating the flow
# =======================================================================
# IMPORTANT: Powerups must be applied to makers BEFORE calling .make()!
# Applying powerups to the workflow after .make() won't work because
# the jobs have already been created.

# Create customized makers with powerup parameters
custom_params = {
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
}

slab_maker = StaticMaker(
    input_set_generator=StaticSetGenerator(user_params=custom_params)
)
adsorbate_maker = StaticMaker(
    input_set_generator=StaticSetGenerator(user_params=custom_params)
)

# Step 2: Create workflow with tier='basic' + custom makers
# ==========================================================
# tier='basic' provides:
#   - PAO.BasisSize: DZP
#   - a2s_kpts: [3, 3, 3]
#   - Mesh.Cutoff: 150 Ry
#   - Fast SCF convergence settings (overridden by custom_params above)

flow_maker = AdsorptionScanFlowMaker(
    dry_run=True,
    slab_static_maker=slab_maker,  # Custom maker with powerups
    adsorbate_static_maker=adsorbate_maker,  # Custom maker with powerups
    grid_size=(2, 2),  # 2×2 grid of adsorption sites
    height=2.0,  # Adsorbate height above surface (Å)
    use_custodian=True,  # Enable automatic error handling
    custodian_max_errors=10,  # Allow up to 10 error corrections
    tier="basic",  # Tier defaults applied to makers
)

# Create the workflow
workflow = flow_maker.make(slab, molecule)

# Step 3: Run the workflow
# =========================
print("=" * 70)
print("ADSORPTION SCANNING: Tier + Powerups + Custodian")
print("=" * 70)
print("\n📊 Workflow Configuration:")
print("  • Grid size: 3×3 (9 adsorption sites)")
print("  • Tier: basic (DZP, [3,3,3], 150 Ry baseline)")
print("  • Powerups: Custom SCF, occupation, output settings")
print("  • Custodian: Enabled (automatic error recovery)")
print("\n⚙️  Base tier='basic' would provide:")
print("  • PAO.BasisSize: DZP")
print("  • K-points: [3, 3, 3]")
print("  • Mesh.Cutoff: 150 Ry")
print("  • SCF.Mixer.Weight: 0.25 (default)")
print("\n🔧 Powerups override/add:")
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

results = run_locally(workflow, create_folders=True, root_dir="08_tier_with_powerups")

print("\n✓ Adsorption scan complete!")
print("\n📈 Results Analysis:")
print("  Check job directories for:")
print("  • siesta.fdf - Verify tier + powerup parameters applied")
print("  • custodian.json - Error handling log (if errors occurred)")
print("  • adsorption_scan_results.json - Adsorption energies")
print("  • adsorption_sites.png - Site visualization")
print("\n💡 Key Takeaway:")
print("  1. Create makers with user_params (powerups)")
print("  2. Pass makers to FlowMaker with tier='basic'")
print("  3. tier merges with user_params (user_params take precedence)")
print("  4. Enable custodian for automatic robustness")
print("  = Production-ready workflow! 🎯")
