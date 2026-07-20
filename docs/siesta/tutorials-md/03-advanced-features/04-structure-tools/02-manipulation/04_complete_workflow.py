"""Tutorial 4: Complete Structure Manipulation Workflow.

This tutorial demonstrates end-to-end workflows combining all structure manipulation
commands for real-world research scenarios:

1. Materials Discovery Pipeline (bulk → screening → analysis)
2. Surface Adsorption Study (bulk → surface → adsorption → DFT)
3. Structure Database Curation (download → standardize → validate → archive)
4. Multi-System Comparison (alloys, polymorphs, derivatives)

Each workflow shows how to chain multiple commands together for
automated, reproducible materials science research.

Commands Covered:
  Tier 1: info, convert, validate, molecule
  Tier 2: supercell, slab, attach, perturb
  Tier 3: remove-species, substitute, sort
  Tier 4: compare, standardize, optimize-cell

Key Concepts:
- Command chaining with bash scripts
- Validation checkpoints
- Error handling
- Quality assurance
"""

from pathlib import Path

from pymatgen.core import Structure

print("=" * 80)
print("Tutorial 4: Complete Structure Manipulation Workflows")
print("=" * 80)

# ============================================================================
# Workflow 1: Materials Discovery Pipeline
# ============================================================================
print("\n" + "=" * 80)
print("Workflow 1: Materials Discovery Pipeline")
print("=" * 80)

print(
    """
Scenario: Screen perovskite materials for solar cell applications

Goal: Generate and analyze 10+ perovskite structures
Pipeline: Create → Validate → Standardize → Compare → Select

Step 1: Create base perovskite structure (CaTiO3)
"""
)

# Create CaTiO3 perovskite
perovskite = Structure.from_spacegroup(
    "Pm-3m",  # Cubic perovskite
    [[3.84, 0, 0], [0, 3.84, 0], [0, 0, 3.84]],
    ["Ca", "Ti", "O"],
    [[0.5, 0.5, 0.5], [0, 0, 0], [0.5, 0, 0]],
)
perovskite.to(filename="CaTiO3.cif", fmt="cif")
print("Created CaTiO3 structure:")
print("  Space group: Pm-3m (221)")
print(f"  Atoms: {perovskite.num_sites}")
print(f"  Formula: {perovskite.formula}")

print(
    """
Step 2: Generate derivative structures by substitution

# Bash script for automated generation:
#!/bin/bash

# Base structure
base="CaTiO3.cif"

# A-site substitutions (Ca → Sr, Ba, Pb)
for element in Sr Ba Pb; do
    atomate2siesta-structure substitute $base Ca $element
    mv substituted_$base ${element}TiO3.cif
done

# B-site substitutions (Ti → Zr, Hf, Sn)
for element in Zr Hf Sn; do
    atomate2siesta-structure substitute $base Ti $element
    mv substituted_$base Ca${element}O3.cif
done

# Mixed substitutions (Ca/Sr + Ti/Zr)
atomate2siesta-structure substitute $base Ca Sr
atomate2siesta-structure substitute substituted_$base Ti Zr
mv substituted_substituted_$base SrZrO3.cif

# Result: 10+ perovskite structures created
"""
)

print(
    """
Step 3: Validate all structures

# Check each structure for errors
for file in *TiO3.cif *ZrO3.cif *HfO3.cif *SnO3.cif; do
    echo "Validating: $file"
    atomate2siesta-structure validate $file
    if [ $? -eq 0 ]; then
        echo "  ✓ Valid"
    else
        echo "  ✗ Invalid - removing"
        rm $file
    fi
done
"""
)

print(
    """
Step 4: Standardize to primitive cells

# Standardize for DFT efficiency
mkdir -p standardized/
for file in *O3.cif; do
    echo "Standardizing: $file"
    atomate2siesta-structure standardize $file --primitive --output standardized/prim_$file
done

cd standardized/
echo "Standardized structures ready for DFT"
"""
)

print(
    """
Step 5: Compare with reference structure

# Compare all structures to CaTiO3 reference
reference="prim_CaTiO3.cif"

for file in prim_*.cif; do
    if [ "$file" != "$reference" ]; then
        echo "Comparing: $file vs $reference"
        atomate2siesta-structure compare $reference $file > comparison_$file.txt
    fi
done

# Extract key metrics
echo "Material,Atoms,Volume,Lattice_a,Lattice_b,Lattice_c" > summary.csv
for file in prim_*.cif; do
    atomate2siesta-structure info $file --format json > info.json
    # Parse JSON and append to summary.csv
done
"""
)

print(
    """
Step 6: Generate DFT workflows for top candidates

# Select structures with volume 50-100 Ų
for file in prim_*.cif; do
    volume=$(atomate2siesta-structure info $file | grep Volume | awk '{print $2}')
    if (( $(echo "$volume > 50 && $volume < 100" | bc -l) )); then
        echo "Generating DFT workflow for: $file"
        atomate2siesta-maker relax $file --preset relax_standard --execution-mode local
    fi
done

# Result: 5-10 ready-to-run DFT scripts
"""
)

# ============================================================================
# Workflow 2: Surface Adsorption Study
# ============================================================================
print("\n" + "=" * 80)
print("Workflow 2: Surface Adsorption Study")
print("=" * 80)

print(
    """
Scenario: CO adsorption on Cu(111) for catalysis research

Goal: Prepare all structures for DFT adsorption energy calculation
Pipeline: Bulk → Surface → Adsorbate → Attach → Optimize → DFT

Complete bash script:
"""
)

print(
    """
#!/bin/bash
set -e  # Exit on error

echo "==================================================================="
echo "CO Adsorption on Cu(111) - Complete Workflow"
echo "==================================================================="

# ===================================================================
# Step 1: Prepare bulk Cu structure
# ===================================================================
echo ""
echo "Step 1: Preparing bulk Cu structure..."

# Download from Materials Project or create
atomate2siesta-structure molecule --formula Cu --lattice-param 3.61
mv structure_Cu.cif cu_bulk.cif

# Validate
atomate2siesta-structure validate cu_bulk.cif
if [ $? -ne 0 ]; then
    echo "Error: Invalid Cu bulk structure"
    exit 1
fi

# Standardize to primitive
atomate2siesta-structure standardize cu_bulk.cif --primitive
mv primitive_cu_bulk.cif cu_bulk_prim.cif

echo "✓ Bulk Cu prepared: cu_bulk_prim.cif"

# ===================================================================
# Step 2: Generate Cu(111) surface
# ===================================================================
echo ""
echo "Step 2: Generating Cu(111) surface..."

atomate2siesta-structure slab cu_bulk_prim.cif \\
    --miller-indices 1,1,1 \\
    --min-slab-size 12.0 \\
    --min-vacuum-size 18.0 \\
    --symmetric

# Standardize surface to primitive
atomate2siesta-structure standardize slab_cu_bulk_prim.cif --primitive

# Optimize cell (orthogonalize for better k-points)
atomate2siesta-structure optimize-cell primitive_slab_cu_bulk_prim.cif --orthogonalize

mv orthogonal_primitive_slab_cu_bulk_prim.cif cu_111_surface.cif

echo "✓ Cu(111) surface prepared: cu_111_surface.cif"

# Validate surface
atomate2siesta-structure info cu_111_surface.cif

# ===================================================================
# Step 3: Create CO molecule
# ===================================================================
echo ""
echo "Step 3: Creating CO molecule..."

atomate2siesta-structure molecule --formula CO --bond-length 1.15

# Validate molecule
atomate2siesta-structure validate molecule_CO.cif
mv molecule_CO.cif co_molecule.cif

echo "✓ CO molecule prepared: co_molecule.cif"

# ===================================================================
# Step 4: Generate adsorption configurations
# ===================================================================
echo ""
echo "Step 4: Generating adsorption configurations..."

# Top site (on-top of surface Cu atom)
atomate2siesta-structure attach cu_111_surface.cif co_molecule.cif \\
    --position top \\
    --distance 2.0 \\
    --axis z

mv attached_cu_111_surface.cif co_cu111_top.cif
echo "  ✓ Top site: co_cu111_top.cif"

# Bridge site (between two Cu atoms)
# This requires manual coordinate adjustment or grid scanning
# For now, we'll use attach with offset
atomate2siesta-structure attach cu_111_surface.cif co_molecule.cif \\
    --position center \\
    --distance 2.0 \\
    --axis z

mv attached_cu_111_surface.cif co_cu111_bridge.cif
echo "  ✓ Bridge site: co_cu111_bridge.cif"

# Hollow site (above 3-fold hollow)
# Use attach at bottom to place in hollow
atomate2siesta-structure attach cu_111_surface.cif co_molecule.cif \\
    --position bottom \\
    --distance 2.0 \\
    --axis z

mv attached_cu_111_surface.cif co_cu111_hollow.cif
echo "  ✓ Hollow site: co_cu111_hollow.cif"

# ===================================================================
# Step 5: Validate all adsorption structures
# ===================================================================
echo ""
echo "Step 5: Validating adsorption structures..."

for site in top bridge hollow; do
    file="co_cu111_${site}.cif"
    echo "  Validating: $file"

    # Check no overlapping atoms (distance > 1.0 Å)
    atomate2siesta-structure validate $file --check-overlaps

    # Verify number of atoms
    n_atoms=$(atomate2siesta-structure info $file | grep "Number of atoms" | awk '{print $4}')
    echo "    Atoms: $n_atoms"

    # Compare to bare surface
    atomate2siesta-structure compare cu_111_surface.cif $file > comparison_${site}.txt
    echo "    Comparison saved: comparison_${site}.txt"
done

# ===================================================================
# Step 6: Generate DFT workflows
# ===================================================================
echo ""
echo "Step 6: Generating DFT workflows..."

# Bare surface (reference)
echo "  Generating workflow: cu_111_surface.cif"
atomate2siesta-maker relax cu_111_surface.cif \\
    --preset surface_standard \\
    --execution-mode local \\
    --output-filename relax_cu111_bare.py

# All adsorption sites
for site in top bridge hollow; do
    file="co_cu111_${site}.cif"
    echo "  Generating workflow: $file"

    atomate2siesta-maker relax $file \\
        --preset surface_standard \\
        --execution-mode local \\
        --output-filename relax_cu111_co_${site}.py
done

# ===================================================================
# Step 7: Summary
# ===================================================================
echo ""
echo "==================================================================="
echo "Workflow Complete!"
echo "==================================================================="
echo ""
echo "Generated structures:"
echo "  - cu_111_surface.cif           (bare surface, reference)"
echo "  - co_cu111_top.cif             (CO on top site)"
echo "  - co_cu111_bridge.cif          (CO on bridge site)"
echo "  - co_cu111_hollow.cif          (CO on hollow site)"
echo ""
echo "DFT workflow scripts:"
echo "  - relax_cu111_bare.py          (reference energy)"
echo "  - relax_cu111_co_top.py        (top site energy)"
echo "  - relax_cu111_co_bridge.py     (bridge site energy)"
echo "  - relax_cu111_co_hollow.py     (hollow site energy)"
echo ""
echo "Next steps:"
echo "  1. Run all DFT calculations:"
echo "     for script in relax_*.py; do python $script; done"
echo ""
echo "  2. Calculate adsorption energies:"
echo "     E_ads = E(CO+surface) - E(surface) - E(CO)"
echo ""
echo "  3. Identify most stable site (lowest E_ads)"
echo ""
echo "==================================================================="
"""
)

# ============================================================================
# Workflow 3: Structure Database Curation
# ============================================================================
print("\n" + "=" * 80)
print("Workflow 3: Structure Database Curation")
print("=" * 80)

print(
    """
Scenario: Curate a database of transition metal oxides from multiple sources

Goal: Download, validate, standardize, and archive structures
Pipeline: Download → Validate → Standardize → Compare → Archive

Complete Python script with error handling:
"""
)

print(
    """
#!/usr/bin/env python3
\"\"\"Structure database curation workflow.\"\"\"

import json
import subprocess
from pathlib import Path
from typing import Dict, List

# ===================================================================
# Configuration
# ===================================================================

# Target materials (formula : expected_space_group)
TARGET_MATERIALS = {
    "TiO2": "P42/mnm",  # Rutile
    "Fe2O3": "R-3c",     # Hematite
    "Al2O3": "R-3c",     # Corundum
    "ZnO": "P63mc",      # Wurtzite
    "CuO": "C2/c",       # Tenorite
    "NiO": "Fm-3m",      # Rocksalt
    "Co3O4": "Fd-3m",    # Spinel
    "MnO2": "P42/mnm",   # Pyrolusite
}

OUTPUT_DIR = Path("curated_oxides")
OUTPUT_DIR.mkdir(exist_ok=True)

# ===================================================================
# Helper functions
# ===================================================================

def run_command(cmd: List[str], capture=True) -> subprocess.CompletedProcess:
    \"\"\"Run shell command and return result.\"\"\"
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True)
    return subprocess.run(cmd)

def validate_structure(file_path: Path) -> bool:
    \"\"\"Validate structure file.\"\"\"
    result = run_command([
        "atomate2siesta-structure", "validate", str(file_path)
    ])
    return result.returncode == 0

def get_structure_info(file_path: Path) -> Dict:
    \"\"\"Get structure information as dictionary.\"\"\"
    result = run_command([
        "atomate2siesta-structure", "info", str(file_path), "--format", "json"
    ])
    if result.returncode == 0:
        return json.loads(result.stdout)
    return {}

def standardize_structure(file_path: Path, output_path: Path) -> bool:
    \"\"\"Standardize structure to primitive cell.\"\"\"
    result = run_command([
        "atomate2siesta-structure", "standardize", str(file_path),
        "--primitive", "--output", str(output_path)
    ])
    return result.returncode == 0

def compare_structures(file1: Path, file2: Path) -> Dict:
    \"\"\"Compare two structures.\"\"\"
    result = run_command([
        "atomate2siesta-structure", "compare", str(file1), str(file2)
    ])
    # Parse output to extract RMSD, lattice differences, etc.
    # (simplified for tutorial)
    return {"identical": result.returncode == 0}

# ===================================================================
# Main curation workflow
# ===================================================================

def curate_database():
    \"\"\"Main curation function.\"\"\"

    print("="*70)
    print("Structure Database Curation Workflow")
    print("="*70)

    summary = {
        "total": len(TARGET_MATERIALS),
        "downloaded": 0,
        "validated": 0,
        "standardized": 0,
        "failed": [],
    }

    for formula, expected_sg in TARGET_MATERIALS.items():
        print(f"\\n{'='*70}")
        print(f"Processing: {formula} (expected SG: {expected_sg})")
        print(f"{'='*70}")

        # Step 1: Download from Materials Project
        # (This is simulated - real version would use MP API)
        raw_file = OUTPUT_DIR / f"{formula}_raw.cif"
        print(f"  [1/5] Downloading {formula}...")

        # Simulate download (in real case, use MP API)
        # For tutorial, we'll create a simple structure
        from pymatgen.core import Structure
        from pymatgen.core import Lattice

        # Create example structures (simplified)
        if formula == "TiO2":
            structure = Structure.from_spacegroup(
                expected_sg,
                [[4.59, 0, 0], [0, 4.59, 0], [0, 0, 2.96]],
                ["Ti", "O"],
                [[0, 0, 0], [0.3, 0.3, 0]],
            )
            structure.to(filename=str(raw_file), fmt="cif")
            summary["downloaded"] += 1
            print(f"    ✓ Downloaded to {raw_file}")
        else:
            print(f"    ✗ Download failed (simulated)")
            summary["failed"].append(f"{formula}: download failed")
            continue

        # Step 2: Validate
        print(f"  [2/5] Validating structure...")
        if not validate_structure(raw_file):
            print(f"    ✗ Validation failed")
            summary["failed"].append(f"{formula}: validation failed")
            continue

        summary["validated"] += 1
        print(f"    ✓ Valid structure")

        # Step 3: Get structure info
        print(f"  [3/5] Analyzing structure...")
        info = get_structure_info(raw_file)
        if info:
            print(f"    Space group: {info.get('space_group', 'Unknown')}")
            print(f"    Atoms: {info.get('num_atoms', 'Unknown')}")
            print(f"    Volume: {info.get('volume', 'Unknown'):.2f} Ų")

            # Check if space group matches
            if info.get('space_group') != expected_sg:
                print(f"    ⚠ Warning: Space group mismatch!")
                print(f"      Expected: {expected_sg}")
                print(f"      Got: {info.get('space_group')}")

        # Step 4: Standardize to primitive cell
        print(f"  [4/5] Standardizing to primitive cell...")
        prim_file = OUTPUT_DIR / f"{formula}_primitive.cif"

        if standardize_structure(raw_file, prim_file):
            summary["standardized"] += 1
            print(f"    ✓ Standardized to {prim_file}")

            # Get primitive cell info
            prim_info = get_structure_info(prim_file)
            if prim_info:
                reduction = info.get('num_atoms', 0) / prim_info.get('num_atoms', 1)
                print(f"    Atom reduction: {reduction:.1f}x")
        else:
            print(f"    ✗ Standardization failed")
            summary["failed"].append(f"{formula}: standardization failed")

        # Step 5: Compare raw vs primitive
        print(f"  [5/5] Comparing raw vs primitive...")
        comparison = compare_structures(raw_file, prim_file)
        if comparison["identical"]:
            print(f"    ✓ Structures consistent")
        else:
            print(f"    ⚠ Structures differ (expected for cell reduction)")

    # ===================================================================
    # Print summary
    # ===================================================================

    print(f"\\n{'='*70}")
    print("Curation Summary")
    print(f"{'='*70}")
    print(f"  Total materials: {summary['total']}")
    print(f"  Downloaded: {summary['downloaded']}")
    print(f"  Validated: {summary['validated']}")
    print(f"  Standardized: {summary['standardized']}")

    if summary['failed']:
        print(f"\\n  Failed ({len(summary['failed'])}):")
        for failure in summary['failed']:
            print(f"    - {failure}")

    print(f"\\n  Success rate: {summary['standardized']/summary['total']*100:.1f}%")
    print(f"\\n  Curated structures saved in: {OUTPUT_DIR}/")
    print(f"{'='*70}\\n")

if __name__ == "__main__":
    curate_database()
"""
)

# ============================================================================
# Workflow 4: Multi-System Comparison
# ============================================================================
print("\n" + "=" * 80)
print("Workflow 4: Multi-System Comparison")
print("=" * 80)

print(
    """
Scenario: Compare different polymorphs of SiO2 (quartz, cristobalite, etc.)

Goal: Systematic comparison of all SiO2 polymorphs
Pipeline: Create/Load → Standardize → Compare → Analyze

Complete workflow:
"""
)

print(
    """
#!/bin/bash

echo "==================================================================="
echo "SiO2 Polymorphs Comparison Workflow"
echo "==================================================================="

# ===================================================================
# Step 1: Create/load all polymorphs
# ===================================================================
echo ""
echo "Step 1: Loading SiO2 polymorphs..."

# This would normally download from databases
# For tutorial, we'll list the expected polymorphs

polymorphs=(
    "quartz"         # α-quartz (P3221)
    "cristobalite"   # β-cristobalite (Fd-3m)
    "tridymite"      # Tridymite (P63/mmc)
    "coesite"        # Coesite (C2/c)
    "stishovite"     # Stishovite (P42/mnm)
)

# Download each (simulated)
for polymorph in "${polymorphs[@]}"; do
    echo "  Loading: $polymorph"
    # mp-query or COD download would go here
    # atomate2siesta-structure download --database MP --formula SiO2 --polymorph $polymorph
done

# ===================================================================
# Step 2: Standardize all structures
# ===================================================================
echo ""
echo "Step 2: Standardizing all polymorphs to primitive cells..."

mkdir -p standardized/

for polymorph in "${polymorphs[@]}"; do
    raw_file="${polymorph}_SiO2.cif"
    prim_file="standardized/prim_${polymorph}.cif"

    if [ -f "$raw_file" ]; then
        echo "  Standardizing: $polymorph"
        atomate2siesta-structure standardize $raw_file \\
            --primitive \\
            --output $prim_file

        # Get info
        atomate2siesta-structure info $prim_file
    fi
done

# ===================================================================
# Step 3: Create comparison matrix
# ===================================================================
echo ""
echo "Step 3: Creating pairwise comparison matrix..."

cd standardized/

# Header
echo "Polymorph1,Polymorph2,RMSD,VolumeDiff,LatticeDiff" > comparison_matrix.csv

# Pairwise comparisons
for poly1 in "${polymorphs[@]}"; do
    file1="prim_${poly1}.cif"

    if [ ! -f "$file1" ]; then continue; fi

    for poly2 in "${polymorphs[@]}"; do
        file2="prim_${poly2}.cif"

        if [ ! -f "$file2" ]; then continue; fi
        if [ "$poly1" == "$poly2" ]; then continue; fi

        echo "  Comparing: $poly1 vs $poly2"

        # Run comparison and parse results
        atomate2siesta-structure compare $file1 $file2 \\
            > comparison_${poly1}_${poly2}.txt

        # Extract metrics (simplified)
        rmsd=$(grep "RMSD" comparison_${poly1}_${poly2}.txt | awk '{print $2}')
        vol_diff=$(grep "Volume" comparison_${poly1}_${poly2}.txt | tail -1 | awk '{print $4}')

        echo "$poly1,$poly2,$rmsd,$vol_diff,0.0" >> comparison_matrix.csv
    done
done

# ===================================================================
# Step 4: Analyze density trends
# ===================================================================
echo ""
echo "Step 4: Analyzing density trends..."

echo "Polymorph,Atoms,Volume,Density" > density_analysis.csv

for polymorph in "${polymorphs[@]}"; do
    file="prim_${polymorph}.cif"

    if [ ! -f "$file" ]; then continue; fi

    # Extract metrics
    n_atoms=$(atomate2siesta-structure info $file | grep "Number of atoms" | awk '{print $4}')
    volume=$(atomate2siesta-structure info $file | grep "Volume" | awk '{print $2}')

    # Calculate density (simplified: atoms/volume)
    density=$(echo "scale=4; $n_atoms / $volume" | bc)

    echo "$polymorph,$n_atoms,$volume,$density" >> density_analysis.csv
    echo "  $polymorph: $density atoms/Ų"
done

# ===================================================================
# Step 5: Generate summary report
# ===================================================================
echo ""
echo "Step 5: Generating summary report..."

cat > summary_report.txt << 'EOF'
=================================================================
SiO2 Polymorphs Comparison Report
=================================================================

Structures Analyzed:
EOF

for polymorph in "${polymorphs[@]}"; do
    file="prim_${polymorph}.cif"
    if [ -f "$file" ]; then
        echo "  - $polymorph" >> summary_report.txt
    fi
done

cat >> summary_report.txt << 'EOF'

Density Analysis:
  See: density_analysis.csv

Pairwise Comparisons:
  See: comparison_matrix.csv

Key Findings:
  1. Stishovite has highest density (high-pressure phase)
  2. Cristobalite has lowest density (high-temperature phase)
  3. Quartz is most common (thermodynamically stable at STP)

Recommended for DFT:
  - All structures standardized to primitive cells
  - Ready for energy calculations
  - Use same k-point density for fair comparison

Next Steps:
  1. Run DFT relaxation on each polymorph
  2. Calculate formation energies
  3. Compare stability (E_form vs density)
  4. Identify phase transitions

=================================================================
EOF

echo "✓ Report saved: summary_report.txt"

cd ..

echo ""
echo "==================================================================="
echo "Workflow Complete!"
echo "==================================================================="
echo ""
echo "Generated files:"
echo "  - standardized/prim_*.cif           (primitive cells)"
echo "  - standardized/comparison_*.txt     (pairwise comparisons)"
echo "  - standardized/comparison_matrix.csv (comparison matrix)"
echo "  - standardized/density_analysis.csv (density trends)"
echo "  - standardized/summary_report.txt   (final report)"
echo ""
echo "==================================================================="
"""
)

# ============================================================================
# Summary and Best Practices
# ============================================================================
print("\n" + "=" * 80)
print("Summary and Best Practices")
print("=" * 80)

print(
    """
Key Workflow Patterns:

1. Sequential Processing:
   create → validate → standardize → analyze

2. Parallel Processing:
   for file in *.cif; do
       process_structure $file &
   done
   wait

3. Error Handling:
   if ! atomate2siesta-structure validate $file; then
       echo "Error: $file invalid"
       exit 1
   fi

4. Quality Checkpoints:
   - Validate after download/creation
   - Compare after transformations
   - Verify before DFT submission

5. Automation:
   - Use bash scripts for batch processing
   - Python scripts for complex logic
   - Make workflows for production

Command Chaining Best Practices:

✓ Always validate inputs
✓ Check return codes ($? or set -e)
✓ Use meaningful filenames
✓ Save intermediate results
✓ Generate summary reports
✓ Clean up temporary files

Example Error-Safe Script:

#!/bin/bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

trap 'echo "Error on line $LINENO"' ERR

input_file="$1"

# Validate input
atomate2siesta-structure validate "$input_file" || exit 1

# Process
atomate2siesta-structure standardize "$input_file" --primitive

# Verify output
if [ -f "primitive_$input_file" ]; then
    echo "✓ Success: primitive_$input_file"
else
    echo "✗ Error: Output file not created"
    exit 1
fi

Integration with DFT Workflows:

1. Structure Preparation:
   standardize → optimize-cell → validate

2. DFT Workflow Generation:
   atomate2siesta-maker relax structure.cif --preset standard

3. Result Analysis:
   compare input.cif output.cif

4. Multi-Structure Studies:
   for structure in standardized/*.cif; do
       atomate2siesta-maker relax $structure
   done

Performance Tips:

- Use primitive cells (faster DFT)
- Batch process similar structures
- Parallelize independent operations
- Cache standardized structures
- Use dry-run mode for testing

Quality Assurance:

✓ Compare before/after transformations
✓ Verify atom counts and compositions
✓ Check lattice parameters
✓ Validate symmetry preservation
✓ Test on small systems first

Documentation:

- Log all commands (script -c "workflow.sh" log.txt)
- Save comparison reports
- Track file provenance
- Version control workflows
- Document parameter choices

Command Reference:

All workflows use these core commands:
  atomate2siesta-structure validate
  atomate2siesta-structure standardize
  atomate2siesta-structure compare
  atomate2siesta-structure info
  atomate2siesta-structure optimize-cell

Full documentation:
  atomate2siesta-structure --help
  atomate2siesta-structure <command> --help
"""
)

# ============================================================================
# Cleanup
# ============================================================================
print("\nCleaning up generated files...")
cleanup_files = [
    "CaTiO3.cif",
]

for f in cleanup_files:
    if Path(f).exists():
        Path(f).unlink()

print("✓ Tutorial complete!")
print("\nAll structure manipulation tutorials:")
print("  01_structure_comparison.py")
print("  02_cell_standardization.py")
print("  03_surface_preparation.py")
print("  04_complete_workflow.py (current)")
print("\nNext: Create README.md for this tutorial series")
