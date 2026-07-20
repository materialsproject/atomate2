"""Templates for surface and catalysis calculations."""

from __future__ import annotations

from typing import Any

from atomate2.siesta.cli.maker.templates.base import WorkflowTemplate


class SurfaceTemplate(WorkflowTemplate):
    """Template for surface energy calculations."""

    def __init__(self):
        super().__init__(
            name="surface",
            description="Surface energy calculation for all terminations",
            runtime_estimate="30-90 minutes",
            output_files=[
                "job_*/siesta.out",
                "surface_energies.png",
                "surface_energy_summary.txt",
                "slab_structures/*.cif",
            ],
        )

    def generate_imports(self, options: dict[str, Any]) -> str:
        """Generate import statements."""
        return """
# ============================================================================
# IMPORTS
# ============================================================================
from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.surface import SurfaceEnergyFlowMaker
"""

    def generate_structure_loading(
        self, structure_file: str, options: dict[str, Any]
    ) -> str:
        """Generate structure loading code."""
        return f"""
# ============================================================================
# LOAD STRUCTURE
# ============================================================================
structure = Structure.from_file("{structure_file}")
print(f"Loaded: {{structure.composition.reduced_formula}}")
print(f"Space group: {{structure.get_space_group_info()}}")
print(f"Atoms: {{len(structure)}}")
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate surface maker initialization."""
        slab_directory = options.get("slab_directory", "./slabs")
        miller = options.get("miller_indices", (1, 0, 0))
        relax_slabs = options.get("relax_slabs", False)
        dry_run = options.get("dry_run", False)

        maker_code = """
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
"""

        # Parse miller indices
        if isinstance(miller, str):
            miller = tuple(
                map(int, miller.replace("(", "").replace(")", "").split(","))
            )

        maker_code += f"""maker = SurfaceEnergyFlowMaker(
    slab_directory="{slab_directory}",
    miller_indices={miller},
"""

        if relax_slabs:
            maker_code += "    slab_relax_maker=RelaxMaker.fixed_cell_relaxation(),\n"

        if dry_run:
            maker_code += "    dry_run=True,\n"

        maker_code += ")\n"

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nGenerated files:")
print("  - surface_energies.png (energy vs termination)")
print("  - surface_energy_summary.txt (detailed results)")
print("  - slab_structures/*.cif (all slab geometries)")
print("\\nNext steps:")
print("  - Check most stable termination in summary")
print("  - Visualize surfaces: ase gui slab_structures/*.cif")
print("  - Use most stable slab for adsorption studies")
"""

    def validate_inputs(self, structure_file: str, options: dict[str, Any]):
        """Validate surface-specific inputs."""
        super().validate_inputs(structure_file, options)

        # Check slab directory exists
        from pathlib import Path

        slab_dir = options.get("slab_directory", "./slabs")
        if not Path(slab_dir).exists():
            import warnings

            warnings.warn(
                f"Slab directory '{slab_dir}' does not exist yet. "
                "Make sure to generate slabs before running this workflow."
            )


class AdsorptionTemplate(WorkflowTemplate):
    """Template for adsorption site scanning."""

    def __init__(self):
        super().__init__(
            name="adsorption",
            description="Adsorption site scanning on surfaces",
            runtime_estimate="1-3 hours",
            output_files=[
                "job_*/siesta.out",
                "adsorption_sites.png",
                "adsorption_summary.txt",
                "adsorption_*.cif (configurations)",
            ],
        )

    def generate_imports(self, options: dict[str, Any]) -> str:
        """Generate import statements."""
        imports = """
# ============================================================================
# IMPORTS
# ============================================================================
from jobflow import run_locally
from pymatgen.core import Structure
"""

        # Add Molecule import if custom molecule specified
        if options.get("adsorbate"):
            imports += "from pymatgen.core import Molecule\n"

        imports += "from atomate2.siesta.flows.surface import AdsorptionScanFlowMaker\n"

        return imports

    def generate_structure_loading(
        self, structure_file: str, options: dict[str, Any]
    ) -> str:
        """Generate structure loading code."""
        from pathlib import Path

        # Convert to absolute path
        abs_structure_file = str(Path(structure_file).resolve())

        code = f"""
# ============================================================================
# LOAD SLAB STRUCTURE
# ============================================================================
slab = Structure.from_file("{abs_structure_file}")
print(f"Loaded slab: {{slab.composition.reduced_formula}}")
print(f"Atoms: {{len(slab)}}")
"""

        # Add adsorbate molecule if specified
        adsorbate = options.get("adsorbate")
        if adsorbate:
            # Convert to absolute path
            abs_adsorbate = str(Path(adsorbate).resolve())
            code += f"""
# Load adsorbate molecule (handles XSF, XYZ, and other formats)
try:
    # Try direct Molecule loading
    adsorbate = Molecule.from_file("{abs_adsorbate}")
except (ValueError, Exception):
    # If that fails, load as Structure first (supports XSF, XYZ, etc.)
    from pymatgen.core import Structure as TempStructure
    temp_struct = TempStructure.from_file("{abs_adsorbate}")
    # Convert Structure to Molecule
    adsorbate = Molecule(temp_struct.species, temp_struct.cart_coords)
print(f"Adsorbate: {{adsorbate.composition.reduced_formula}}")
"""

        return code

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate adsorption maker initialization."""
        grid_size = options.get("grid_size", (3, 3))
        height = options.get("height", 2.0)
        miller = options.get("miller_indices", (1, 0, 0))
        dry_run = options.get("dry_run", False)

        # Parse grid size (now a tuple from CLI nargs=2)
        if isinstance(grid_size, str):
            # Fallback for backward compatibility
            grid_size = tuple(map(int, grid_size.split("x")))
        elif not isinstance(grid_size, tuple):
            # Ensure it's a tuple
            grid_size = tuple(grid_size)

        # Parse miller indices
        if isinstance(miller, str):
            miller = tuple(
                map(int, miller.replace("(", "").replace(")", "").split(","))
            )

        maker_code = """
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
"""

        maker_code += f"""maker = AdsorptionScanFlowMaker(
    grid_size={grid_size},
    height={height},
    miller_indices={miller},
"""

        if dry_run:
            maker_code += "    dry_run=True,\n"

        maker_code += ")\n"

        return maker_code

    def generate_execution(self, options: dict[str, Any]) -> str:
        """Generate execution code (override to pass slab and adsorbate)."""
        dry_run = options.get("dry_run", False)
        remote = options.get("remote", False)

        if remote:
            return """
# ============================================================================
# SUBMIT TO REMOTE
# ============================================================================
from jobflow_remote import submit_flow

# Create workflow
flow = maker.make(slab, adsorbate)

# Submit to remote worker
response = submit_flow(flow, worker="{worker}")
print(f"✓ Submitted to remote worker: {{response}}")
""".format(worker=options.get("worker", "default"))

        if dry_run:
            return """
# ============================================================================
# RUN WORKFLOW (DRY-RUN MODE)
# ============================================================================
# Dry-run mode: Only generates input files, does not run calculations
flow = maker.make(slab, adsorbate)
results = run_locally(flow, create_folders=True, ensure_success=True)

print("\\n✓ Dry-run complete! Check output directories for generated input files.")
"""

        return """
# ============================================================================
# RUN WORKFLOW
# ============================================================================
flow = maker.make(slab, adsorbate)
results = run_locally(flow, create_folders=True, ensure_success=True)

print("\\n✓ Workflow complete!")
"""

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nGenerated files:")
print("  - adsorption_sites.png (site map with energies)")
print("  - adsorption_summary.txt (energies for all sites)")
print("  - adsorption_*.cif (all configurations)")
print("\\nNext steps:")
print("  - Identify most favorable adsorption site")
print("  - Optimize geometry at best site")
print("  - Calculate reaction barriers with NEB")
"""

    def validate_inputs(self, structure_file: str, options: dict[str, Any]):
        """Validate adsorption-specific inputs."""
        super().validate_inputs(structure_file, options)

        # Validate grid size
        grid = options.get("grid_size", (3, 3))
        if isinstance(grid, tuple):
            if grid[0] < 1 or grid[1] < 1:
                raise ValueError(f"Grid size must be positive, got {grid}")

        # Check adsorbate file if specified
        adsorbate = options.get("adsorbate")
        if adsorbate:
            from pathlib import Path

            if not Path(adsorbate).exists():
                raise FileNotFoundError(f"Adsorbate file not found: {adsorbate}")


class MultiSurfaceTemplate(WorkflowTemplate):
    """Template for multiple surface energy calculation."""

    def __init__(self):
        super().__init__(
            name="multi-surface",
            description="Calculate surface energies for multiple Miller indices",
            runtime_estimate="1-3 hours",
            output_files=[
                "multi_surface_*.png",
                "surface_energy_comparison.txt",
                "job_*/siesta.out",
            ],
        )

    def generate_imports(self, options: dict[str, Any]) -> str:
        """Generate import statements."""
        return """
# ============================================================================
# IMPORTS
# ============================================================================
from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.flows.surface import MultiSurfaceEnergyFlowMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        miller_list = options.get("miller_list", [(1, 0, 0), (1, 1, 0), (1, 1, 1)])
        dry_run = options.get("dry_run", False)

        maker_code = f"""
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
maker = MultiSurfaceEnergyFlowMaker(
    miller_indices_list={miller_list},
"""
        if dry_run:
            maker_code += "    dry_run=True,\n"

        maker_code += ")\n"

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nOutput files:")
print("  - multi_surface_energies.png (comparison plot)")
print("  - surface_energy_comparison.txt (detailed results)")
print("  - slab_structures/*.cif (all slab geometries)")
"""


class AdsorptionOptimizationTemplate(WorkflowTemplate):
    """Template for adsorption optimization at best site."""

    def __init__(self):
        super().__init__(
            name="adsorption-optimize",
            description="Optimize adsorption geometry at best site from scan",
            runtime_estimate="30-90 minutes",
            output_files=[
                "optimized_structure.cif",
                "adsorption_energy.txt",
                "job_*/siesta.out",
            ],
        )

    def generate_imports(self, options: dict[str, Any]) -> str:
        """Generate import statements."""
        imports = """
# ============================================================================
# IMPORTS
# ============================================================================
from jobflow import run_locally
from pymatgen.core import Structure
"""
        if options.get("adsorbate"):
            imports += "from pymatgen.core import Molecule\n"

        imports += "from atomate2.siesta.flows.surface import AdsorptionOptimizationFlowMaker\n"

        return imports

    def generate_structure_loading(
        self, structure_file: str, options: dict[str, Any]
    ) -> str:
        """Generate structure loading code."""
        from pathlib import Path

        # Convert to absolute path
        abs_structure_file = str(Path(structure_file).resolve())

        code = f"""
# ============================================================================
# LOAD SLAB STRUCTURE
# ============================================================================
slab = Structure.from_file("{abs_structure_file}")
print(f"Loaded slab: {{slab.composition.reduced_formula}}")
"""

        adsorbate = options.get("adsorbate")
        if adsorbate:
            # Convert to absolute path
            abs_adsorbate = str(Path(adsorbate).resolve())
            code += f"""
# Load adsorbate molecule (handles XSF, XYZ, and other formats)
try:
    # Try direct Molecule loading
    adsorbate = Molecule.from_file("{abs_adsorbate}")
except (ValueError, Exception):
    # If that fails, load as Structure first (supports XSF, XYZ, etc.)
    from pymatgen.core import Structure as TempStructure
    temp_struct = TempStructure.from_file("{abs_adsorbate}")
    # Convert Structure to Molecule
    adsorbate = Molecule(temp_struct.species, temp_struct.cart_coords)
print(f"Adsorbate: {{adsorbate.composition.reduced_formula}}")
"""

        return code

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        site = options.get("site", (0.5, 0.5))
        height = options.get("height", 2.0)
        dry_run = options.get("dry_run", False)

        maker_code = f"""
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
maker = AdsorptionOptimizationFlowMaker(
    site={site},
    height={height},
"""
        if dry_run:
            maker_code += "    dry_run=True,\n"

        maker_code += ")\n"

        return maker_code

    def generate_execution(self, options: dict[str, Any]) -> str:
        """Generate execution code (override to pass slab and adsorbate)."""
        dry_run = options.get("dry_run", False)
        remote = options.get("remote", False)

        if remote:
            return """
# ============================================================================
# SUBMIT TO REMOTE
# ============================================================================
from jobflow_remote import submit_flow

# Create workflow
flow = maker.make(slab, adsorbate)

# Submit to remote worker
response = submit_flow(flow, worker="{worker}")
print(f"✓ Submitted to remote worker: {{response}}")
""".format(worker=options.get("worker", "default"))

        if dry_run:
            return """
# ============================================================================
# RUN WORKFLOW (DRY-RUN MODE)
# ============================================================================
# Dry-run mode: Only generates input files, does not run calculations
flow = maker.make(slab, adsorbate)
results = run_locally(flow, create_folders=True, ensure_success=True)

print("\\n✓ Dry-run complete! Check output directories for generated input files.")
"""

        return """
# ============================================================================
# RUN WORKFLOW
# ============================================================================
flow = maker.make(slab, adsorbate)
results = run_locally(flow, create_folders=True, ensure_success=True)

print("\\n✓ Workflow complete!")
"""

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nOutput files:")
print("  - optimized_structure.cif (final geometry)")
print("  - adsorption_energy.txt (binding energy)")
print("  - job_*/siesta.XV (relaxed structure)")
"""
