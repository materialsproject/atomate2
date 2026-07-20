"""Templates for transition state calculations (NEB)."""

from __future__ import annotations

from typing import Any

from atomate2.siesta.cli.maker.templates.base import WorkflowTemplate


class NebTemplate(WorkflowTemplate):
    """Template for NEB (Nudged Elastic Band) calculations."""

    def __init__(self) -> None:
        super().__init__(
            name="neb",
            description="Nudged elastic band calculation for transition state search",
            runtime_estimate="2-6 hours",
            output_files=[
                "job_*/siesta.out",
                "neb_*.cif (NEB images)",
                "neb_trajectory.xyz",
                "neb_energies.txt",
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
from atomate2.siesta.flows.neb import NebDirectFlowMaker
"""

    def generate_structure_loading(
        self, structure_file: str, options: dict[str, Any]
    ) -> str:
        """Generate structure loading code for NEB (requires two structures)."""
        final_structure = options.get("final_structure")

        if not final_structure:
            raise ValueError(
                "NEB workflow requires both initial and final structures. "
                "Use: atomate2siesta-maker neb initial.cif final.cif"
            )

        return f"""
# ============================================================================
# LOAD STRUCTURES
# ============================================================================
initial = Structure.from_file("{structure_file}")
final = Structure.from_file("{final_structure}")

print(f"Initial: {{initial.composition.reduced_formula}}")
print(f"Final: {{final.composition.reduced_formula}}")
print(f"Atoms: {{len(initial)}}")
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate NEB maker initialization."""
        number_of_images = options.get("number_of_images", 5)
        relax_endpoints = options.get("relax_endpoints", False)
        interpolation = options.get("interpolation", "idpp")
        dry_run = options.get("dry_run", False)

        maker_code = """
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
"""

        # Start maker initialization
        params = [f"number_of_images={number_of_images}"]

        if relax_endpoints:
            params.append("relax_endpoints=True")

        if interpolation != "idpp":
            params.append(f'interpolation_method="{interpolation}"')

        if dry_run:
            params.append("dry_run=True")

        maker_code += "maker = NebDirectFlowMaker(\n"
        for i, param in enumerate(params):
            if i < len(params) - 1:
                maker_code += f"    {param},\n"
            else:
                maker_code += f"    {param},\n"
        maker_code += ")\n"

        return maker_code

    def generate_execution(self, options: dict[str, Any]) -> str:
        """Generate execution code for NEB."""
        remote = options.get("remote", False)

        if remote:
            return """
# ============================================================================
# SUBMIT TO REMOTE
# ============================================================================
from jobflow_remote import submit_flow

# Create workflow
flow = maker.make(initial_structure=initial, final_structure=final)

# Submit to remote worker
response = submit_flow(flow, worker="{worker}")
print(f"✓ Submitted to remote worker: {{response}}")
""".format(worker=options.get("worker", "default"))

        return """
# ============================================================================
# RUN WORKFLOW
# ============================================================================
flow = maker.make(initial_structure=initial, final_structure=final)
results = run_locally(flow, create_folders=True, ensure_success=True)

print("\\n✓ NEB calculation complete!")
"""

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section for NEB."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nGenerated files:")
print("  - neb_*.cif (NEB image structures)")
print("  - neb_trajectory.xyz (all images)")
print("  - neb_energies.txt (energy along path)")
print("  - job_*/siesta.out (SIESTA output files)")
print("\\nNext steps:")
print("  - Check barrier height in neb_energies.txt")
print("  - Visualize path: ase gui neb_trajectory.xyz")
print("  - Analyze transition state structure")
"""

    def validate_inputs(self, structure_file: str, options: dict[str, Any]) -> None:
        """Validate NEB-specific inputs."""
        super().validate_inputs(structure_file, options)

        # Check that final structure is provided
        final_structure = options.get("final_structure")
        if not final_structure:
            raise ValueError(
                "NEB workflow requires both initial and final structures.\\n"
                "Usage: atomate2siesta-maker neb initial.cif final.cif"
            )

        # Validate final structure file exists
        from pathlib import Path

        if not Path(final_structure).exists():
            raise FileNotFoundError(
                f"Final structure file not found: {final_structure}"
            )

        # Validate number of images
        n_images = options.get("number_of_images", 5)
        if n_images < 1:
            raise ValueError(f"number_of_images must be >= 1, got {n_images}")
        if n_images > 20:
            import warnings

            warnings.warn(
                f"Large number of images ({n_images}) may be computationally expensive",
                stacklevel=2,
            )
