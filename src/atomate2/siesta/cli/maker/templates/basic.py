"""Basic workflow templates (relax, static, bands, dos)."""

from __future__ import annotations

from typing import Any

from atomate2.siesta.cli.maker.templates.base import WorkflowTemplate


class RelaxTemplate(WorkflowTemplate):
    """Template for structure relaxation."""

    def __init__(self) -> None:
        super().__init__(
            name="relax",
            description="Structure relaxation to find equilibrium geometry",
            runtime_estimate="5-15 minutes",
            output_files=[
                "job_*/siesta.out",
                "job_*/siesta.XV (relaxed structure)",
                "job_*/siesta.fdf (input file)",
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
from atomate2.siesta.jobs.core import RelaxMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        cell_type = options.get("cell_type", "fixed")
        preset = options.get("preset")
        dry_run = options.get("dry_run", False)

        maker_code = """
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
"""

        if cell_type == "fixed":
            maker_code += "maker = RelaxMaker.fixed_cell_relaxation("
        else:
            maker_code += "maker = RelaxMaker.variable_cell_relaxation("

        if dry_run:
            maker_code += "dry_run=True"

        maker_code += ")\n"

        if preset:
            maker_code += f"""
# Apply tier preset
from atomate2.siesta.sets.tiers import apply_tier_preset
maker = apply_tier_preset(maker, "{preset}")
"""

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nOutput files in job_* directories:")
print("  - siesta.out (calculation output)")
print("  - siesta.XV (relaxed structure)")
print("  - siesta.fdf (input parameters)")
"""


class StaticTemplate(WorkflowTemplate):
    """Template for static (single-point) calculation."""

    def __init__(self) -> None:
        super().__init__(
            name="static",
            description="Single-point energy calculation",
            runtime_estimate="2-5 minutes",
            output_files=[
                "job_*/siesta.out",
                "job_*/siesta.fdf",
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
from atomate2.siesta.jobs.core import StaticMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        preset = options.get("preset")
        dry_run = options.get("dry_run", False)

        maker_code = """
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
"""

        if dry_run:
            maker_code += "maker = StaticMaker(dry_run=True)\n"
        else:
            maker_code += "maker = StaticMaker()\n"

        if preset:
            maker_code += f"""
# Apply tier preset
from atomate2.siesta.sets.tiers import apply_tier_preset
maker = apply_tier_preset(maker, "{preset}")
"""

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nOutput files in job_* directory:")
print("  - siesta.out (energy and forces)")
print("  - siesta.fdf (input parameters)")
"""


class BandsTemplate(WorkflowTemplate):
    """Template for band structure calculation."""

    def __init__(self) -> None:
        super().__init__(
            name="bands",
            description="Electronic band structure calculation with automatic k-path",
            runtime_estimate="10-20 minutes",
            output_files=[
                "job_*/siesta.out",
                "job_*/siesta.bands (band structure data)",
                "band_structure.png (plot)",
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
from atomate2.siesta.jobs.core import BandStructureMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        kpath_density = options.get("kpath_density", 20)
        preset = options.get("preset")

        maker_code = f"""
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
maker = BandStructureMaker(
    kpath_density={kpath_density},  # K-points per Å⁻¹ along path
)
"""

        if preset:
            maker_code += f"""
# Apply tier preset to both relax and bands makers
from atomate2.siesta.sets.tiers import apply_tier_preset
maker.relax_maker = apply_tier_preset(maker.relax_maker, "{preset}")
maker.bands_maker = apply_tier_preset(maker.bands_maker, "{preset}")
"""

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nOutput files:")
print("  - job_relax_*/siesta.XV (relaxed structure)")
print("  - job_bands_*/siesta.bands (band structure data)")
print("  - band_structure.png (automatic plot)")
"""


class DOSTemplate(WorkflowTemplate):
    """Template for density of states calculation."""

    def __init__(self) -> None:
        super().__init__(
            name="dos",
            description="Density of states (DOS) calculation",
            runtime_estimate="10-20 minutes",
            output_files=[
                "job_*/siesta.out",
                "job_*/siesta.DOS (DOS data)",
                "dos.png (plot)",
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
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        options.get("energy_range", 10.0)
        preset = options.get("preset")

        maker_code = """
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
# First relax, then calculate DOS
from jobflow import Flow

relax_maker = RelaxMaker.fixed_cell_relaxation()
static_maker = StaticMaker()

# For DOS, need dense k-point mesh
dos_params = {
    "a2s_kpts": [8, 8, 8],  # Dense mesh for accurate DOS
    "fdf_arguments": {
        "WriteProjectedDOS": True,
    },
}

from atomate2.siesta.powerups import update_user_siesta_settings
static_maker = update_user_siesta_settings(static_maker, dos_params)
"""

        if preset:
            maker_code += f"""
# Apply tier preset
from atomate2.siesta.sets.tiers import apply_tier_preset
relax_maker = apply_tier_preset(relax_maker, "{preset}")
static_maker = apply_tier_preset(static_maker, "{preset}")
"""

        maker_code += """
# Create workflow: relax then DOS
relax_job = relax_maker.make(structure)
dos_job = static_maker.make(relax_job.output.structure)
maker = Flow([relax_job, dos_job], output=dos_job.output)
"""

        return maker_code

    def generate_execution(self, options: dict[str, Any]) -> str:
        """Generate execution code."""
        return """
# ============================================================================
# RUN WORKFLOW
# ============================================================================
# maker is already a Flow, so run it directly
results = run_locally(maker, create_folders=True, ensure_success=True)

print("\\n✓ DOS calculation complete!")
"""

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nOutput files:")
print("  - job_relax_*/siesta.XV (relaxed structure)")
print("  - job_dos_*/siesta.DOS (density of states)")
print("  - job_dos_*/siesta.PDOS* (projected DOS)")
"""


class PDOSTemplate(WorkflowTemplate):
    """Template for projected density of states (PDOS) calculation."""

    def __init__(self) -> None:
        super().__init__(
            name="pdos",
            description="Projected density of states (PDOS) calculation",
            runtime_estimate="10-20 minutes",
            output_files=[
                "job_*/siesta.out",
                "job_*/siesta.PDOS* (projected DOS files)",
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
from atomate2.siesta.jobs.core import PDOSMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        preset = options.get("preset")
        dry_run = options.get("dry_run", False)

        maker_code = """
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
"""
        if dry_run:
            maker_code += "maker = PDOSMaker(dry_run=True)\n"
        else:
            maker_code += "maker = PDOSMaker()\n"

        if preset:
            maker_code += f"""
# Apply tier preset
from atomate2.siesta.sets.tiers import apply_tier_preset
maker = apply_tier_preset(maker, "{preset}")
"""

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nOutput files:")
print("  - siesta.PDOS (total projected DOS)")
print("  - siesta.PDOS* (orbital-resolved DOS)")
"""


class OpticalTemplate(WorkflowTemplate):
    """Template for optical properties calculation."""

    def __init__(self) -> None:
        super().__init__(
            name="optical",
            description="Optical properties (absorption, dielectric function)",
            runtime_estimate="15-30 minutes",
            output_files=[
                "job_*/siesta.out",
                "job_*/siesta.EPSIMG (imaginary dielectric)",
                "job_*/siesta.EPSREAL (real dielectric)",
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
from atomate2.siesta.jobs.core import OpticalMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        preset = options.get("preset")
        dry_run = options.get("dry_run", False)

        maker_code = """
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
"""
        if dry_run:
            maker_code += "maker = OpticalMaker(dry_run=True)\n"
        else:
            maker_code += "maker = OpticalMaker()\n"

        if preset:
            maker_code += f"""
# Apply tier preset
from atomate2.siesta.sets.tiers import apply_tier_preset
maker = apply_tier_preset(maker, "{preset}")
"""

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nOutput files:")
print("  - siesta.EPSIMG (imaginary dielectric function)")
print("  - siesta.EPSREAL (real dielectric function)")
print("  - Absorption spectrum can be plotted from EPSIMG")
"""
