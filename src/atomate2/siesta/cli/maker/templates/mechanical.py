"""Mechanical property templates (eos, elastic)."""

from __future__ import annotations

from typing import Any

from atomate2.siesta.cli.maker.templates.base import WorkflowTemplate


class EOSTemplate(WorkflowTemplate):
    """Template for equation of state calculation."""

    def __init__(self):
        super().__init__(
            name="eos",
            description="Equation of state (EOS) for bulk modulus and equilibrium volume",
            runtime_estimate="15-30 minutes",
            output_files=[
                "eos_plot.png",
                "eos_summary.txt",
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
from atomate2.siesta.flows.eos import SiestaEosFlowMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        number_of_frames = options.get("number_of_frames", 7)
        strain_range = options.get("strain_range", 0.05)
        preset = options.get("preset")

        maker_code = f"""
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
maker = SiestaEosFlowMaker(
    number_of_frames={number_of_frames},  # Volume sampling points
    linear_strain=(-{strain_range}, {strain_range}),  # ±{int(strain_range*100)}% volume variation
)
"""

        if preset:
            maker_code += f"""
# Apply tier preset
from atomate2.siesta.sets.tiers import apply_tier_preset
maker.relax_maker = apply_tier_preset(maker.relax_maker, "{preset}")
"""

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nGenerated files:")
print("  - eos_plot.png (E vs V curve)")
print("  - eos_summary.txt (fitted parameters)")
print("\\nEOS provides:")
print("  - Equilibrium volume V₀")
print("  - Bulk modulus B₀")
print("  - Pressure derivative B₀'")
print("  - Ground state energy E₀")
"""


class ElasticTemplate(WorkflowTemplate):
    """Template for elastic constants calculation."""

    def __init__(self):
        super().__init__(
            name="elastic",
            description="Elastic constants and mechanical properties",
            runtime_estimate="30-60 minutes",
            output_files=[
                "elastic_tensor.txt",
                "elastic_properties.txt",
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
from atomate2.siesta.flows.elastic import ElasticFlowMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        strain_magnitude = options.get("strain_magnitude", 0.005)
        preset = options.get("preset")

        maker_code = f"""
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
maker = ElasticFlowMaker(
    strain_magnitude={strain_magnitude},  # Applied strain (0.5%)
    # Number of deformations determined by symmetry
)
"""

        if preset:
            maker_code += f"""
# Apply tier preset
from atomate2.siesta.sets.tiers import apply_tier_preset
maker.relax_maker = apply_tier_preset(maker.relax_maker, "{preset}")
"""

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nGenerated files:")
print("  - elastic_tensor.txt (full elastic tensor)")
print("  - elastic_properties.txt (derived properties)")
print("\\nElastic properties calculated:")
print("  - Elastic tensor Cᵢⱼ")
print("  - Bulk modulus B")
print("  - Shear modulus G")
print("  - Young's modulus E")
print("  - Poisson's ratio ν")
"""


class BulkModulusTemplate(WorkflowTemplate):
    """Template for bulk modulus calculation (simplified EOS)."""

    def __init__(self):
        super().__init__(
            name="bulk-modulus",
            description="Quick bulk modulus calculation from EOS",
            runtime_estimate="10-20 minutes",
            output_files=[
                "eos_plot.png",
                "bulk_modulus.txt",
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
from atomate2.siesta.flows.eos import SiestaEosFlowMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        preset = options.get("preset")

        maker_code = """
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
# Quick bulk modulus using fewer points
maker = SiestaEosFlowMaker(
    number_of_frames=5,  # Fewer points for speed
    linear_strain=(-0.03, 0.03),  # ±3% volume variation
)
"""

        if preset:
            maker_code += f"""
# Apply tier preset
from atomate2.siesta.sets.tiers import apply_tier_preset
maker.relax_maker = apply_tier_preset(maker.relax_maker, "{preset}")
"""

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nGenerated files:")
print("  - eos_plot.png")
print("  - Bulk modulus B₀ from EOS fit")
"""
