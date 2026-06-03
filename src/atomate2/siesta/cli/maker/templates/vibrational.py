"""Vibrational property templates (phonon, gruneisen, qha)."""

from __future__ import annotations

from typing import Any

from atomate2.siesta.cli.maker.templates.base import WorkflowTemplate


class PhononTemplate(WorkflowTemplate):
    """Template for phonon calculation."""

    def __init__(self):
        super().__init__(
            name="phonon",
            description="Phonon calculation with automatic plotting",
            runtime_estimate="30-60 minutes",
            output_files=[
                "phonon_bands.png",
                "phonon_dos.png",
                "thermal_properties.png",
                "phonon_summary.txt",
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
from atomate2.siesta.jobs.core import SiestaPhononMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        supercell = options.get("supercell")
        min_length = options.get("min_length", 10.0)
        displacement = options.get("displacement", 0.01)
        preset = options.get("preset")
        custom_params = options.get("custom_params", False)
        dry_run = options.get("dry_run", False)

        maker_code = """
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
"""

        if custom_params:
            # Show example of separate relax/force parameters
            maker_code += """# Separate parameters for relaxation vs force calculations
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.powerups import update_user_siesta_settings

# Relaxation: moderate parameters (finds geometry)
relax_params = {
    "PAO.BasisSize": "DZP",
    "PAO.EnergyShift": "0.01 Ry",
    "a2s_kpts": [6, 6, 6],
    "MeshCutoff": "300 Ry",
}

# Forces: tight parameters (critical for phonons!)
force_params = {
    "PAO.BasisSize": "DZP",
    "PAO.EnergyShift": "0.005 Ry",  # Tighter
    "a2s_kpts": [8, 8, 8],  # Denser
    "MeshCutoff": "400 Ry",  # Higher
    "DM.Tolerance": 1e-6,
}

relax_maker = RelaxMaker.variable_cell_relaxation()
static_maker = StaticMaker()

relax_maker = update_user_siesta_settings(relax_maker, relax_params)
static_maker = update_user_siesta_settings(static_maker, force_params)

maker = SiestaPhononMaker(
    relax_maker=relax_maker,
    static_maker=static_maker,
"""
        else:
            maker_code += "maker = SiestaPhononMaker(\n"

        # Supercell specification
        if supercell:
            # supercell is now a tuple (nx, ny, nz) from CLI nargs=3
            if isinstance(supercell, (list, tuple)) and len(supercell) == 3:
                a, b, c = supercell
                maker_code += "    # Supercell\n"
                maker_code += (
                    f"    supercell_matrix=[[{a}, 0, 0], [0, {b}, 0], [0, 0, {c}]],\n"
                )
            else:
                # Fallback for backward compatibility with string format
                parts = str(supercell).replace("x", ",").split(",")
                if len(parts) == 3:
                    a, b, c = [int(x.strip()) for x in parts]
                    maker_code += "    # Supercell\n"
                    maker_code += f"    supercell_matrix=[[{a}, 0, 0], [0, {b}, 0], [0, 0, {c}]],\n"
        else:
            maker_code += "    # Supercell (auto-generated)\n"
            maker_code += f"    min_length={min_length},\n"

        maker_code += f"""    prefer_90_degrees=True,
    # Displacement
    displacement={displacement},
    use_symmetry=True,
    # Analysis
    mesh=(30, 30, 30),
    create_thermal_properties=True,
    t_min=0,
    t_max=1000,
    t_step=10,
    # Automatic plotting
    generate_plots=True,
    plot_band_structure=True,
    plot_dos=True,
    plot_thermal=True,
    write_summary=True,"""

        # Add dry_run parameter if enabled
        if dry_run:
            maker_code += "\n    # Dry-run mode\n"
            maker_code += "    dry_run=True,\n"

        maker_code += ")\n"

        if preset and not custom_params:
            maker_code += f"""
# Apply tier preset
from atomate2.siesta.sets.tiers import apply_tier_preset
if maker.relax_maker:
    maker.relax_maker = apply_tier_preset(maker.relax_maker, "{preset}")
maker.static_maker = apply_tier_preset(maker.static_maker, "{preset}")
"""

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nGenerated files:")
print("  - phonon_bands.png (phonon band structure)")
print("  - phonon_dos.png (phonon density of states)")
print("  - thermal_properties.png (Cv, S, F vs T)")
print("  - phonon_summary.txt (comprehensive summary)")
"""


class GruneisenTemplate(WorkflowTemplate):
    """Template for Grüneisen parameters calculation."""

    def __init__(self):
        super().__init__(
            name="gruneisen",
            description="Grüneisen parameters and thermal expansion",
            runtime_estimate="2-4 hours",
            output_files=[
                "gruneisen_bands.png",
                "gruneisen_dos.png",
                "thermal_expansion.png",
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
from atomate2.siesta.flows.phonon import SiestaGruneisenFlowMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        min_length = options.get("min_length", 10.0)
        preset = options.get("preset")

        maker_code = f"""
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
maker = SiestaGruneisenFlowMaker(
    min_length={min_length},  # Supercell size
    displacement=0.01,
    use_symmetry=True,
    mesh=(30, 30, 30),
    # Thermal properties
    create_thermal_properties=True,
    t_min=0,
    t_max=1000,
    t_step=10,
    # Automatic plotting
    generate_plots=True,
)
"""

        if preset:
            maker_code += f"""
# Apply tier preset
from atomate2.siesta.sets.tiers import apply_tier_preset
maker.eos_maker.relax_maker = apply_tier_preset(
    maker.eos_maker.relax_maker, "{preset}"
)
"""

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nGenerated files:")
print("  - gruneisen_bands.png (Grüneisen parameters)")
print("  - gruneisen_dos.png (Grüneisen DOS)")
print("  - thermal_expansion.png (thermal expansion vs T)")
"""


class QHATemplate(WorkflowTemplate):
    """Template for quasi-harmonic approximation."""

    def __init__(self):
        super().__init__(
            name="qha",
            description="Quasi-harmonic approximation (QHA) for thermal properties",
            runtime_estimate="4-8 hours",
            output_files=[
                "qha_thermal_properties.png",
                "qha_gibbs_free_energy.png",
                "qha_bulk_modulus_vs_T.png",
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
from atomate2.siesta.flows.phonon import SiestaQhaFlowMaker
"""

    def generate_maker(self, options: dict[str, Any]) -> str:
        """Generate maker initialization."""
        min_length = options.get("min_length", 10.0)
        preset = options.get("preset")

        maker_code = f"""
# ============================================================================
# WORKFLOW SETUP
# ============================================================================
maker = SiestaQhaFlowMaker(
    min_length={min_length},  # Supercell size
    displacement=0.01,
    use_symmetry=True,
    mesh=(30, 30, 30),
    # EOS parameters
    number_of_frames=7,  # Volume sampling points
    # Thermal properties
    t_min=0,
    t_max=1000,
    t_step=10,
    # Pressure
    pressure=0.0,  # GPa
    # Automatic plotting
    generate_plots=True,
)
"""

        if preset:
            maker_code += f"""
# Apply tier preset to all makers
from atomate2.siesta.sets.tiers import apply_tier_preset
for eos_maker in maker.eos_phonon_makers:
    if eos_maker.relax_maker:
        eos_maker.relax_maker = apply_tier_preset(eos_maker.relax_maker, "{preset}")
"""

        return maker_code

    def generate_results_section(self, options: dict[str, Any]) -> str:
        """Generate results section."""
        return """
# ============================================================================
# RESULTS
# ============================================================================
print("\\nGenerated files:")
print("  - qha_thermal_properties.png")
print("  - qha_gibbs_free_energy.png")
print("  - qha_bulk_modulus_vs_T.png")
print("\\nQHA provides temperature-dependent properties:")
print("  - Gibbs free energy G(T)")
print("  - Thermal expansion α(T)")
print("  - Heat capacity Cp(T)")
print("  - Bulk modulus B(T)")
"""
