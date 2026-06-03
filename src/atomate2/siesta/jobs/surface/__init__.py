"""Surface-related job modules for SIESTA calculations."""

# Re-export everything for backward compatibility
# This allows: from atomate2.siesta.jobs.surface import ...
# Instead of: from atomate2.siesta.jobs.surface.core import ...

from atomate2.siesta.jobs.surface.adsorption import (
    add_adsorbate_to_slab,
    calculate_adsorption_energy_single_site,
    generate_adsorption_sites,
)
from atomate2.siesta.jobs.surface.core import (
    calculate_surface_energies,
    plot_surface_energies,
    write_surface_energy_summary,
)
from atomate2.siesta.jobs.surface.slab_generation import (
    generate_slabs_for_all_miller_indices,
    generate_slabs_for_miller_index,
)

__all__ = [
    # From adsorption.py
    "add_adsorbate_to_slab",
    "generate_adsorption_sites",
    "calculate_adsorption_energy_single_site",
    # From core.py (was surface.py)
    "calculate_surface_energies",
    "plot_surface_energies",
    "write_surface_energy_summary",
    # From slab_generation.py
    "generate_slabs_for_miller_index",
    "generate_slabs_for_all_miller_indices",
]
