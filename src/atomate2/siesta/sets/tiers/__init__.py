"""Material-specific tier presets for SIESTA calculations.

This module provides predefined tier configurations optimized for different
types of calculations and materials. Each preset specifies:
- Base tier level (basic, intermediate, advanced, expert)
- Additional modules to enable beyond the tier
- Modules to disable from the tier
- Recommended user_params for the calculation type

The tier system enables automatic activation of appropriate SIESTA parameter
modules based on the calculation complexity and material type.

Organization
------------
The tier system is organized into several submodules:

- **defaults**: Base parameters for each tier level (basic/intermediate/advanced/expert)
- **presets/**: Individual preset categories:
    - structural: Relaxation and bulk calculations
    - surface: Surface and interface calculations
    - molecular: Isolated molecules and adsorbate screening
    - magnetic: Spin-polarized and correlated systems
    - phonon: Vibrational property calculations
    - optical: Optical and electronic structure
    - performance: Large systems and HPC optimization
    - two_dimension: 2D materials with z-direction vacuum
- **categories**: Preset organization for documentation
- **core**: Main API functions (apply_tier_preset, get_tier_preset, etc.)

Usage Examples
--------------

Basic relaxation (default intermediate tier):
    >>> from atomate2.siesta.jobs.core import RelaxMaker
    >>> maker = RelaxMaker.fixed_cell_relaxation()  # Uses intermediate tier

Surface energy calculation:
    >>> from atomate2.siesta.sets.tiers import apply_tier_preset
    >>> maker = RelaxMaker.fixed_cell_relaxation()
    >>> maker = apply_tier_preset(maker, "surface_metal")

Phonon calculation:
    >>> from atomate2.siesta.jobs.core import SiestaPhononMaker
    >>> maker = SiestaPhononMaker(tier="phonon_high_accuracy")

Custom tier configuration:
    >>> from atomate2.siesta.sets.base import SiestaInputGenerator
    >>> input_gen = SiestaInputGenerator(
    ...     tier="advanced",
    ...     enabled_modules=["phonons", "optical"],
    ...     disabled_modules=["dftu"],
    ... )

List all available presets:
    >>> from atomate2.siesta.sets.tiers import list_tier_presets
    >>> presets = list_tier_presets()
    >>> for name, description in presets.items():
    ...     print(f"{name}: {description}")

Display presets in a formatted table:
    >>> from atomate2.siesta.sets.tiers import print_tier_presets
    >>> print_tier_presets()
"""

from __future__ import annotations

# Categories
from .categories import TIER_CATEGORIES, get_presets_by_category

# Core functions
from .core import (
    apply_tier_preset,
    get_tier_preset,
    list_tier_presets,
    print_tier_presets,
)

# Default tier parameters
from .defaults import TIER_DEFAULTS

# All presets
from .presets import (
    MAGNETIC_PRESETS,
    MOLECULAR_PRESETS,
    OPTICAL_PRESETS,
    PERFORMANCE_PRESETS,
    PHONON_PRESETS,
    STRUCTURAL_PRESETS,
    SURFACE_PRESETS,
    TIER_PRESETS,
    TWO_DIMENSION_PRESETS,
)

__all__ = [
    "MAGNETIC_PRESETS",
    "MOLECULAR_PRESETS",
    "OPTICAL_PRESETS",
    "PERFORMANCE_PRESETS",
    "PHONON_PRESETS",
    "STRUCTURAL_PRESETS",
    "SURFACE_PRESETS",
    "TIER_CATEGORIES",
    "TIER_DEFAULTS",
    "TIER_PRESETS",
    "TWO_DIMENSION_PRESETS",
    "apply_tier_preset",
    "get_presets_by_category",
    "get_tier_preset",
    "list_tier_presets",
    "print_tier_presets",
]
