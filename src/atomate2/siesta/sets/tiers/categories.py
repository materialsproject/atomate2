"""Tier preset categorization.

Organization of tier presets into logical categories for documentation
and discovery purposes.
"""

from __future__ import annotations

# ==============================================================================
# TIER CATEGORIZATION (for documentation and organization)
# ==============================================================================

TIER_CATEGORIES = {
    "Structural": [
        "relax_dirty",
        "relax_standard",
        "relax_high_accuracy",
        "relax_bulk_metal",
        "relax_bulk_semiconductor",
        "molecule_gas_phase",
        "adsorbate_screening",
    ],
    "Surface": [
        "surface_basic",
        "surface_dirty",
        "surface_metal",
        "surface_semiconductor",
    ],
    "2D Materials": [
        "2d_metal",
        "2d_metal_rough_auto",
        "2d_semiconductor",
        "2d_insulator",
        "2d_magnetic",
        "2d_vdw",
        "2d_vdw_dirty",
        "2d_optical",
        "2d_screening",
    ],
    "Magnetic": [
        "magnetic_2d",
        "magnetic_correlated",
    ],
    "Phonon": [
        "phonon_dirty",
        "phonon_standard",
        "phonon_high_accuracy",
    ],
    "Optical": [
        "optical_response",
        "band_structure",
    ],
    "Defects": [
        "defect_dirty",
        "defect_standard",
        "defect_accurate",
        "defect_metal",
        "defect_oxide",
    ],
    "Electrocatalysis": [
        "electrocatalysis_dirty",
        "electrocatalysis_basic",
        "electrocatalysis_intermediate",
        "electrocatalysis_gas_phase",
    ],
    "Performance": [
        "large_system",
        "parallel_hpc",
    ],
    "Testing": [
        "convergence_test",
    ],
}


def get_presets_by_category(category: str) -> list[str]:
    """
    Get all tier presets in a specific category.

    Parameters
    ----------
    category : str
        Category name (e.g., "Structural", "Surface", "Magnetic")

    Returns
    -------
    list[str]
        List of preset names in the category

    Raises
    ------
    ValueError
        If category is not found

    Examples
    --------
    >>> presets = get_presets_by_category("Phonon")
    >>> print(presets)
    ['phonon_standard', 'phonon_high_accuracy']
    """
    if category not in TIER_CATEGORIES:
        available = list(TIER_CATEGORIES.keys())
        raise ValueError(f"Unknown category '{category}'. Available: {available}")

    return TIER_CATEGORIES[category]
