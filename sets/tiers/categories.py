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
        "basic_relax",
        "relax_standard",
        "high_accuracy_relax",
        "bulk_metal",
        "bulk_semiconductor",
        "molecule_gas_phase",
        "adsorbate_screening",
    ],
    "Surface": [
        "surface_metal",
        "surface_semiconductor",
    ],
    "2D Materials": [
        "2d_metal",
        "2d_semiconductor",
        "2d_insulator",
        "2d_magnetic",
        "2d_vdw",
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
