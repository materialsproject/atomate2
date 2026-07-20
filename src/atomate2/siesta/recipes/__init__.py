"""
Workflow Recipe Book for atomate2siesta.

This module provides high-level recipe functions that create complete workflows
for common materials characterization tasks. Recipes encapsulate best practices
and reduce complex multi-step workflows to simple one-liners.

Available Recipe Categories:
    - complete: Complete material characterization workflows
    - electronic: Electronic structure properties (bands, DOS, etc.)
    - mechanical: Mechanical properties (elastic constants, etc.)
    - thermal: Thermal properties (phonons, QHA, Grüneisen)
    - catalysis: Surface energy and adsorption studies
    - defect: Point defect studies (vacancies, substitutions, interstitials)
    - convergence: Parameter convergence testing

Example Usage:
    >>> from atomate2.siesta.recipes import RecipeBook
    >>> from pymatgen.core import Structure
    >>>
    >>> structure = Structure.from_file("POSCAR")
    >>>
    >>> # Complete characterization in one line
    >>> flow = RecipeBook.complete_material_study(structure)
    >>>
    >>> # Specific property calculations
    >>> flow = RecipeBook.electronic_properties(structure)
    >>> flow = RecipeBook.thermal_properties(structure)
"""

from typing import Any

from atomate2.siesta.recipes.base import RecipeBook
from atomate2.siesta.recipes.catalysis import (
    adsorption_scanning_workflow,
    catalysis_study,
    surface_energy_workflow,
)
from atomate2.siesta.recipes.complete import (
    battery_cathode_screening,
    complete_material_study,
    high_temperature_ceramic,
    magnetic_material_study,
    quick_characterization,
    semiconductor_device_study,
    structural_phase_transition,
    thermoelectric_analysis,
)
from atomate2.siesta.recipes.convergence import (
    basis_convergence,
    complete_convergence,
    convergence_suite,
    kpoints_convergence,
)
from atomate2.siesta.recipes.defect import (
    antisite_study,
    complete_defect_study,
    interstitial_study,
    substitution_study,
    vacancy_study,
)
from atomate2.siesta.recipes.electronic import (
    band_structure_workflow,
    dos_workflow,
    electronic_properties,
)
from atomate2.siesta.recipes.mechanical import (
    elastic_constants_workflow,
    eos_workflow,
    mechanical_properties,
)
from atomate2.siesta.recipes.thermal import (
    gruneisen_workflow,
    phonon_workflow,
    qha_workflow,
    thermal_properties,
)

# Dynamically attach recipe methods to RecipeBook class.
# Use an ``Any``-typed alias (the same class object) so that mypy does not flag
# the dynamically assigned attributes; this is a no-op at runtime.
_recipe_book: Any = RecipeBook
_recipe_book.complete_material_study = staticmethod(complete_material_study)
_recipe_book.quick_characterization = staticmethod(quick_characterization)
_recipe_book.battery_cathode_screening = staticmethod(battery_cathode_screening)
_recipe_book.thermoelectric_analysis = staticmethod(thermoelectric_analysis)
_recipe_book.high_temperature_ceramic = staticmethod(high_temperature_ceramic)
_recipe_book.magnetic_material_study = staticmethod(magnetic_material_study)
_recipe_book.semiconductor_device_study = staticmethod(semiconductor_device_study)
_recipe_book.structural_phase_transition = staticmethod(structural_phase_transition)
_recipe_book.electronic_properties = staticmethod(electronic_properties)
_recipe_book.band_structure_workflow = staticmethod(band_structure_workflow)
_recipe_book.dos_workflow = staticmethod(dos_workflow)
_recipe_book.mechanical_properties = staticmethod(mechanical_properties)
_recipe_book.elastic_constants_workflow = staticmethod(elastic_constants_workflow)
_recipe_book.eos_workflow = staticmethod(eos_workflow)
_recipe_book.thermal_properties = staticmethod(thermal_properties)
_recipe_book.phonon_workflow = staticmethod(phonon_workflow)
_recipe_book.qha_workflow = staticmethod(qha_workflow)
_recipe_book.gruneisen_workflow = staticmethod(gruneisen_workflow)
_recipe_book.surface_energy_workflow = staticmethod(surface_energy_workflow)
_recipe_book.adsorption_scanning_workflow = staticmethod(adsorption_scanning_workflow)
_recipe_book.catalysis_study = staticmethod(catalysis_study)
_recipe_book.convergence_suite = staticmethod(convergence_suite)
_recipe_book.kpoints_convergence = staticmethod(kpoints_convergence)
_recipe_book.basis_convergence = staticmethod(basis_convergence)
_recipe_book.complete_convergence = staticmethod(complete_convergence)
_recipe_book.complete_defect_study = staticmethod(complete_defect_study)
_recipe_book.vacancy_study = staticmethod(vacancy_study)
_recipe_book.substitution_study = staticmethod(substitution_study)
_recipe_book.antisite_study = staticmethod(antisite_study)
_recipe_book.interstitial_study = staticmethod(interstitial_study)

__all__ = [
    "RecipeBook",
    "adsorption_scanning_workflow",
    "antisite_study",
    "band_structure_workflow",
    "basis_convergence",
    "battery_cathode_screening",
    "catalysis_study",
    "complete_convergence",
    "complete_defect_study",
    "complete_material_study",
    "convergence_suite",
    "dos_workflow",
    "elastic_constants_workflow",
    "electronic_properties",
    "eos_workflow",
    "gruneisen_workflow",
    "high_temperature_ceramic",
    "interstitial_study",
    "kpoints_convergence",
    "magnetic_material_study",
    "mechanical_properties",
    "phonon_workflow",
    "qha_workflow",
    "quick_characterization",
    "semiconductor_device_study",
    "structural_phase_transition",
    "substitution_study",
    "surface_energy_workflow",
    "thermal_properties",
    "thermoelectric_analysis",
    "vacancy_study",
]
