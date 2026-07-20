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

# Dynamically attach recipe methods to RecipeBook class
RecipeBook.complete_material_study = staticmethod(complete_material_study)
RecipeBook.quick_characterization = staticmethod(quick_characterization)
RecipeBook.battery_cathode_screening = staticmethod(battery_cathode_screening)
RecipeBook.thermoelectric_analysis = staticmethod(thermoelectric_analysis)
RecipeBook.high_temperature_ceramic = staticmethod(high_temperature_ceramic)
RecipeBook.magnetic_material_study = staticmethod(magnetic_material_study)
RecipeBook.semiconductor_device_study = staticmethod(semiconductor_device_study)
RecipeBook.structural_phase_transition = staticmethod(structural_phase_transition)
RecipeBook.electronic_properties = staticmethod(electronic_properties)
RecipeBook.band_structure_workflow = staticmethod(band_structure_workflow)
RecipeBook.dos_workflow = staticmethod(dos_workflow)
RecipeBook.mechanical_properties = staticmethod(mechanical_properties)
RecipeBook.elastic_constants_workflow = staticmethod(elastic_constants_workflow)
RecipeBook.eos_workflow = staticmethod(eos_workflow)
RecipeBook.thermal_properties = staticmethod(thermal_properties)
RecipeBook.phonon_workflow = staticmethod(phonon_workflow)
RecipeBook.qha_workflow = staticmethod(qha_workflow)
RecipeBook.gruneisen_workflow = staticmethod(gruneisen_workflow)
RecipeBook.surface_energy_workflow = staticmethod(surface_energy_workflow)
RecipeBook.adsorption_scanning_workflow = staticmethod(adsorption_scanning_workflow)
RecipeBook.catalysis_study = staticmethod(catalysis_study)
RecipeBook.convergence_suite = staticmethod(convergence_suite)
RecipeBook.kpoints_convergence = staticmethod(kpoints_convergence)
RecipeBook.basis_convergence = staticmethod(basis_convergence)
RecipeBook.complete_convergence = staticmethod(complete_convergence)
RecipeBook.complete_defect_study = staticmethod(complete_defect_study)
RecipeBook.vacancy_study = staticmethod(vacancy_study)
RecipeBook.substitution_study = staticmethod(substitution_study)
RecipeBook.antisite_study = staticmethod(antisite_study)
RecipeBook.interstitial_study = staticmethod(interstitial_study)

__all__ = [
    "RecipeBook",
    # Complete workflows
    "complete_material_study",
    "quick_characterization",
    "battery_cathode_screening",
    "thermoelectric_analysis",
    "high_temperature_ceramic",
    "magnetic_material_study",
    "semiconductor_device_study",
    "structural_phase_transition",
    # Electronic properties
    "electronic_properties",
    "band_structure_workflow",
    "dos_workflow",
    # Mechanical properties
    "mechanical_properties",
    "elastic_constants_workflow",
    "eos_workflow",
    # Thermal properties
    "thermal_properties",
    "phonon_workflow",
    "qha_workflow",
    "gruneisen_workflow",
    # Catalysis
    "surface_energy_workflow",
    "adsorption_scanning_workflow",
    "catalysis_study",
    # Convergence
    "convergence_suite",
    "kpoints_convergence",
    "basis_convergence",
    "complete_convergence",
    # Defect studies
    "complete_defect_study",
    "vacancy_study",
    "substitution_study",
    "antisite_study",
    "interstitial_study",
]
