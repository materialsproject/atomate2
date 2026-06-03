"""
Defect workflow module for atomate2siesta.

State-of-the-art point defect calculations with multiple correction schemes,
automated supercell optimization, and comprehensive analysis tools.
"""

from __future__ import annotations

from atomate2.siesta.flows.defects.analysis import (
    FormationEnergyDiagramFlowMaker,
    write_combined_defect_summary,
)
from atomate2.siesta.flows.defects.comparison import CorrectionComparisonFlowMaker
from atomate2.siesta.flows.defects.core import DefectFlowMaker
from atomate2.siesta.flows.defects.generation import (
    DefectSite,
    LayerInfo,
    SiestaInterstitialGenerator,
    SiestaSubstitutionGenerator,
    SiestaVacancyGenerator,
    SurfaceInterstitialGenerator,
    SurfaceSubstitutionGenerator,
    SurfaceVacancyGenerator,
    create_vacancy_with_ghost,
    create_vacancy_with_ghost_from_site,
)
from atomate2.siesta.flows.defects.makers import DefectRelaxMaker, DefectStaticMaker
from atomate2.siesta.flows.defects.schemas import DefectDocument
from atomate2.siesta.flows.defects.utils import (
    calculate_planar_average,
    find_vt_files,
    plot_potential_alignment,
    prepare_density_data,
    prepare_freysoldt_potential_data,
    read_siesta_density,
    read_siesta_grid_file,
)

__all__ = [
    # Main workflows
    "DefectFlowMaker",
    "FormationEnergyDiagramFlowMaker",
    "CorrectionComparisonFlowMaker",
    # Makers
    "DefectRelaxMaker",
    "DefectStaticMaker",
    # Schemas
    "DefectDocument",
    # Analysis jobs
    "write_combined_defect_summary",
    # Generation utilities
    "create_vacancy_with_ghost",
    "create_vacancy_with_ghost_from_site",
    "SiestaVacancyGenerator",
    "SurfaceVacancyGenerator",
    "SurfaceInterstitialGenerator",
    "SurfaceSubstitutionGenerator",
    "SiestaSubstitutionGenerator",
    "SiestaInterstitialGenerator",
    "DefectSite",
    "LayerInfo",
    # File I/O utilities
    "read_siesta_grid_file",
    "read_siesta_density",
    "prepare_density_data",
    "prepare_freysoldt_potential_data",
    "find_vt_files",
    # Analysis utilities
    "calculate_planar_average",
    "plot_potential_alignment",
    # Future workflows (to be implemented)
    # "DefectMigrationFlowMaker",
    # "ComplexDefectFlowMaker",
]
