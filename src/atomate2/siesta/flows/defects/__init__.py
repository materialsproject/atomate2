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
    "CorrectionComparisonFlowMaker",
    "DefectDocument",
    "DefectFlowMaker",
    "DefectRelaxMaker",
    "DefectSite",
    "DefectStaticMaker",
    "FormationEnergyDiagramFlowMaker",
    "LayerInfo",
    "SiestaInterstitialGenerator",
    "SiestaSubstitutionGenerator",
    "SiestaVacancyGenerator",
    "SurfaceInterstitialGenerator",
    "SurfaceSubstitutionGenerator",
    "SurfaceVacancyGenerator",
    "calculate_planar_average",
    "create_vacancy_with_ghost",
    "create_vacancy_with_ghost_from_site",
    "find_vt_files",
    "plot_potential_alignment",
    "prepare_density_data",
    "prepare_freysoldt_potential_data",
    "read_siesta_density",
    "read_siesta_grid_file",
    "write_combined_defect_summary",
    # Future workflows (to be implemented)
    # "DefectMigrationFlowMaker",
    # "ComplexDefectFlowMaker",
]
