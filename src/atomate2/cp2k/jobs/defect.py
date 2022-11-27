"""Jobs for defect calculations."""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
from copy import deepcopy
from tkinter import W
from numpy.typing import NDArray

from pymatgen.core import Structure
from pymatgen.analysis.defects.core import Defect, Vacancy
from atomate2.cp2k.sets.base import Cp2kInputGenerator, recursive_update
from atomate2.cp2k.sets.defect import (
    DefectSetGenerator, DefectStaticSetGenerator, DefectRelaxSetGenerator, DefectCellOptSetGenerator, 
    DefectHybridStaticSetGenerator, DefectHybridRelaxSetGenerator, DefectHybridCellOptSetGenerator
)
from atomate2.cp2k.jobs.base import BaseCp2kMaker, cp2k_job
from atomate2.cp2k.jobs.core import HybridStaticMaker, HybridRelaxMaker, HybridCellOptMaker

logger = logging.getLogger(__name__)

DEFECT_TASK_DOC = {
    "average_v_hartree": True,
    "store_volumetric_data": ("v_hartree",)
}

@dataclass 
class BaseDefectMaker(BaseCp2kMaker):

    task_document_kwargs: dict = field(default_factory=lambda: DEFECT_TASK_DOC)
    supercell_matrix: NDArray = field(default=None)
    min_atoms: int = field(default=80)
    max_atoms: int = field(default=240)
    min_length: int = field(default=10)
    force_diagonal: bool = field(default=False)

    @cp2k_job
    def make(self, defect: Defect | Structure, charge: int = 0, prev_cp2k_dir: str | Path | None = None):
        if isinstance(defect, Defect):
            if isinstance(defect, Vacancy):
                defect = GhostVacancy(
                    structure=defect.structure, site=defect.site,
                    multiplicity=defect.multiplicity, oxi_state=defect.oxi_state,
                    symprec=defect.symprec, angle_tolerance=defect.angle_tolerance
                    )
            structure = defect.get_supercell_structure(
                sc_mat=self.supercell_matrix, 
                dummy_species=None, 
                min_atoms=self.min_atoms,
                max_atoms=self.max_atoms,
                min_length=self.min_length,
                force_diagonal=self.force_diagonal,
            )

            # provenance stuff
            recursive_update(self.write_additional_data, {
                "info.json": {
                    "defect": deepcopy(defect), 
                    "defect_charge": charge, 
                    "sc_mat": self.supercell_matrix
                    }
                }
            )
            
        else:
            structure = deepcopy(defect)
        structure.set_charge(charge)
        return super().make.original(self, structure=structure, prev_cp2k_dir=prev_cp2k_dir)

@dataclass
class DefectStaticMaker(BaseDefectMaker):

    name: str = field(default="defect static")
    input_set_generator: DefectSetGenerator = field(
        default_factory=DefectStaticSetGenerator
        )

@dataclass
class DefectRelaxMaker(BaseDefectMaker):
    """
    Maker to create a relaxation job for point defects.

    Adds an initial random perturbation and ensures that the output contains
    the hartree potential for finite size corrections.
    """

    name: str = field(default="defect relax")
    input_set_generator: Cp2kInputGenerator = field(default_factory=DefectRelaxSetGenerator)
    transformations: tuple[str, ...] = field(default=("PerturbStructureTransformation",))
    transformation_params: tuple[dict, ...] | None = field(default=({"distance": 0.01},))

@dataclass
class DefectCellOptMaker(BaseDefectMaker):
    """
    Maker to create a cell for point defects.

    Adds an initial random perturbation and ensures that the output contains
    the hartree potential for finite size corrections.
    """

    name: str = field(default="defect relax")
    input_set_generator: Cp2kInputGenerator = field(default_factory=DefectCellOptSetGenerator)
    transformations: tuple[str, ...] = field(default=("PerturbStructureTransformation",))
    transformation_params: tuple[dict, ...] | None = field(default=({"distance": 0.01},))

@dataclass
class DefectHybridStaticMaker(DefectStaticMaker, HybridStaticMaker):
    
    name: str = field(default="defect hybrid static")
    input_set_generator: DefectSetGenerator = field(default_factory=DefectHybridStaticSetGenerator)

@dataclass
class DefectHybridRelaxMaker(DefectRelaxMaker, HybridRelaxMaker):

    name: str = field(default="defect hybrid relax")
    input_set_generator: DefectSetGenerator = field(default_factory=DefectHybridRelaxSetGenerator)

@dataclass
class DefectHybridCellOptMaker(DefectCellOptMaker, HybridCellOptMaker):

    name: str = field(default="defect hybrid cell opt")
    input_set_generator: DefectSetGenerator = field(default_factory=DefectHybridCellOptSetGenerator)

class GhostVacancy(Vacancy):
    """Custom override of vacancy to deal with basis set superposition error."""

    @property
    def defect_structure(self):
        """Returns the defect structure with the proper oxidation state"""
        struct = self.structure.copy()
        struct.add_site_property("ghost", [i == self.defect_site_index for i in range(len(struct))])
        return struct