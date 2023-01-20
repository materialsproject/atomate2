"""Jobs for defect calculations."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from numpy.typing import NDArray
from pymatgen.analysis.defects.core import Defect, Vacancy
from pymatgen.core import Structure

from atomate2.cp2k.jobs.base import BaseCp2kMaker, cp2k_job
from atomate2.cp2k.sets.base import Cp2kInputGenerator, recursive_update
from atomate2.cp2k.sets.defect import (
    DefectCellOptSetGenerator,
    DefectHybridCellOptSetGenerator,
    DefectHybridRelaxSetGenerator,
    DefectHybridStaticSetGenerator,
    DefectRelaxSetGenerator,
    DefectStaticSetGenerator,
)

logger = logging.getLogger(__name__)

DEFECT_TASK_DOC = {
    "average_v_hartree": True,
    "store_volumetric_data": ("v_hartree",),
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
    def make(self, defect: Defect | Structure, prev_cp2k_dir: str | Path | None = None):
        if isinstance(defect, Defect):

            structure = defect.get_supercell_structure(
                sc_mat=self.supercell_matrix,
                dummy_species=defect.site.species
                if isinstance(defect, Vacancy)
                else None,
                min_atoms=self.min_atoms,
                max_atoms=self.max_atoms,
                min_length=self.min_length,
                force_diagonal=self.force_diagonal,
            )

            if isinstance(defect, Vacancy):
                structure.add_site_property(
                    "ghost", [False] * (len(structure.sites) - 1) + [True]
                )

            if defect.user_charges:
                if len(defect.user_charges) > 1:
                    raise ValueError(
                        "Multiple user charges found. Individual defect jobs can only contain 1."
                    )
                else:
                    charge = defect.user_charges[0]
            else:
                charge = 0

            # provenance stuff
            recursive_update(
                self.write_additional_data,
                {
                    "info.json": {
                        "defect": deepcopy(defect),
                        "sc_mat": self.supercell_matrix,
                    }
                },
            )

        else:
            structure = deepcopy(defect)
            charge = structure.charge

        structure.set_charge(charge)
        return super().make.original(
            self, structure=structure, prev_cp2k_dir=prev_cp2k_dir
        )


@dataclass
class DefectStaticMaker(BaseDefectMaker):

    name: str = field(default="defect static")
    input_set_generator: Cp2kInputGenerator = field(
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
    input_set_generator: Cp2kInputGenerator = field(
        default_factory=DefectRelaxSetGenerator
    )
    transformations: tuple[str, ...] = field(
        default=("PerturbStructureTransformation",)
    )
    transformation_params: tuple[dict, ...] | None = field(
        default=({"distance": 0.01},)
    )


@dataclass
class DefectCellOptMaker(BaseDefectMaker):
    """
    Maker to create a cell for point defects.

    Adds an initial random perturbation and ensures that the output contains
    the hartree potential for finite size corrections.
    """

    name: str = field(default="defect relax")
    input_set_generator: Cp2kInputGenerator = field(
        default_factory=DefectCellOptSetGenerator
    )
    transformations: tuple[str, ...] = field(
        default=("PerturbStructureTransformation",)
    )
    transformation_params: tuple[dict, ...] | None = field(
        default=({"distance": 0.01},)
    )


@dataclass
class DefectHybridStaticMaker(BaseDefectMaker):

    name: str = field(default="defect hybrid static")
    hybrid_functional: str = "PBE0"
    input_set_generator: Cp2kInputGenerator = field(
        default_factory=DefectHybridStaticSetGenerator
    )


@dataclass
class DefectHybridRelaxMaker(BaseDefectMaker):

    name: str = field(default="defect hybrid relax")
    hybrid_functional: str = "PBE0"
    input_set_generator: Cp2kInputGenerator = field(
        default_factory=DefectHybridRelaxSetGenerator
    )
    transformations: tuple[str, ...] = field(
        default=("PerturbStructureTransformation",)
    )
    transformation_params: tuple[dict, ...] | None = field(
        default=({"distance": 0.01},)
    )


@dataclass
class DefectHybridCellOptMaker(BaseDefectMaker):

    name: str = field(default="defect hybrid cell opt")
    hybrid_functional: str = "PBE0"
    input_set_generator: Cp2kInputGenerator = field(
        default_factory=DefectHybridCellOptSetGenerator
    )
    transformations: tuple[str, ...] = field(
        default=("PerturbStructureTransformation",)
    )
    transformation_params: tuple[dict, ...] | None = field(
        default=({"distance": 0.01},)
    )
