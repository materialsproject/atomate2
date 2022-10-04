
"""Flows used in the calculation of defect properties."""

from __future__ import annotations
from copy import deepcopy

import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal
from pathlib import Path
from numpy.typing import NDArray

from jobflow import Flow, Job, Maker, OutputReference, job
from pymatgen.core.structure import Structure
from pymatgen.analysis.defects.core import Defect
from pymatgen.analysis.defects.supercells import get_sc_fromstruct

from atomate2.cp2k.jobs.base import BaseCp2kMaker 
from atomate2.cp2k.jobs.core import StaticMaker, HybridStaticMaker, RelaxMaker, HybridRelaxMaker, CellOptMaker, HybridCellOptMaker

from atomate2.cp2k.sets.core import (
    StaticSetGenerator, RelaxSetGenerator, CellOptSetGenerator
)

from atomate2.cp2k.sets.defect import (
    DefectStaticSetGenerator, DefectRelaxSetGenerator, DefectCellOptSetGenerator,
    DefectHybridStaticSetGenerator, DefectHybridRelaxSetGenerator, DefectHybridCellOptSetGenerator
)
from atomate2.cp2k.jobs.defect import (
    BaseDefectMaker, DefectStaticMaker, DefectRelaxMaker, DefectCellOptMaker,
    DefectHybridStaticMaker, DefectHybridRelaxMaker, DefectHybridCellOptMaker
)

from atomate2.cp2k.flows.core import HybridStaticFlowMaker, HybridRelaxFlowMaker, HybridCellOptFlowMaker

logger = logging.getLogger(__name__)

@dataclass
class FormationEnergyMaker(Maker):
    """
    Run a collection of defect jobs and (possibly) the bulk supercell
    for determination of defect formation energies.

    Parameters
    ----------
    name: This flow's name. i.e. "defect formation energy" 
    run_bulk: whether to run the bulk supercell as a static ("static")
        calculation, a full relaxation ("relax"), or to skip it (False)
    hybrid_functional: If provided, this activates hybrid version of the
        workflow. Provide functional as a parameter that the input set
        can recognize. e.g. "PBE0" or "HSE06"
    initialize_with_pbe: If hybrid functional is provided, this enables
        the use of a static PBE run before the hybrid calc to provide a
        starting guess for CP2K HF module.
    supercell_matrix: If provided, the defect supercell wil lbe created
        by this 3x3 matrix. Else other parameters will be used.
    max_atoms: Maximum number of atoms allowed in the supercell.
    min_atoms: Minimum number of atoms allowed in the supercell.
    min_length: Minimum length of the smallest supercell lattice vector.
    force_diagonal: If True, return a transformation with a diagonal transformation matrix.
    """

    name: str = "defect formation energy"
    run_bulk: Literal["static", "relax"] | bool = field(default="static") 
    hybrid_functional: str | None = field(default=None)
    initialize_with_pbe: bool = field(default=True)
    
    supercell_matrix: NDArray = field(default=None)
    min_atoms: int = field(default=80)
    max_atoms: int = field(default=240)
    min_length: int = field(default=10)
    force_diagonal: bool = field(default=False)

    def __post_init__(self):

        if self.run_bulk:
            if self.run_bulk == 'relax':
                if self.hybrid_functional:
                    self.bulk_maker = HybridCellOptFlowMaker(
                        initialize_with_pbe=self.initialize_with_pbe,
                        hybrid_functional=self.hybrid_functional,
                        hybrid_maker=HybridCellOptMaker(
                            input_set_generator=DefectHybridCellOptSetGenerator()
                        )
                        )
                else:
                    self.bulk_maker = CellOptMaker(
                        input_set_generator=DefectCellOptSetGenerator()
                        )
            elif self.run_bulk == "static":
                if self.hybrid_functional:
                    self.bulk_maker = HybridStaticFlowMaker( 
                        hybrid_functional=self.hybrid_functional,
                        hybrid_maker=HybridStaticMaker(
                            input_set_generator=DefectHybridStaticSetGenerator()
                        )
                        )
                else:
                    self.bulk_maker = StaticMaker(
                        input_set_generator=DefectStaticSetGenerator()
                        )

        if self.hybrid_functional:
            self.def_maker = HybridRelaxFlowMaker(
                hybrid_functional=self.hybrid_functional,
                initialize_with_pbe=self.initialize_with_pbe,
                initialize_maker=DefectStaticMaker(),
                hybrid_maker=HybridRelaxMaker()
            )
        else:
            self.def_maker = DefectRelaxMaker()


        self.def_maker.supercell_matrix = self.supercell_matrix
        self.def_maker.max_atoms = self.max_atoms
        self.def_maker.min_atoms = self.min_atoms
        self.def_maker.min_length = self.min_length
        self.def_maker.force_diagonal = self.force_diagonal

    def make(self, defects: Iterable[Defect], prev_cp2k_dir: str | Path | None = None):
        """Make a flow to run multiple defects in order to calculate their formation 
        energy diagram.

        Parameters
        ----------
        defects: list[Defect]
            List of defects objects to calculate the formation energy diagram for.
        prev_cp2k_dir: str | Path | None
            If provided, this acts as prev_dir for the bulk calculation only
        Returns
        -------
        flow: Flow
            The workflow to calculate the formation energy diagram.
        """
        jobs = []
        bulk_structure = ensure_defects_same_structure(defects)

        sc_mat = self.supercell_matrix if self.supercell_matrix else \
                    get_sc_fromstruct(
                        bulk_structure, self.min_atoms, 
                        self.max_atoms, self.min_length, 
                        self.force_diagonal,)

        if self.run_bulk:
            bulk_job = self.bulk_maker.make(bulk_structure * sc_mat, prev_cp2k_dir=prev_cp2k_dir)
            jobs.append(bulk_job)

        for defect in defects:
            # write some provenances data in info.json file
            info = {"defect": deepcopy(defect), "supercell_matrix": sc_mat}
            defect_job = self.def_maker.make(defect)
            defect_job.update_maker_kwargs(
                {"_set": {"write_additional_data->info:json": info}}, dict_mod=True
            )
            jobs.append(defect_job)

        return Flow(
            jobs=jobs,
            name=self.name,
            output=[j.output for j in jobs],
        )

def ensure_defects_same_structure(defects: Iterable[Defect]):
    """Ensure that the defects are valid.
    Parameters
    ----------
    defects
        The defects to check.
    Raises
    ------
    ValueError
        If any defect is invalid.
    """
    struct = None
    for defect in defects:
        if struct is None:
            struct = defect.structure
        elif struct != defect.structure:
            raise ValueError("All defects must have the same host structure.")
    return struct