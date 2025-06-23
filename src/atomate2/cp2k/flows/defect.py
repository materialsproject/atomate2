"""Flows used in the calculation of defect properties."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Mapping

from jobflow import Flow, Maker, OutputReference, job
from numpy.typing import NDArray
from pymatgen.analysis.defects.core import Defect
from pymatgen.analysis.defects.supercells import get_sc_fromstruct
from pymatgen.analysis.defects.thermo import DefectEntry
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.common import VolumetricData

from atomate2.cp2k.flows.core import (
    HybridCellOptFlowMaker,
    HybridRelaxFlowMaker,
    HybridStaticFlowMaker,
)
from atomate2.cp2k.jobs.base import BaseCp2kMaker
from atomate2.cp2k.jobs.defect import (
    DefectCellOptMaker,
    DefectHybridCellOptMaker,
    DefectHybridRelaxMaker,
    DefectHybridStaticMaker,
    DefectRelaxMaker,
    DefectStaticMaker,
)

logger = logging.getLogger(__name__)


@dataclass
class DefectHybridStaticFlowMaker(HybridStaticFlowMaker):

    pbe_maker: BaseCp2kMaker = field(default_factory=DefectStaticMaker)
    hybrid_maker: BaseCp2kMaker = field(
        default=DefectHybridStaticMaker(
            copy_cp2k_kwargs={"additional_cp2k_files": ("info.json",)}
        )
    )


@dataclass
class DefectHybridRelaxFlowMaker(HybridRelaxFlowMaker):

    pbe_maker: BaseCp2kMaker = field(default_factory=DefectStaticMaker)
    hybrid_maker: BaseCp2kMaker = field(
        default=DefectHybridRelaxMaker(
            copy_cp2k_kwargs={"additional_cp2k_files": ("info.json",)}
        )
    )


@dataclass
class DefectHybridCellOptFlowMaker(HybridCellOptFlowMaker):

    pbe_maker: BaseCp2kMaker = field(default_factory=DefectStaticMaker)
    hybrid_maker: BaseCp2kMaker = field(
        default=DefectHybridCellOptMaker(
            copy_cp2k_kwargs={"additional_cp2k_files": ("info.json",)}
        )
    )


# TODO close to being able to put this in common. Just need a switch that decides
# which core flow/job to use based on software
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
    supercell_matrix: If provided, the defect supercell will be created
        by this 3x3 matrix. Else other parameters will be used.
    max_atoms: Maximum number of atoms allowed in the supercell.
    min_atoms: Minimum number of atoms allowed in the supercell.
    min_length: Minimum length of the smallest supercell lattice
        vector.
    force_diagonal: If True, return a transformation with a
        diagonal transformation matrix.
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
        if self.run_bulk == "relax":
            if self.hybrid_functional:
                self.bulk_maker = DefectHybridCellOptMaker(
                    name="bulk hybrid relax",
                    transformations=None,
                    initialize_with_pbe=self.initialize_with_pbe,
                    hybrid_functional=self.hybrid_functional,
                )
            else:
                self.bulk_maker = DefectCellOptMaker(
                    name="bulk relax", transformations=None
                )

        elif self.run_bulk == "static":
            if self.hybrid_functional:
                self.bulk_maker = DefectHybridStaticFlowMaker(
                    name="bulk hybrid static",
                    initialize_with_pbe=self.initialize_with_pbe,
                    hybrid_functional=self.hybrid_functional,
                )
            else:
                self.bulk_maker = DefectStaticMaker(name="bulk static")

        if self.hybrid_functional:
            self.def_maker = DefectHybridRelaxFlowMaker(
                hybrid_functional=self.hybrid_functional,
                initialize_with_pbe=self.initialize_with_pbe,
            )
            self.def_maker.pbe_maker.supercell_matrix = self.supercell_matrix
            self.def_maker.hybrid_maker.supercell_matrix = self.supercell_matrix

            self.def_maker.pbe_maker.max_atoms = self.max_atoms
            self.def_maker.hybrid_maker.max_atoms = self.max_atoms

            self.def_maker.pbe_maker.min_atoms = self.min_atoms
            self.def_maker.hybrid_maker.min_atoms = self.min_atoms

            self.def_maker.pbe_maker.min_length = self.min_length
            self.def_maker.hybrid_maker.min_length = self.min_length

            self.def_maker.pbe_maker.force_diagonal = self.force_diagonal
            self.def_maker.hybrid_maker.force_diagonal = self.force_diagonal

        else:
            self.def_maker = DefectRelaxMaker()
            self.def_maker.supercell_matrix = self.supercell_matrix
            self.def_maker.max_atoms = self.max_atoms
            self.def_maker.min_atoms = self.min_atoms
            self.def_maker.min_length = self.min_length
            self.def_maker.force_diagonal = self.force_diagonal

    def make(
        self,
        defects: Iterable[Defect],
        charges: bool | Iterable[int] = False,
        dielectric: NDArray | int | float | None = None,
        prev_cp2k_dir: str | Path | None = None,
        collect_outputs: bool = True,
    ):
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
        defect_outputs: dict[str, dict[int, tuple[Defect, OutputReference]]] = {
            defect.name: {} for defect in defects
        }  # TODO DEFECT NAMES ARE NOT UNIQUE HASHES
        bulk_structure = ensure_defects_same_structure(defects)

        sc_mat = (
            self.supercell_matrix
            if self.supercell_matrix
            else get_sc_fromstruct(
                bulk_structure,
                self.min_atoms,
                self.max_atoms,
                self.min_length,
                self.force_diagonal,
            )
        )

        if self.run_bulk:
            s = bulk_structure.copy()
            s.make_supercell(sc_mat)
            bulk_job = self.bulk_maker.make(
                bulk_structure * sc_mat, prev_cp2k_dir=prev_cp2k_dir
            )
            jobs.append(bulk_job)

        for defect in defects:
            if charges is True:
                chgs = defect.get_charge_states()
            else:
                chgs = charges if charges else [0]
            for charge in chgs:
                dfct = deepcopy(defect)
                dfct.user_charges = [charge]
                defect_job = self.def_maker.make(dfct)
                jobs.append(defect_job)
                defect_outputs[defect.name][int(charge)] = (defect, defect_job.output)

        if self.run_bulk and defects and collect_outputs:
            collect_job = collect_defect_outputs(
                defect_outputs=defect_outputs,
                bulk_output=bulk_job.output if self.run_bulk else None,
                dielectric=dielectric,
            )
            jobs.append(collect_job)
        else:
            collect_job = None
        return Flow(
            jobs=jobs,
            name=self.name,
            output=jobs[-1].output if collect_job else None,
        )


# TODO this is totally code agnostic and should be in common
@job
def collect_defect_outputs(
    defect_outputs: Mapping[str, Mapping[int, OutputReference]],
    bulk_output: OutputReference,
    dielectric: NDArray | int | float | None,
) -> dict:
    """Collect all the outputs from the defect calculations.
    This job will combine the structure and entry fields to create a
    ComputerStructureEntry object.
    Parameters
    ----------
    defects_output:
        The output from the defect calculations.
    bulk_sc_dir:
        The directory containing the bulk supercell calculation.
    dielectric:
        The dielectric constant used to construct the formation energy diagram.
    """
    outputs: dict[str, dict[str, dict]] = {"results": {}}
    if not dielectric:
        logger.warn(
            "Dielectric constant not provided. Defect formation energies will be uncorrected."
        )
    for defect_name, defects_with_charges in defect_outputs.items():
        defect_entries = []
        fnv_plots = {}
        for charge, defect_and_output in defects_with_charges.items():
            defect, output_with_charge = defect_and_output
            logger.info(f"Processing {defect_name} with charge state={charge}")
            defect_entry = DefectEntry(
                defect=defect,
                charge_state=charge,
                sc_entry=ComputedStructureEntry(
                    structure=bulk_output.structure,
                    energy=output_with_charge.output.energy - bulk_output.output.energy,
                ),
            )
            defect_entries.append(defect_entry)
            plot_data = defect_entry.get_freysoldt_correction(
                defect_locpot=VolumetricData.from_dict(
                    output_with_charge.cp2k_objects["v_hartree"]
                ),
                bulk_locpot=VolumetricData.from_dict(
                    output_with_charge.cp2k_objects["v_hartree"]
                ),
                dielectric=dielectric,
            )
            fnv_plots[int(charge)] = plot_data
        outputs["results"][defect_name] = dict(
            defect=defect, defect_entries=defect_entries, fnv_plots=fnv_plots
        )
    return outputs


# TODO should be in common
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
