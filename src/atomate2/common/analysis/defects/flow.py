"""Flows used in the calculation of defect properties."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from jobflow import Flow, Maker
from numpy.typing import NDArray
from pymatgen.analysis.defects.core import Defect

from atomate2.vasp.jobs.core import RelaxMaker
from atomate2.vasp.jobs.defect import (
    bulk_supercell_calculation,
    get_supercell_from_prv_calc,
    spawn_defect_calcs,
)

logger = logging.getLogger(__name__)


@dataclass
class FormationEnergyMaker(Maker):
    """Maker class to help calculate of the formation energy diagram.

    Maker class to calculate formation energy diagrams. The main settings for
    this maker is the `relax_maker` which contains the settings for the atomic
    relaxations that each defect supercell will undergo. The `relax_maker`
    uses a `ChargeStateRelaxSetGenerator` by default but more complex makers
    like the `HSEDoubleRelaxMaker` can be used for more accurate (but expensive)
    calculations.
    If the `validate_maker` is set to True, the maker will check for some basic
    settings in the `relax_maker` to make sure the calculations are done correctly.

    Attributes
    ----------
    name: str
        The name of the flow created by this maker.
    relax_maker: .BaseVaspMaker or None
        A maker to perform a atomic-position-only relaxation on the defect charge
        states. If None, the defaults will be used.
    validate_maker: bool
        If True, the code will check the relax_maker for specific settings.
    """

    relax_maker: RelaxMaker
    name: str = "formation energy"
    validate_maker: bool = True

    def make(
        self,
        defect: Defect,
        dielectric: float | NDArray | None = None,
        bulk_supercell_dir: str | Path | None = None,
        supercell_matrix: NDArray | None = None,
        defect_index: int | str = "",
    ):
        """Make a flow to calculate the formation energy diagram.

        Start a series of charged supercell relaxations from a single defect
        structure. Since the standard finite size correction (Freysoldt) requires
        a bulk supercell calculation (to obtain the pristine electrostatic potentia),
        this maker will either perform a bulk supercell calculation or use a existing
        one if provided.
        If a value for the dielectric constant is provided, the Freysoldt correction
        will be applied to the formation energy.

        Parameters
        ----------
        defects: Defect
            List of defects objects to calculate the formation energy diagram for.
        dielectric: float | NDArray | None
            The dielectric constant or tensor used to calculate the
            finite-size correction. If None (default), no finite-size correction will be
            applied.
        bulk_supercell_dir: str | Path | None
            If provided, the bulk supercell calculation will be skipped.
        supercell_matrix: NDArray | None
            The supercell transformation matrix. If None, the supercell matrix
            will be computed automatically.  If `bulk_supercell_dir` is provided,
            this parameter will be ignored.
        defect_index : int | str
            Additional index to give unique names to the defect calculations.
            Useful for external bookkeeping of symmetry distinct defects.

        Returns
        -------
        flow: Flow
            The workflow to calculate the formation energy diagram.
        """
        jobs = []

        if bulk_supercell_dir is None:
            get_sc_job = bulk_supercell_calculation(
                uc_structure=defect.structure,
                relax_maker=self.relax_maker,
                sc_mat=supercell_matrix,
            )
            sc_mat = get_sc_job.output["sc_mat"]
            bulk_supercell_dir = get_sc_job.output["dir_name"]
        else:
            get_sc_job = get_supercell_from_prv_calc(
                defect.structure, bulk_supercell_dir, supercell_matrix
            )
            sc_mat = get_sc_job.output["sc_mat"]

        spawn_output = spawn_defect_calcs(
            defect=defect,
            sc_mat=sc_mat,
            relax_maker=self.relax_maker,
            defect_index=defect_index,
        )
        jobs.extend([get_sc_job, spawn_output])

        return Flow(
            jobs=jobs,
            name=self.name,
        )
