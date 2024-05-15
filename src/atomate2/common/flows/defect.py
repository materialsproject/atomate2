"""Flows used in the calculation of defect properties."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jobflow import Flow, Job, Maker, OutputReference

from atomate2.common.jobs.defect import (
    bulk_supercell_calculation,
    get_ccd_documents,
    get_charged_structures,
    get_defect_entry,
    get_supercell_from_prv_calc,
    spawn_defect_q_jobs,
    spawn_energy_curve_calcs,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    from emmet.core.tasks import TaskDoc
    from pymatgen.analysis.defects.core import Defect
    from pymatgen.core.structure import Structure
    from pymatgen.entries.computed_entries import ComputedStructureEntry

logger = logging.getLogger(__name__)

DEFAULT_DISTORTIONS = (-1, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 1)


@dataclass
class ConfigurationCoordinateMaker(Maker):
    """Maker to generate a configuration coordinate diagram.

    Parameters
    ----------
    name: str
        The name of the flow created by this maker.
    relax_maker: Maker
        A maker to perform a atomic-position-only relaxation on the defect charge
        states.
    static_maker: Maker
        A maker to perform the single-shot static calculation of the distorted
        structures.
    distortions: tuple[float, ...]
        The distortions, as a fraction of ΔQ, to use in the calculation of the
        configuration coordinate diagram.
    """

    relax_maker: Maker
    static_maker: Maker
    name: str = "config coordinate"
    distortions: tuple[float, ...] = DEFAULT_DISTORTIONS

    def make(
        self,
        structure: Structure,
        charge_state1: int,
        charge_state2: int,
    ) -> Flow:
        """Make a job for the calculation of the configuration coordinate diagram.

        Parameters
        ----------
        structure
            A structure.
        charge_state1
            The reference charge state of the defect.
        charge_state2
            The excited charge state of the defect

        Returns
        -------
        Flow
            The full workflow for the calculation of the configuration coordinate
            diagram.
        """
        # use a more descriptive name when possible
        if not isinstance(structure, OutputReference):
            name = f"{self.name}: {structure.formula}"
            if not (
                isinstance(charge_state1, OutputReference)
                or isinstance(charge_state2, OutputReference)
            ):
                name = (
                    f"{self.name}: {structure.formula}({charge_state1}-{charge_state2})"
                )

        # need to wrap this up in a job so that references to undone calculations can
        # be passed in
        charged_structures = get_charged_structures(
            structure, [charge_state1, charge_state2]
        )

        relax1: Job = self.relax_maker.make(structure=charged_structures.output[0])
        relax2: Job = self.relax_maker.make(structure=charged_structures.output[1])
        relax1.append_name(" q1")
        relax2.append_name(" q2")

        dir1 = relax1.output.dir_name
        dir2 = relax2.output.dir_name
        struct1 = relax1.output.structure
        struct2 = relax2.output.structure

        deformations1 = spawn_energy_curve_calcs(
            struct1,
            struct2,
            distortions=self.distortions,
            static_maker=self.static_maker,
            prev_dir=dir1,
            add_name="q1",
            add_info={"relaxed_uuid": relax1.uuid, "distorted_uuid": relax2.uuid},
        )

        deformations2 = spawn_energy_curve_calcs(
            struct2,
            struct1,
            distortions=self.distortions,
            static_maker=self.static_maker,
            prev_dir=dir2,
            add_name="q2",
            add_info={"relaxed_uuid": relax2.uuid, "distorted_uuid": relax1.uuid},
        )

        deformations1.append_name(" q1")
        deformations2.append_name(" q2")

        # distortion index with smallest absolute value
        min_abs_index = min(
            range(len(self.distortions)), key=lambda i: abs(self.distortions[i])
        )

        ccd_job = get_ccd_documents(
            deformations1.output, deformations2.output, undistorted_index=min_abs_index
        )

        return Flow(
            jobs=[
                charged_structures,
                relax1,
                relax2,
                deformations1,
                deformations2,
                ccd_job,
            ],
            output=ccd_job.output,
            name=name,
        )


@dataclass
class FormationEnergyMaker(Maker, ABC):
    """Maker class to help calculate of the formation energy diagram.

    Maker class to calculate formation energy diagrams. The main settings for
    this maker is the `defect_relax_maker` which contains the settings for the atomic
    relaxations that each defect supercell will undergo.

    Attributes
    ----------
    defect_relax_maker: Maker
        A maker to perform a atomic-position-only relaxation on the defect charge
        states. Since these calculations are expensive and the settings might get
        messy, it is recommended for each implementation of this maker to check
        some of the most important settings in the `relax_maker`. Please see
        `FormationEnergyMaker.validate_maker` for more details.

    bulk_relax_maker: Maker
        If None, the same `defect_relax_maker` will be used for the bulk supercell.
        A maker to used to perform the bulk supercell calculation. For marginally
        converged calculations, it might be desirable to perform an additional
        lattice relaxation on the bulk supercell to make sure the energies are more
        reliable. However, if you do relax the bulk supercell, you can inadvertently
        change the grid size used in the calculation and thus the representation
        of the electrostatic potential which will affect calculation of the Freysoldt
        finite-size correction. Therefore, if you do want to perform a bulk supercell
        lattice relaxation, you should manually set the grid size.

        .. code-block:: python
            relax_set = MPRelaxSet(defect.get_supercell_structure())
            ng, ngf = relax_set.calculate_ng()
            params = ["NGX", "NGY", "NGZ", "NGXF", "NGYF", "NGZF"]
            ng_settings = dict(zip(params, ng + ngf))
            relax_maker = update_user_incar_settings(relax_maker, ng_settings)

    name: str
        The name of the flow created by this maker.

    relax_radius:
        The radius to include around the defect site for the relaxation.
        If "auto", the radius will be set to the maximum that will fit inside
        a periodic cell. If None, all atoms will be relaxed.

    perturb:
        The amount to perturb the sites in the supercell. Only perturb the
        sites with selective dynamics set to True. So this setting only works
        with `relax_radius`.

    validate_charge: bool
        Whether to validate the charge of the defect. If True (default), the charge
        of the output structure will have to match the charge of the input defect.
        This helps catch situations where the charge of the output defect is either
        improperly set or improperly parsed before the data is stored in the
        database.

    collect_defect_entry_data: bool
        Whether to collect the defect entry data at the end of the flow.
        If True, the output of all the charge states for each symmetry distinct
        defect will be collected into a list of dictionaries that can be used
        to create a DefectEntry. The data here can be trivially combined with
        phase diagram data from the materials project API to create the formation
        energy diagrams.

        .. note::
        Once we remove the requirement for explicit bulk supercell calculations,
        this setting will be removed. It is only needed because the bulk supercell
        locpot is currently needed for the finite-size correction calculation.

        Output format for the DefectEntry data:
        .. code-block:: python
        [
            {
                'bulk_dir_name': 'computer1:/folder1',
                'bulk_locpot': {...},
                'bulk_uuid': '48fb6da7-dc2b-4dcb-b1c8-1203c0f72ce3',
                'defect_dir_name': 'computer1:/folder2',
                'defect_entry': {...},
                'defect_locpot': {...},
                'defect_uuid': 'e9af2725-d63c-49b8-a01f-391540211750'
            },
            {
                'bulk_dir_name': 'computer1:/folder3',
                'bulk_locpot': {...},
                'bulk_uuid': '48fb6da7-dc2b-4dcb-b1c8-1203c0f72ce3',
                'defect_dir_name': 'computer1:/folder4',
                'defect_entry': {...},
                'defect_locpot': {...},
                'defect_uuid': 'a1c31095-0494-4eed-9862-95311f80a993'
            }
        ]
    """

    defect_relax_maker: Maker
    bulk_relax_maker: Maker | None = None
    name: str = "formation energy"
    relax_radius: float | str | None = None
    perturb: float | None = None
    validate_charge: bool = True
    collect_defect_entry_data: bool = False

    def __post_init__(self) -> None:
        """Apply post init updates."""
        self.validate_maker()
        if self.bulk_relax_maker is None:
            self.bulk_relax_maker = self.defect_relax_maker

    def make(
        self,
        defect: Defect,
        bulk_supercell_dir: str | Path | None = None,
        supercell_matrix: npt.NDArray | None = None,
        defect_index: int | str = "",
    ) -> Flow:
        """Make a flow to calculate the formation energy diagram.

        Start a series of charged supercell relaxations from a single defect
        structure.

        Parameters
        ----------
        defect: Defect
            A `Defect` object representing the Defect we are calculating the
            formation energy diagram for.
        bulk_supercell_dir: str | Path | None
            If provided, the bulk supercell calculation will be skipped.
        supercell_matrix: NDArray | None
            The supercell transformation matrix. If None, the supercell matrix
            will be computed automatically. If `bulk_supercell_dir` is provided,
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
                relax_maker=self.bulk_relax_maker,
                sc_mat=supercell_matrix,
                get_planar_locpot=self.get_planar_locpot,
            )
            sc_mat = get_sc_job.output["sc_mat"]
            lattice = get_sc_job.output["sc_struct"].lattice
            bulk_supercell_dir = get_sc_job.output["dir_name"]
        else:
            # all additional reader functions need to be in this job
            # b/c they might receive Response objects instead of data.
            get_sc_job = get_supercell_from_prv_calc(
                uc_structure=defect.structure,
                prv_calc_dir=bulk_supercell_dir,
                sc_entry_and_locpot_from_prv=self.sc_entry_and_locpot_from_prv,
                sc_mat_ref=supercell_matrix,
            )
            sc_mat = get_sc_job.output["sc_mat"]
            lattice = get_sc_job.output["lattice"]

        spawn_output = spawn_defect_q_jobs(
            defect=defect,
            sc_mat=sc_mat,
            relax_maker=self.defect_relax_maker,
            relaxed_sc_lattice=lattice,
            defect_index=defect_index,
            add_info={
                "bulk_supercell_dir": bulk_supercell_dir,
                "bulk_supercell_matrix": sc_mat,
                "bulk_supercell_uuid": get_sc_job.uuid,
            },
            relax_radius=self.relax_radius,
            perturb=self.perturb,
            validate_charge=self.validate_charge,
        )
        jobs.extend([get_sc_job, spawn_output])

        if self.collect_defect_entry_data:
            collection_job = get_defect_entry(
                charge_state_summary=spawn_output.output,
                bulk_summary=get_sc_job.output,
            )
            jobs.append(collection_job)

        return Flow(
            jobs=jobs,
            output=spawn_output.output,
            name=self.name,
        )

    @abstractmethod
    def sc_entry_and_locpot_from_prv(
        self, previous_dir: str
    ) -> tuple[ComputedStructureEntry, dict]:
        """Copy the output ComputedStructureEntry and Locpot from previous directory.

        Parameters
        ----------
        previous_dir: str
            The directory to copy from.

        Returns
        -------
        entry: ComputedStructureEntry
        """

    @abstractmethod
    def get_planar_locpot(self, task_doc: TaskDoc) -> dict:
        """Get the Planar Locpot from the TaskDoc.

        This is needed just in case the planar average locpot is stored in different
        part of the TaskDoc for different codes.

        Parameters
        ----------
        task_doc: TaskDoc
            The task document.

        Returns
        -------
        planar_locpot: dict
            The planar average locpot.
        """

    @abstractmethod
    def validate_maker(self) -> None:
        """Check some key settings in the relax maker.

        Since this workflow is pretty complex but allows you to use any
        relax maker, it can be easy to make mistakes in the settings.
        This method should check the most important settings and raise
        an error if something is wrong.

        Example:  For VASP, the relax maker should have:
            `ISIF = 2` and `use_structure_charge = True`
        """
