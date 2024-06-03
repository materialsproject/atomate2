"""Flows used in the calculation of defect properties."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from emmet.core.tasks import TaskDoc
from jobflow import Flow, Maker, OutputReference
from jobflow.core.maker import recursive_call

from atomate2.common.flows import defect as defect_flows
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.jobs.defect import calculate_finite_diff
from atomate2.vasp.sets.defect import (
    SPECIAL_KPOINT,
    ChargeStateRelaxSetGenerator,
    ChargeStateStaticSetGenerator,
    HSEChargeStateRelaxSetGenerator,
)

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    from atomate2.common.schemas.defects import CCDDocument
    from atomate2.vasp.jobs.base import BaseVaspMaker

logger = logging.getLogger(__name__)


DEFECT_INCAR_SETTINGS = {
    "ISMEAR": 0,
    "LWAVE": True,
    "SIGMA": 0.05,
    "KSPACING": None,
    "ENCUT": 500,
}
DEFECT_KPOINT_SETTINGS = {"reciprocal_density": 64}

DEFECT_RELAX_GENERATOR = ChargeStateRelaxSetGenerator(
    use_structure_charge=True,
    user_incar_settings=DEFECT_INCAR_SETTINGS,
    user_kpoints_settings=DEFECT_KPOINT_SETTINGS,
)
DEFECT_STATIC_GENERATOR = ChargeStateStaticSetGenerator(
    user_incar_settings=DEFECT_INCAR_SETTINGS,
    user_kpoints_settings=DEFECT_KPOINT_SETTINGS,
)
HSE_DOUBLE_RELAX = DoubleRelaxMaker(
    relax_maker1=RelaxMaker(
        input_set_generator=ChargeStateRelaxSetGenerator(
            user_kpoints_settings=SPECIAL_KPOINT
        )
    ),
    relax_maker2=RelaxMaker(
        input_set_generator=HSEChargeStateRelaxSetGenerator(
            user_kpoints_settings=SPECIAL_KPOINT
        ),
        task_document_kwargs={"store_volumetric_data": ["locpot"]},
        copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)},
    ),
)
GRID_KEYS = ["NGX", "NGY", "NGZ", "NGXF", "NGYF", "NGZF"]


@dataclass
class FormationEnergyMaker(defect_flows.FormationEnergyMaker):
    """Maker class to help calculate of the formation energy diagram.

    Maker class to calculate formation energy diagrams. The main settings for
    this maker is the `relax_maker` which contains the settings for the atomic
    relaxations that each defect supercell will undergo. The `relax_maker`
    uses a `ChargeStateRelaxSetGenerator` by default but more complex makers
    like the `HSE_DOUBLE_RELAX` can be used for more accurate (but expensive)
    calculations.
    If the `validate_maker` is set to True, the maker will check for some basic
    settings in the `relax_maker` to make sure the calculations are done correctly.

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

    defect_relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=ChargeStateRelaxSetGenerator(
                user_kpoints_settings=SPECIAL_KPOINT
            ),
            task_document_kwargs={"average_locpot": True},
        )
    )
    bulk_relax_maker: BaseVaspMaker | None = None
    name: str = "formation energy"

    def sc_entry_and_locpot_from_prv(
        self, previous_dir: str
    ) -> tuple[ComputedStructureEntry, dict]:
        """Copy the output structure from previous directory.

        Read the vasprun.xml file from the previous directory
        and return the structure.

        Parameters
        ----------
        previous_dir: str
            The directory to copy from.

        Returns
        -------
        ComputedStructureEntry
        """
        task_doc = TaskDoc.from_directory(previous_dir)
        return task_doc.structure_entry, task_doc.calcs_reversed[0].output.locpot

    def get_planar_locpot(self, task_doc: TaskDoc) -> dict:
        """Get the planar-averaged electrostatic potential."""
        return task_doc.calcs_reversed[0].output.locpot

    def validate_maker(self) -> None:
        """Check some key settings in the relax maker.

        Since this workflow is pretty complex but allows you to use any
        relax maker, it can be easy to make mistakes in the settings.
        This method should check the most important settings and raise
        an error if something is wrong.

        Example:  For VASP, the relax maker should have:
            `ISIF = 2` and `use_structure_charge = True`
        """

        def check_defect_relax_maker(relax_maker: RelaxMaker) -> RelaxMaker:
            input_gen = relax_maker.input_set_generator
            if input_gen.use_structure_charge is False:
                raise ValueError("use_structure_charge should be set to True")
            isif_ = input_gen.get_incar_updates(None).get("ISIF", None)
            isif = input_gen.user_incar_settings.get("ISIF", isif_)
            if isif != 2:
                raise ValueError("ISIF should be set to 2")
            return relax_maker

        recursive_call(
            self.defect_relax_maker,
            func=check_defect_relax_maker,
            class_filter=RelaxMaker,
            nested=True,
        )


@dataclass
class ConfigurationCoordinateMaker(defect_flows.ConfigurationCoordinateMaker):
    """Maker to generate a configuration coordinate diagram.

    Parameters
    ----------
    name: str
        The name of the flow created by this maker.
    relax_maker: BaseVaspMaker or None
        A maker to perform a atomic-position-only relaxation on the defect charge
        states.
    static_maker: BaseVaspMaker or None
        A maker to perform the single-shot static calculation of the distorted
        structures.
    distortions: tuple[float, ...]
        The distortions, as a fraction of ΔQ, to use in the calculation of the
        configuration coordinate diagram.
    """

    relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=DEFECT_RELAX_GENERATOR,
        )
    )
    static_maker: BaseVaspMaker = field(
        default_factory=lambda: StaticMaker(input_set_generator=DEFECT_STATIC_GENERATOR)
    )
    name: str = "config coordinate"


@dataclass
class NonRadiativeMaker(Maker):
    """Maker to calculate non-radiative defect capture.

    Parameters
    ----------
    name: str
        The name of the flow created by this maker.
    ccd_maker: ConfigurationCoordinateMaker
        A maker to perform the calculation of the configuration coordinate diagram.
    """

    ccd_maker: ConfigurationCoordinateMaker
    name: str = "non-radiative"

    def make(
        self,
        structure: Structure,
        charge_state1: int,
        charge_state2: int,
    ) -> Flow:
        """Create the job for Non-Radiative defect capture.

        Make a job for the calculation of the configuration coordinate diagram.
        Also calculate the el-phon matrix elements for 1-D special phonon.

        Parameters
        ----------
        structure
            A structure.
        charge_state1
            The reference charge state of the defect.
        charge_state2
            The excited charge state of the defect
        """
        if not isinstance(structure, OutputReference):
            name = f"{self.name}: {structure.formula}"
            if not (
                isinstance(charge_state1, OutputReference)
                or isinstance(charge_state2, OutputReference)
            ):
                name = (
                    f"{self.name}: {structure.formula}({charge_state1}-{charge_state2})"
                )

        flow = self.ccd_maker.make(
            structure=structure,
            charge_state1=charge_state1,
            charge_state2=charge_state2,
        )
        ccd: CCDDocument = flow.output

        finite_diff_job1 = calculate_finite_diff(
            distorted_calc_dirs=ccd.static_dirs1,
            ref_calc_index=ccd.relaxed_index1,
            run_vasp_kwargs=self.ccd_maker.static_maker.run_vasp_kwargs,
        )
        finite_diff_job2 = calculate_finite_diff(
            distorted_calc_dirs=ccd.static_dirs2,
            ref_calc_index=ccd.relaxed_index2,
            run_vasp_kwargs=self.ccd_maker.static_maker.run_vasp_kwargs,
        )

        finite_diff_job1.name = "finite diff q1"
        finite_diff_job2.name = "finite diff q2"

        output = {
            charge_state1: finite_diff_job1.output,
            charge_state2: finite_diff_job2.output,
        }
        return Flow(
            jobs=[flow, finite_diff_job1, finite_diff_job2], output=output, name=name
        )
