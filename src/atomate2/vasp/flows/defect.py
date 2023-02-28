"""Flows used in the calculation of defect properties."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, OutputReference
from jobflow.core.maker import recursive_call
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Vasprun

from atomate2.common.analysis.defects import flows as defect_flows
from atomate2.common.files import get_zfile
from atomate2.common.schemas.defects import CCDDocument
from atomate2.utils.file_client import FileClient
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.jobs.defect import calculate_finite_diff
from atomate2.vasp.powerups import update_user_incar_settings
from atomate2.vasp.sets.defect import (
    SPECIAL_KPOINT,
    ChargeStateRelaxSetGenerator,
    ChargeStateStaticSetGenerator,
    HSEChargeStateRelaxSetGenerator,
)

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
        copy_vasp_kwargs={
            "additional_vasp_files": ("WAVECAR",),
        },
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
    bulk_incar_update: dict
        A dictionary of incar settings to update the bulk job with. This is
        useful if you want to change the ISIF setting for example. Default is
        {"ISIF": 3}.
    """

    name: str = "formation energy"
    relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=ChargeStateRelaxSetGenerator(
                user_kpoints_settings=SPECIAL_KPOINT
            ),
            task_document_kwargs={"average_locpot": True},
        )
    )
    bulk_incar_update: dict = field(default_factory=lambda: {"ISIF": 3})

    def update_bulk_maker(self, relax_maker: Maker):
        """Update the bulk job with settings from `self.bulk_incar_update`."""
        return update_user_incar_settings(relax_maker, self.bulk_incar_update)

    def structure_from_prv(self, previous_dir: str):
        """
        Read the vasprun.xml file from the previous directory
        and return the structure.
        """
        fc = FileClient()
        files = fc.listdir(previous_dir)
        vasprun_file = Path(previous_dir) / get_zfile(files, "vasprun.xml")
        vasprun = Vasprun(vasprun_file)
        return vasprun.final_structure

    def grid_update_from_prv(self, previous_dir):
        """Read the previous directory and get the grid update."""
        fc = FileClient()
        files = fc.listdir(previous_dir)
        vasprun_file = Path(previous_dir) / get_zfile(files, "vasprun.xml")
        vasprun = Vasprun(vasprun_file)
        params = vasprun.parameters
        return {k: params[k] for k in GRID_KEYS}

    def validate_maker(self):
        def check_func(relax_maker: RelaxMaker):
            input_gen = relax_maker.input_set_generator
            if input_gen.use_structure_charge is False:
                raise ValueError("use_structure_charge should be set to True")

        recursive_call(
            self.relax_maker, func=check_func, class_filter=RelaxMaker, nested=True
        )


@dataclass
class ConfigurationCoordinateMaker(defect_flows.ConfigurationCoordinateMaker):
    relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=DEFECT_RELAX_GENERATOR,
        )
    )
    static_maker: BaseVaspMaker = field(
        default_factory=lambda: StaticMaker(input_set_generator=DEFECT_STATIC_GENERATOR)
    )
    name: str = "config. coordinate"


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
    ):
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
