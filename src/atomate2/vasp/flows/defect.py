"""Flows used in the calculation of defect properties."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from jobflow import Flow, Job, Maker, OutputReference, job
from numpy.typing import NDArray
from pymatgen.analysis.defect.core import Defect, get_sc_fromstruct
from pymatgen.analysis.defect.generators import DefectGenerator
from pymatgen.core.structure import Structure

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.jobs.defect import (
    FiniteDifferenceMaker,
    collect_outputs,
    get_ccd_documents,
    perform_defect_calculations,
    spawn_energy_curve_calcs,
)
from atomate2.vasp.schemas.defect import CCDDocument
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.sets.defect import AtomicRelaxSetGenerator

logger = logging.getLogger(__name__)

################################################################################
# Default settings                                                            ##
################################################################################

DEFAULT_DISTORTIONS = (-1, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 1)
DEFECT_INCAR_SETTINGS = {
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "KSPACING": None,
    "ENCUT": 500,
}
DEFECT_KPOINT_SETTINGS = {"reciprocal_density": 64}

DEFECT_RELAX_GENERATOR: AtomicRelaxSetGenerator = AtomicRelaxSetGenerator(
    use_structure_charge=True,
    user_incar_settings=DEFECT_INCAR_SETTINGS,
    user_kpoints_settings=DEFECT_KPOINT_SETTINGS,
)
DEFECT_STATIC_GENERATOR: StaticSetGenerator = StaticSetGenerator(
    user_incar_settings=DEFECT_INCAR_SETTINGS,
    user_kpoints_settings=DEFECT_KPOINT_SETTINGS,
)


################################################################################
# Formation Energy                                                            ##
################################################################################


@dataclass
class FormationEnergyMaker(Maker):
    """Class to generate VASP input sets for the calculation of the formation energy diagram.

    Parameters
    ----------
    name: str
        The name of the flow created by this maker.
    relax_maker: .BaseVaspMaker or None
        A maker to perform a atomic-position-only relaxation on the defect charge states.
        If None, the defaults will be used.
    """

    name: str = "formation energy"
    relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=DEFECT_RELAX_GENERATOR,
        )
    )

    def make(self, defect_gen: DefectGenerator, sc_mat: NDArray | None = None):
        """Make a flow to calculate the formation energy diagram."""
        bulk_structure = defect_gen.structure
        self.relax_maker.input_set_generator.user_incar_settings["LVHAR"] = True
        if sc_mat is None:
            sc_mat = get_sc_fromstruct(bulk_structure)
        bulk_relax: Job = self.relax_maker.make(bulk_structure * sc_mat)
        bulk_relax.name = "bulk relax"
        defect_calcs = []
        defect: Defect
        output = dict()
        for i, defect in enumerate(defect_gen):
            defect_job = perform_defect_calculations(
                defect,
                sc_mat=sc_mat,
                prev_vasp_dir=bulk_relax.output.dir_name,
            )
            defect_calcs.append(defect_job)
            output[f"defect.name_{i}"] = defect_job.output

        collected = collect_outputs(output)
        return Flow(
            jobs=[bulk_relax] + defect_calcs,
            name=self.name,
            output=collected,
        )


@dataclass
class ConfigurationCoordinateMaker(Maker):
    """Class to generate VASP input sets for the calculation of the configuration coordinate diagram.

    Parameters
    ----------
    name: str
        The name of the flow created by this maker.
    relax_maker: .BaseVaspMaker or None
        A maker to perform a atomic-position-only relaxation on the defect charge states.
        If None, the defaults will be used.
    static_maker: .BaseVaspMaker or None
        A maker to perform the single-shot static calculation of the distorted structures.
        If None, the defaults will be used.
    distortions: tuple[float, ...]
        The distortions to use in the calculation of the configuration coordinate diagram.
    """

    name: str = "config. coordinate"
    relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=DEFECT_RELAX_GENERATOR,
        )
    )
    static_maker: BaseVaspMaker = field(
        default_factory=lambda: StaticMaker(input_set_generator=DEFECT_STATIC_GENERATOR)
    )
    distortions: tuple[float, ...] = DEFAULT_DISTORTIONS

    def make(
        self,
        structure: Structure,
        charge_state1: int,
        charge_state2: int,
    ):
        """
        Make a job for the calculation of the configuration coordinate diagram.

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
            The full workflow for the calculation of the configuration coordinate diagram.
        """
        # Make sure the static makers stores the wavecar
        self.static_maker.input_set_generator.user_incar_settings["LWAVE"] = True
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

        # need to wrap this up in a job so that references to undone calculations can be passed in
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
            prev_vasp_dir=dir1,
            add_name="q1",
            add_info={"relaxed_uuid": relax1.uuid, "distorted_uuid": relax2.uuid},
        )

        deformations2 = spawn_energy_curve_calcs(
            struct2,
            struct1,
            distortions=self.distortions,
            static_maker=self.static_maker,
            prev_vasp_dir=dir2,
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


@job
def get_charged_structures(structure: Structure, charges: Iterable):
    """Adding charges to structure.

    This needs to be a job so the results of other jobs can be passed in.

    Parameters
    ----------
    structure
        A structure.
    charges
        A list of charges on the structure

    Returns
    -------
    dict
        A dictionary with the two structures with the charge states added.
    """
    structs_out = [structure.copy() for _ in charges]
    for i, q in enumerate(charges):
        structs_out[i].set_charge(q)
    return structs_out


@dataclass
class NonRadiativeMaker(Maker):
    """Class to generate workflows for the calculation of the non-radiative defect capture.

    Parameters
    ----------
    name: str
        The name of the flow created by this maker.
    ccd_maker: ConfigurationCoordinateMaker
        A maker to perform the calculation of the configuration coordinate diagram.
    fdiff_maker: FiniteDifferenceMaker
        A maker to perform the calculation of the finite difference using wavefunction overlaps from VASP.
    """

    ccd_maker: ConfigurationCoordinateMaker
    name: str = "non-radiative"
    fdiff_maker: FiniteDifferenceMaker = field(default_factory=FiniteDifferenceMaker)

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

        dirs0 = ccd.static_dirs1
        dirs1 = ccd.static_dirs2
        mid_index0 = len(self.ccd_maker.distortions) // 2
        mid_index1 = len(self.ccd_maker.distortions) // 2

        finite_diff_job1 = self.fdiff_maker.make(
            ref_calc_dir=dirs0[mid_index0], distorted_calc_dirs=dirs0
        )
        finite_diff_job2 = self.fdiff_maker.make(
            ref_calc_dir=dirs1[mid_index1], distorted_calc_dirs=dirs1
        )
        finite_diff_job1.append_name(" q1")
        finite_diff_job2.append_name(" q2")

        output = {
            charge_state1: finite_diff_job1.output,
            charge_state2: finite_diff_job2.output,
        }
        return Flow(
            jobs=[flow, finite_diff_job1, finite_diff_job2], output=output, name=name
        )
