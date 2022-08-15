"""Flows used in the calculation of defect properties."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from jobflow import Flow, Job, Maker, OutputReference, job

# from jobflow.core.maker import recursive_call
from numpy.typing import NDArray
from pymatgen.analysis.defects.generators import DefectGenerator
from pymatgen.core.structure import Lattice, Structure
from pymatgen.io.vasp.inputs import Kpoints, Kpoints_supported_modes

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.jobs.defect import (
    bulk_supercell_calculation,
    calculate_finite_diff,
    collect_defect_outputs,
    get_ccd_documents,
    get_supercell_from_prv_calc,
    spawn_defect_calcs,
    spawn_energy_curve_calcs,
)
from atomate2.vasp.schemas.defect import CCDDocument
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.sets.defect import AtomicRelaxSetGenerator

logger = logging.getLogger(__name__)

# Defaults
DUMMY_STRUCT = Structure(Lattice.cubic(3.6), ["Si", "Si"], [[0.5, 0.5, 0.5], [0, 0, 0]])

SPECIAL_KPOINT = Kpoints(
    comment="special k-point",
    num_kpts=1,
    style=Kpoints_supported_modes.Reciprocal,
    kpts=((0.25, 0.25, 0.25),),
    kpts_shift=(0, 0, 0),
    kpts_weights=[1],
)

SPECIAL_KPOINT_GAMMA = Kpoints(
    comment="special k-point",
    num_kpts=1,
    style=Kpoints_supported_modes.Reciprocal,
    kpts=((0.25, 0.25, 0.25), (0.25, 0.25, 0.25)),
    kpts_shift=(0, 0, 0),
    kpts_weights=[1, 0],
)

# Default relax should use: PBE -> HSE to save computation time
PBE_GEN: AtomicRelaxSetGenerator = AtomicRelaxSetGenerator(
    use_structure_charge=True,
    user_incar_settings={"LVHAR": False},
    user_kpoints_settings=SPECIAL_KPOINT,
)

HSE_GEN: AtomicRelaxSetGenerator = AtomicRelaxSetGenerator(
    use_structure_charge=True,
    user_incar_settings={
        "LAECHG": False,
        "LREAL": False,
        "ALGO": "Normal",
        "NSW": 99,
        "LCHARG": False,
        "HFSCREEN": 0.2,
        "LHFCALC": True,
        "PRECFOCK": "Fast",
        "LASPH": True,
        "LDAU": False,
    },
    user_kpoints_settings=SPECIAL_KPOINT,
)

HSE_DOUBLE_RELAX = DoubleRelaxMaker(
    relax_maker1=RelaxMaker(input_set_generator=PBE_GEN),
    relax_maker2=RelaxMaker(
        input_set_generator=HSE_GEN,
        task_document_kwargs={"store_volumetric_data": ["locpot"]},
    ),
)

DEFECT_STATIC_GENERATOR: StaticSetGenerator = StaticSetGenerator(
    user_kpoints_settings=SPECIAL_KPOINT,
)

CCD_DEFAULT_DISTORTIONS = (-1, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 1)


# Define RelaxMaker Errors
def check_relax_maker(maker: Maker):
    """Check specific RelaxMaker settings.

    Check all nested RelaxMaker in to make sure specific Setting is set.

    Parameters
    ----------
    maker : Maker
        The Maker object to check.
    """

    def check_func(relax_maker: RelaxMaker):
        input_gen = relax_maker.input_set_generator
        vis = input_gen.get_input_set(DUMMY_STRUCT, potcar_spec=True)
        if input_gen.use_structure_charge is False:
            raise ValueError("use_structure_charge should be set to True")
        if vis.incar["LVHAR"] is False:
            raise ValueError("LVHAR should be set to True")
        if vis.incar["ISIF"] != 2:
            raise ValueError("ISIF should be 2")

    # TODO: add this check after JobFlow is updated to support recursive_call
    # recursive_call(maker, func=check_func, class_filter=RelaxMaker, nested=True)


@dataclass
class FormationEnergyMaker(Maker):
    """Maker class to help calculate of the formation energy diagram.

    Attributes
    ----------
    name: str
        The name of the flow created by this maker.
    relax_maker: .BaseVaspMaker or None
        A maker to perform a atomic-position-only relaxation on the defect charge
        states. If None, the defaults will be used.
    validate_maker: bool
        If True, the code will check the relax_maker for specific settings
    """

    name: str = "formation energy"
    validate_maker: bool = True
    relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=PBE_GEN,
            task_document_kwargs={"store_volumetric_data": ["locpot"]},
        )
    )

    def __post_init__(self):
        """Check the calculation settings."""
        if self.validate_maker:
            check_relax_maker(self.relax_maker)

    def make(
        self,
        defect_gen: DefectGenerator,
        dielectric: float | NDArray,
        bulk_sc_dir: str | Path | None = None,
        sc_mat: NDArray | None = None,
    ):
        """Make a flow to calculate the formation energy diagram."""
        jobs = []

        if bulk_sc_dir is None:
            get_sc_job = bulk_supercell_calculation(
                uc_structure=defect_gen.structure,
                relax_maker=self.relax_maker,
                sc_mat=sc_mat,
            )
        else:
            get_sc_job = get_supercell_from_prv_calc(
                defect_gen.structure, bulk_sc_dir, sc_mat
            )

        spawn_output = spawn_defect_calcs(
            defect_gen=defect_gen,
            sc_mat=get_sc_job.output["sc_mat"],
            relax_maker=self.relax_maker,
        )
        jobs.extend([get_sc_job, spawn_output])

        collect_job = collect_defect_outputs(
            spawn_output.output, get_sc_job.output["dir_name"], dielectric
        )
        jobs.append(collect_job)

        return Flow(
            jobs=jobs,
            name=self.name,
            output=collect_job.output,
        )


@dataclass
class ConfigurationCoordinateMaker(Maker):
    """Maker to generate a configuration coordinate diagram.

    Parameters
    ----------
    name: str
        The name of the flow created by this maker.
    relax_maker: .BaseVaspMaker or None
        A maker to perform a atomic-position-only relaxation on the defect charge
        states.
    static_maker: .BaseVaspMaker or None
        A maker to perform the single-shot static calculation of the distorted
        structures.
    distortions: tuple[float, ...]
        The distortions, as a fraction of ΔQ, to use in the calculation of the
        configuration coordinate diagram.
    """

    name: str = "config. coordinate"
    relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=PBE_GEN,
        )
    )
    static_maker: BaseVaspMaker = field(
        default_factory=lambda: StaticMaker(input_set_generator=DEFECT_STATIC_GENERATOR)
    )
    distortions: tuple[float, ...] = CCD_DEFAULT_DISTORTIONS

    def make(
        self,
        structure: Structure,
        charge_state1: int,
        charge_state2: int,
    ):
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
    """Add charges to a structure.

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
