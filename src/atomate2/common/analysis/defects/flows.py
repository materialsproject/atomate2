"""Flows used in the calculation of defect properties."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy.typing as npt
from jobflow import Flow, Job, Maker, OutputReference
from pymatgen.analysis.defects.core import Defect
from pymatgen.core.structure import Structure

from atomate2.common.analysis.defects.jobs import (
    bulk_supercell_calculation,
    get_ccd_documents,
    get_charged_structures,
    get_supercell_from_prv_calc,
    spawn_defect_calcs,
    spawn_energy_curve_calcs,
)

logger = logging.getLogger(__name__)

DEFAULT_DISTORTIONS = (-1, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 1)


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

    relax_maker: Maker
    static_maker: Maker
    name: str = "config. coordinate"
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


@dataclass
class FormationEnergyMaker(Maker, ABC):
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
    relax_maker: Maker
        A maker to perform a atomic-position-only relaxation on the defect charge
        states.
    name: str
        The name of the flow created by this maker.
    validate_maker: bool
        If True, the code will check the relax_maker for specific settings.
    """

    relax_maker: Maker
    name: str = "formation energy"
    validate_maker: bool = True

    def make(
        self,
        defect: Defect,
        bulk_supercell_dir: str | Path | None = None,
        supercell_matrix: npt.NDArray | None = None,
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
                update_maker=self.update_maker,
            )
            sc_mat = get_sc_job.output["sc_mat"]
            lattice = get_sc_job.output["sc_struct"].lattice
            bulk_supercell_dir = get_sc_job.output["dir_name"]
        else:
            get_sc_job = get_supercell_from_prv_calc(
                uc_structure=defect.structure,
                prv_calc_dir=bulk_supercell_dir,
                sc_mat_ref=supercell_matrix,
                structure_from_prv=self.structure_from_prv,
            )
            sc_mat = get_sc_job.output["sc_mat"]
            lattice = get_sc_job.output["lattice"]

        spawn_output = spawn_defect_calcs(
            defect=defect,
            sc_mat=sc_mat,
            relax_maker=self.relax_maker,
            relaxed_sc_lattice=lattice,
            defect_index=defect_index,
            add_info={
                "bulk_supercell_dir": bulk_supercell_dir,
                "bulk_supercell_matrix": sc_mat,
                "bulk_supercell_uuid": get_sc_job.uuid,
            },
        )
        jobs.extend([get_sc_job, spawn_output])

        return Flow(
            jobs=jobs,
            name=self.name,
        )

    @abstractmethod
    def update_maker(self, relax_maker: Maker):
        """Update the maker for the bulk job.

        Common usage case:
        While almost all of the settings for the bulk relaxation and defect
        relaxation should be the same, it is usually desirable to allow lattice
        relaxation for the bulk job only.
        Assuming the `relax_maker` is only allows atomic relaxations, this method
        will update the bulk job to allow lattice relaxations.

        Parameters
        ----------
        relax_maker: Maker
            The maker used to create the defect job.

        Returns:
        --------
        Maker:
            The updated maker.
        """
        raise NotImplementedError("This method is not implemented yet.")

    @abstractmethod
    def structure_from_prv(self, previous_dir: str):
        """Copy the previous directory to the new one.

        Parameters
        ----------
        previous_dir: str
            The directory to copy from.

        Returns
        -------
        new_dir: str
            The new directory.
        """
        raise NotImplementedError("This method is not implemented yet.")
