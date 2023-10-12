"""
Module defining equation of state workflows.

Modeled on the atomate bulk_modulus workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Job, Maker

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import TransmuterMaker
from atomate2.vasp.powerups import update_user_incar_settings
from atomate2.vasp.sets.core import EOSSetGenerator
from atomate2.vasp.sets.mp import MPGGAEOSSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure


@dataclass
class EOSMaker(Maker):
    """
    Workflow to generate energy vs. volume data for EOS fitting.

    First relax a structure using relax_maker.
    Then perform a series of deformations to the relaxed structure, and
    evaluate single-point energies with static_maker.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation, defaults to DoubleRelaxMaker
    transmuter_maker : .BaseVaspMaker
        Maker to generate deformations + single-points, defaults to TransmuterMaker
    static_maker : .BaseVaspMaker
        Optional Maker to generate statics from transmutation.
        Original atomate workflow did not include statics, including it here
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6
    """

    name: str = "EOS Maker"
    relax_maker: Maker | None = field(
        default_factory=lambda: DoubleRelaxMaker(
            input_set_generator=EOSSetGenerator(
                user_incar_settings={"ISIF": 3, "LWAVE": True}
            )
        )
    )
    transmuter_maker: Maker = field(
        default_factory=lambda: TransmuterMaker(
            input_set_generator=EOSSetGenerator(),
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)},
        )
    )
    static_maker: Maker | None = None
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Run an EOS VASP job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        """
        relax_jobs: list[Job] = []
        static_jobs: list[Job] = []

        relax_flow = self.relax_maker.make(
            structure=structure, prev_vasp_dir=prev_vasp_dir
        )
        relax_flow.name = "EOS equilibrium relaxation"

        if self.static_maker:
            equil_static = self.static_maker.make(
                structure=relax_flow.output.structure,
                prev_vasp_dir=relax_flow.output.dir_name,
            )
            equil_static.name = "EOS equilibrium static"

        strain_l = np.linspace(
            self.linear_strain[0], self.linear_strain[1], self.number_of_frames
        )
        insert_equil_index = np.searchsorted(strain_l, 0, side="left")
        deformation_l = [(np.identity(3) * (1 + eps)).tolist() for eps in strain_l]

        # Doubly ensure that relaxations are done at fixed volume --> ISIF = 2
        deform_relax_maker = update_user_incar_settings(
            flow=self.transmuter_maker, incar_updates={"ISIF": 2}
        )

        for ideformation, deformation in enumerate(deformation_l):
            deform_relax_job = deform_relax_maker(
                transformations=("DeformStructureTransformation"),
                transformation_params=({"deformation": deformation}),
            ).make(
                structure=relax_flow.output.structure,
                prev_vasp_dir=relax_flow.output.dir_name,
            )
            deform_relax_job.name = f"EOS Deformation Relax {ideformation}"

            if ideformation == insert_equil_index:
                relax_jobs += [relax_flow]
                if self.static_maker:
                    static_jobs += [equil_static]

            relax_jobs += [deform_relax_job]
            if self.static_maker:
                deform_static_job = self.static_maker.make(
                    structure=deform_relax_job.output.structure,
                    prev_vasp_dir=deform_relax_job.output.dir_name,
                )
                deform_static_job.name = f"EOS Static {ideformation}"
                static_jobs += [deform_static_job]

        flow_output: dict[str, list] = {"relax": []}
        for iframe in range(self.number_of_frames + 1):
            flow_output["relax"].append(
                [relax_jobs[iframe].output.volume, relax_jobs[iframe].output.energy]
            )

        if self.static_maker:
            flow_output["static"] = []
            for iframe in range(self.number_of_frames + 1):
                flow_output["static"].append(
                    [
                        static_jobs[iframe].output.volume,
                        static_jobs[iframe].output.energy,
                    ]
                )

        return Flow(jobs=relax_jobs + static_jobs, output=flow_output, name=self.name)


@dataclass
class MPGGAEOSMaker(Maker):
    """
    Workflow to generate MP-compatible energy vs. volume data for EOS fitting.

    First relax a structure using relax_maker.
    Then perform a series of deformations to the relaxed structure, and
    evaluate single-point energies with static_maker.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation, defaults to DoubleRelaxMaker
    transmuter_maker : .BaseVaspMaker
        Maker to generate deformations + single-points, defaults to TransmuterMaker
    static_maker : .BaseVaspMaker
        Optional Maker to generate statics from transmutation.
        Original atomate workflow did not include statics, including it here
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6
    """

    name: str = "EOS Maker"
    relax_maker: Maker | None = field(
        default_factory=lambda: DoubleRelaxMaker(
            input_set_generator=MPGGAEOSSetGenerator(
                user_incar_settings={"ISIF": 3, "LWAVE": True}
            )
        )
    )
    transmuter_maker: Maker = field(
        default_factory=lambda: TransmuterMaker(
            input_set_generator=MPGGAEOSSetGenerator(),
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)},
        )
    )
    static_maker: Maker | None = None
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6
