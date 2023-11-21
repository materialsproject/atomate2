"""
Module defining equation of state workflows using default atomate2 parameters.

Modeled on the atomate bulk_modulus workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Job, Maker

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.EOS.base import (
    DeformationMaker,
    EosRelaxMaker,
    postprocess_EOS,
)
from atomate2.vasp.powerups import update_user_incar_settings

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class EosDoubleRelaxMaker(DoubleRelaxMaker):
    """
    Workflow to generate initial double relaxation for EOS fitting.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to use to generate the second relaxation.

    NB: WAVECARs are copied over to speed up calculation
    """

    name: str = "EOS double relax"
    relax_maker1: BaseVaspMaker | None = field(default_factory=EosRelaxMaker)
    relax_maker2: BaseVaspMaker = field(
        default_factory=lambda: EosRelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )


@dataclass
class EosMaker(Maker):
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
    deformation_maker : .BaseVaspMaker
        Maker to generate deformations + single-points, defaults to TransmuterMaker
    static_maker : .BaseVaspMaker
        Optional Maker to generate statics from transmutation.
        Original atomate workflow did not include statics, including it here
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6
    postprocessor : .job
        optional postprocessing step
    """

    name: str = "EOS Maker"
    relax_maker: Maker = field(default_factory=EosDoubleRelaxMaker)
    deformation_maker: Maker = field(
        default_factory=lambda: DeformationMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    static_maker: Maker | None = None
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6
    postprocessor: Job = postprocess_EOS

    def make(self, structure: Structure, prev_dir: str | Path | None = None):
        """
        Run an EOS VASP job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        """
        jobs: dict[str, list[Job]] = {"relax": [], "static": []}

        relax_flow = self.relax_maker.make(structure=structure, prev_dir=prev_dir)
        relax_flow.name = "EOS equilibrium relaxation"
        equil_props = {
            "relax": {
                "E0": relax_flow.output.calcs_reversed[0].output.energy,
                "V0": relax_flow.output.calcs_reversed[0].output.structure.volume,
            }
        }

        if self.static_maker:
            equil_static = self.static_maker.make(
                structure=relax_flow.output.structure,
                prev_dir=relax_flow.output.dir_name,
            )
            equil_static.name = "EOS equilibrium static"
            equil_props["static"] = {
                "E0": equil_static.output.calcs_reversed[0].output.energy,
                "V0": equil_static.output.calcs_reversed[0].output.structure.volume,
            }

        strain_l = np.linspace(
            self.linear_strain[0], self.linear_strain[1], self.number_of_frames
        )
        insert_equil_index = np.searchsorted(strain_l, 0, side="left")
        deformation_l = [(np.identity(3) * (1 + eps)).tolist() for eps in strain_l]

        for ideformation, deformation in enumerate(deformation_l):
            # Doubly ensure that relaxations are done at fixed volume --> ISIF = 2
            deform_relax_maker = update_user_incar_settings(
                flow=self.deformation_maker, incar_updates={"ISIF": 2}
            )
            deform_relax_job = deform_relax_maker.make(
                structure=relax_flow.output.structure,
                deformation_matrix=deformation,
                prev_dir=relax_flow.output.dir_name,
            )
            deform_relax_job.name = f"EOS Deformation Relax {ideformation}"

            if ideformation == insert_equil_index:
                jobs["relax"] += [relax_flow]
                if self.static_maker:
                    jobs["static"] += [equil_static]

            jobs["relax"] += [deform_relax_job]
            if self.static_maker:
                deform_static_job = self.static_maker.make(
                    structure=deform_relax_job.output.structure,
                    prev_dir=deform_relax_job.output.dir_name,
                )
                deform_static_job.name = f"EOS Static {ideformation}"
                jobs["static"] += [deform_static_job]

        flow_output: dict = {}
        if self.postprocessor:
            for jobtype in jobs:
                if len(jobs[jobtype]) == 0:
                    continue

                flow_output[jobtype] = {
                    "energies": [],
                    "volumes": [],
                    **equil_props[jobtype],
                }
                for iframe in range(self.number_of_frames + 1):
                    flow_output[jobtype]["energies"].append(
                        jobs[jobtype][iframe].output.calcs_reversed[0].output.energy
                    )
                    flow_output[jobtype]["volumes"].append(
                        jobs[jobtype][iframe]
                        .output.calcs_reversed[0]
                        .output.structure.volume
                    )

            postprocess = self.postprocessor(flow_output)
            postprocess.name = self.name + "_" + postprocess.name
            flow_output = postprocess.output

        return Flow(
            jobs=jobs["relax"] + jobs["static"] + [postprocess],
            output=flow_output,
            name=self.name,
        )
