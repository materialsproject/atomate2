"""Define common EOS flow agnostic to electronic-structure code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Maker

from atomate2.common.jobs.eos import postprocess_EOS

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Job
    from pymatgen.core import Structure


@dataclass
class CommonEosMaker(Maker):
    """
    Workflow to generate energy vs. volume data for EOS fitting.

    First relax a structure using relax_maker.
    Then perform a series of deformations to the relaxed structure, and
    evaluate single-point energies with static_maker.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .Maker | None
        Maker to relax the input structure, defaults to None
    deformation_maker : .Maker
        Maker to relax deformationed structures
    static_maker : .Maker
        Optional Maker to generate statics from transmutation.
        Original atomate VASP workflow did not include statics, including it here
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6
    postprocessor : .job
        optional postprocessing step
    """

    name: str = "EOS Maker"
    relax_maker: Maker = None
    deformation_maker: Maker = None
    static_maker: Maker = None
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
        jobs: dict[str, list[Job]] = {
            key: [] for key in ("relax", "static", "postprocess")
        }

        relax_flow = self.relax_maker.make(structure=structure, prev_dir=prev_dir)
        relax_flow.name = "EOS equilibrium relaxation"
        equil_props = {
            "relax": {
                "E0": relax_flow.output.output.energy,
                "V0": relax_flow.output.structure.volume,
            }
        }

        if self.static_maker:
            equil_static = self.static_maker.make(
                structure=relax_flow.output.structure,
                prev_dir=relax_flow.output.dir_name,
            )
            equil_static.name = "EOS equilibrium static"
            equil_props["static"] = {
                "E0": equil_static.output.output.energy,
                "V0": equil_static.output.structure.volume,
            }

        strain_l, strain_delta = np.linspace(
            self.linear_strain[0],
            self.linear_strain[1],
            self.number_of_frames,
            retstep=True,
        )

        # Cell without applied strain already included from relax/equilibrium steps.
        # Perturb this point (or these points) if included
        if self.relax_maker:
            zero_strain_mask = np.abs(strain_l) < 1.0e-15
            if np.any(zero_strain_mask):
                nzs = len(strain_l[zero_strain_mask])
                shift = strain_delta / (nzs + 1.0) * np.linspace(-1.0, 1.0, nzs)
                strain_l[np.abs(strain_l) < 1.0e-15] += shift

        insert_equil_index = np.searchsorted(strain_l, 0, side="left")
        deformation_l = [(np.identity(3) * (1 + eps)).tolist() for eps in strain_l]

        for ideformation, deformation in enumerate(deformation_l):
            """
            deform_relax_job = relax_deformation(
                structure = relax_flow.output.structure,
                deformation_matrix = deformation,
                relax_maker = self.deformation_maker,
                relax_maker_kwargs = {"prev_dir": relax_flow.output.dir_name}
            )

            deform_relax_job = RelaxDeformation(
                relax_maker = self.deformation_maker
            ).make(
                structure = relax_flow.output.structure,
                deformation_matrix = deformation,
                prev_dir = relax_flow.output.dir_name
            )
            """
            deform_relax_job = self.deformation_maker.make(
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
            assert self.number_of_frames >= 3, (
                "To perform least squares EOS fit with four parameters, "
                "you must specify self.number_of_frames >= 3."
            )
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
                        jobs[jobtype][iframe].output.output.energy
                    )
                    flow_output[jobtype]["volumes"].append(
                        jobs[jobtype][iframe].output.structure.volume
                    )

            postprocess = self.postprocessor(flow_output)
            postprocess.name = self.name + "_" + postprocess.name
            flow_output = postprocess.output
            jobs["postprocess"] += [postprocess]

        return Flow(
            jobs=jobs["relax"] + jobs["static"] + jobs["postprocess"],
            output=flow_output,
            name=self.name,
        )
