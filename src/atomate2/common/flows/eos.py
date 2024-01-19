"""Define common EOS flow agnostic to electronic-structure code."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Maker, Response, job
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

from atomate2.common.jobs.eos import (
    postprocess_EOS, 
    apply_strain_to_structure
)

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Job
    from pymatgen.core import Structure


@dataclass
class CommonEosMaker(Maker):
    """
    Generate equation of state data.

    First relax a structure using relax_maker.
    Then perform a series of deformations to the relaxed structure, and
    evaluate single-point energies with static_maker.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    initial_relax_maker : .Maker | None
        Maker to relax the input structure, defaults to None (no initial relaxation).
    eos_relax_maker : .Maker
        Maker to relax deformationed structures for the EOS fit.
    static_maker : .Maker | None
        Maker to generate statics after each relaxation, defaults to None.
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%.
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6.
    postprocessor : .job
        Optional postprocessing step, defaults to `atomate2.common.jobs.postprocess_EOS`.
    _store_transformation_information : .bool = False
        Whether to store the information about transformations. Unfortunately
        needed at present to handle issues with emmet and pydantic validation
        TODO: remove this when clash is fixed
    """

    name: str = "EOS Maker"
    initial_relax_maker: Maker = None
    eos_relax_maker: Maker = None
    static_maker: Maker = None
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6
    postprocessor: Job = postprocess_EOS
    _store_transformation_information : bool = False

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """
        Run an EOS flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        """
        jobs: dict[str, list[Job]] = {
            key: [] for key in ("relax", "static", "utility")
        }

        equil_props = {}
        # First step: optional relaxation of structure
        if self.initial_relax_maker:
            relax_flow = self.initial_relax_maker.make(
                structure=structure, prev_dir=prev_dir
            )
            relax_flow.name = "EOS equilibrium relaxation"
            equil_props["relax"] = {
                "E0": relax_flow.output.output.energy,
                "V0": relax_flow.output.structure.volume,
            }
            structure = relax_flow.output.structure
            prev_dir = relax_flow.output.dir_name
            jobs["relax"].append(relax_flow)

            if self.static_maker:
                equil_static = self.static_maker.make(
                    structure=structure, prev_dir=prev_dir
                )
                equil_static.name = "EOS equilibrium static"
                equil_props["static"] = {
                    "E0": equil_static.output.output.energy,
                    "V0": equil_static.output.structure.volume,
                }
                jobs["static"].append(equil_static)

        strain_l, strain_delta = np.linspace(
            self.linear_strain[0],
            self.linear_strain[1],
            self.number_of_frames,
            retstep=True,
        )

        # Cell without applied strain already included from relax/equilibrium steps.
        # Perturb this point (or these points) if included
        zero_strain_mask = np.abs(strain_l) < 1.0e-15
        if np.any(zero_strain_mask):
            nzs = len(strain_l[zero_strain_mask])
            shift = strain_delta / (nzs + 1.0) * np.linspace(-1.0, 1.0, nzs)
            strain_l[np.abs(strain_l) < 1.0e-15] += shift

        deformation_l = [(np.identity(3) * (1.0 + eps)).tolist() for eps in strain_l]

        # apply strain to structures, return list of transformations
        transformations = apply_strain_to_structure(structure,deformation_l)
        jobs["utility"] += [transformations]

        """
        jobs["deformation"] += [
            EosDeformationMaker(
                eos_relax_maker=self.eos_relax_maker, static_maker=self.static_maker
            ).make(
                structure=structure,
                deformations=deformation_l,
                prev_dir=prev_dir,
            )
        ]
        """
        job_types = ("relax", "static") if self.static_maker else ("relax")
        deformation_output = {
            key: {"energies": [], "volumes": []} for key in job_types
        }
        for idef in range(self.number_of_frames):

            if self._store_transformation_information:
                with contextlib.suppress(Exception):
                    # write details of the transformation to the transformations.json file
                    # this file will automatically get added to the task document and allow
                    # the elastic builder to reconstruct the elastic document; note the ":" is
                    # automatically converted to a "." in the filename.
                    self.eos_relax_maker.write_additional_data[
                        "transformations:json"
                    ] = transformations.output[idef]

            relax_job = self.eos_relax_maker.make(
                structure = transformations.output[idef].final_structure, 
                prev_dir=prev_dir
            )
            relax_job.name += f" deformation {idef}"
            jobs["relax"].append(relax_job)

            if self.static_maker:
                static_job = self.static_maker.make(
                    structure=relax_job.output.structure,
                    prev_dir=relax_job.output.dir_name,
                )
                static_job.name += f" {idef}"
                jobs["static"].append(static_job)

            for key in job_types:
                deformation_output[key]["energies"].append(jobs[key][-1].output.output.energy)
                deformation_output[key]["volumes"].append(jobs[key][-1].output.structure.volume)

        flow_output = {
            "equilibrium": equil_props,
            "deformation": deformation_output,
        }
        if self.postprocessor:
            if self.number_of_frames < 3:
                raise ValueError(
                    "To perform least squares EOS fit with four parameters, "
                    "you must specify self.number_of_frames >= 3."
                )

            postprocess = self.postprocessor(equil_props, jobs["deformation"][0].output)
            postprocess.name = self.name + "_" + postprocess.name
            flow_output = postprocess.output
            jobs["utility"] += [postprocess]

        joblist = []
        for key in jobs:
            joblist += jobs[key]

        return Flow(jobs=joblist, output=flow_output, name=self.name)


@dataclass
class EosDeformationMaker(Maker):
    """
    Flow that computes deformations on input structure.

    Based on atomate2.common.jobs.elastic.run_elastic_deformations,
    mostly different structure of I/O

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    eos_relax_maker : .Maker | None
        Maker to relax the deformed structure.
    static_maker : .Maker | None
        Maker to perform an optional static on the relaxed structure,
        defaults to None (no statics).
    """

    name: str = "EOS Deformation Relax"
    eos_relax_maker: Maker = None
    static_maker: Maker = None

    @job
    def make(
        self,
        structure: Structure,
        deformations: list,
        prev_dir: str | Path = None,
    ) -> Response:
        """
        Relax an input structure after applying a deformation.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure.
        deformations : list of .Deformation objects or 3x3 matrices
            The deformations to apply.
        prev_dir : str or Path or None
            A previous directory to use for copying outputs.
        """
        job_types = ["relax"] + (["static"] if self.static_maker else [])
        jobs = {key: [] for key in job_types}
        output = {key: {"energies": [], "volumes": []} for key in job_types}

        for idef, deformation in enumerate(deformations):
            # deform the structure
            dst = DeformStructureTransformation(deformation=deformation)
            ts = TransformedStructure(structure, transformations=[dst])
            deformed_structure = ts.final_structure

            with contextlib.suppress(Exception):
                # write details of the transformation to the transformations.json file
                # this file will automatically get added to the task document and allow
                # the elastic builder to reconstruct the elastic document; note the ":" is
                # automatically converted to a "." in the filename.
                self.eos_relax_maker.write_additional_data["transformations:json"] = ts

            relax_job = self.eos_relax_maker.make(deformed_structure, prev_dir=prev_dir)
            relax_job.name += f" deformation {idef}"
            jobs["relax"].append(relax_job)

            if self.static_maker:
                static_job = self.static_maker.make(
                    structure=relax_job.output.structure,
                    prev_dir=relax_job.output.dir_name,
                )
                static_job.name += f" {idef}"
                jobs["static"].append(static_job)

            for key in job_types:
                output[key]["energies"].append(jobs[key][idef].output.output.energy)
                output[key]["volumes"].append(jobs[key][idef].output.structure.volume)

        return Response(
            replace=Flow(jobs["relax"] + jobs.get("static", []), output=output)
        )
