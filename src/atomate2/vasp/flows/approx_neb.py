"""Define the ApproxNEB VASP flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.common.jobs.electrode import get_charge_density_job
from atomate2.vasp.flows.electrode import ElectrodeInsertionMaker
from atomate2.vasp.jobs.approx_neb import (
    ApproxNEBHostRelaxMaker,
    ApproxNEBImageRelaxMaker,
    collate_results,
    get_endpoint_input_structs,
    get_image_input_structures,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from jobflow import Job
    from pymatgen.core import Structure


@dataclass
class ApproxNEBMaker(Maker):
    """Maker for an ApproxNEB flow."""

    name: str = "ApproxNEB"
    host_relax_maker: Maker = field(default_factory=ApproxNEBHostRelaxMaker)
    image_relax_maker: Maker = field(default_factory=ApproxNEBImageRelaxMaker)
    selective_dynamics_scheme: Literal["fix_two_atoms"] | None = None

    def make(
        self,
        host_structure: Structure,
        working_ion: str,
        inserted_coords_dict: dict | list,
        inserted_coords_combo: list,
        n_images: int = 5,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Make an ApproxNEB flow.

        Parameters
        ----------
        host_structure: Structure
            the (supercell) structure of the empty host with no working ion
        working_ion: str
            the mobile species in ApproxNEB
        inserted_coords_dict: dict
            a dictionary containing site coords (endpoints) for working ions
            in the simulation cell
        inserted_coords_combo: list
            a list of combo strings "a+b" to designate run calculations between
            endpoints a and b
        n_image: int
            number of images for the ApproxNEB calculation
        selective_dynamics: str
            the scheme for adding selective dynamics to image relaxations

        Returns
        -------
        Flow
            A flow performing AppoxNEB calculations
        """
        # compatibility with legacy input (list)
        if isinstance(inserted_coords_dict, list):
            inserted_coords_dict = dict(enumerate(inserted_coords_dict))

        jobs: list[Job] = []

        # assign job to relax host structure
        if self.host_relax_maker:
            host_relax_job = self.host_relax_maker.make(
                host_structure, prev_dir=prev_dir
            )
            host_relax_job.name = "ApproxNEB Relax Host Structure"
            jobs += [host_relax_job]
            host_structure = host_relax_job.output.structure
            prev_dir = host_relax_job.output.dir_name

        # assign jobs to relax endpoint structures
        ep_relax_input = get_endpoint_input_structs(
            host_structure=host_structure,
            working_ion=working_ion,
            endpoint_coords_dict=inserted_coords_dict,
            inserted_coords_combo=inserted_coords_combo,
        )
        ep_relax_jobs = {
            key: self.image_relax_maker.make(val) for key, val in ep_relax_input.items()
        }
        jobs += list(ep_relax_jobs.values())
        ep_structures = {}
        for idx, job in ep_relax_jobs.items():
            if (job_output := getattr(job, "output", None)) is not None:
                ep_structures[idx] = job_output.structure

        # get charge density of host structure for pathfinder
        host_chgcar_job = get_charge_density_job(
            prev_dir, ElectrodeInsertionMaker.get_charge_density
        )
        host_chgcar = host_chgcar_job.output
        jobs.append(host_chgcar_job)

        # run pathfinder (and selective dynamics) to get image structure input
        image_input_structures = get_image_input_structures(
            working_ion=working_ion,
            ep_structures=ep_structures,
            inserted_combo_list=inserted_coords_combo,
            n_images=n_images,
            host_chgcar=host_chgcar,
            selective_dynamics_scheme=self.selective_dynamics_scheme,
        ).output

        # make image relaxation jobs
        relax_image_jobs = {
            idx: [self.image_relax_maker.make(image) for image in images]
            for idx, images in image_input_structures.items()
        }
        for image_calcs in relax_image_jobs.values():
            jobs += image_calcs

        collect_output = collate_results(
            {idx: calc.output for idx, calc in ep_relax_jobs.items()},
            {
                idx: [calc.output for calc in images]
                for idx, images in relax_image_jobs.items()
            },
        )
        jobs += collect_output

        return Flow(jobs, output=collect_output.output)
