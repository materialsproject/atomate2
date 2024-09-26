"""Define the ApproxNEB VASP flow."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker, job

from atomate2.common.jobs.electrode import get_charge_density_job
from atomate2.vasp.flows.electrode import ElectrodeInsertionMaker
from atomate2.vasp.jobs.approx_neb import (
    ApproxNEBHostRelaxMaker,
    ApproxNEBImageRelaxMaker,
    get_endpoint_input_structs,
    get_image_input_structures,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class ApproxNEBMaker(Maker):
    """Maker for an ApproxNEB flow."""

    name: str = "ApproxNEB"
    host_relax_maker: Maker = field(default_factory=ApproxNEBHostRelaxMaker)
    image_relax_maker: Maker = field(default_factory=ApproxNEBImageRelaxMaker)

    def make(
        self,
        host_structure: Structure,
        working_ion: str,
        inserted_coords_dict: dict | list,
        inserted_coords_combo: list,
        n_images: int = 5,
        selective_dynamics_scheme: str = "fix_two_atoms",
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

        jobs: list[job] = []

        # assign job to relax host structure
        if self.host_relax_maker:
            host_relax_job = self.host_relax_maker.make(
                host_structure,
            )
            host_relax_job.name = "host_structure_relax"
            jobs += [host_relax_job]
            host_struct_output = host_relax_job.output.structure
            host_relax_dir = host_relax_job.output.dir_name

        # assign jobs to relax endpoint structures
        ep_relax_input = get_endpoint_input_structs(
            host_structure=host_struct_output,
            working_ion=working_ion,
            endpoint_coords_dict=inserted_coords_dict,
            inserted_coords_combo=inserted_coords_combo,
        )
        ep_relax_jobs = {
            key: self.image_relax_maker.make(val) for key, val in ep_relax_input.items()
        }
        jobs += ep_relax_jobs.values()
        ep_jobs_output = {
            ind: getattr(job, "output", None) for ind, job in ep_relax_jobs.items()
        }

        # get charge density of host structure for pathfinder
        host_chgcar_job = get_charge_density_job(
            host_relax_dir, ElectrodeInsertionMaker.get_charge_density
        )
        host_chgcar = host_chgcar_job.output
        jobs.append(host_chgcar_job)

        # run pathfinder (and selective dynamics) to get image structure input
        image_input_structures_dict = get_image_input_structures(
            working_ion=working_ion,
            ep_jobs_output=ep_jobs_output,
            inserted_combo_list=inserted_coords_combo,
            n_images=n_images,
            host_chgcar=host_chgcar,
            selective_dynamics_scheme=selective_dynamics_scheme,
        )

        # make image relaxation jobs
        image_relax_jobs = [
            [self.image_relax_maker.make(image) for image in v]
            for v in image_input_structures_dict.values()
        ]
        jobs.append(image_relax_jobs)

        # TODO: output documents?

        return Flow(jobs)
