"""Define the ApproxNEB VASP flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.vasp.jobs.approx_neb import (
    ApproxNEBHostRelaxMaker,
    ApproxNEBImageRelaxMaker,
    collate_results,
    get_endpoints_and_relax,
    get_images_and_relax,
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
    host_relax_maker: Maker | None = field(default_factory=ApproxNEBHostRelaxMaker)
    image_relax_maker: Maker = field(default_factory=ApproxNEBImageRelaxMaker)
    selective_dynamics_scheme: Literal["fix_two_atoms"] | None = None

    def make(
        self,
        host_structure: Structure,
        working_ion: str,
        inserted_coords_dict: dict | list,
        inserted_coords_combo: list,
        n_images: int = 5,
        use_aeccar : bool = False,
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
        ep_relax_jobs = get_endpoints_and_relax(
            host_structure=host_structure,
            working_ion=working_ion,
            endpoint_coords_dict=inserted_coords_dict,
            inserted_coords_combo=inserted_coords_combo,
            relax_maker=self.image_relax_maker,
        )

        # run pathfinder (and selective dynamics) to get image structure input
        image_relax_jobs = get_images_and_relax(
            working_ion=working_ion,
            ep_structures=ep_relax_jobs.output,
            inserted_combo_list=inserted_coords_combo,
            n_images=n_images,
            host_calc_path=prev_dir,
            relax_maker=self.image_relax_maker,
            selective_dynamics_scheme=self.selective_dynamics_scheme,
        )

        collect_output = collate_results(
            ep_relax_jobs.output,
            image_relax_jobs.output,
        )

        return Flow(
            [*jobs, ep_relax_jobs, image_relax_jobs, collect_output],
            output=collect_output.output,
        )