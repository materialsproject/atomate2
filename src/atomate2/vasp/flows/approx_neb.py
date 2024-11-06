"""Define the ApproxNEB VASP flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker, OnMissing

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
    """Run an ApproxNEB workflow.

    Parameters
    ----------
    name : str = "ApproxNEB"
        Name of the workflow
    host_relax_maker : Maker
        Optional, a maker to relax the input host structure.
        Defaults to atomate2.vasp.jobs.approx_neb.ApproxNEBHostRelaxMaker
    image_relax_maker : maker
        Required, a maker to relax the ApproxNEB endpoints and images.
        Defaults to atomate2.vasp.jobs.approx_neb.ApproxNEBImageRelaxMaker
    selective_dynamics_scheme : "fix_two_atoms" or None
        If "fix_two_atoms", uses the default selective dynamics scheme of ApproxNEB,
        wherein the migrating ion and the ion farthest from it are the only
        ions whose positions can relax.
    min_hop_distance : float or None (default)
        If a float, skips any hops where the working ion moves a distance less
        than min_hop_distance. This situation can happen when a migration graph
        mistakenly identifies a periodic image outside the computational cell.
    """

    name: str = "ApproxNEB"
    host_relax_maker: Maker | None = field(default_factory=ApproxNEBHostRelaxMaker)
    image_relax_maker: Maker = field(default_factory=ApproxNEBImageRelaxMaker)
    selective_dynamics_scheme: Literal["fix_two_atoms"] | None = None
    min_hop_distance: float | None = None

    def make(
        self,
        host_structure: Structure,
        working_ion: str,
        inserted_coords_dict: dict | list,
        inserted_coords_combo: list,
        n_images: int = 5,
        use_aeccar: bool = False,
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
            host_relax_job.append_name("host structure ", prepend=True)
            jobs += [host_relax_job]
            host_structure = host_relax_job.output.structure
            prev_dir = host_relax_job.output.dir_name

        # assign jobs to relax endpoint structures
        ep_relax_jobs = get_endpoints_and_relax(
            host_structure=host_structure,
            working_ion=working_ion,
            endpoint_coords=inserted_coords_dict,
            inserted_coords_combo=inserted_coords_combo,
            relax_maker=self.image_relax_maker,
        )

        # run pathfinder (and selective dynamics) to get image structure input
        image_relax_jobs = get_images_and_relax(
            working_ion=working_ion,
            ep_output=ep_relax_jobs.output,
            inserted_combo_list=inserted_coords_combo,
            n_images=n_images,
            host_calc_path=prev_dir,
            relax_maker=self.image_relax_maker,
            selective_dynamics_scheme=self.selective_dynamics_scheme,
            min_hop_distance=self.min_hop_distance,
        )

        collect_output = collate_results(
            host_structure,
            working_ion,
            ep_relax_jobs.output,
            image_relax_jobs.output,
        )

        # to permit the flow to succeed even when prior jobs fail
        collect_output.config.on_missing_references = OnMissing.NONE

        return Flow(
            [*jobs, ep_relax_jobs, image_relax_jobs, collect_output],
            output=collect_output.output,
        )
