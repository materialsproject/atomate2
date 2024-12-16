"""Define ApproxNEB flows for all calculators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jobflow import Flow, Maker, OnMissing

from atomate2.common.jobs.approx_neb import (
    collate_images_single_hop,
    get_images_and_relax,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Literal

    from jobflow import Job
    from pymatgen.core import Structure
    from pymatgen.io.common import VolumetricData
    from pymatgen.util.typing import CompositionLike


@dataclass
class ApproxNebFromEndpointsMaker(Maker):
    """
    Create an ApproxNEB flow from specified endpoints.

    image_relax_maker : Maker
        Maker to relax both endpoints and images
    selective_dynamics_scheme : "fix_two_atoms" (default) or None
        If "fix_two_atoms", uses the default selective dynamics scheme of ApproxNEB,
        wherein the migrating ion and the ion farthest from it are the only
        ions whose positions can relax.
    min_images_per_hop : int or None
        If an int, the minimum number of image calculations per hop that
        must succeed to mark a hop as successfully calculated.
    min_hop_distance : float or bool (default = True)
        If a float, skips any hops where the working ion moves a distance less
        than min_hop_distance.
        If True, min_hop_distance is set to twice the average ionic radius.
        If False, no checks are made.
    """

    image_relax_maker: Maker
    selective_dynamics_scheme: Literal["fix_two_atoms"] | None = "fix_two_atoms"
    min_images_per_hop: int | None = 3
    min_hop_distance: float | bool = True

    def make(
        self,
        working_ion: CompositionLike,
        end_point_structures: list[Structure],
        charge_density_path: str | Path,
        n_images: int = 5,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Run an ApproxNEB flow for a single hop.

        working_ion : CompositionLike
            The element which migrates.
        end_point_structures : list of pymatgen .Structure
            The two endpoint structures
        charge_density_path: str or .Path
            Path to the directory containing the charge density file(s).
        n_images: int = 5
            number of images for the ApproxNEB calculation
        prev_dir : str, .Path, or None
            If not None, the path to a previous calculation to
            copy outputs from.

        Returns
        -------
        Flow
            A flow performing an AppoxNEB calculation for a single hop.
        """
        ep_jobs: list[Job] = []
        ep_output: dict[str, dict[str, Any]] = {}
        for idx, ep in enumerate(end_point_structures):
            job = self.image_relax_maker.make(ep, prev_dir=prev_dir)
            job.name = f"ApproxNEB relax endpoint {idx}"
            ep_output[str(idx)] = {
                "initial_structure": ep,
                "structure": job.output.structure,
                "energy": job.output.output.energy,
            }
            ep_jobs.append(job)

        image_calcs = get_images_and_relax(
            working_ion=working_ion,
            ep_output=ep_output,
            inserted_combo_list=["0+1"],
            n_images=n_images,
            charge_density_path=charge_density_path,
            get_charge_density=self.get_charge_density,
            relax_maker=self.image_relax_maker,
            selective_dynamics_scheme=self.selective_dynamics_scheme,
            min_hop_distance=self.min_hop_distance,
        )

        collate_job = collate_images_single_hop(
            working_ion=working_ion,
            endpoint_calc_output=ep_output,
            image_calc_output=image_calcs.output,
            min_images_per_hop=self.min_images_per_hop,
        )
        collate_job.config.on_missing_references = OnMissing.NONE
        return Flow([*ep_jobs, image_calcs, collate_job], output=collate_job.output)

    def get_charge_density(self, prev_dir: str | Path) -> VolumetricData:
        """Obtain charge density from a specified path.

        Parameters
        ----------
        prev_dir : str or Path
            Path to the calculation containing the previous charge density.

        Returns
        -------
            VolumetricData
                The charge density
        """
        raise NotImplementedError
