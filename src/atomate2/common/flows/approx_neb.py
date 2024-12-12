from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jobflow import Maker, Flow, OnMissing

from atomate2.common.jobs.approx_neb import (
    collate_images_single_hop,
    get_images_and_relax,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Literal, Callable

    from jobflow import Job
    from pymatgen.core import Structure
    from pymatgen.io.common import VolumetricData
    from pymatgen.util.typing import CompositionLike

@dataclass
class ApproxNebFromEndpointsMaker(Maker):

    image_relax_maker : Maker
    selective_dynamics_scheme: Literal["fix_two_atoms"] | None = "fix_two_atoms"
    min_images_per_hop: int | None = 3,
    min_hop_distance: float | bool = True

    def make(
        self,
        working_ion : CompositionLike,
        end_points : list[Structure],
        charge_density_path : str | Path,
        n_images : int = 5,
        prev_dir : str | Path | None = None,
    ):

        jobs : list[Job] = []
        ep_output : dict[str,dict[str,Any]] = {}
        for idx, ep in enumerate(end_points):
            job = self.image_relax_maker.make(ep,prev_dir=prev_dir)
            job.name = f"ApproxNEB relax endpoint {idx}"
            ep_output[str(idx)] = {
                "initial_structure": ep,
                "structure": job.output.structure,
                "energy": job.output.output.energy,
            }
            jobs.append(job)
        
        image_calcs = get_images_and_relax(
            working_ion = working_ion,
            ep_output = ep_output,
            inserted_combo_list = ["0+1"],
            n_images = n_images,
            charge_density_path = charge_density_path,
            get_charge_density = self.get_charge_density,
            relax_maker = self.image_relax_maker,
            selective_dynamics_scheme = self.selective_dynamics_scheme,
            min_hop_distance = self.min_hop_distance,
        )

        collate_job = collate_images_single_hop(
            working_ion = working_ion,
            endpoint_calc_output = ep_output,
            image_calc_output = image_calcs.output,
            min_images_per_hop = self.min_images_per_hop,
        )
        collate_job.config.on_missing_references = OnMissing.NONE
        return Flow(jobs + [image_calcs, collate_job], output=collate_job.output)


    def get_charge_density(self, prev_dir : str | Path) -> VolumetricData:
        """Obtain charge density from a specified path.
        
        Parameters
        -----------
        prev_dir : str or Path
            Path to the calculation containing the previous charge density.
        
        Returns
        -----------
            VolumetricData
                The charge density
        """
        raise NotImplementedError