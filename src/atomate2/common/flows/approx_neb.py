"""Define ApproxNEB flows for all calculators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from emmet.core.mobility.migrationgraph import MigrationGraphDoc
from jobflow import Flow, Maker, OnMissing

from atomate2.common.jobs.approx_neb import (
    collate_images_single_hop,
    collate_results,
    get_endpoints_and_relax,
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
class CommonApproxNebMaker(Maker):
    """Run an ApproxNEB workflow.

    Parameters
    ----------
    name : str = "ApproxNEB"
        Name of the workflow
    host_relax_maker : Maker
        Optional, a maker to relax the input host structure.
    image_relax_maker : Maker
        Required, a maker to relax the ApproxNEB endpoints and images.
    endpoint_relax_maker : Maker or None (default)
        Optional maker to relax the endpoints that could differ from the
        relax maker used on the intermediate images.
        If None, this is set to `image_relax_maker`.
    selective_dynamics_scheme : "fix_two_atoms" (default) or None
        If "fix_two_atoms", uses the default selective dynamics scheme of ApproxNEB,
        wherein the migrating ion and the ion farthest from it are the only
        ions whose positions can relax.
    min_hop_distance : float or bool (default = True)
        If a float, skips any hops where the working ion moves a distance less
        than min_hop_distance.
        If True, min_hop_distance is set to twice the average ionic radius.
        If False, no checks are made.
    """

    name: str = "ApproxNEB"
    host_relax_maker: Maker | None = None
    image_relax_maker: Maker = None
    endpoint_relax_maker: Maker | None = None
    selective_dynamics_scheme: Literal["fix_two_atoms"] | None = "fix_two_atoms"
    min_hop_distance: float | bool = True

    def make(
        self,
        host_structure: Structure,
        working_ion: str,
        inserted_coords_dict: dict | list,
        inserted_coords_combo: list,
        n_images: int = 5,
        min_images_per_hop: int | None = 3,
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
        inserted_coords_dict: dict or list
            a dictionary containing site coords (endpoints) for working ions
            in the simulation cell
        inserted_coords_combo: list
            a list of combo strings "a+b" to designate run calculations between
            endpoints a and b
        n_images: int = 5
            number of images for the ApproxNEB calculation
        min_images_per_hop : int or None, default = 3
            If an int, the minimum number of image calculations per hop that
            must succeed to mark a hop as successfully calculated.
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
            relax_maker=self.endpoint_relax_maker or self.image_relax_maker,
        )

        # run pathfinder (and selective dynamics) to get image structure input
        image_relax_jobs = get_images_and_relax(
            working_ion=working_ion,
            ep_output=ep_relax_jobs.output,
            inserted_combo_list=inserted_coords_combo,
            n_images=n_images,
            charge_density_path=prev_dir,
            get_charge_density=self.get_charge_density,
            relax_maker=self.image_relax_maker,
            selective_dynamics_scheme=self.selective_dynamics_scheme,
            min_hop_distance=self.min_hop_distance,
        )

        collect_output = collate_results(
            host_structure,
            working_ion,
            ep_relax_jobs.output,
            image_relax_jobs.output,
            min_images_per_hop=min_images_per_hop,
        )

        # to permit the flow to succeed even when prior jobs fail
        collect_output.config.on_missing_references = OnMissing.NONE

        return Flow(
            [*jobs, ep_relax_jobs, image_relax_jobs, collect_output],
            output=collect_output.output,
        )

    def make_from_migration_graph_doc(
        self,
        migration_graph_doc: MigrationGraphDoc,
        n_images: int = 5,
        prev_dir: str | Path | None = None,
        atomate_compat_labels: bool = False,
    ) -> Flow:
        """
        Make an ApproxNEB flow from an emmet MigrationGraphDoc.

        Parameters
        ----------
        migration_graph_doc: MigrationGraph
            Migration graph containing information about the host structure,
            inserted coordinates, etc.
        n_images: int
            number of images for the ApproxNEB calculation
        prev_dir: str or .Path or None (default)
            A previous calculation directory to copy outputs from.
        atomate_compat_labels : bool = False
            Whether to use atomate style labeling of the endpoints (True)
            or the original labels from the MigrationGraphDoc (False, default)

        Returns
        -------
        Flow
            A flow performing AppoxNEB calculations
        """
        inserted_coords, inserted_coords_combo, mapping = (
            MigrationGraphDoc.get_distinct_hop_sites(
                migration_graph_doc.inserted_ion_coords,
                migration_graph_doc.insert_coords_combo,
            )
        )
        if not atomate_compat_labels:
            inserted_coords_combo = [mapping[k] for k in inserted_coords_combo]
            site_idx_map = {}
            for k, v in mapping.items():
                start_idx_new, end_idx_new = k.split("+")
                start_idx_old, end_idx_old = v.split("+")
                site_idx_map.update(
                    {start_idx_new: start_idx_old, end_idx_new: end_idx_old}
                )
            inserted_coords = {
                site_idx_map[str(idx)]: coords
                for idx, coords in enumerate(inserted_coords)
            }

        composition = migration_graph_doc.working_ion_entry.composition.remove_charges()
        working_ion = next(ele.value for ele in composition)

        return self.make(
            host_structure=migration_graph_doc.matrix_supercell_structure,
            working_ion=working_ion,
            inserted_coords_dict=inserted_coords,
            inserted_coords_combo=inserted_coords_combo,
            n_images=n_images,
            prev_dir=prev_dir,
        )

    def get_charge_density(self, *args, **kwargs) -> VolumetricData:
        """Get charge density, to be implemented in subclasses.

        Returns
        -------
        pymatgen VolumetricData
        """
        raise NotImplementedError


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
    name: str = "ApproxNEB single hop from endpoints maker"
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
            endpoint_calc_output=[ep_output[str(idx)] for idx in range(2)],
            image_calc_output=image_calcs.output["0+1"],
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
