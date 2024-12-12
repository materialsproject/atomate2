"""Define the ApproxNEB VASP flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from emmet.core.mobility.migrationgraph import MigrationGraphDoc
from jobflow import Flow, Maker, OnMissing
from pymatgen.io.vasp.outputs import Chgcar

from atomate2.common.jobs.approx_neb import get_images_and_relax
from atomate2.common.flows.approx_neb import ApproxNebFromEndpointsMaker
from atomate2.vasp.jobs.approx_neb import (
    ApproxNebHostRelaxMaker,
    ApproxNebImageRelaxMaker,
    collate_results,
    get_charge_density,
    get_endpoints_and_relax,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from emmet.core.mobility.migrationgraph import MigrationGraph
    from jobflow import Job
    from pymatgen.core import Structure


@dataclass
class ApproxNebMaker(Maker):
    """Run an ApproxNEB workflow.

    Parameters
    ----------
    name : str = "ApproxNEB"
        Name of the workflow
    host_relax_maker : Maker
        Optional, a maker to relax the input host structure.
        Defaults to atomate2.vasp.jobs.approx_neb.ApproxNebHostRelaxMaker
    image_relax_maker : maker
        Required, a maker to relax the ApproxNEB endpoints and images.
        Defaults to atomate2.vasp.jobs.approx_neb.ApproxNebImageRelaxMaker
    selective_dynamics_scheme : "fix_two_atoms" (default) or None
        If "fix_two_atoms", uses the default selective dynamics scheme of ApproxNEB,
        wherein the migrating ion and the ion farthest from it are the only
        ions whose positions can relax.
    use_aeccar : bool = False
        If True, the sum of the host structure AECCAR0 (pseudo-core charge density)
        and AECCAR2 (valence charge density) are used in image pathfinding.
        If False (default), the CHGCAR (valence charge density) is used.
    min_hop_distance : float or bool (default = True)
        If a float, skips any hops where the working ion moves a distance less
        than min_hop_distance.
        If True, min_hop_distance is set to twice the average ionic radius.
        If False, no checks are made.
    """

    name: str = "ApproxNEB"
    host_relax_maker: Maker | None = field(default_factory=ApproxNebHostRelaxMaker)
    image_relax_maker: Maker = field(default_factory=ApproxNebImageRelaxMaker)
    selective_dynamics_scheme: Literal["fix_two_atoms"] | None = "fix_two_atoms"
    use_aeccar: bool = False
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
        inserted_coords_dict: dict
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
            relax_maker=self.image_relax_maker,
        )

        # run pathfinder (and selective dynamics) to get image structure input
        image_relax_jobs = get_images_and_relax(
            working_ion=working_ion,
            ep_output=ep_relax_jobs.output,
            inserted_combo_list=inserted_coords_combo,
            n_images=n_images,
            charge_density_path=prev_dir,
            get_charge_density = self.get_charge_density,
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

    def make_from_migration_graph(
        self,
        migration_graph: MigrationGraph,
        n_images: int = 5,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Make an ApproxNEB flow from a migration graph.

        Parameters
        ----------
        migration_graph: MigrationGraph
            Migration graph containing information about the host structure,
            inserted coordinates, etc.
        n_images: int
            number of images for the ApproxNEB calculation
        selective_dynamics: str
            the scheme for adding selective dynamics to image relaxations

        Returns
        -------
        Flow
            A flow performing AppoxNEB calculations
        """
        inserted_coords_dict, inserted_coords_combo, _ = (
            MigrationGraphDoc.get_distinct_hop_sites(
                migration_graph.inserted_ion_coords, migration_graph.insert_coords_combo
            )
        )
        working_ion = next(
            ele.value
            for ele in migration_graph.working_ion_entry.composition.remove_charges()
        )
        return self.make(
            host_structure=migration_graph.matrix_supercell_structure,
            working_ion=working_ion,
            inserted_coords_dict=inserted_coords_dict,
            inserted_coords_combo=inserted_coords_combo,
            n_images=n_images,
            prev_dir=prev_dir,
        )
    
    def get_charge_density(self,prev_dir : str | Path) -> Chgcar:
        """Get charge density from a prior VASP calculation.

        Parameters
        ----------
        prev_dir : str or Path
            Path to the previous VASP calculation

        Returns
        -------
        pymatgen Chgcar object
        """
        return get_charge_density(prev_dir, use_aeccar=self.use_aeccar)


@dataclass
class ApproxNebSingleHopMaker(ApproxNebFromEndpointsMaker):

    image_relax_maker : Maker = field(
        default_factory=ApproxNebImageRelaxMaker
    )
    use_aeccar: bool = False

    def get_charge_density(self, prev_dir : str | Path) -> Chgcar:
        """Get charge density from a prior VASP calculation.

        Parameters
        ----------
        prev_dir : str or Path
            Path to the previous VASP calculation

        Returns
        -------
        pymatgen Chgcar object
        """
        return get_charge_density(prev_dir, use_aeccar=self.use_aeccar)