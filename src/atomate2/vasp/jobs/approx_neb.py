"""Define ApproxNEB jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import job
from pymatgen.analysis.diffusion.neb.pathfinder import ChgcarPotential, NEBPathfinder

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import RelaxMaker
from atomate2.vasp.sets.approxneb import ApproxNEBSetGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Chgcar

    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class ApproxNEBHostRelaxMaker(DoubleRelaxMaker):
    """Maker to perform a double relaxation on an ApproxNEB host structure."""

    name: str = "approxneb_host_relax"
    relax_maker1: BaseVaspMaker | None = field(
        default_factory=lambda: RelaxMaker(input_set_generator=ApproxNEBSetGenerator())
    )
    relax_maker2: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(input_set_generator=ApproxNEBSetGenerator())
    )


@dataclass
class ApproxNEBImageRelaxMaker(DoubleRelaxMaker):
    """Maker to perform a double relaxation on an ApproxNEB endpoint/image structure."""

    name: str = "approxneb_image_relax"
    relax_maker1: BaseVaspMaker | None = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=ApproxNEBSetGenerator(set_type="image")
        )
    )
    relax_maker2: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=ApproxNEBSetGenerator(set_type="image")
        )
    )


@job
def get_endpoint_input_structs(
    host_structure: Structure,
    working_ion: str,
    endpoint_coords_dict: dict,
    inserted_coords_combo: list,
) -> dict:
    """Get the input structures for endpoint relaxations."""
    ep_distinct = []
    for one_combo in inserted_coords_combo:
        try:
            combo = one_combo.split("+")
            ini, fin = [int(combo[0]), int(combo[1])]
        except ValueError as error:
            raise ValueError(
                f"{one_combo} inserted_coords_combo input is incorrect"
            ) from error
        ep_distinct.extend([ini, fin])
    ep_distinct = list(set(ep_distinct))

    ep_relax_input_structs = {}
    for ep_index, ep_coords in endpoint_coords_dict.items():
        if int(ep_index) in ep_distinct:
            ep_inserted_struct = host_structure.copy()
            ep_inserted_struct.insert(0, working_ion, ep_coords)
            ep_relax_input_structs[ep_index] = ep_inserted_struct

    return ep_relax_input_structs


@job
def get_image_input_structures(
    working_ion: str,
    ep_jobs_output: dict,
    inserted_combo_list: list,
    n_images: int,
    host_chgcar: Chgcar,
    selective_dynamics_scheme: str | None = "fix_two_atoms",
) -> dict:
    """Get image relaxation jobs."""
    image_jobs_dict = {}
    for combo in inserted_combo_list:
        ini_ind, fin_ind = map(int, combo.split("+"))
        # potential place for uuid logic if depth first si desirable
        pf_struct_ini = ep_jobs_output[ini_ind].structure
        pf_struct_fin = ep_jobs_output[fin_ind].structure
        pathfinder_output = get_pathfiner_results(
            pf_struct_ini, pf_struct_fin, working_ion, n_images, host_chgcar
        )
        images_list = pathfinder_output.output["images"]

        # add selective dynamics to structure
        if selective_dynamics_scheme:
            images_list = [
                add_selective_dynamics(
                    image,
                    pathfinder_output.output["mobile_site_index"],
                    selective_dynamics_scheme,
                ).output
                for image in pathfinder_output.output["images"]
            ]

        image_jobs_dict[combo] = images_list

    return image_jobs_dict


@job
def get_pathfiner_results(
    pf_struct_ini: Structure,
    pf_struct_fin: Structure,
    working_ion: str,
    n_images: int,
    host_chgcar: Chgcar,
) -> dict:
    """Get interpolated images from the pathfinder algorithm."""
    _, ini_wi_ind = _get_wi_coords_ind(pf_struct_ini, working_ion).output
    _, fin_wi_ind = _get_wi_coords_ind(pf_struct_fin, working_ion).output

    if ini_wi_ind != fin_wi_ind:
        raise ValueError(
            "Inserted site indexes of end point structures must match for NEBPathfinder"
        )

    # get potential gradient v from host chgcar
    v_chgcar = ChgcarPotential(host_chgcar)
    host_v = v_chgcar.get_v()

    # perform pathfinding and get images
    neb_pf = NEBPathfinder(
        pf_struct_ini,
        pf_struct_fin,
        relax_sites=ini_wi_ind,
        v=host_v,
        n_images=n_images + 1,
    )
    # note NEBPathfinder currently returns n_images+1 images (rather than n_images)
    # and the first and last images generated are very similar to the end points
    # provided so they are discarded

    return {
        "images": neb_pf.images[1:-1],
        "mobile_site_index": ini_wi_ind,
    }


@job
def add_selective_dynamics(
    structure: Structure,
    fixed_index: int,
    fixed_specie_name: str,
    selective_dynamics_scheme: str,
) -> Structure:
    """Add selective dynamics to input structure according to scheme."""
    if selective_dynamics_scheme not in ["fix_two_atoms"]:
        raise ValueError(
            "selective_dynamics_scheme does match any supported schemes, "
            "check input value"
        )

    if structure[fixed_index].specie.name != fixed_specie_name:
        raise ValueError(
            f"The chosen fixed atom at index {fixed_index} is not a "
            f"{fixed_specie_name} atom"
        )

    # removes site properties to avoid error
    if structure.site_properties != {}:
        for p in structure.site_properties:
            structure.remove_site_property(p)

    # add selectives dynamics with fix_two_atoms scheme
    # fix the atom at fixed_index and the furthest atom in the structure
    if selective_dynamics_scheme == "fix_two_atoms":
        sd_structure = structure.copy()
        sd_array = [[True, True, True] for i in range(sd_structure.num_sites)]
        sd_array[fixed_index] = [False, False, False]
        ref_site = sd_structure.sites[fixed_index]
        distances = [site.distance(ref_site) for site in sd_structure.sites]
        farthest_index = distances.index(max(distances))
        sd_array[farthest_index] = [False, False, False]
        sd_structure.add_site_property("selective_dynamics", sd_array)

    return sd_structure


@job
def _get_wi_coords_ind(structure: Structure, working_ion: str) -> tuple[list, int]:
    coords = []
    indices = []
    for ind, site in enumerate(structure):
        if site.speice.name == working_ion:
            coords.append(site.frac_coords)
            indices.append(ind)

    # assume that only the lowest indexed working ion is mobile
    return coords[0], indices[0]
