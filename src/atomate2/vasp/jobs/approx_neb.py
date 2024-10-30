"""Define ApproxNEB jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from emmet.core.neb import NebMethod
from jobflow import Flow, Maker, Response, job
from monty.os.path import zpath
from pymatgen.analysis.diffusion.neb.pathfinder import ChgcarPotential, NEBPathfinder
from pymatgen.io.vasp.outputs import Chgcar

from atomate2.common.schemas.neb import NebPathwayResult, NebResult
from atomate2.utils.path import strip_hostname
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import RelaxMaker
from atomate2.vasp.run import JobType
from atomate2.vasp.sets.approx_neb import ApproxNEBSetGenerator

if TYPE_CHECKING:
    from typing import Literal

    from pymatgen.core import Structure
    from pymatgen.util.typing import CompositionLike

    from atomate2.vasp.jobs.base import BaseVaspMaker
    from atomate2.vasp.sets.base import VaspInputGenerator


@dataclass
class ApproxNEBHostRelaxMaker(DoubleRelaxMaker):
    """Maker to perform a double relaxation on an ApproxNEB host structure."""

    name: str = "ApproxNEB host relax"
    relax_maker1: BaseVaspMaker | None = field(
        default_factory=lambda: RelaxMaker(input_set_generator=ApproxNEBSetGenerator())
    )
    relax_maker2: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(input_set_generator=ApproxNEBSetGenerator())
    )


@dataclass
class ApproxNEBImageRelaxMaker(RelaxMaker):
    """
    Maker to perform a double relaxation on an ApproxNEB endpoint/image structure.

    Very important here - we are doing a double relaxation in the atomate style,
    where one job maps to two VASP calculations.
    """

    name: str = "ApproxNEB image relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: ApproxNEBSetGenerator(set_type="image")
    )
    run_vasp_kwargs: dict = field(
        default_factory=lambda: {
            "job_type": JobType.DOUBLE_RELAXATION,
        }
    )


@job
def get_endpoints_and_relax(
    host_structure: Structure,
    working_ion: CompositionLike,
    endpoint_coords: dict,
    inserted_coords_combo: list[str],
    relax_maker: Maker,
) -> Response:
    """
    Get and relax endpoint structures.

    Parameters
    ----------
    host_structure : pymatgen Structure
        The structure in which to insert a working ion
    working_ion : pymatgen CompositionLike (string, Element, or Species)
        The name of the element to insert
    endpoint_coords : dict
        Dict of endpoint index to the coordinates where the
        working ion will be inserted
    inserted_coords_combo : list
        List of hops indicated by < endpoint index 1 > + < endpoint index 2 >
    relax_maker : Maker
        Maker to relax the endpoints

    Returns
    -------
    Response:
        The flow relaxes all required endpoints, but the output of this flow
        is a dict containing the energies and relaxed structures of the
        endpoints, with the endpoint indices as keys.
    """
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

    ep_relax_output = {}
    ep_relax_jobs = []
    for ep_index, ep_coords in endpoint_coords.items():
        if int(ep_index) in ep_distinct:
            ep_inserted_struct = host_structure.copy()
            ep_inserted_struct.insert(0, working_ion, ep_coords)

            relax_job = relax_maker.make(ep_inserted_struct)
            relax_job.append_name(f" endpoint {ep_index}")
            ep_relax_jobs.append(relax_job)
            ep_relax_output[ep_index] = {
                "energy": relax_job.output.output.energy,
                "structure": relax_job.output.structure,
            }

    flow = Flow(ep_relax_jobs, output=ep_relax_output)

    return Response(replace=flow)


@job
def get_images_and_relax(
    working_ion: str,
    ep_output: dict[str, dict],
    inserted_combo_list: list[str],
    n_images: int | list[int],
    host_calc_path: str | Path,
    relax_maker: Maker,
    selective_dynamics_scheme: Literal["fix_two_atoms"] | None = "fix_two_atoms",
    use_aeccar: bool = False,
) -> Response:
    """
    Get and relax image input structures.

    Parameters
    ----------
    ep_output : dict
        Output of get_endpoints_and_relax
    inserted_combo_list : list
        List of hops to perform ApproxNEB on
    n_images : int
        The number of images to use along each hop.
        If an int, the number to use for every hop.
        If a list of ints, the number of images to use in that hop,
    host_calc_path: str | Path
        The path to the calculation of the host_structure
    relax_maker : Maker
        Maker to relax images
    selective_dynamics_scheme : "fix_two_atoms" or None
        Whether to use a pre-defined selective dynamics scheme or relax
        all ionic positions in the images.
    use_aeccar : bool = False
        If True, the sum of the host structure AECCAR0 (pseudo-core charge density)
        and AECCAR2 (valence charge density) are used in image pathfinding.
        If False (default), the CHGCAR (valence charge density) is used.

    Returns
    -------
    Response : a series of image relaxations with output containing the
        relaxed structures and energies of each image in each hop in a dict:
        {
            hop_index : [
                {"energy": energy of image 1, "structure": relaxed image structure 1},
                ...
            ]
        }
    """
    # remove failed output first
    ep_structures = {
        k: calc["structure"]
        for k, calc in ep_output.items()
        if calc["structure"] is not None
    }

    host_chgcar = get_charge_density(host_calc_path, use_aeccar=use_aeccar)

    image_relax_jobs = []
    image_relax_output: dict[str, list] = {}

    if isinstance(n_images, int):
        n_images = [n_images for _ in inserted_combo_list]

    for hop_idx, combo in enumerate(inserted_combo_list):
        ini_ind, fin_ind = combo.split("+")

        if not all(ep_structures.get(idx) for idx in [ini_ind, fin_ind]):
            # cannot proceed with this hop calculation
            continue

        # potential place for uuid logic if depth first is desirable
        pathfinder_output = get_pathfinder_results(
            ep_structures[ini_ind],
            ep_structures[fin_ind],
            working_ion,
            n_images[hop_idx],
            host_chgcar,
        )
        images_list = pathfinder_output["images"]

        # add selective dynamics to structure
        if selective_dynamics_scheme == "fix_two_atoms":
            images_list = [
                add_selective_dynamics_two_fixed_sites(
                    image,
                    pathfinder_output["mobile_site_index"],
                    working_ion,
                )
                for image in pathfinder_output["images"]
            ]

        elif selective_dynamics_scheme is not None:
            raise ValueError(f"Unknown {selective_dynamics_scheme=}.")

        image_relax_output[combo] = []
        for image_idx, image in enumerate(images_list):
            relax_job = relax_maker.make(image)
            relax_job.append_name(f" hop {combo} image {image_idx+1}")
            image_relax_jobs.append(relax_job)
            image_relax_output[combo].append(
                {
                    "structure": relax_job.output.structure,
                    "energy": relax_job.output.output.energy,
                }
            )

    relax_flow = Flow(image_relax_jobs, output=image_relax_output)

    return Response(replace=relax_flow)


def get_pathfinder_results(
    pf_struct_ini: Structure,
    pf_struct_fin: Structure,
    working_ion: CompositionLike,
    n_images: int,
    host_chgcar: Chgcar,
) -> dict:
    """
    Get interpolated images from the pathfinder algorithm.

    Parameters
    ----------
    pf_struct_ini : pymatgen Structure
        First NEB endpoint structure
    pf_struct_fin : pymatgen Structure
        Second/final endpoint structure
    working_ion : CompositionLike
        The element which migrates
    n_images : int
        The number of images to be created along the path
    host_chgcar : pymatgen.io.vasp.outputs Chgcar
        The charge density of the host structure

    Returns
    ---------
    dict containing the images along the path and the mobile_site index
    """
    ini_wi_ind = get_working_ion_index(pf_struct_ini, working_ion)
    fin_wi_ind = get_working_ion_index(pf_struct_fin, working_ion)

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
        relax_sites=[ini_wi_ind],
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


def add_selective_dynamics_two_fixed_sites(
    structure: Structure,
    fixed_index: int,
    fixed_species_name: CompositionLike,
) -> Structure:
    """Add selective dynamics to input structure.
    
    Parameters
    ----------
    structure : pymatgen Structure
        Structure to add selective dynamics tags to
    fixed_index : int
        Index of the site to add selective dynamics to
    fixed_species_name : CompositionLike
        The expected element at the indexed site - used
        to check validity of fixed_index

    Returns
    ---------
    Copy of the input structure with selective dynamics tags
    """
    if structure[fixed_index].specie.name != fixed_species_name:
        raise ValueError(
            f"The chosen fixed atom at index {fixed_index} is not a "
            f"{fixed_species_name} atom"
        )

    # removes site properties to avoid error
    for p in structure.site_properties:
        structure.remove_site_property(p)

    # add selectives dynamics with fix_two_atoms scheme
    # fix the atom at fixed_index and the furthest atom in the structure
    ref_site = structure.sites[fixed_index]
    distances = [site.distance(ref_site) for site in structure.sites]
    farthest_index = distances.index(max(distances))
    sd_array = [
        [False, False, False]
        if idx in {fixed_index, farthest_index}
        else [True, True, True]
        for idx in range(structure.num_sites)
    ]
    structure.add_site_property("selective_dynamics", sd_array)

    return structure


def get_working_ion_index(structure: Structure, working_ion: CompositionLike) -> int | None:
    """Get the index of the working ion in a structure.
    
    Parameters
    ----------
    structure : pymatgen Structure
    working_ion : CompositionLike
        Name of the element to identify

    Returns
    ----------
    int - the first site in the structure containing the working ion
    None - no sites in the structure contain the working ion
    """
    for ind, site in enumerate(structure):
        if site.species_string == working_ion:
            # assume that only the lowest indexed working ion is mobile
            return ind
    return None


def get_charge_density(prev_dir: str | Path, use_aeccar: bool = False) -> Chgcar:
    """Get charge density from a prior VASP calculation.
    
    Parameters
    ----------
    prev_dir : str or Path
        Path to the previous VASP calculation
    use_aeccar : bool = False
        True: use AECCAR0 and AECCAR2 (pseudo-all electron charge density)
        rather than CHGCAR (valence electron density only - False)

    Returns
    ----------
    pymatgen Chgcar object
    """
    prev_dir = Path(strip_hostname(prev_dir))
    if use_aeccar:
        aeccar0 = Chgcar.from_file(zpath(str(prev_dir / "AECCAR0")))
        aeccar2 = Chgcar.from_file(zpath(str(prev_dir / "AECCAR2")))
        return aeccar0 + aeccar2
    return Chgcar.from_file(zpath(str(prev_dir / "CHGCAR")))


@job
def collate_results(
    host_structure: Structure,
    working_ion: CompositionLike,
    endpoint_calc_output: dict,
    image_calc_output: dict[str, list],
) -> NebPathwayResult:
    """Collect output from an ApproxNEB workflow.
    
    Parameters
    ----------
    host_structure : pymatgen Structure
    working_ion : CompositionLike
        Used in combination with host_structure to estimate
        the linear distance traversed in a given ApproxNEB hop
    endpoint_calc_output : dict
        Output of get_endpoints_and_relax
    image_calc_output : dict[str,list]
        Output of get_images_and_relax

    Returns
    ----------
    NebPathwayResult - a collection of NEB calculations and analysis
    """
    hop_dict = {}
    hop_dist = {}
    for combo_name, images in image_calc_output.items():
        endpoint_calcs = [endpoint_calc_output[idx] for idx in combo_name.split("+")]
        hop = [endpoint_calcs[0], *images, endpoint_calcs[1]]
        hop_dict[combo_name] = NebResult(
            images=[calc["structure"] for calc in hop if calc["structure"] is not None],
            energies=[calc["energy"] for calc in hop if calc["energy"] is not None],
            ionic_steps=None,  # [calc.output.ionic_steps for calc in hop],
            method=NebMethod.APPROX,
        )

        working_ion_sites = [
            [
                site
                for site in endpoint_calcs[ep_idx]["structure"]
                if site.species_string == working_ion
            ]
            for ep_idx in range(2)
        ]
        hop_dist[combo_name] = max(
            np.linalg.norm(site_a.coords - site_b.coords)
            for site_a in working_ion_sites[0]
            for site_b in working_ion_sites[1]
        )

    return NebPathwayResult(
        hops=hop_dict, host_structure=host_structure, hop_distances=hop_dist
    )
