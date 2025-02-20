"""Define common utility jobs needed for ApproxNEB flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from emmet.core.neb import NebMethod
from emmet.core.vasp.task_valid import TaskState
from jobflow import Flow, Response, job
from pymatgen.analysis.diffusion.neb.pathfinder import ChgcarPotential, NEBPathfinder
from pymatgen.core import Element

from emmet.core.neb import HopFailureReason, NebPathwayResult, NebResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from typing import Any, Literal

    from jobflow import Maker
    from pymatgen.core import Structure
    from pymatgen.io.common import VolumetricData
    from pymatgen.util.typing import CompositionLike

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

    # In principle, it makes sense to use the magmoms from the host
    # structure to initialize the host + inserted working ion calcs
    # In practice, this throws unfixable "Bravais" errors in VASP
    # (the actual reciprocal lattice does not have the expected
    # symmetry of the ideal reciprocal lattice.)
    if host_structure.site_properties.get("magmom") is not None:
        host_structure.remove_site_property("magmom")

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
                "initial_structure": relax_job.output.input.structure,
            }

    flow = Flow(ep_relax_jobs, output=ep_relax_output)

    return Response(replace=flow)


@job
def collate_results(
    host_structure: Structure,
    working_ion: CompositionLike,
    endpoint_calc_output: dict,
    image_calc_output: dict[str, list],
    min_images_per_hop: int | None = None,
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
    min_images_per_hop : int or None (default)
        If an integer, the minimum number of successful image calculations
        to mark a calculation as successful

    Returns
    -------
    NebPathwayResult - a collection of NEB calculations and analysis
    """
    hop_dict = {}
    hop_dist = {}

    endpoint_idxs = []
    for combo_name in image_calc_output:
        endpoint_idxs.extend(combo_name.split("+"))
    endpoint_idxs = sorted(set(endpoint_idxs))

    for combo_name, entry in image_calc_output.items():
        metadata = {}
        task_state = TaskState.SUCCESS
        images = entry
        if all(isinstance(v, str) for v in entry):
            # hop calculation failed
            metadata = {"failure_reasons": entry}
            task_state = TaskState.FAILED
            if HopFailureReason.ENDPOINT.value in entry:
                # Cannot populate any NEB fields, skip entirely
                hop_dict[combo_name] = NebResult(state=task_state, metadata=metadata)
                continue
            images = []

        endpoint_calcs = [endpoint_calc_output[idx] for idx in combo_name.split("+")]
        hop = [endpoint_calcs[0], *images, endpoint_calcs[1]]

        if min_images_per_hop is not None and task_state == TaskState.SUCCESS:
            num_success_calcs = len(
                [calc for calc in hop if calc["structure"] is not None]
            )
            if num_success_calcs < min_images_per_hop:
                task_state = TaskState.FAILED
                if "failure_reasons" not in metadata:
                    metadata["failure_reasons"] = []
                metadata["failure_reasons"].append(HopFailureReason.MIN_IMAGE.value)

        hop_dict[combo_name] = NebResult(
            images=[calc["structure"] for calc in hop if calc["structure"] is not None],
            initial_images=[
                calc["initial_structure"]
                for calc in hop
                if calc["initial_structure"] is not None
            ],
            energies=[calc["energy"] for calc in hop if calc["energy"] is not None],
            method=NebMethod.APPROX,
            state=task_state,
            metadata=metadata if len(metadata) > 0 else None,
            endpoint_indices=combo_name.split("+"),
        )

        hop_dist[combo_name] = get_hop_distance_from_endpoints(
            [ep_calc["structure"] for ep_calc in endpoint_calcs], working_ion
        )

    return NebPathwayResult(
        hops=hop_dict,
        host_structure=host_structure,
        hop_distances=hop_dist,
        initial_endpoints={
            idx: entry["initial_structure"]
            for idx, entry in endpoint_calc_output.items()
        },
        relaxed_endpoints={
            idx: entry["structure"] for idx, entry in endpoint_calc_output.items()
        },
    )


@job
def get_images_and_relax(
    working_ion: str,
    ep_output: dict[str, dict],
    inserted_combo_list: list[str],
    n_images: int | list[int],
    charge_density_path: str | Path,
    get_charge_density: Callable[[str | Path], VolumetricData],
    relax_maker: Maker,
    selective_dynamics_scheme: Literal["fix_two_atoms"] | None = "fix_two_atoms",
    min_hop_distance: float | bool = True,
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
    charge_density_path: str | Path
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
    min_hop_distance : float or bool (default = True)
        If a float, skips any hops where the working ion moves a distance less
        than min_hop_distance.
        If True, min_hop_distance is set to twice the average ionic radius.
        If False, no checks are made.

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
    # remove failed output and strip magmoms to avoid "Bravais" errors
    ep_structures = {}
    for k, calc in ep_output.items():
        if calc["structure"] is None:
            continue
        ep_struct = calc["structure"].copy()
        if ep_struct.site_properties.get("magmom") is not None:
            ep_struct.remove_site_property("magmom")
        ep_structures[k] = ep_struct

    host_chgcar = get_charge_density(charge_density_path)

    image_relax_jobs = []
    image_relax_output: dict[str, list] = {}

    if isinstance(n_images, int):
        n_images = [n_images for _ in inserted_combo_list]

    if isinstance(min_hop_distance, bool) and min_hop_distance:
        _wion = Element(str(working_ion))
        if (ionic_radius := getattr(_wion, "average_ionic_radius", None)) is not None:
            min_hop_distance = 2 * ionic_radius
        elif (atomic_radius := getattr(_wion, "atomic_radius", None)) is not None:
            # all elements have an atomic radius in pymatgen
            min_hop_distance = atomic_radius

    for hop_idx, combo in enumerate(inserted_combo_list):
        ini_ind, fin_ind = combo.split("+")

        # See if we can proceed with this hop calculation:
        skip_reasons = []
        if not all(ep_structures.get(idx) for idx in [ini_ind, fin_ind]):
            # At least one endpoint calculation failed
            skip_reasons.append(HopFailureReason.ENDPOINT)
        if (
            isinstance(min_hop_distance, float)
            and get_hop_distance_from_endpoints(
                [ep_structures[ini_ind], ep_structures[fin_ind]], working_ion
            )
            < min_hop_distance
        ):
            # The working ion hop distance is below the specified threshold
            skip_reasons.append(HopFailureReason.MIN_DIST)

        if len(skip_reasons) > 0:
            image_relax_output[combo] = [reason.value for reason in skip_reasons]
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
            relax_job.append_name(f" hop {combo} image {image_idx + 1}")
            image_relax_jobs.append(relax_job)
            image_relax_output[combo].append(
                {
                    "energy": relax_job.output.output.energy,
                    "structure": relax_job.output.structure,
                    "initial_structure": relax_job.output.input.structure,
                }
            )

    relax_flow = Flow(image_relax_jobs, output=image_relax_output)

    return Response(replace=relax_flow)


def get_pathfinder_results(
    pf_struct_ini: Structure,
    pf_struct_fin: Structure,
    working_ion: CompositionLike,
    n_images: int,
    host_charge_density: VolumetricData,
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
    host_charge_density : VolumetricData, like pymatgen.io.vasp.outputs Chgcar
        The charge density of the host structure

    Returns
    -------
    dict containing the images along the path and the mobile_site index
    """
    ini_wi_ind = get_working_ion_index(pf_struct_ini, working_ion)
    fin_wi_ind = get_working_ion_index(pf_struct_fin, working_ion)

    if ini_wi_ind != fin_wi_ind:
        raise ValueError(
            "Inserted site indexes of end point structures must match for NEBPathfinder"
        )

    # get potential gradient v from host chgcar
    v_chgcar = ChgcarPotential(host_charge_density)
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
    -------
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


def get_working_ion_index(
    structure: Structure, working_ion: CompositionLike
) -> int | None:
    """Get the index of the working ion in a structure.

    Parameters
    ----------
    structure : pymatgen Structure
    working_ion : CompositionLike
        Name of the element to identify

    Returns
    -------
    int - the first site in the structure containing the working ion
    None - no sites in the structure contain the working ion
    """
    for ind, site in enumerate(structure):
        if site.species_string == working_ion:
            # assume that only the lowest indexed working ion is mobile
            return ind
    return None


def get_hop_distance_from_endpoints(
    endpoint_structures: Sequence[Structure], working_ion: CompositionLike
) -> float:
    """
    Find the hop distance of a working ion from two endpoint structures.

    Parameters
    ----------
    endpoint_structures : Sequence of pymatgen .Structure
        The two endpoint structures defining a hop.
    working_ion : pymatgen .CompositionLike
        The species name of the working ion.

    Returns
    -------
    float - the hop distance
    """
    working_ion_sites = [
        [
            site
            for site in endpoint_structures[ep_idx]
            if site.species_string == working_ion
        ]
        for ep_idx in range(2)
    ]

    return max(
        np.linalg.norm(site_a.coords - site_b.coords)
        for site_a in working_ion_sites[0]
        for site_b in working_ion_sites[1]
    )


@job
def collate_images_single_hop(
    working_ion: CompositionLike,
    endpoint_calc_output: list[dict[str, Any]],
    image_calc_output: list[dict[str, Any]],
    min_images_per_hop: int | None = None,
) -> NebResult:
    """
    Collect output from an ApproxNEB flow.

    Parameters
    ----------
    working_ion: CompositionLike
        The mobile ion.
    endpoint_calc_output: list of dict
        Output from the endpoint relaxations.
    image_calc_output: list of dict
        Output from the image relaxations.
    min_images_per_hop: int | None = None,
        If an int, the minimum number of image calculations per hop that
        must succeed to mark a hop as successfully calculated.

    Returns
    -------
    NebResult
    """
    metadata: dict[str, Any] = {}
    task_state = TaskState.SUCCESS

    calcs: list[dict[str, Any]] = []
    if image_calc_output is None:
        task_state = TaskState.FAILED
        metadata["failure_reasons"] = ["No image calculation output."]
    else:
        calcs += image_calc_output

    if endpoint_calc_output is not None:
        calcs = [endpoint_calc_output[0], *calcs, endpoint_calc_output[1]]

    if min_images_per_hop is not None:
        num_success_calcs = len(
            [calc for calc in calcs if calc["structure"] is not None]
        )
        if num_success_calcs < min_images_per_hop:
            task_state = TaskState.FAILED
            if "failure_reasons" not in metadata:
                metadata["failure_reasons"] = []
            metadata["failure_reasons"].append(HopFailureReason.MIN_IMAGE.value)

    hop_dist = None
    if endpoint_calc_output is not None and working_ion is not None:
        hop_dist = get_hop_distance_from_endpoints(
            [ep_calc["structure"] for ep_calc in endpoint_calc_output], working_ion
        )

    return NebResult(
        images=[calc["structure"] for calc in calcs if calc["structure"] is not None],
        initial_images=[
            calc["initial_structure"]
            for calc in calcs
            if calc["initial_structure"] is not None
        ],
        energies=[calc["energy"] for calc in calcs if calc["energy"] is not None],
        method=NebMethod.APPROX,
        state=task_state,
        metadata=metadata if len(metadata) > 0 else None,
        hop_distance=hop_dist,
    )
