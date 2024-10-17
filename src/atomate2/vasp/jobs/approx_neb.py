"""Define ApproxNEB jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from emmet.core.neb import NebMethod
from jobflow import Flow, Maker, Response, job
from monty.os.path import zpath
from pymatgen.analysis.diffusion.neb.pathfinder import ChgcarPotential, NEBPathfinder
from pymatgen.io.vasp.outputs import Chgcar

from atomate2.common.schemas.neb import NebPathwayResult, NebResult
from atomate2.utils.path import strip_hostname
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import RelaxMaker
from atomate2.vasp.sets.approx_neb import ApproxNEBSetGenerator

if TYPE_CHECKING:
    from typing import Literal

    from pymatgen.core import Structure

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
def get_endpoints_and_relax(
    host_structure: Structure,
    working_ion: str,
    endpoint_coords_dict: dict,
    inserted_coords_combo: list,
    relax_maker: Maker,
) -> Response:
    """Get and relax endpoint structures."""
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
    for ep_index, ep_coords in endpoint_coords_dict.items():
        if int(ep_index) in ep_distinct:
            ep_inserted_struct = host_structure.copy()
            ep_inserted_struct.insert(0, working_ion, ep_coords)

            relax_job = relax_maker.make(ep_inserted_struct)
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
    ep_output: dict,
    inserted_combo_list: list,
    n_images: int,
    host_calc_path: str | Path,
    relax_maker: Maker,
    selective_dynamics_scheme: Literal["fix_two_atoms"] | None = "fix_two_atoms",
    use_aeccar: bool = False,
) -> Response:
    """Get and relax image input structures."""
    # remove failed output first
    ep_structures = {
        k: calc["structure"]
        for k, calc in ep_output.items()
        if calc["structure"] is not None
    }

    host_chgcar = get_charge_density(host_calc_path, use_aeccar=use_aeccar)

    image_relax_jobs = []
    image_relax_output: dict[str, list] = {}
    for combo in inserted_combo_list:
        ini_ind, fin_ind = map(int, combo.split("+"))

        if not all(ep_structures.get(idx) for idx in [ini_ind, fin_ind]):
            # cannot proceed with this hop calculation
            continue

        # potential place for uuid logic if depth first is desirable
        pathfinder_output = get_pathfinder_results(
            ep_structures[ini_ind],
            ep_structures[fin_ind],
            working_ion,
            n_images,
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
        for image in images_list:
            relax_job = relax_maker.make(image)
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
    working_ion: str,
    n_images: int,
    host_chgcar: Chgcar,
) -> dict:
    """Get interpolated images from the pathfinder algorithm."""
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


def add_selective_dynamics_two_fixed_sites(
    structure: Structure,
    fixed_index: int,
    fixed_species_name: str,
) -> Structure:
    """Add selective dynamics to input structure."""
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


def get_working_ion_index(structure: Structure, working_ion: str) -> int | None:
    """Get the index of the working ion in a structure."""
    for ind, site in enumerate(structure):
        if site.species_string == working_ion:
            # assume that only the lowest indexed working ion is mobile
            return ind
    return None


def get_charge_density(prev_dir: str | Path, use_aeccar: bool = False) -> Chgcar:
    """Get charge density from a prior VASP calculation."""
    prev_dir = Path(strip_hostname(prev_dir))
    if use_aeccar:
        aeccar0 = Chgcar.from_file(zpath(str(prev_dir / "AECCAR0")))
        aeccar2 = Chgcar.from_file(zpath(str(prev_dir / "AECCAR2")))
        return aeccar0 + aeccar2
    return Chgcar.from_file(zpath(str(prev_dir / "CHGCAR")))


@job
def collate_results(
    endpoint_calc_output: dict, image_calc_output: dict[str, list]
) -> NebPathwayResult:
    """Collect output from an ApproxNEB workflow."""
    hop_dict = {}
    for combo_name, images in image_calc_output.items():
        endpoint_calcs = [endpoint_calc_output[idx] for idx in combo_name.split("+")]
        hop = [endpoint_calcs[0], *images, endpoint_calcs[1]]
        hop_dict[combo_name] = NebResult(
            images=[calc["structure"] for calc in hop],
            energies=[calc["energy"] for calc in hop],
            ionic_steps=None,  # [calc.output.ionic_steps for calc in hop],
            method=NebMethod.APPROX,
        )
    return NebPathwayResult(hops=hop_dict)
