"""Define ApproxNEB jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from emmet.core.neb import NebMethod
from emmet.core.vasp.task_valid import TaskState
from jobflow import Flow, Maker, Response, job
from monty.os.path import zpath
from pymatgen.io.vasp.outputs import Chgcar

from atomate2.common.jobs.approx_neb import (
    HopFailureReason,
    get_hop_distance_from_endpoints,
)
from atomate2.common.schemas.neb import NebPathwayResult, NebResult
from atomate2.utils.path import strip_hostname
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import RelaxMaker
from atomate2.vasp.run import JobType
from atomate2.vasp.sets.approx_neb import ApproxNebSetGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.util.typing import CompositionLike

    from atomate2.vasp.jobs.base import BaseVaspMaker
    from atomate2.vasp.sets.base import VaspInputGenerator


@dataclass
class ApproxNebHostRelaxMaker(DoubleRelaxMaker):
    """Maker to perform a double relaxation on an ApproxNEB host structure."""

    name: str = "ApproxNEB host relax"
    relax_maker1: BaseVaspMaker | None = field(
        default_factory=lambda: RelaxMaker(input_set_generator=ApproxNebSetGenerator())
    )
    relax_maker2: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(input_set_generator=ApproxNebSetGenerator())
    )


@dataclass
class ApproxNebImageRelaxMaker(RelaxMaker):
    """
    Maker to perform a double relaxation on an ApproxNEB endpoint/image structure.

    Very important here - we are doing a double relaxation in the atomate style,
    where one job maps to two VASP calculations.
    """

    name: str = "ApproxNEB image relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: ApproxNebSetGenerator(set_type="image")
    )
    run_vasp_kwargs: dict = field(
        default_factory=lambda: {
            "job_type": JobType.DOUBLE_RELAXATION,
        }
    )


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
    -------
    pymatgen Chgcar object
    """
    prev_dir = Path(strip_hostname(prev_dir))
    if use_aeccar:
        aeccar0 = Chgcar.from_file(zpath(str(prev_dir / "AECCAR0")))
        aeccar2 = Chgcar.from_file(zpath(str(prev_dir / "AECCAR2")))
        return aeccar0 + aeccar2
    return Chgcar.from_file(zpath(str(prev_dir / "CHGCAR")))


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
            ionic_steps=None,
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
