"""Job used in the Ferroelectric wflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jobflow import Flow, Job, Response, job
from pymatgen.analysis.ferroelectricity.polarization import get_total_ionic_dipole

from atomate2.vasp.schemas.ferroelectric import PolarizationDocument

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker

logger = logging.getLogger(__name__)

__all__ = ["polarization_analysis"]


@job(output_schema=PolarizationDocument)
def polarization_analysis(
    np_lcalcpol_output: dict[str, Any],
    p_lcalcpol_output: dict[str, Any],
    interp_lcalcpol_outputs: dict[str, Any],
) -> PolarizationDocument:
    """
    Recover the same branch polarization and the spontaneous polarization.

    Parameters
    ----------
    np_lcalcpol_output : dict
        Output from previous nonpolar lcalcpol job.
    p_lcalcpol_output : dict
        Output from previous polar lcalcpol job.
    interp_lcalcpol_outputs : dict
        Output from previous interpolation lcalcpol jobs.

    Returns
    -------
    PolarizationDocument
        Document containing the polarization analysis results.

    """
    # order previous calculations from nonpolar to polar
    ordered_keys = [
        f"interpolation_{i}" for i in reversed(range(len(interp_lcalcpol_outputs)))
    ]

    polarization_tasks = [np_lcalcpol_output]
    polarization_tasks += [interp_lcalcpol_outputs[k] for k in ordered_keys]
    polarization_tasks += [p_lcalcpol_output]

    task_lbls = []
    structures = []
    energies_per_atom = []
    energies = []
    job_dirs = []
    uuids = []

    for i, p in enumerate(polarization_tasks):
        energies_per_atom.append(p["energy_per_atom"])
        energies.append(p["energy"])
        task_lbls.append(p["task_label"] or str(i))
        structures.append(p["structure"])
        job_dirs.append(p["job_dir"])
        uuids.append(p["uuid"])

    # If LCALCPOL = True then Outcar will parse and store the pseudopotential zvals.
    zval_dict = p["zval_dict"]

    # Assumes that we want to calculate the ionic contribution to the dipole moment.
    # VASP's ionic contribution is sometimes strange.
    # See pymatgen.analysis.ferroelectricity.polarization.Polarization for details.
    p_elecs = [p["p_elecs"] for p in polarization_tasks]
    p_ions = [get_total_ionic_dipole(st, zval_dict) for st in structures]

    return PolarizationDocument.from_pol_output(
        p_elecs,
        p_ions,
        structures,
        energies,
        energies_per_atom,
        zval_dict,
        task_lbls,
        job_dirs,
        uuids,
    )


@job
def interpolate_structures(p_st: Structure, np_st: Structure, nimages: int) -> list:
    """
    Interpolate linearly the polar and the nonpolar structures with nimages structures.

    Parameters
    ----------
    p_st : Structure
        A pymatgen structure of polar phase.
    np_st : Structure
        A pymatgen structure of nonpolar phase.
    nimages : int
        Number of interpolatated structures calculated
        from polar to nonpolar structures.

    Returns
    -------
    List of interpolated structures
    """
    # adding +1 to nimages to match convention used in the interpolate
    # func where nonpolar is (weirdly) included in the nimages count
    return p_st.interpolate(
        np_st, nimages + 1, interpolate_lattices=True, autosort_tol=0.0
    )


@job
def add_interpolation_flow(
    interp_structures: list[Structure], lcalcpol_maker: BaseVaspMaker
) -> Response:
    """
    Generate the interpolations jobs and add them to the main ferroelectric flow.

    Parameters
    ----------
    interp_structures: List[Structure]
        List of interpolated structures
    lcalcpol_maker : BaseVaspMaker
        Vasp maker to compute the polarization of each structure.

    Returns
    -------
    Response
        Job response containing the interpolation flow.
    """
    jobs = []
    outputs = {}

    for i, interp_structure in enumerate(interp_structures[1:-1]):
        lcalcpol_maker.write_additional_data["structures:json"] = {
            "st_polar": interp_structures[0],
            "st_nonpolar": interp_structures[-1],
            "st_interp_idx": i + 1,
        }
        interpolation = lcalcpol_maker.make(interp_structure)
        interpolation.append_name(f" interpolation_{i}")
        jobs.append(interpolation)
        output = get_polarization_output(interpolation)
        outputs.update({f"interpolation_{i}": output})

    interp_flow = Flow(jobs, outputs)
    return Response(replace=interp_flow)


def get_polarization_output(job: Job) -> dict:
    """
    Extract from lcalcpol job all the relevant output to compute the polarization.

    Parameters
    ----------
    job : Job
        Job from which to extract relevant quantities.

    Returns
    -------
    dict
        Dictionary containing the extracted polarization data.
    """
    p = job.output
    outcar = p.calcs_reversed[0].output.outcar

    return {
        "energy_per_atom": p.calcs_reversed[0].output.energy_per_atom,
        "energy": p.calcs_reversed[0].output.energy,
        "task_label": p.task_label,
        "structure": p.structure,
        "zval_dict": outcar["zval_dict"],
        "p_elecs": outcar["p_elec"],
        "job_dir": p.dir_name,
        "uuid": p.uuid,
    }
