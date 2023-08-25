"""Job used in the Ferroelectric wflow."""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from jobflow import Flow, Response, job
from monty.serialization import dumpfn,loadfn

from pymatgen.analysis.ferroelectricity.polarization import (
    EnergyTrend,
    Polarization,
    get_total_ionic_dipole,
)

from atomate2.vasp.jobs.core import PolarizationMaker
from atomate2.vasp.schemas.ferroelectric import PolarizationDocument

logger = logging.getLogger(__name__)

__all__ = ["polarization_analysis"]


@job(output_schema=PolarizationDocument)
def polarization_analysis(np_lcalcpol_output,
                          p_lcalcpol_output,
                          interp_lcalcpol_outputs):
    """
    Recovers the same branch polarization and the spontaneous polarization
    for a ferroelectric workflow.

    Parameters
    ----------
    np_lcalcpol_outputs: output from previous nonpolar lcalcpol job
    p_lcalcpol_outputs: output from previous polar lcalcpol job
    interp_lcalcpol_outputs: output from previous interpolation lcalcpol jobs
   
    """  # noqa: D205
    # order previous calculations from nonpolar to polar
    ordered_keys = [
        f"interpolation_{i}" for i in reversed(range(len(interp_lcalcpol_outputs)))
    ]

    polarization_tasks = [np_lcalcpol_output.dict()]
    polarization_tasks += [interp_lcalcpol_outputs[k].dict() for k in ordered_keys]
    polarization_tasks += [p_lcalcpol_output.dict()]

    tasks = []
    outcars = []
    structures = []
    energies_per_atom = []
    energies = []
    zval_dicts = []
            
    for i, p in enumerate(polarization_tasks):
        energies_per_atom.append(p["calcs_reversed"][0]["output"]["energy_per_atom"])
        energies.append(p["calcs_reversed"][0]["output"]["energy"])
        tasks.append(p["task_label"] or str(i))
        outcars.append(p["calcs_reversed"][0]["output"]["outcar"])
        structures.append(p["calcs_reversed"][0]["input"]["structure"])
        zval_dicts.append(p["calcs_reversed"][0]["output"]["outcar"]["zval_dict"])

    # structures = [Structure.from_dict(structure) for structure in structure_dicts]

    # If LCALCPOL = True then Outcar will parse and store the pseudopotential zvals.
    zval_dict = zval_dicts.pop()

    # Assumes that we want to calculate the ionic contribution to the dipole moment.
    # VASP's ionic contribution is sometimes strange.
    # See pymatgen.analysis.ferroelectricity.polarization.Polarization for details.
    p_elecs = [outcar["p_elec"] for outcar in outcars]
    p_ions = [get_total_ionic_dipole(structure, zval_dict) for structure in structures]

    polarization_doc = PolarizationDocument.from_pol_output(
        p_elecs, p_ion, structures, energies,
        energies_per_atom, zval_dicts, tasks,
    )

    return polarization_doc

@job
def interpolate_structures(p_st, np_st, nimages):
    """
    Interpolate linearly the polar and the nonpolar structures with nimages structures.

    Parameters
    ----------
    polar_structure : .Structure
        A pymatgen structure of polar phase.
    nonpolar_structure : .Structure
        A pymatgen structure of nonpolar phase.
    nimages: int
        Number of interpolations calculated from polar to nonpolar structures,
        including the nonpolar.
    """
    interp_structures = p_st.interpolate(np_st, nimages, True)
    dumpfn(interp_structures, "interp_structures.json")

    return Path.cwd()

@job
def add_interpolation_flow(prev_dir,lcalcpol_maker):
    """
    Generate the interpolations jobs and add them to the main ferroelectric flow

    Parameters
    ----------
    prev_dir: str
        Previous directory where interpolated structures were created
    lcalcpol_maker: BaseVaspMaker
       Vasp maker to compute the polarization of each structure        
    """    

    interp_structures = loadfn(f"{prev_dir}/interp_structures.json")
    
    jobs = []
    outputs = {}

    for i, interp_structure in enumerate(interp_structures[1:]):
        interpolation = lcalcpol_maker.make(interp_structure)
        interpolation.append_name(f" interpolation_{i}")
        jobs.append(interpolation)

        outputs.update({f"interpolation_{i}": interpolation.output})

    interp_flow = Flow(jobs, outputs)
    return Response(replace=interp_flow)
