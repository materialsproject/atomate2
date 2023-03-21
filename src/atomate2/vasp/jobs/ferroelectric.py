"""Jobs used in the calculation of elastic tensors."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from jobflow import Flow, Response, job
from monty.serialization import dumpfn
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.ferroelectricity.polarization import (
    EnergyTrend,
    Polarization,
    get_total_ionic_dipole,
)

from atomate2 import SETTINGS
from atomate2.vasp.schemas.ferroelectric import PolarizationDocument
from atomate2.vasp.jobs.core import PolarizationMaker

logger = logging.getLogger(__name__)

__all__ = ["polarization_analysis"]

@job(output_schema=PolarizationDocument)
def polarization_analysis(lcalcpol_outputs):
    """
    Recovers the same branch polarization and spontaneous polarization
    for a ferroelectric workflow.
    """

    # oreder previous calculations from nonpolar to polar
    ordered_keys = [f'interpolation_{i}' for i in reversed(range(len(lcalcpol_outputs[2])))]


    polarization_tasks = [lcalcpol_outputs[0].dict()]    
    polarization_tasks += [lcalcpol_outputs[2][k].dict() for k in ordered_keys]
    polarization_tasks += [lcalcpol_outputs[1].dict()]
    
    tasks = []
    outcars = []
    structures = []
    energies_per_atom = []
    energies = []
    zval_dicts = []

    for i,p in enumerate(polarization_tasks):
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
    p_ions = [
        get_total_ionic_dipole(structure, zval_dict) for structure in structures
    ]

    polarization = Polarization(p_elecs, p_ions, structures)

    p_change = np.ravel(polarization.get_polarization_change()).tolist()
    p_norm = polarization.get_polarization_change_norm()
    same_branch = polarization.get_same_branch_polarization_data(
        convert_to_muC_per_cm2=True
    )
    raw_elecs, raw_ions = polarization.get_pelecs_and_pions()
    quanta = polarization.get_lattice_quanta(convert_to_muC_per_cm2=True)

    if len(structures) > 3:
        energy_trend = EnergyTrend(energies_per_atom)
        energy_max_spline_jumps = energy_trend.max_spline_jump()
        polarization_max_spline_jumps = polarization.max_spline_jumps()
    else:
        energy_max_spline_jumps = None
        polarization_max_spline_jumps = None
                
    polarization_dict = {}

    def split_abc(var):
        d = {}
        for i, j in enumerate("abc"):
            d.update({f"{j}": np.ravel(var[:, i]).tolist()})
        return d

    # General information
    polarization_dict.update(
        {"pretty_formula": structures[0].composition.reduced_formula}
    )
    #polarization_dict.update({"wfid": wfid})
    polarization_dict.update({"task_label_order": tasks})

    # Polarization information
    polarization_dict.update({"polarization_change": p_change})
    polarization_dict.update({"polarization_change_norm": p_norm})
    polarization_dict.update(
        {"polarization_max_spline_jumps": polarization_max_spline_jumps}
    )
    polarization_dict.update({"same_branch_polarization": split_abc(same_branch)})
    polarization_dict.update({"raw_electron_polarization": split_abc(raw_elecs)})
    polarization_dict.update({"raw_ion_polarization": split_abc(raw_ions)})
    polarization_dict.update({"polarization_quanta": split_abc(quanta)})
    polarization_dict.update({"zval_dict": zval_dict})

    # Energy information
    polarization_dict.update(
        {"energy_per_atom_max_spline_jumps": energy_max_spline_jumps}
    )
    polarization_dict.update({"energies": energies})
    polarization_dict.update({"energies_per_atom": energies_per_atom})
    polarization_dict.update({"outcars": outcars})
    polarization_dict.update({"structures": structures})

    dumpfn(polarization_dict,'polarization_doc.json')
    return PolarizationDocument(**polarization_dict)


@job
def interpolate_structures(p_st,np_st,nimages):
    """
    Interpolate polar and nonpolar structures with nimages points


    Parameters
    ----------
    polar_structure : .Structure
        A pymatgen structure of polar phase.
    nonpolar_structure : .Structure
        A pymatgen structure of nonpolar phase.
    """
    
    interp_structures = p_st.interpolate(np_st,nimages,True)

    jobs = []
    outputs = {}
    
    for i,interp_structure in enumerate(interp_structures[1:]):
        interpolation = PolarizationMaker().make(interp_structure)
        interpolation.append_name(f' interpolation_{i}')
        jobs.append(interpolation)
        
        outputs.update({f'interpolation_{i}':interpolation.output})

    interp_flow = Flow(jobs,outputs)
    return Response(replace=interp_flow)
