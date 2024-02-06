"""Define common jobs used in EOS workflows, electronic-structure code agnostic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jobflow import job
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.analysis.eos import EOS, EOSError
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure


@job
def postprocess_eos(output: dict, eos_models: list[str] = None) -> dict:
    """
    General-purpose postprocessing step to fit to EOS models in pymatgen.

    Parameters
    ----------
    output : dict
        dict of energy and volume data for equilibrium calculations,
        structured as
        {
            "relax" or "static" : {
                key : float for key in ("E0", "V0"),
                "energies": [list of energies],
                "volumes": [list of volumes]
            }
        }
    eos_models : list[str] or None (default)
        Custom user-defined list of EOS names (string) to fit to

    Returns
    -------
    dict
        A dict containing a copy of the input dict, and fitted EOS
        parameters for each of the models in `eos_models`
    """
    eos_models = eos_models or [
        "murnaghan",
        "birch",
        "birch_murnaghan",
        "pourier_tarantola",
        "vinet",
    ]

    jobtypes = list(output)

    for jobtype in jobtypes:
        if len(list(output.get(jobtype, []))) == 0:
            continue
        if "E0" and "V0" in output[jobtype]:
            output[jobtype]["energies"].append(output[jobtype]["E0"])
            output[jobtype]["volumes"].append(output[jobtype]["V0"])
        for key in ("energies", "volumes"):
            output[jobtype][key] = np.array(output[jobtype][key])

        sort_vol_indx = np.argsort(output[jobtype]["volumes"])
        for key in ("energies", "volumes"):
            output[jobtype][key] = list(output[jobtype][key][sort_vol_indx])

        output[jobtype]["EOS"] = {}
        for eos_name in eos_models:
            try:
                eos = EOS(eos_name=eos_name).fit(
                    output[jobtype]["volumes"], output[jobtype]["energies"]
                )
                output[jobtype]["EOS"][eos_name] = {
                    **eos.results,
                    "b0 GPa": float(eos.b0_GPa),
                }
            except EOSError:
                output[jobtype]["EOS"][eos_name] = {}

    return output


@job
def apply_strain_to_structure(structure: Structure, deformations: list) -> list:
    """
    Apply strain(s) to input structure and return transformation(s) as list.

    Parameters
    ----------
    structure: .Structure
        Input structure to apply strain to
    deformations: list[.Deformation]
        A list of deformations to apply **independently** to the input
        structure, in anticipation of performing an EOS fit.
        Deformations should be of the form of a 3x3 matrix, e.g.,
        [[1.2, 0., 0.], [0., 1.2, 0.], [0., 0., 1.2]]
        or
        ((1.2, 0., 0.), (0., 1.2, 0.), (0., 0., 1.2))

    Returns
    -------
    list
        A list of .TransformedStructure objects corresponding to the
        list of input deformations.
    """
    transformations = []
    for deformation in deformations:
        # deform the structure
        ts = TransformedStructure(
            structure,
            transformations=[DeformStructureTransformation(deformation=deformation)],
        )
        transformations += [ts]
    return transformations


@job
def extract_eos_sampling_data(
    output: dict,
) -> dict:  # TODO specify the dictionary format
    """
    Extracts the energy, volume, and pressure (if available) data from the output of an EOS flow.

    Parameters
    ----------
    output : dict
        The output of an EOS flow.

    Returns
    -------
    dict
        The energy, volume, pressure data dictionary from the output of an EOS flow.
    """

    eos_tags = ("energies", "volumes", "pressure")

    flow_fit_outputs = {}

    for key in eos_tags:
        try:
            flow_fit_outputs[key] = output["relax"][key]
        except KeyError:
            flow_fit_outputs[key] = []

    flow_fit_outputs["V0"] = output["relax"]["EOS"]["birch_murnaghan"]["V0"]
    flow_fit_outputs["Vmax"] = max(output["relax"]["volumes"])
    flow_fit_outputs["Vmin"] = min(output["relax"]["volumes"])

    return flow_fit_outputs
