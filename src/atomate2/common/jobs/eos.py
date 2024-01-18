"""Define common jobs used in EOS workflows, electronic-structure code agnostic."""

from __future__ import annotations

import numpy as np
from jobflow import job
from pymatgen.analysis.eos import EOS, EOSError


@job
def postprocess_EOS(
    equilibrium: dict, deformation: dict, EOS_models: list[str] | None = None
):
    """
    General-purpose postprocessing step that fits to the EOS models in pymatgen.

    Parameters
    ----------
    equilibrium : dict
        dict of energy and volume data for equilibrium calculations,
        structured as
        {
            "relax" or "static" : {
                key : float for key in ("E0", "V0")
            }
        }
    deformation : dict
        dict of energy and volume data for deformation calculations,
        structured as
        {
            "relax" or "static": {
                "energies": [list of energies],
                "volumes": [list of volumes]
            }
        }
    EOS_models : list[str] or None (default)
        Custom user-defined list of EOS names as strings to fit to
    """
    if EOS_models is None:
        EOS_models = [
            "murnaghan",
            "birch",
            "birch_murnaghan",
            "pourier_tarantola",
            "vinet",
        ]

    output = {**equilibrium, **deformation}
    jobtypes = list(output)

    for jobtype in jobtypes:
        if len(list(output.get(jobtype, []))) == 0:
            continue

        output[jobtype]["energies"].append(output[jobtype]["E0"])
        output[jobtype]["volumes"].append(output[jobtype]["V0"])
        for key in ("energies", "volumes"):
            output[jobtype][key] = np.array(output[jobtype][key])

        sort_vol_indx = np.argsort(output[jobtype]["volumes"])
        for key in ("energies", "volumes"):
            output[jobtype][key] = list(output[jobtype][key][sort_vol_indx])

        output[jobtype]["EOS"] = {}
        for eos_name in EOS_models:
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
