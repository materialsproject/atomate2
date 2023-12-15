"""Define common jobs used in EOS workflows, electronic-structure code agnostic."""

from __future__ import annotations

from jobflow import job
from pymatgen.analysis.eos import EOS, EOSError


@job
def postprocess_EOS(data_dict: dict, EOS_models: list[str] | None = None):
    """
    General-purpose postprocessing step that fits to the EOS models in pymatgen.

    Parameters
    ----------
    data_dict : dict
        dict of energy and volume data structured as
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

    output = data_dict.copy()
    jobtypes = list(output)
    for jobtype in jobtypes:
        if len(list(data_dict.get(jobtype, []))) == 0:
            continue
        output[jobtype]["EOS"] = {}
        for eos_name in EOS_models:
            try:
                eos = EOS(eos_name=eos_name).fit(
                    data_dict[jobtype]["volumes"], data_dict[jobtype]["energies"]
                )
                output[jobtype]["EOS"][eos_name] = {
                    **eos.results,
                    "b0 GPa": float(eos.b0_GPa),
                }
            except EOSError:
                output[jobtype]["EOS"][eos_name] = {}

    return output
