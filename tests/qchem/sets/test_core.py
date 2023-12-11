"""Confirm with @janosh before changing any of the expected values below."""

import pytest

from atomate2.qchem.sets.base import QCInputGenerator
from atomate2.qchem.sets.core import (
    SinglePointSetGenerator,
    #OptSetGenerator,
    #TransitionStateSetGenerator,
    #ForceSetGenerator,
    #FreqSetGenerator,
    #PESScanSetGenerator,
)


@pytest.mark.parametrize(
    "set_generator",
    [
        SinglePointSetGenerator,
        #OptSetGenerator,
        #TransitionStateSetGenerator,
        #ForceSetGenerator,
        #FreqSetGenerator,
        #PESScanSetGenerator,
    ],
)
def test_qc_sets(set_generator: QCInputGenerator) -> None:
    qc_set: QCInputGenerator = set_generator()
    assert {*qc_set.as_dict()} >= {
        "job_type",
        "basis_set",
        "scf_algorithm",
        "dft_rung",
        "pcm_dielectric",
        "smd_solvent",
        "custom_smd",
        "opt_variables",
        "scan_variables",
        "max_scf_cycles",
        "geom_opt_max_cycles",
        "plot_cubes",
        "nbo_params",
        "new_geom_opt",
        "overwrite_inputs",
        "vdw_mode",
        "rem_dict",
        "opt_dict",
        "pcm_dict",
        "solv_dict",
        "smx_dict",
        "scan_dict",
        "vdw_dict",
        "plots_dict",
        "nbo_dict",
        "geom_opt_dict",

    }
    assert qc_set.scf_algorithm == "diis"
    assert qc_set.job_type == 'sp'