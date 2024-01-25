import pytest

from atomate2.qchem.sets.base import QCInputGenerator
from atomate2.qchem.sets.core import (
    ForceSetGenerator,
    FreqSetGenerator,
    OptSetGenerator,
    PESScanSetGenerator,
    SinglePointSetGenerator,
    TransitionStateSetGenerator,
)


@pytest.mark.parametrize(
    "set_generator, expected_job_type",
    [
        (SinglePointSetGenerator, "sp"),
        (OptSetGenerator, "opt"),
        (TransitionStateSetGenerator, "ts"),
        (ForceSetGenerator, "force"),
        (FreqSetGenerator, "freq"),
        (PESScanSetGenerator, "pes_scan"),
    ],
)
def test_qc_sets(set_generator: QCInputGenerator, expected_job_type: str) -> None:
    qc_set: QCInputGenerator = set_generator()
    assert {*qc_set.__dict__} >= {
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
    assert qc_set.job_type == expected_job_type
    assert qc_set.basis_set == "def2-tzvppd"
    assert isinstance(qc_set.rem_dict, dict)
