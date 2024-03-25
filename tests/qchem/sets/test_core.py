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
        "opt_dict",
        "scan_dict",
        "max_scf_cycles",
        "geom_opt_max_cycles",
        "plot_cubes",
        "nbo_params",
        "new_geom_opt",
        "overwrite_inputs",
        "vdw_mode",
        "rem_dict",
        "pcm_dict",
        "solv_dict",
        "smx_dict",
        "vdw_dict",
        "plots_dict",
    }
    assert qc_set.scf_algorithm == "diis"
    assert qc_set.job_type == expected_job_type
    assert qc_set.basis_set == "def2-tzvppd"
    assert isinstance(qc_set.rem_dict, dict)


@pytest.mark.parametrize(
    "set_generator_pcm_d3, expected_job_type",
    [
        (SinglePointSetGenerator, "sp"),
        (OptSetGenerator, "opt"),
        (TransitionStateSetGenerator, "ts"),
        (ForceSetGenerator, "force"),
        (FreqSetGenerator, "freq"),
        (PESScanSetGenerator, "pes_scan"),
    ],
)
def test_extra_params_pcm(
    set_generator_pcm_d3: QCInputGenerator, expected_job_type: str
) -> None:
    qc_set: QCInputGenerator = set_generator_pcm_d3(dft_rung=2, pcm_dielectric=78.39)
    assert qc_set.rem_dict["dft_D"] == "D3_BJ"
    assert qc_set.rem_dict["solvent_method"] == "pcm"

    pcm_defaults = {
        "heavypoints": "194",
        "hpoints": "194",
        "radii": "uff",
        "theory": "cpcm",
        "vdwscale": "1.1",
    }

    assert qc_set.pcm_dict == pcm_defaults
    assert qc_set.solv_dict["dielectric"] == qc_set.pcm_dielectric
    assert qc_set.scf_algorithm == "diis"
    assert qc_set.job_type == expected_job_type
    assert qc_set.basis_set == "def2-tzvppd"
    assert isinstance(qc_set.rem_dict, dict)


@pytest.mark.parametrize(
    "set_generator_smd, expected_job_type",
    [
        (SinglePointSetGenerator, "sp"),
        (OptSetGenerator, "opt"),
        (TransitionStateSetGenerator, "ts"),
        (ForceSetGenerator, "force"),
        (FreqSetGenerator, "freq"),
        (PESScanSetGenerator, "pes_scan"),
    ],
)
def test_extra_params_smd(
    set_generator_smd: QCInputGenerator, expected_job_type: str
) -> None:
    qc_set: QCInputGenerator = set_generator_smd(smd_solvent="water")
    assert qc_set.rem_dict["solvent_method"] == "smd"
    assert qc_set.rem_dict["ideriv"] == "1"
    assert qc_set.smx_dict["solvent"] == "water"
    assert qc_set.scf_algorithm == "diis"
    assert qc_set.job_type == expected_job_type
    assert qc_set.basis_set == "def2-tzvppd"
    assert isinstance(qc_set.rem_dict, dict)


@pytest.mark.parametrize(
    "set_generator_plots, expected_job_type",
    [
        (SinglePointSetGenerator, "sp"),
        (OptSetGenerator, "opt"),
        (TransitionStateSetGenerator, "ts"),
        (ForceSetGenerator, "force"),
        (FreqSetGenerator, "freq"),
        (PESScanSetGenerator, "pes_scan"),
    ],
)
def test_extra_params_plots(
    set_generator_plots: QCInputGenerator, expected_job_type: str
) -> None:
    qc_set: QCInputGenerator = set_generator_plots(plot_cubes=True)
    assert qc_set.plots_dict == {"grid_spacing": "0.05", "total_density": "0"}
    assert qc_set.rem_dict["plots"] == "true"
    assert qc_set.rem_dict["make_cube_files"] == "true"
    assert qc_set.scf_algorithm == "diis"
    assert qc_set.job_type == expected_job_type
    assert qc_set.basis_set == "def2-tzvppd"


@pytest.mark.parametrize(
    "set_generator_nbo, expected_job_type",
    [
        (SinglePointSetGenerator, "sp"),
        (OptSetGenerator, "opt"),
        (TransitionStateSetGenerator, "ts"),
        (ForceSetGenerator, "force"),
        (FreqSetGenerator, "freq"),
        (PESScanSetGenerator, "pes_scan"),
    ],
)
def test_extra_params_nbo(
    set_generator_nbo: QCInputGenerator, expected_job_type: str
) -> None:
    qc_set: QCInputGenerator = set_generator_nbo(
        nbo_params={"version": 7, "plots": "PLOT"}
    )

    assert qc_set.rem_dict["nbo"] == "true"
    assert qc_set.rem_dict["nbo_external"] == "true"
    assert qc_set.scf_algorithm == "diis"
    assert qc_set.job_type == expected_job_type
    assert qc_set.basis_set == "def2-tzvppd"
