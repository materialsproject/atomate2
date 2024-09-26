from pathlib import Path

import pytest
from jobflow import run_locally

from atomate2.qchem.flows.core import FrequencyOptFlatteningMaker, FrequencyOptMaker

fake_run_qchem_kwargs = {}


def test_frequency_opt_maker(mock_qchem, clean_dir, qchem_test_dir, h2o_molecule):
    ref_paths = {
        "Geometry Optimization": Path(qchem_test_dir)
        / "ffopt"
        / "geometry_optimization",
        "Frequency Analysis": Path(qchem_test_dir) / "ffopt" / "frequency_analysis_1",
    }
    mock_qchem(ref_paths, fake_run_qchem_kwargs)

    flow = FrequencyOptMaker().make(h2o_molecule)
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    output = {job.name: responses[job.uuid][1].output for job in flow}

    ref_total_energy = -76.346601
    assert output["Geometry Optimization"].output.final_energy == pytest.approx(
        ref_total_energy, rel=1e-6
    )

    assert output["Frequency Analysis"].output.final_energy == pytest.approx(
        ref_total_energy, rel=1e-6
    )
    ref_freq = [1587.39, 3864.9, 3969.87]
    assert all(
        freq == pytest.approx(ref_freq[i], abs=1e-2)
        for i, freq in enumerate(output["Frequency Analysis"].output.frequencies)
    )
    assert (
        output["Geometry Optimization"].output.optimized_molecule
        == output["Frequency Analysis"].output.initial_molecule
    )
    assert output["Frequency Analysis"].output.optimized_molecule is None


def test_frequency_opt_flattening_maker(
    mock_qchem, clean_dir, qchem_test_dir, h2o_molecule
):
    ref_paths = {
        k: Path(qchem_test_dir) / "ffopt" / f"{k.lower().replace(' ', '_')}"
        for k in ("Geometry Optimization", "Frequency Analysis 1")
    }

    mock_qchem(ref_paths, fake_run_qchem_kwargs)
    flow = FrequencyOptFlatteningMaker().make(h2o_molecule)
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # first get job name / uuid pairs from dynamic flow
    uuid_to_name = {}
    for resp in responses.values():
        if replace_flow := getattr(resp[1], "replace", None):
            uuid_to_name.update({job.uuid: job.name for job in replace_flow.jobs})

    # then get job output
    output = {}
    for uuid, job_name in uuid_to_name.items():
        output[job_name] = responses[uuid][1].output

    ref_total_energy = -76.346601
    assert output["Geometry Optimization"].output.final_energy == pytest.approx(
        ref_total_energy, rel=1e-6
    )

    # because the initial frequency analysis has no negative frequencies,
    # the workflow only performs one frequency analysis
    assert output["Frequency Analysis 1"].output.final_energy == pytest.approx(
        ref_total_energy, rel=1e-6
    )
    ref_freq = [1587.39, 3864.9, 3969.87]
    assert all(
        freq == pytest.approx(ref_freq[i], abs=1e-2)
        for i, freq in enumerate(output["Frequency Analysis 1"].output.frequencies)
    )
    assert (
        output["Geometry Optimization"].output.optimized_molecule
        == output["Frequency Analysis 1"].output.initial_molecule
    )
    assert output["Frequency Analysis 1"].output.optimized_molecule is None
