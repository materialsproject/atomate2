from pathlib import Path

import numpy as np
import pytest
from jobflow import JobStore, run_locally
from maggma.stores import MemoryStore
from pymatgen.core import Molecule

from atomate2.qchem.flows.core import FrequencyOptFlatteningMaker, FrequencyOptMaker
from atomate2.qchem.jobs.core import FreqMaker, OptMaker
from atomate2.qchem.sets.core import FreqSetGenerator, OptSetGenerator

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


def _out_of_plane_height(molecule: Molecule) -> float:
    """Distance of the first atom from the plane of the next three atoms."""
    center, *others = (site.coords for site in molecule)
    normal = np.cross(others[1] - others[0], others[2] - others[0])
    return abs(np.dot(center - others[0], normal / np.linalg.norm(normal)))


def test_frequency_opt_flattening_maker(mock_qchem, clean_dir, qchem_test_dir):
    # planar NH3 is the transition state of the NH3 inversion. By symmetry the first
    # optimization stays planar and the frequency analysis finds one imaginary mode,
    # so the workflow must perturb along that mode and reoptimize to pyramidal NH3.
    planar_nh3 = Molecule(
        species=["N", "H", "H", "H"],
        coords=[
            [0.0, 0.0, 0.0],
            [1.012, 0.0, 0.0],
            [-0.506, 0.8764292328, 0.0],
            [-0.506, -0.8764292328, 0.0],
        ],
    )
    ref_dir = Path(qchem_test_dir) / "ffopt_planar_nh3"
    ref_paths = {
        name: ref_dir / name.lower().replace(" ", "_")
        for name in (
            "Geometry Optimization 1",
            "Frequency Analysis 1",
            "Geometry Optimization 2",
            "Frequency Analysis 2",
        )
    }
    mock_qchem(ref_paths, fake_run_qchem_kwargs)

    # the reference calculations use def2-SVPD to keep the test data small
    maker = FrequencyOptFlatteningMaker(
        opt_maker=OptMaker(input_set_generator=OptSetGenerator(basis_set="def2-svpd")),
        freq_maker=FreqMaker(
            input_set_generator=FreqSetGenerator(basis_set="def2-svpd")
        ),
    )
    flow = maker.make(planar_nh3)
    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    responses = run_locally(flow, store=store, create_folders=True, ensure_success=True)

    # get the QChem jobs of the dynamic flow in the order they ran
    jobs = []
    for resp in responses.values():
        if replace_flow := getattr(resp[1], "replace", None):
            jobs += [
                job
                for job in replace_flow.jobs
                if job.name not in (maker.name, "store_inputs")
            ]
    assert [job.name for job in jobs] == list(ref_paths)
    opt1, freq1, opt2, freq2 = (responses[job.uuid][1].output for job in jobs)

    # the numbered job names are not used as the task type
    assert [doc.task_type for doc in (opt1, freq1, opt2, freq2)] == [
        "Geometry Optimization",
        "Frequency Analysis",
    ] * 2

    # the first cycle stays at the planar transition state
    assert opt1.output.final_energy == pytest.approx(-56.479811309, abs=1e-8)
    assert _out_of_plane_height(opt1.output.optimized_molecule) == pytest.approx(
        0.0, abs=1e-3
    )
    assert freq1.output.frequencies[:3] == pytest.approx(
        [-743.32, 1540.92, 1543.11], abs=1e-2
    )
    assert sum(freq < 0 for freq in freq1.output.frequencies) == 1

    # the second cycle reaches the pyramidal minimum
    assert opt2.output.final_energy == pytest.approx(-56.486273422, abs=1e-8)
    assert opt2.output.final_energy < opt1.output.final_energy
    assert _out_of_plane_height(opt2.output.optimized_molecule) == pytest.approx(
        0.375, abs=1e-3
    )
    assert freq2.output.frequencies[:3] == pytest.approx(
        [976.56, 1615.17, 1617.33], abs=1e-2
    )
    assert all(freq > 0 for freq in freq2.output.frequencies)

    # each frequency analysis runs at the optimized geometry and copies files from
    # the optimization directory
    for job, opt, freq in ((jobs[1], opt1, freq1), (jobs[3], opt2, freq2)):
        assert opt.output.optimized_molecule == freq.output.initial_molecule
        assert freq.output.optimized_molecule is None
        assert job.function_kwargs["prev_dir"] == opt.dir_name

    # the flow output should be the final frequency analysis task document
    flow_output = flow.output.resolve(store)
    assert flow_output is not None
    assert flow_output.output.final_energy == pytest.approx(
        freq2.output.final_energy, abs=1e-8
    )
    assert flow_output.output.frequencies == pytest.approx(
        freq2.output.frequencies, abs=1e-2
    )
