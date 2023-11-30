from pathlib import Path

import pytest
from jobflow import run_locally
from pytest import approx, importorskip

from atomate2.forcefields.jobs import (
    CHGNetRelaxMaker,
    CHGNetStaticMaker,
    GAPRelaxMaker,
    GAPStaticMaker,
    M3GNetRelaxMaker,
    M3GNetStaticMaker,
    MACERelaxMaker,
    MACEStaticMaker,
)
from atomate2.forcefields.schemas import ForceFieldTaskDocument


def test_chgnet_static_maker(si_structure):
    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    job = CHGNetStaticMaker(task_document_kwargs=task_doc_kwargs).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.6275062, rel=1e-4)
    assert output1.output.ionic_steps[-1].magmoms is None
    assert output1.output.n_steps == 1


def test_chgnet_relax_maker(si_structure):
    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    # generate job
    job = CHGNetRelaxMaker(steps=25).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.6274, rel=1e-4)
    assert output1.output.ionic_steps[-1].magmoms[0] == approx(0.00303572, rel=1e-4)
    assert output1.output.n_steps >= 12


def test_m3gnet_static_maker(si_structure):
    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    job = M3GNetStaticMaker(task_document_kwargs=task_doc_kwargs).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.8, abs=0.2)
    assert output1.output.n_steps == 1


def test_m3gnet_relax_maker(si_structure):
    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    # generate job
    job = M3GNetRelaxMaker(steps=25).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.8, abs=0.2)
    assert output1.output.n_steps == 14


mace_paths = pytest.mark.parametrize(
    "model",
    [
        # None, # TODO uncomment once https://github.com/ACEsuit/mace/pull/230 is merged
        # to test loading MACE checkpoint on the fly from figshare
        f"{Path(__file__).parent.parent}/test_data/forcefields/mace/MACE.model",
    ],
)


@mace_paths
def test_mace_static_maker(si_structure, test_dir, model):
    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    # NOTE the test model is not trained on Si, so the energy is not accurate
    job = MACEStaticMaker(model=model, task_document_kwargs=task_doc_kwargs).make(
        si_structure
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-0.068231, rel=1e-4)
    assert output1.output.n_steps == 1


@mace_paths
def test_mace_relax_maker(si_structure, test_dir, model):
    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    # generate job
    # NOTE the test model is not trained on Si, so the energy is not accurate
    job = MACERelaxMaker(
        model=model,
        steps=25,
        optimizer_kwargs={"optimizer": "BFGSLineSearch"},
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validating the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-0.051912, rel=1e-4)
    assert output1.output.n_steps == 4


def test_gap_static_maker(si_structure, test_dir):
    importorskip("quippy")

    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    # Test files have been provided by Yuanbin Liu (University of Oxford)
    job = GAPStaticMaker(
        potential_args_str="IP GAP",
        potential_param_file_name=test_dir / "forcefields" / "gap" / "gap_file.xml",
        task_document_kwargs=task_doc_kwargs,
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.8523, rel=1e-4)
    assert output1.output.n_steps == 1


def test_gap_relax_maker(si_structure, test_dir):
    importorskip("quippy")

    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    # generate job
    # Test files have been provided by Yuanbin Liu (University of Oxford)
    job = GAPRelaxMaker(
        potential_param_file_name=test_dir / "forcefields" / "gap" / "gap_file.xml",
        steps=25,
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validating the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.8523, rel=1e-4)
    assert output1.output.n_steps == 17
