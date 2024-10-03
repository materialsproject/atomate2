from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from emmet.core.openmm import OpenMMInterchange, OpenMMTaskDocument
from jobflow import Flow
from MDAnalysis import Universe
from monty.json import MontyDecoder
from openmm.app import PDBFile

from atomate2.openmm.flows.core import OpenMMFlowMaker
from atomate2.openmm.jobs import EnergyMinimizationMaker, NPTMaker, NVTMaker


def test_anneal_maker(interchange, run_job):
    # Create an instance of AnnealMaker with custom parameters
    anneal_maker = OpenMMFlowMaker.anneal_flow(
        name="test_anneal",
        anneal_temp=500,
        final_temp=300,
        n_steps=30,
        temp_steps=1,
        job_names=("heat", "hold", "cool"),
        platform_name="CPU",
    )

    # Run the AnnealMaker flow
    anneal_flow = anneal_maker.make(interchange)

    task_doc = run_job(anneal_flow)

    # Check the output task document
    assert isinstance(task_doc, OpenMMTaskDocument)
    assert task_doc.state == "successful"
    assert len(task_doc.calcs_reversed) == 1

    # Check the individual jobs in the flow
    raise_temp_job = anneal_flow.jobs[0]
    assert raise_temp_job.maker.name == "heat"
    assert raise_temp_job.maker.n_steps == 10
    assert raise_temp_job.maker.temperature == 500
    assert raise_temp_job.maker.temp_steps == 1

    nvt_job = anneal_flow.jobs[1]
    assert nvt_job.maker.name == "hold"
    assert nvt_job.maker.n_steps == 10
    assert nvt_job.maker.temperature == 500

    lower_temp_job = anneal_flow.jobs[2]
    assert lower_temp_job.maker.name == "cool"
    assert lower_temp_job.maker.n_steps == 10
    assert lower_temp_job.maker.temperature == 300
    assert lower_temp_job.maker.temp_steps == 1


# @pytest.mark.skip("Reporting to HDF5 is broken in MDA upstream.")
def test_hdf5_writing(interchange, run_job):
    # Create an instance of AnnealMaker with custom parameters
    import MDAnalysis
    from packaging.version import Version

    if Version(MDAnalysis.__version__) < Version("2.8.0"):
        return

    anneal_maker = OpenMMFlowMaker.anneal_flow(
        name="test_anneal",
        n_steps=3,
        temp_steps=1,
        platform_name="CPU",
        traj_file_type="h5md",
        report_velocities=True,
        traj_interval=1,
    )

    # Run the AnnealMaker flow
    anneal_maker.collect_outputs = True
    anneal_flow = anneal_maker.make(interchange)

    task_doc = run_job(anneal_flow)

    calc_output_names = [calc.output.traj_file for calc in task_doc.calcs_reversed]
    assert len(list(Path(task_doc.dir_name).iterdir())) == 5
    assert set(calc_output_names) == {
        "trajectory3.h5md",
        "trajectory2.h5md",
        "trajectory.h5md",
    }


def test_collect_outputs(interchange, run_job):
    # Create an instance of ProductionMaker with custom parameters
    production_maker = OpenMMFlowMaker(
        name="test_production",
        tags=["test"],
        makers=[
            EnergyMinimizationMaker(max_iterations=1, save_structure=True),
            NVTMaker(n_steps=5),
        ],
        collect_outputs=True,
    )

    # Run the ProductionMaker flow
    production_flow = production_maker.make(interchange)
    run_job(production_flow)


def test_flow_maker(interchange, run_job):
    # Create an instance of ProductionMaker with custom parameters
    production_maker = OpenMMFlowMaker(
        name="test_production",
        tags=["test"],
        makers=[
            EnergyMinimizationMaker(max_iterations=1, state_interval=1),
            NPTMaker(n_steps=5, pressure=1.0, state_interval=1, traj_interval=1),
            OpenMMFlowMaker.anneal_flow(anneal_temp=400, final_temp=300, n_steps=5),
            NVTMaker(n_steps=5),
        ],
    )

    # Run the ProductionMaker flow
    production_flow = production_maker.make(interchange)
    task_doc = run_job(production_flow)

    # Check the output task document
    assert isinstance(task_doc, OpenMMTaskDocument)
    assert task_doc.state == "successful"
    assert len(task_doc.calcs_reversed) == 6
    assert task_doc.calcs_reversed[-1].task_name == "energy minimization"
    assert task_doc.calcs_reversed[0].task_name == "nvt simulation"
    assert task_doc.tags == ["test"]
    assert len(task_doc.job_uuids) == 6
    assert task_doc.job_uuids[0] is not None

    # Check the individual jobs in the flow
    energy_job = production_flow.jobs[0]
    assert isinstance(energy_job.maker, EnergyMinimizationMaker)

    npt_job = production_flow.jobs[1]
    assert isinstance(npt_job.maker, NPTMaker)
    assert npt_job.maker.n_steps == 5
    assert npt_job.maker.pressure == 1.0

    anneal_flow = production_flow.jobs[2]
    assert isinstance(anneal_flow, Flow)
    assert anneal_flow.jobs[0].maker.temperature == 400
    assert anneal_flow.jobs[2].maker.temperature == 300

    nvt_job = production_flow.jobs[3]
    assert isinstance(nvt_job.maker, NVTMaker)
    assert nvt_job.maker.n_steps == 5

    # Test length of state attributes in calculation output
    calc_output = task_doc.calcs_reversed[0].output
    assert len(calc_output.steps_reported) == 5

    all_steps = [calc.output.steps_reported for calc in task_doc.calcs_reversed]
    assert all_steps == [
        [11, 12, 13, 14, 15],
        [10],
        [8, 9],
        [6, 7],
        [1, 2, 3, 4, 5],
        [0],
    ]
    # Test that the state interval is respected
    assert calc_output.steps_reported == list(range(11, 16))
    assert calc_output.traj_file == "trajectory5.dcd"
    assert calc_output.state_file == "state5.csv"

    interchange = OpenMMInterchange.model_validate_json(task_doc.interchange)
    topology = PDBFile(io.StringIO(interchange.topology)).getTopology()
    u = Universe(topology, str(Path(task_doc.dir_name) / "trajectory5.dcd"))

    assert len(u.trajectory) == 5


def test_traj_blob_embed(interchange, run_job, tmp_path):
    nvt = NVTMaker(n_steps=2, traj_interval=1, embed_traj=True)

    # Run the ProductionMaker flow
    nvt_job = nvt.make(interchange)
    task_doc = run_job(nvt_job)

    interchange = OpenMMInterchange.model_validate_json(task_doc.interchange)
    topology = PDBFile(io.StringIO(interchange.topology)).getTopology()

    u = Universe(topology, str(Path(task_doc.dir_name) / "trajectory.dcd"))

    assert len(u.trajectory) == 2

    calc_output = task_doc.calcs_reversed[0].output
    assert calc_output.traj_blob is not None

    # Write the bytes back to a file
    with open(tmp_path / "doc_trajectory.dcd", "wb") as f:
        f.write(bytes.fromhex(calc_output.traj_blob))

    u2 = Universe(topology, str(tmp_path / "doc_trajectory.dcd"))

    assert np.all(u.atoms.positions == u2.atoms.positions)

    with open(Path(task_doc.dir_name) / "taskdoc.json") as file:
        task_dict = json.load(file, cls=MontyDecoder)
        task_doc_parsed = OpenMMTaskDocument.model_validate(task_dict)

    parsed_output = task_doc_parsed.calcs_reversed[0].output

    assert parsed_output.traj_blob == calc_output.traj_blob


@pytest.mark.skip("for local testing and debugging")
def test_fireworks(interchange):
    # Create an instance of ProductionMaker with custom parameters

    production_maker = OpenMMFlowMaker(
        name="test_production",
        tags=["test"],
        makers=[
            EnergyMinimizationMaker(max_iterations=1),
            NPTMaker(n_steps=5, pressure=1.0, state_interval=1, traj_interval=1),
            OpenMMFlowMaker.anneal_flow(anneal_temp=400, final_temp=300, n_steps=5),
            NVTMaker(n_steps=5),
        ],
    )

    interchange_json = interchange.json()
    # interchange_bytes = interchange_json.encode("utf-8")

    # Run the ProductionMaker flow
    production_flow = production_maker.make(interchange_json)

    from fireworks import LaunchPad
    from jobflow.managers.fireworks import flow_to_workflow

    wf = flow_to_workflow(production_flow)

    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf)

    # from fireworks.core.rocket_launcher import launch_rocket
    #
    # launch_rocket(lpad)
