from jobflow import Flow

from atomate2.classical_md.openmm.flows.core import AnnealMaker, ProductionMaker
from atomate2.classical_md.openmm.jobs.core import (
    EnergyMinimizationMaker,
    NPTMaker,
    NVTMaker,
)
from atomate2.classical_md.schemas import ClassicalMDTaskDocument


def test_anneal_maker(interchange, tmp_path, run_job):
    # Create an instance of AnnealMaker with custom parameters
    anneal_maker = AnnealMaker.from_temps_and_steps(
        name="test_anneal",
        anneal_temp=500,
        final_temp=300,
        n_steps=30,
        temp_steps=1,
        job_names=("heat", "hold", "cool"),
        platform_name="CPU",
    )

    # Run the AnnealMaker flow
    anneal_flow = anneal_maker.make(interchange, output_dir=tmp_path)

    task_doc = run_job(anneal_flow)

    # Check the output task document
    assert isinstance(task_doc, ClassicalMDTaskDocument)
    assert task_doc.state == "successful"
    assert len(task_doc.calcs_reversed) == 3

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


def test_hdf5_writing(interchange, tmp_path, run_job):
    # Create an instance of AnnealMaker with custom parameters
    anneal_maker = AnnealMaker.from_temps_and_steps(
        name="test_anneal",
        n_steps=3,
        temp_steps=1,
        platform_name="CPU",
        traj_file_type="h5",
        report_velocities=True,
    )

    # Run the AnnealMaker flow
    anneal_flow = anneal_maker.make(interchange, output_dir=tmp_path)

    task_doc = run_job(anneal_flow)

    calc_output_names = [calc.output.dcd_file for calc in task_doc.calcs_reversed]
    assert len(list(tmp_path.iterdir())) == 5
    assert set(calc_output_names) == {
        "trajectory3_h5",
        "trajectory2_h5",
        "trajectory_h5",
    }


def test_production_maker(interchange, tmp_path, run_job):
    # Create an instance of ProductionMaker with custom parameters
    production_maker = ProductionMaker(
        name="test_production",
        energy_maker=EnergyMinimizationMaker(max_iterations=1),
        npt_maker=NPTMaker(n_steps=5, pressure=1.0, state_interval=1),
        anneal_maker=AnnealMaker.from_temps_and_steps(
            anneal_temp=400, final_temp=300, n_steps=5
        ),
        nvt_maker=NVTMaker(n_steps=5),
    )

    # Run the ProductionMaker flow
    production_flow = production_maker.make(interchange, output_dir=tmp_path)
    task_doc = run_job(production_flow)

    # Check the output task document
    assert isinstance(task_doc, ClassicalMDTaskDocument)
    assert task_doc.state == "successful"
    assert len(task_doc.calcs_reversed) == 6

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
    assert len(calc_output.steps) == 5

    all_steps = [calc.output.steps for calc in task_doc.calcs_reversed]
    assert all_steps == [
        [1, 2, 3, 4, 5],
        [1],
        [1, 2],
        [1, 2],
        [1, 2, 3, 4, 5],
        None,
    ]
    # Test that the state interval is respected
    assert calc_output.steps == list(range(1, 6))
