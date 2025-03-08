from __future__ import annotations

from emmet.core.openmm import OpenMMTaskDocument

from atomate2.openmm.flows.dynamic import (
    DynamicOpenMMFlowMaker,
    default_should_continue,
)
from atomate2.openmm.jobs import NPTMaker


def test_should_continue(interchange, run_job):
    maker = NPTMaker(n_steps=300, pressure=1.0, state_interval=10, traj_interval=10)
    base_job = maker.make(interchange)
    npt_task_doc = run_job(base_job)

    # use low threshold to for should_continue=True
    should_continue = default_should_continue(
        [npt_task_doc],
        stage_index=0,
        max_stages=15,
        physical_property="potential_energy",
        threshold=1e-64,
    )
    should_continue_task_doc = run_job(should_continue)
    assert isinstance(should_continue_task_doc, OpenMMTaskDocument)
    assert should_continue_task_doc.should_continue

    # use high threshold to false should_continue=False
    should_continue = default_should_continue(
        [npt_task_doc],
        stage_index=0,
        max_stages=15,
        physical_property="potential_energy",
        threshold=1e64,
    )
    should_continue_task_doc = run_job(should_continue)
    assert isinstance(should_continue_task_doc, OpenMMTaskDocument)
    assert not should_continue_task_doc.should_continue


def test_dynamic_flow_maker(interchange, run_dynamic_job):
    from functools import partial

    from atomate2.openmm.flows.dynamic import _get_final_jobs, default_should_continue

    should_continue = partial(
        default_should_continue,
        physical_property="potential_energy",
        threshold=1e-3,
    )
    should_continue.__name__ = "should_continue"

    # Create an instance of DynamicFlowMaker with custom parameters
    dynamic_flow_maker = DynamicOpenMMFlowMaker(
        name="test dynamic equilibration",
        tags=["test"],
        maker=NPTMaker(n_steps=300, pressure=1.0, state_interval=10, traj_interval=10),
        max_stages=15,
        should_continue=should_continue,
    )

    dynamic_flow = dynamic_flow_maker.make(interchange)

    # run_job not general for dynamic flow, use run_locally
    task_doc = run_dynamic_job(dynamic_flow)

    assert isinstance(task_doc, OpenMMTaskDocument)
    assert task_doc.state == "successful"
    assert (len(task_doc.calcs_reversed) - 1) <= dynamic_flow_maker.max_stages
    assert task_doc.calcs_reversed[-1].task_name == "npt simulation"
    assert task_doc.calcs_reversed[0].task_name == "npt simulation"
    assert task_doc.tags == ["test"]
    assert task_doc.job_uuids[0] is not None

    ## Check the individual jobs in the flow
    job_list = _get_final_jobs(dynamic_flow)
    npt_job_0 = job_list[0]
    assert isinstance(npt_job_0.maker, NPTMaker)

    npt_stages = 0
    for job in job_list:
        if isinstance(job.maker, NPTMaker):
            npt_stages += 1

    assert (npt_stages - 1) <= dynamic_flow_maker.max_stages
    assert task_doc.calcs_reversed[0].output.traj_file == f"trajectory{npt_stages}.dcd"
    assert task_doc.calcs_reversed[0].output.state_file == f"state{npt_stages}.csv"
