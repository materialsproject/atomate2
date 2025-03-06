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
from atomate2.openmm.flows.dynamic import DynamicOpenMMFlowMaker
from atomate2.openmm.jobs import EnergyMinimizationMaker, NPTMaker, NVTMaker

def test_dynamic_flow_maker(interchange, run_job):
    from functools import partial

    from jobflow import run_locally

    from atomate2.openmm.flows.dynamic import _get_final_jobs, default_should_continue

    should_continue = partial(
        default_should_continue,
        physical_property="potential_energy",
        threshold=1e-2,
    )
    should_continue.__name__ = "should_continue"

    # Create an instance of DynamicFlowMaker with custom parameters
    dynamic_flow_maker = DynamicOpenMMFlowMaker(
        name="test dynamic equilibration",
        tags=["test"],
        maker=NPTMaker(n_steps=200, pressure=1.0, state_interval=10, traj_interval=10),
        max_stages=10,
        should_continue=should_continue,
    )

    production_flow = dynamic_flow_maker.make(interchange)
    response_dict = run_locally(Flow([production_flow]))
    task_doc = list(response_dict.values())[-1][2].output

    assert isinstance(task_doc, OpenMMTaskDocument)
    assert task_doc.state == "successful"
    assert (len(task_doc.calcs_reversed) - 1) <= dynamic_flow_maker.max_stages
    assert task_doc.calcs_reversed[-1].task_name == "npt simulation"
    assert task_doc.calcs_reversed[0].task_name == "npt simulation"
    assert task_doc.tags == ["test"]
    assert task_doc.job_uuids[0] is not None

    ## Check the individual jobs in the flow
    job_list = _get_final_jobs(production_flow)
    npt_job_0 = job_list[0]
    assert isinstance(npt_job_0.maker, NPTMaker)

    npt_stages = 0
    for job in job_list:
        if isinstance(job.maker, NPTMaker):
            npt_stages += 1

    assert (npt_stages - 1) <= dynamic_flow_maker.max_stages
    assert task_doc.calcs_reversed[0].output.traj_file == f"trajectory{npt_stages}.dcd"
    assert task_doc.calcs_reversed[0].output.traj_file == f"state{npt_stages}.csv"


