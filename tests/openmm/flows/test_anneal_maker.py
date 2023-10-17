from atomate2.openmm.flows.anneal_maker import AnnealMaker
from atomate2.openmm.jobs.temp_change_maker import TempChangeMaker


def test_anneal_maker(alchemy_input_set, job_store):
    from atomate2.openmm.jobs.nvt_maker import NVTMaker
    from atomate2.openmm.schemas.openmm_task_document import OpenMMTaskDocument
    from jobflow import run_locally

    anneal_maker = AnnealMaker(
        raise_temp_maker=TempChangeMaker(steps=100, temp_steps=10, final_temp=310),
        nvt_maker=NVTMaker(steps=100, temperature=310),
        lower_temp_maker=TempChangeMaker(steps=100, temp_steps=10),
    )

    anneal_flow = anneal_maker.make(input_set=alchemy_input_set)

    responses = run_locally(flow=anneal_flow)

    for job_response in responses.values():
        assert isinstance(job_response[1].output, OpenMMTaskDocument)
