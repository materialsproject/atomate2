import pytest
from jobflow import run_locally
from pymatgen.core.structure import Structure

from atomate2.common.jobs.gruneisen import shrink_expand_structure


def test_shrink_expand_structure(clean_dir, si_structure: Structure):
    job = shrink_expand_structure(si_structure, perc_vol=0.01)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    assert isinstance(responses[job.output.uuid][1].output["plus"], Structure)
    assert isinstance(responses[job.output.uuid][1].output["minus"], Structure)

    assert responses[job.output.uuid][1].output[
        "plus"
    ].volume / si_structure.volume == pytest.approx(1.01)
    assert responses[job.output.uuid][1].output[
        "minus"
    ].volume / si_structure.volume == pytest.approx(0.99)
