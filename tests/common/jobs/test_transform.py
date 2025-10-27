"""Test transformation jobs."""

try:
    import icet
except ImportError:
    icet = None

import numpy as np
import pytest
from jobflow import run_locally
from pymatgen.core import Structure
from pymatgen.transformations.advanced_transformations import SQSTransformation

from atomate2.common.jobs.transform import SQS
from atomate2.common.schemas.transform import SQSTask


@pytest.fixture(scope="module")
def simple_alloy() -> Structure:
    """Hexagonal close-packed 50-50 Mg-Al alloy."""
    return Structure(
        3.5
        * np.array(
            [
                [0.5, -(3.0 ** (0.5)) / 2.0, 0.0],
                [0.5, 3.0 ** (0.5) / 2.0, 0.0],
                [0.0, 0.0, 8 ** (0.5) / 3.0],
            ]
        ),
        [{"Mg": 0.5, "Al": 0.5}, {"Mg": 0.5, "Al": 0.5}],
        [[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.5]],
    )


@pytest.mark.skipif(
    icet is None, reason="`icet` must be installed to perform this test."
)
def test_sqs(tmp_dir, simple_alloy):
    # Probably most common use case - just get one "best" SQS
    sqs_trans = SQSTransformation(
        scaling=4,
        best_only=False,
        sqs_method="icet-enumeration",
    )
    job = SQS(transformation=sqs_trans).make(simple_alloy)

    output = run_locally(job)[job.uuid][1].output
    assert isinstance(output, SQSTask)
    assert output.final_structure.composition.as_dict() == {"Mg": 4, "Al": 4}
    assert isinstance(output.final_structure, Structure)
    assert output.final_structure.is_ordered
    assert all(
        getattr(output, attr) is None for attr in ("sqs_structures", "sqs_scores")
    )
    assert isinstance(output.transformation, SQSTransformation)

    # Now simulate retrieving multiple SQSes
    sqs_trans = SQSTransformation(
        scaling=4,
        best_only=False,
        sqs_method="icet-monte_carlo",
        instances=3,
        icet_sqs_kwargs={"n_steps": 10},  # only 10-step search
        remove_duplicate_structures=False,  # needed just to simulate output
    )

    # return up to the two best structures
    job = SQS(transformation=sqs_trans).make(simple_alloy, return_ranked_list=2)
    output = run_locally(job)[job.uuid][1].output

    assert isinstance(output, SQSTask)

    # return_ranked_list - 1 structures and objective functions should be here
    assert all(
        len(getattr(output, attr)) == 1 for attr in ("sqs_structures", "sqs_scores")
    )

    assert all(
        struct.composition.as_dict() == {"Mg": 4, "Al": 4}
        and isinstance(struct, Structure)
        for struct in output.sqs_structures
    )
