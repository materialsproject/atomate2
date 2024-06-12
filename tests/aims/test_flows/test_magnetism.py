from __future__ import annotations

import pytest
from jobflow import run_locally
from pymatgen.analysis.magnetism.analyzer import Ordering
from pymatgen.io.aims.sets.core import StaticSetGenerator

from atomate2.aims.jobs.core import RelaxMaker, StaticMaker
from atomate2.common.flows.magnetism import MagneticOrderingsMaker
from atomate2.common.schemas.magnetism import MagneticOrderingsDocument


def test_magnetic_orderings(test_dir, species_dir, mg2mn4o8):
    parameters = {
        "k_grid": [2, 2, 2],
        "species_dir": (species_dir / "light").as_posix(),
        "spin": "collinear",
    }

    maker = MagneticOrderingsMaker(
        static_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(user_params=dict(**parameters))
        ),
        relax_maker=RelaxMaker.full_relaxation(user_params=dict(**parameters)),
    )
    flow = maker.make(mg2mn4o8)

    responses = run_locally(flow, create_folders=True, ensure_success=True)

    final_output = responses[flow.jobs[-1].uuid][1].output

    assert isinstance(final_output, MagneticOrderingsDocument)
    assert len(final_output.outputs) == 3
    assert (
        final_output.ground_state_uuid
        == min(final_output.outputs, key=lambda doc: doc.energy_per_atom).uuid
    )
    assert final_output.ground_state_ordering == Ordering.AFM
    assert final_output.ground_state_energy == pytest.approx(-104.29910777)
    assert final_output.ground_state_energy_per_atom == pytest.approx(-7.44993626929)
