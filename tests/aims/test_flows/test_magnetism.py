from __future__ import annotations

import os

import numpy as np
import pytest
from jobflow import run_locally
from pymatgen.analysis.magnetism.analyzer import Ordering
from pymatgen.io.aims.sets.magnetism import MagneticStaticSetGenerator

from atomate2.aims.jobs.magnetism import MagneticRelaxMaker, MagneticStaticMaker
from atomate2.common.flows.magnetism import MagneticOrderingsMaker
from atomate2.common.schemas.magnetism import MagneticOrderingsDocument

cwd = os.getcwd()


def test_magnetic_orderings(mock_aims, tmp_path, species_dir, mg2mn4o8):
    parameters = {
        "k_grid": [2, 2, 2],
        "species_dir": (species_dir / "light").as_posix(),
    }

    ref_paths = {
        "Magnetic Relaxation calculation 1/3 (fm)": "MgMn2O4_magnetic/relax_1_3_(fm)",
        "Magnetic Relaxation calculation 2/3 (afm)": "MgMn2O4_magnetic/relax_2_3_(afm)",
        "Magnetic Relaxation calculation 3/3 (afm)": "MgMn2O4_magnetic/relax_3_3_(afm)",
        "Magnetic SCF Calculation 1/3 (fm)": "MgMn2O4_magnetic/static_1_3_(fm)",
        "Magnetic SCF Calculation 2/3 (afm)": "MgMn2O4_magnetic/static_2_3_(afm)",
        "Magnetic SCF Calculation 3/3 (afm)": "MgMn2O4_magnetic/static_3_3_(afm)",
    }

    fake_run_aims_kwargs = {}
    mock_aims(ref_paths, fake_run_aims_kwargs)

    maker = MagneticOrderingsMaker(
        static_maker=MagneticStaticMaker(
            input_set_generator=MagneticStaticSetGenerator(
                user_params=dict(**parameters)
            )
        ),
        relax_maker=MagneticRelaxMaker.full_relaxation(user_params=dict(**parameters)),
    )

    flow = maker.make(mg2mn4o8)

    os.chdir(tmp_path)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    final_output = responses[flow.jobs[-1].uuid][1].output

    assert isinstance(final_output, MagneticOrderingsDocument)
    assert len(final_output.outputs) == 3
    assert (
        final_output.ground_state_uuid
        == min(final_output.outputs, key=lambda doc: doc.energy_per_atom).uuid
    )
    magmoms = np.round(
        [
            0.0,
            0.0,
            3.757125,
            3.757112,
            -3.75709,
            -3.757075,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        3,
    )

    assert final_output.ground_state_ordering == Ordering.FiM
    assert final_output.ground_state_energy == pytest.approx(-153874.652021512)
    assert final_output.ground_state_energy_per_atom == pytest.approx(
        -10991.046572965144
    )
    assert np.allclose(np.round(final_output.outputs[2].magmoms, 3), magmoms)
