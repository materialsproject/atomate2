import os

import numpy as np
from jobflow import run_locally
from pymatgen.io.aims.sets.core import StaticSetGenerator

from atomate2.aims.flows.phonons import PhononMaker
from atomate2.aims.jobs.core import RelaxMaker, StaticMaker
from atomate2.aims.jobs.phonons import PhononDisplacementMaker
from atomate2.common.flows.anharmonicity import BaseAnharmonicityMaker

cwd = os.get_cwd()


def test_anharmonic_quantification(si, tmp_path, species_dir):
    parameters = {
        "k_grid": [2, 2, 2],
        "species_dir": (species_dir / "light").as_posix(),
    }

    parameters_phonon_disp = dict(compute_forces=True, **parameters)

    phonon_maker = PhononMaker(
        bulk_relax_maker=RelaxMaker.full_relaxation(user_params=parameters),
        static_energy_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(user_params=parameters)
        ),
        use_symmetrized_structure="primitive",
        phonon_displacement_maker=PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters_phonon_disp,
            )
        ),
    )

    maker = BaseAnharmonicityMaker(
        phonon_maker=phonon_maker,
    )
    maker.name = "anharmonicity"
    flow = maker.make(
        si,
        supercell_matrix=np.array([-1, 1, 1, 1, -1, 1, 1, 1, -1]).reshape((3, 3)),
    )

    # run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    assert np.round(responses[flow.job_uuids[-1]][1].output, 3) == 0.163
