import numpy as np
from jobflow import run_locally
from pymatgen.io.aims.sets.core import StaticSetGenerator

from atomate2.aims.flows.phonons import PhononMaker
from atomate2.aims.jobs.core import RelaxMaker, StaticMaker
from atomate2.aims.jobs.phonons import PhononDisplacementMaker
from atomate2.common.flows.anharmonicity import BaseAnharmonicityMaker

from pathlib import Path
from pymatgen.core.structure import Structure, Lattice

def test_anharmonic_quantification(si, tmp_path, mock_aims, species_dir):
    # species_dir = Path(
    #     "/Users/kevinbeck/Software/fhi-aims.231212/species_defaults/defaults_2020"
    # )
    # si = Structure(
    #     lattice=Lattice(
    #         [[0.0, 2.815, 2.815], [2.815, 0.0, 2.815], [2.815, 2.815, 0.0]]
    #     ),
    #     species=["Si", "Si"],
    #     coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    # )

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
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    assert np.round(responses[flow.job_uuids[-1]][1].output, 3) == 0.104

