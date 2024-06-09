import os

import numpy as np
import pytest
from jobflow import run_locally
from pymatgen.io.aims.sets.core import SocketIOSetGenerator, StaticSetGenerator

from atomate2.aims.flows.anharmonicity import AnharmonicityMaker
from atomate2.aims.flows.phonons import PhononMaker
from atomate2.aims.jobs.core import RelaxMaker, StaticMaker
from atomate2.aims.jobs.phonons import (
    PhononDisplacementMaker,
    PhononDisplacementMakerSocket,
)

cwd = os.getcwd()

def test_anharmonic_quantification_oneshot(si, tmp_path, mock_aims, species_dir):
    # mapping from job name to directory containing test files
    ref_paths = {
        "Relaxation calculation": "phonon-relax-si",
        "phonon static aims 1/1": "phonon-disp-si",
        "SCF Calculation": "phonon-energy-si",
        "phonon static aims anharmonicity quant. 1/1": "anharm-os-si",
    }

    # settings passed to fake_run_aims; adjust these to check for certain input settings
    fake_run_aims_kwargs = {}

    # automatically use fake FHI-aims
    mock_aims(ref_paths, fake_run_aims_kwargs)

    parameters = {
        "species_dir": (species_dir / "light").as_posix(),
        "rlsy_symmetry": "all",
        "sc_accuracy_rho": 1e-06,
        "sc_accuracy_forces": 0.0001,
        "relativistic": "atomic_zora scalar",
    }

    parameters_phonon_disp = dict(compute_forces=True, **parameters)
    parameters_phonon_disp["rlsy_symmetry"] = None

    phonon_maker = PhononMaker(
        bulk_relax_maker=RelaxMaker.full_relaxation(user_params=parameters, user_kpoints_settings={"density": 5.0}),
        static_energy_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters,
                user_kpoints_settings={"density": 5.0}
            )
        ),
        use_symmetrized_structure="primitive",
        phonon_displacement_maker=PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters_phonon_disp,
                user_kpoints_settings={"density": 5.0}
            )
        ),
    )

    maker = AnharmonicityMaker(
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
    assert np.round(responses[flow.job_uuids[-1]][1].output.sigma_A, 3) == 0.104

def test_anharmonic_quantification_full(si, tmp_path, mock_aims, species_dir):
    ref_paths = {
        "Relaxation calculation": "phonon-relax-si-full",
        "phonon static aims 1/1": "phonon-disp-si-full",
        "SCF Calculation": "phonon-energy-si-full",
        "phonon static aims anharmonicity quant. 1/1": "anharm-si-full",
    }

    # settings passed to fake_run_aims; adjust these to check for certain input settings
    fake_run_aims_kwargs = {}

    # automatically use fake FHI-aims
    mock_aims(ref_paths, fake_run_aims_kwargs)

    parameters = {
        "species_dir": (species_dir / "light").as_posix(),
        "rlsy_symmetry": "all",
        "sc_accuracy_rho": 1e-06,
        "sc_accuracy_forces": 0.0001,
        "relativistic": "atomic_zora scalar",
    }

    parameters_phonon_disp = dict(compute_forces=True, **parameters)
    parameters_phonon_disp["rlsy_symmetry"] = None

    phonon_maker = PhononMaker(
        bulk_relax_maker=RelaxMaker.full_relaxation(user_params=parameters, user_kpoints_settings={"density": 5.0}),
        static_energy_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters,
                user_kpoints_settings={"density": 5.0}
            )
        ),
        use_symmetrized_structure="primitive",
        phonon_displacement_maker=PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters_phonon_disp,
                user_kpoints_settings={"density": 5.0}
            )
        ),
    )

    maker = AnharmonicityMaker(
        phonon_maker=phonon_maker,
    )
    maker.name = "anharmonicity"
    flow = maker.make(
        si,
        supercell_matrix=np.array([-1, 1, 1, 1, -1, 1, 1, 1, -1]).reshape((3, 3)),
        one_shot_approx=False,
        seed=1234
    )

    # run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    os.chdir(cwd)
    # return responses[flow.job_uuids[-1]][1].output.sigma_A
    assert np.round(responses[flow.job_uuids[-1]][1].output.sigma_A, 3) == 0.120


@pytest.mark.skip(reason="Currently not mocked and needs FHI-aims binary")
def test_anharmonic_quantification_socket_oneshot(si, tmp_path, species_dir):
    # mapping from job name to directory containing test files
    parameters = {
        "k_grid": [2, 2, 2],
        "species_dir": (species_dir / "light").as_posix(),
        "rlsy_symmetry": "all",
        "sc_accuracy_rho": 1e-06,
        "sc_accuracy_forces": 0.0001,
        "relativistic": "atomic_zora scalar",
    }

    parameters_phonon_disp = dict(
        compute_forces=True, use_pimd_wrapper=("localhost", 12347), **parameters
    )
    parameters_phonon_disp["rlsy_symmetry"] = None

    phonon_maker = PhononMaker(
        bulk_relax_maker=RelaxMaker.full_relaxation(user_params=parameters),
        static_energy_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(user_params=parameters)
        ),
        use_symmetrized_structure="primitive",
        phonon_displacement_maker=PhononDisplacementMakerSocket(
            input_set_generator=SocketIOSetGenerator(
                user_params=parameters_phonon_disp,
            )
        ),
        socket=True,
    )

    maker = AnharmonicityMaker(
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
    assert np.round(responses[flow.job_uuids[-1]][1].output.sigma_A, 3) == 0.125
