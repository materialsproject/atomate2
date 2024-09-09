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


def test_anharmonic_quantification_oneshot(si, clean_dir, mock_aims, species_dir):
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
        bulk_relax_maker=RelaxMaker.full_relaxation(
            user_params=parameters, user_kpoints_settings={"density": 5.0}
        ),
        static_energy_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters, user_kpoints_settings={"density": 5.0}
            )
        ),
        use_symmetrized_structure="primitive",
        phonon_displacement_maker=PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters_phonon_disp,
                user_kpoints_settings={"density": 5.0},
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
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    dct = responses[flow.job_uuids[-1]][1].output.sigma_dict
    assert np.round(dct["one-shot"], 3) == 0.104


def test_anharmonic_quantification_full(si, clean_dir, mock_aims, species_dir):
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
        bulk_relax_maker=RelaxMaker.full_relaxation(
            user_params=parameters, user_kpoints_settings={"density": 5.0}
        ),
        static_energy_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters, user_kpoints_settings={"density": 5.0}
            )
        ),
        use_symmetrized_structure="primitive",
        phonon_displacement_maker=PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters_phonon_disp,
                user_kpoints_settings={"density": 5.0},
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
        seed=1234,
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    dct = responses[flow.job_uuids[-1]][1].output.sigma_dict
    assert pytest.approx(dct["full"], 0.001) == 0.12012


def test_mode_resolved_anharmonic_quantification(si, clean_dir, mock_aims, species_dir):
    # mapping from job name to directory containing test files
    ref_paths = {
        "Relaxation calculation": "phonon-relax-mode-resolved",
        "SCF Calculation": "phonon-energy-mode-resolved",
        "phonon static aims 1/1": "phonon-disp-mode-resolved",
        "phonon static aims anharmonicity quant. 1/3": "anharm-mode-resolved_1",
        "phonon static aims anharmonicity quant. 2/3": "anharm-mode-resolved_2",
        "phonon static aims anharmonicity quant. 3/3": "anharm-mode-resolved_3",
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
        bulk_relax_maker=RelaxMaker.full_relaxation(
            user_params=parameters, user_kpoints_settings={"density": 5.0}
        ),
        static_energy_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters, user_kpoints_settings={"density": 5.0}
            )
        ),
        use_symmetrized_structure="primitive",
        phonon_displacement_maker=PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters_phonon_disp,
                user_kpoints_settings={"density": 5.0},
            )
        ),
    )

    maker = AnharmonicityMaker(
        phonon_maker=phonon_maker,
    )
    maker.name = "anharmonicity"
    flow = maker.make(
        si,
        supercell_matrix=np.array([-2, 2, 2, 2, -2, 2, 2, 2, -2]).reshape((3, 3)),
        one_shot_approx=False,
        seed=1234,
        mode_resolved=True,
        n_samples=3,
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    mode_resolved_vals = responses[flow.job_uuids[-1]][1].output.sigma_dict[
        "mode-resolved"
    ]
    mode_resolved_vals_rounded = np.round(mode_resolved_vals, 3)
    sigmas = mode_resolved_vals_rounded[:, 1]
    assert [3.7470, 5.5000 * (10 ** (-2))] in mode_resolved_vals_rounded
    assert pytest.approx(sigmas.mean(), 0.01) == 0.186
    assert pytest.approx(sigmas.std(), 0.01) == 0.213


def test_site_resolved_anharmonic_quantification(
    nacl, clean_dir, mock_aims, species_dir
):
    # mapping from job name to directory containing test files
    ref_paths = {
        "Relaxation calculation": "phonon-relax-nacl",
        "phonon static aims 1/2": "phonon-disp-nacl_1",
        "phonon static aims 2/2": "phonon-disp-nacl_2",
        "SCF Calculation": "phonon-energy-nacl",
        "phonon static aims anharmonicity quant. 1/1": "anharm-nacl",
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
        bulk_relax_maker=RelaxMaker.full_relaxation(
            user_params=parameters, user_kpoints_settings={"density": 5.0}
        ),
        static_energy_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters, user_kpoints_settings={"density": 5.0}
            )
        ),
        use_symmetrized_structure="primitive",
        phonon_displacement_maker=PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters_phonon_disp,
                user_kpoints_settings={"density": 5.0},
            )
        ),
    )

    maker = AnharmonicityMaker(
        phonon_maker=phonon_maker,
    )
    maker.name = "anharmonicity"
    flow = maker.make(
        nacl,
        supercell_matrix=np.array([-1, 1, 1, 1, -1, 1, 1, 1, -1]).reshape((3, 3)),
        site_resolved=True,
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    nacl_sigma_vals = responses[flow.job_uuids[-1]][1].output.sigma_dict
    nacl_sigma_rounded_sites = [
        (next(iter(arr[0].keys())), np.round(arr[1], 3))
        for arr in nacl_sigma_vals["site-resolved"]
    ]
    assert ("a", 0.076) in nacl_sigma_rounded_sites
    assert ("b", 0.072) in nacl_sigma_rounded_sites


def test_element_resolved_anharmonic_quantification(
    nacl, clean_dir, mock_aims, species_dir
):
    # mapping from job name to directory containing test files
    ref_paths = {
        "Relaxation calculation": "phonon-relax-nacl",
        "phonon static aims 1/2": "phonon-disp-nacl_1",
        "phonon static aims 2/2": "phonon-disp-nacl_2",
        "SCF Calculation": "phonon-energy-nacl",
        "phonon static aims anharmonicity quant. 1/1": "anharm-nacl",
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
        bulk_relax_maker=RelaxMaker.full_relaxation(
            user_params=parameters, user_kpoints_settings={"density": 5.0}
        ),
        static_energy_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters, user_kpoints_settings={"density": 5.0}
            )
        ),
        use_symmetrized_structure="primitive",
        phonon_displacement_maker=PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters_phonon_disp,
                user_kpoints_settings={"density": 5.0},
            )
        ),
    )

    maker = AnharmonicityMaker(
        phonon_maker=phonon_maker,
    )
    maker.name = "anharmonicity"
    flow = maker.make(
        nacl,
        supercell_matrix=np.array([-1, 1, 1, 1, -1, 1, 1, 1, -1]).reshape((3, 3)),
        element_resolved=True,
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    nacl_sigma_vals = responses[flow.job_uuids[-1]][1].output.sigma_dict
    nacl_sigma_rounded_elements = [
        (arr[0], np.round(arr[1], 3)) for arr in nacl_sigma_vals["element-resolved"]
    ]
    assert ("Na", 0.076) in nacl_sigma_rounded_elements
    assert ("Cl", 0.072) in nacl_sigma_rounded_elements


@pytest.mark.skip(reason="Currently not mocked and needs FHI-aims binary")
def test_anharmonic_quantification_socket_oneshot(si, clean_dir, species_dir):
    # mapping from job name to directory containing test files
    parameters = {
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
                user_kpoints_settings={"density": 5.0, "even": True},
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
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    dct = responses[flow.job_uuids[-1]][1].output.sigma_dict
    assert pytest.approx(dct["one-shot"], 0.01) == 0.127
