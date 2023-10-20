import numpy as np
import pytest
from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

from atomate2.common.schemas.phonons import (
    PhononBSDOSDoc,
    PhononComputationalSettings,
    PhononJobDirs,
    PhononUUIDs,
    ThermalDisplacementData,
)
from atomate2.vasp.flows.phonons import PhononMaker


def test_phonon_wf_only_displacements3(mock_vasp, clean_dir):
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # mapping from job name to directory containing test files
    ref_paths = {
        "phonon static 1/1": "Si_phonons_2/phonon_static_1_1",
        "static": "Si_phonons_2/static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # !!! Generate job
    job = PhononMaker(
        min_length=3.0,
        bulk_relax_maker=None,
        born_maker=None,
        use_symmetrized_structure="conventional",
        create_thermal_displacements=False,
        store_force_constants=False,
        prefer_90_degrees=False,
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
    ).make(structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # !!! validation on the outputs
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [
            5774.603553771463,
            5616.334060911681,
            4724.766198084037,
            3044.208072582665,
            696.3373193497828,
        ],
    )

    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonon_bandstructure,
        PhononBandStructureSymmLine,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.phonon_dos, PhononDos)
    assert responses[job.jobs[-1].uuid][1].output.thermal_displacement_data is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.structure, Structure)
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert responses[job.jobs[-1].uuid][1].output.force_constants is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert np.isclose(
        responses[job.jobs[-1].uuid][1].output.total_dft_energy, -5.74555232
    )
    assert responses[job.jobs[-1].uuid][1].output.born is None
    assert responses[job.jobs[-1].uuid][1].output.epsilon_static is None
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        (
            (0, 0.5000000000000001, 0.5000000000000001),
            (0.5000000000000001, 0.0, 0.5000000000000001),
            (0.5000000000000001, 0.5000000000000001, 0.0),
        ),
    )
    assert responses[job.jobs[-1].uuid][1].output.code == "vasp"
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonopy_settings,
        PhononComputationalSettings,
    )
    assert responses[job.jobs[-1].uuid][1].output.phonopy_settings.npoints_band == 101
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpath_scheme
        == "seekpath"
    )
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpoint_density_dos
        == 7000
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.entropies,
        [
            0.0,
            4.786689009990746,
            13.02544271435008,
            20.360935069065423,
            26.39830736008501,
        ],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [
            0.0,
            8.047577577838897,
            15.97117761314484,
            19.97051059716143,
            21.87494655884403,
        ],
    )

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.internal_energies,
        [
            5774.603553771463,
            6095.002960937394,
            7329.854739783488,
            9152.488591840654,
            11255.660261586278,
        ],
    )
    assert responses[job.jobs[-1].uuid][1].output.chemsys == "Si"


# structure will be kept in the format that was transferred
def test_phonon_wf_only_displacements_no_structural_transformation(
    mock_vasp, clean_dir
):
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # mapping from job name to directory containing test files
    ref_paths = {
        "phonon static 1/1": "Si_phonons_3/phonon_static_1_1",
        "static": "Si_phonons_3/static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # !!! Generate job
    job = PhononMaker(
        min_length=3.0,
        bulk_relax_maker=None,
        born_maker=None,
        use_symmetrized_structure=None,
        create_thermal_displacements=False,
        store_force_constants=False,
        prefer_90_degrees=False,
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
    ).make(structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # !!! validation on the outputs
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [
            5774.566996471001,
            5616.29786373465,
            4724.736849262271,
            3044.193412800876,
            696.3435315493462,
        ],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.entropies,
        [
            0.0,
            4.78666294741712,
            13.025332342984333,
            20.36075467024152,
            26.398072464162844,
        ],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [
            0.0,
            8.047497695382027,
            15.971019069215203,
            19.970326488158854,
            21.874752681396565,
        ],
    )

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.internal_energies,
        [
            5774.566996471001,
            6094.964157503006,
            7329.8033166885825,
            9152.419812411707,
            11255.57251541699,
        ],
    )

    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonon_bandstructure,
        PhononBandStructureSymmLine,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.phonon_dos, PhononDos)
    assert responses[job.jobs[-1].uuid][1].output.thermal_displacement_data is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.structure, Structure)
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert responses[job.jobs[-1].uuid][1].output.force_constants is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert np.isclose(
        responses[job.jobs[-1].uuid][1].output.total_dft_energy, -5.74525804
    )
    assert responses[job.jobs[-1].uuid][1].output.born is None
    assert responses[job.jobs[-1].uuid][1].output.epsilon_static is None
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        ((-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)),
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        (
            (1.0000000000000002, 0.0, 0.0),
            (0.0, 1.0000000000000002, 0.0),
            (0.0, 0.0, 1.0000000000000002),
        ),
    )
    assert responses[job.jobs[-1].uuid][1].output.code == "vasp"
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonopy_settings,
        PhononComputationalSettings,
    )
    assert responses[job.jobs[-1].uuid][1].output.phonopy_settings.npoints_band == 101
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpath_scheme
        == "seekpath"
    )
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpoint_density_dos
        == 7000
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.entropies,
        [
            0.0,
            4.78666294741712,
            13.025332342984333,
            20.36075467024152,
            26.398072464162844,
        ],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [
            0.0,
            8.047497695382027,
            15.971019069215203,
            19.970326488158854,
            21.874752681396565,
        ],
    )

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.internal_energies,
        [
            5774.566996471001,
            6094.964157503006,
            7329.8033166885825,
            9152.419812411707,
            11255.57251541699,
        ],
    )


# this will test all kpath schemes in combination with primitive cell
@pytest.mark.parametrize(
    "kpathscheme", ["seekpath", "hinuma", "setyawan_curtarolo", "latimer_munro"]
)
def test_phonon_wf_only_displacements_kpath(mock_vasp, clean_dir, kpathscheme):
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # mapping from job name to directory containing test files
    ref_paths = {"phonon static 1/1": "Si_phonons_1/phonon_static_1_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # !!! Generate job
    job = PhononMaker(
        min_length=3.0,
        bulk_relax_maker=None,
        static_energy_maker=None,
        born_maker=None,
        use_symmetrized_structure="primitive",
        kpath_scheme=kpathscheme,
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
    ).make(structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # !!! validation on the outputs
    # print(type(responses))
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [
            5776.1499503440455,
            5617.747377776762,
            4725.502693639196,
            3043.818276263367,
            694.4907835517783,
        ],
    )

    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonon_bandstructure,
        PhononBandStructureSymmLine,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.phonon_dos, PhononDos)
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.thermal_displacement_data,
        ThermalDisplacementData,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.structure, Structure)
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert np.isclose(
        responses[job.jobs[-1].uuid][1].output.force_constants.force_constants[0][0][0][
            0
        ],
        13.032324,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert responses[job.jobs[-1].uuid][1].output.total_dft_energy is None
    assert responses[job.jobs[-1].uuid][1].output.born is None
    assert responses[job.jobs[-1].uuid][1].output.epsilon_static is None
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert responses[job.jobs[-1].uuid][1].output.code == "vasp"
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonopy_settings,
        PhononComputationalSettings,
    )
    assert responses[job.jobs[-1].uuid][1].output.phonopy_settings.npoints_band == 101
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpath_scheme
        == kpathscheme
    )
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpoint_density_dos
        == 7000
    )


# test supply of born charges, epsilon, dft energy, supercell
def test_phonon_wf_only_displacements_add_inputs_raises(mock_vasp, clean_dir):
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # mapping from job name to directory containing test files
    ref_paths = {"phonon static 1/1": "Si_phonons_1/phonon_static_1_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    born = [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1]],
    ]
    epsilon_static = [
        [5.25, 0.0, 0.0],
        [0.0, 5.25, -0.0],
        [0.0, 0.0, 5.25],
    ]
    total_dft_energy_per_formula_unit = -5

    job = PhononMaker(
        min_length=3.0,
        bulk_relax_maker=None,
        static_energy_maker=None,
        born_maker=None,
        use_symmetrized_structure="primitive",
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
    ).make(
        structure=structure,
        total_dft_energy_per_formula_unit=total_dft_energy_per_formula_unit,
        born=born,
        epsilon_static=epsilon_static,
    )
    with pytest.raises(RuntimeError):
        run_locally(job, create_folders=True, ensure_success=True)


# test supply of born charges, epsilon, dft energy, supercell
def test_phonon_wf_only_displacements_add_inputs(mock_vasp, clean_dir):
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # mapping from job name to directory containing test files
    ref_paths = {"phonon static 1/1": "Si_phonons_1/phonon_static_1_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    born = [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
    epsilon_static = [
        [5.25, 0.0, 0.0],
        [0.0, 5.25, -0.0],
        [0.0, 0.0, 5.25],
    ]
    total_dft_energy_per_formula_unit = -5
    job = PhononMaker(
        min_length=3.0,
        bulk_relax_maker=None,
        static_energy_maker=None,
        born_maker=None,
        use_symmetrized_structure="primitive",
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
    ).make(
        structure=structure,
        total_dft_energy_per_formula_unit=total_dft_energy_per_formula_unit,
        born=born,
        epsilon_static=epsilon_static,
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # !!! validation on the outputs
    # print(type(responses))
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [
            5776.1499503440455,
            5617.747377776762,
            4725.502693639196,
            3043.818276263367,
            694.4907835517783,
        ],
    )

    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonon_bandstructure,
        PhononBandStructureSymmLine,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.phonon_dos, PhononDos)
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.thermal_displacement_data,
        ThermalDisplacementData,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.structure, Structure)
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert np.isclose(
        responses[job.jobs[-1].uuid][1].output.force_constants.force_constants[0][0][0][
            0
        ],
        13.032324,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert np.allclose(responses[job.jobs[-1].uuid][1].output.born, born)
    assert np.isclose(
        responses[job.jobs[-1].uuid][1].output.total_dft_energy,
        total_dft_energy_per_formula_unit,
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.epsilon_static, epsilon_static
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert responses[job.jobs[-1].uuid][1].output.code == "vasp"
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonopy_settings,
        PhononComputationalSettings,
    )
    assert responses[job.jobs[-1].uuid][1].output.phonopy_settings.npoints_band == 101
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpath_scheme
        == "seekpath"
    )
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpoint_density_dos
        == 7000
    )


# test optional parameters
def test_phonon_wf_only_displacements_optional_settings(mock_vasp, clean_dir):
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # mapping from job name to directory containing test files
    ref_paths = {"phonon static 1/1": "Si_phonons_1/phonon_static_1_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # !!! Generate job
    job = PhononMaker(
        min_length=3.0,
        bulk_relax_maker=None,
        static_energy_maker=None,
        born_maker=None,
        use_symmetrized_structure="primitive",
        create_thermal_displacements=False,
        store_force_constants=False,
        prefer_90_degrees=False,
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
    ).make(structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # !!! validation on the outputs
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [
            5776.1499503440455,
            5617.747377776762,
            4725.502693639196,
            3043.818276263367,
            694.4907835517783,
        ],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.entropies,
        [
            0.0,
            4.790668183967239,
            13.034706214127153,
            20.374002842560785,
            26.414254898744545,
        ],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [
            0.0,
            8.053736263830405,
            15.980056690395037,
            19.980312349314378,
            21.885134767453195,
        ],
    )

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.internal_energies,
        [
            5776.1499503440455,
            6096.814195199836,
            7332.443935293648,
            9156.019127569401,
            11260.192741251365,
        ],
    )

    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonon_bandstructure,
        PhononBandStructureSymmLine,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.phonon_dos, PhononDos)
    assert responses[job.jobs[-1].uuid][1].output.thermal_displacement_data is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.structure, Structure)
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert responses[job.jobs[-1].uuid][1].output.force_constants is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert responses[job.jobs[-1].uuid][1].output.total_dft_energy is None
    assert responses[job.jobs[-1].uuid][1].output.born is None
    assert responses[job.jobs[-1].uuid][1].output.epsilon_static is None
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert responses[job.jobs[-1].uuid][1].output.code == "vasp"
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonopy_settings,
        PhononComputationalSettings,
    )
    assert responses[job.jobs[-1].uuid][1].output.phonopy_settings.npoints_band == 101
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpath_scheme
        == "seekpath"
    )
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpoint_density_dos
        == 7000
    )


# test run including all steps of the computation for Si
def test_phonon_wf_all_steps(mock_vasp, clean_dir):
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # mapping from job name to directory containing test files
    ref_paths = {
        "phonon static 1/1": "Si_phonons_4/phonon_static_1_1",
        "static": "Si_phonons_4/static",
        "tight relax 1": "Si_phonons_4/tight_relax_1",
        "tight relax 2": "Si_phonons_4/tight_relax_2",
        "dielectric": "Si_phonons_4/dielectric",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
        "dielectric": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    job = PhononMaker(
        min_length=3.0,
        use_symmetrized_structure=None,
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
    ).make(structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # !!! validation on the outputs
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [
            5853.741503991992,
            5692.290895556432,
            4798.677849195808,
            3122.482960037922,
            782.1734533334413,
        ],
    )

    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonon_bandstructure,
        PhononBandStructureSymmLine,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.phonon_dos, PhononDos)
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.thermal_displacement_data,
        ThermalDisplacementData,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.structure, Structure)
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert np.isclose(
        responses[job.jobs[-1].uuid][1].output.force_constants.force_constants[0][0][0][
            0
        ],
        13.411855999999997,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.born,
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    assert np.isclose(
        responses[job.jobs[-1].uuid][1].output.total_dft_energy, -5.746290585
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.epsilon_static,
        ((13.19242034, -0.0, 0.0), (-0.0, 13.19242034, 0.0), (0.0, 0.0, 13.19242034)),
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert responses[job.jobs[-1].uuid][1].output.code == "vasp"
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonopy_settings,
        PhononComputationalSettings,
    )
    assert responses[job.jobs[-1].uuid][1].output.phonopy_settings.npoints_band == 101
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpath_scheme
        == "seekpath"
    )
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpoint_density_dos
        == 7000
    )


# use a structure where Born charges are actually useful for the computation and change
# the values


# test raises?
# would be good to check if ValueErrors are raised when certain kpath schemes are
# combined with non-standard-primitive structures
# this will test all kpath schemes in combination with primitive cell
@pytest.mark.parametrize(
    "kpathscheme", ["hinuma", "setyawan_curtarolo", "latimer_munro"]
)
def test_phonon_wf_only_displacements_kpath_raises_no_cell_change(
    mock_vasp, clean_dir, kpathscheme
):
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # mapping from job name to directory containing test files
    ref_paths = {"phonon static 1/1": "Si_phonons_1/phonon_static_1_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    with pytest.raises(ValueError):
        PhononMaker(
            min_length=3.0,
            bulk_relax_maker=None,
            static_energy_maker=None,
            born_maker=None,
            use_symmetrized_structure=None,
            kpath_scheme=kpathscheme,
            generate_frequencies_eigenvectors_kwargs={"tstep": 100},
        ).make(structure)


@pytest.mark.parametrize(
    "kpathscheme", ["hinuma", "setyawan_curtarolo", "latimer_munro"]
)
def test_phonon_wf_only_displacements_kpath_raises(mock_vasp, clean_dir, kpathscheme):
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # mapping from job name to directory containing test files
    ref_paths = {"phonon static 1/1": "Si_phonons_1/phonon_static_1_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)
    with pytest.raises(ValueError):
        # !!! Generate job
        PhononMaker(
            min_length=3.0,
            bulk_relax_maker=None,
            static_energy_maker=None,
            born_maker=None,
            use_symmetrized_structure="conventional",
            kpath_scheme=kpathscheme,
            generate_frequencies_eigenvectors_kwargs={"tstep": 100},
        ).make(structure)


def test_phonon_wf_all_steps_na_cl(mock_vasp, clean_dir):
    from jobflow import run_locally
    from pymatgen.core.structure import Structure

    from atomate2.vasp.flows.phonons import PhononMaker

    structure = Structure(
        lattice=[
            [5.691694, 0.000000, 0.000000],
            [-0.000000, 5.691694, 0.000000],
            [0.000000, 0.000000, 5.691694],
        ],
        species=["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ],
    )

    # mapping from job name to directory containing test files
    ref_paths = {
        "dielectric": "NaCl_phonons/dielectric",
        "phonon static 1/2": "NaCl_phonons/phonon_static_1_2",
        "phonon static 2/2": "NaCl_phonons/phonon_static_2_2",
        "static": "NaCl_phonons/static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "dielectric": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 1/2": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 2/2": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    phonon_flow = PhononMaker(min_length=3.0, bulk_relax_maker=None).make(structure)

    # run the job
    responses = run_locally(phonon_flow, create_folders=True, ensure_success=True)

    # !!! validation on the outputs
    assert isinstance(responses[phonon_flow.jobs[-1].uuid][1].output, PhononBSDOSDoc)
    assert np.allclose(
        responses[phonon_flow.jobs[-1].uuid][1].output.born,
        [
            ((1.1033, 0.0, 0.0), (0.0, 1.1033, -0.0), (0.0, 0.0, 1.1033)),
            ((-1.1033, 0.0, -0.0), (-0.0, -1.1033, 0.0), (-0.0, 0.0, -1.1033)),
        ],
    )

    def test_phonon_wf_all_steps_na_cl(mock_vasp, clean_dir):
        from jobflow import run_locally
        from pymatgen.core.structure import Structure

        from atomate2.vasp.flows.phonons import PhononMaker

        structure = Structure(
            lattice=[
                [2.3003714889113018, -3.9843602950772405, 0.0000000000000000],
                [2.3003714889113018, 3.9843602950772405, 0.0000000000000000],
                [0.0000000000000000, 0.0000000000000000, 7.2813299999999996],
            ],
            species=["Mg", "Mg", "Mg", "Sb", "Sb"],
            coords=[
                [0.0, 0.0, 0.0],
                [0.3333333333333333, 0.6666666666666666, 0.3683250000000000],
                [0.6666666666666667, 0.3333333333333334, 0.6316750000000000],
                [0.3333333333333333, 0.6666666666666666, 0.7747490000000000],
                [0.6666666666666667, 0.3333333333333334, 0.2252510000000000],
            ],
        )

        ref_paths = {
            "dielectric": "Mg3Sb2_phonons/dielectric",
            "phonon static 1/10": "Mg3Sb2_phonons/phonon_static_1_10",
            "phonon static 10/10": "Mg3Sb2_phonons/phonon_static_10_10",
            "phonon static 2/10": "Mg3Sb2_phonons/phonon_static_2_10",
            "phonon static 3/10": "Mg3Sb2_phonons/phonon_static_3_10",
            "phonon static 4/10": "Mg3Sb2_phonons/phonon_static_4_10",
            "phonon static 5/10": "Mg3Sb2_phonons/phonon_static_5_10",
            "phonon static 6/10": "Mg3Sb2_phonons/phonon_static_6_10",
            "phonon static 7/10": "Mg3Sb2_phonons/phonon_static_7_10",
            "phonon static 8/10": "Mg3Sb2_phonons/phonon_static_8_10",
            "phonon static 9/10": "Mg3Sb2_phonons/phonon_static_9_10",
        }

        # settings passed to fake_run_vasp; adjust these to check for certain INCAR
        # settings
        fake_run_vasp_kwargs = {
            "dielectric": {"incar_settings": ["NSW", "ISMEAR"]},
            "phonon static 1/10": {"incar_settings": ["NSW", "ISMEAR"]},
            "phonon static 10/10": {"incar_settings": ["NSW", "ISMEAR"]},
            "phonon static 2/10": {"incar_settings": ["NSW", "ISMEAR"]},
            "phonon static 3/10": {"incar_settings": ["NSW", "ISMEAR"]},
            "phonon static 4/10": {"incar_settings": ["NSW", "ISMEAR"]},
            "phonon static 5/10": {"incar_settings": ["NSW", "ISMEAR"]},
            "phonon static 6/10": {"incar_settings": ["NSW", "ISMEAR"]},
            "phonon static 7/10": {"incar_settings": ["NSW", "ISMEAR"]},
            "phonon static 8/10": {"incar_settings": ["NSW", "ISMEAR"]},
            "phonon static 9/10": {"incar_settings": ["NSW", "ISMEAR"]},
        }

        # automatically use fake VASP and write POTCAR.spec during the test
        mock_vasp(ref_paths, fake_run_vasp_kwargs)

        phonon_flow = PhononMaker(
            min_length=3.0,
            bulk_relax_maker=None,
            static_energy_maker=None,
            use_symmetrized_structure="primitive",
            kpath_scheme="setyawan_curtarolo",
        ).make(structure)

        # run the job
        responses = run_locally(phonon_flow, create_folders=True, ensure_success=True)

        # !!! validation on the outputs
        assert isinstance(
            responses[phonon_flow.jobs[-1].uuid][1].output, PhononBSDOSDoc
        )
