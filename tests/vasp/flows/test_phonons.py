import numpy as np
import pytest
from jobflow import run_locally
from numpy.testing import assert_allclose
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


def test_phonon_wf_vasp_only_displacements3(
    mock_vasp, clean_dir, si_structure: Structure
):
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

    job = PhononMaker(
        min_length=3.0,
        bulk_relax_maker=None,
        born_maker=None,
        use_symmetrized_structure="conventional",
        create_thermal_displacements=False,
        store_force_constants=False,
        prefer_90_degrees=False,
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate the outputs
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [6115.980051, 6059.749756, 5490.929122, 4173.234384, 2194.164562],
    )

    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonon_bandstructure,
        PhononBandStructureSymmLine,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.phonon_dos, PhononDos)
    assert responses[job.jobs[-1].uuid][1].output.thermal_displacement_data is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.structure, Structure)
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert responses[job.jobs[-1].uuid][1].output.force_constants is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.total_dft_energy, -5.74555232
    )
    assert responses[job.jobs[-1].uuid][1].output.born is None
    assert responses[job.jobs[-1].uuid][1].output.epsilon_static is None
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        np.eye(3),
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        (np.ones((3, 3)) - np.eye(3)) / 2,
        atol=1e-8,
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
        == 7_000
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.entropies,
        [0.0, 2.194216, 9.478603, 16.687079, 22.702177],
        atol=1e-6,
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [0.0, 5.750113, 15.408866, 19.832123, 21.842104],
        atol=1e-6,
    )

    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.internal_energies,
        [6115.980051, 6279.17132, 7386.649622, 9179.358187, 11275.035523],
        atol=1e-6,
    )
    assert responses[job.jobs[-1].uuid][1].output.chemsys == "Si"


# structure will be kept in the format that was transferred
def test_phonon_wf_vasp_only_displacements_no_structural_transformation(
    mock_vasp, clean_dir, si_structure: Structure
):
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

    job = PhononMaker(
        min_length=3.0,
        bulk_relax_maker=None,
        born_maker=None,
        use_symmetrized_structure=None,
        create_thermal_displacements=False,
        store_force_constants=False,
        prefer_90_degrees=False,
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate the outputs
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [5927.157337, 5905.309813, 5439.530414, 4207.379685, 2297.576147],
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.entropies,
        [0.0, 1.256496, 8.511348, 15.928285, 22.063785],
        atol=1e-6,
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [0.0, 4.958763, 15.893881, 20.311967, 22.196143],
    )

    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.internal_energies,
        [5927.157337, 6030.959432, 7141.800004, 8985.865319, 11123.090225],
    )

    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonon_bandstructure,
        PhononBandStructureSymmLine,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.phonon_dos, PhononDos)
    assert responses[job.jobs[-1].uuid][1].output.thermal_displacement_data is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.structure, Structure)
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert responses[job.jobs[-1].uuid][1].output.force_constants is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.total_dft_energy, -5.74525804
    )
    assert responses[job.jobs[-1].uuid][1].output.born is None
    assert responses[job.jobs[-1].uuid][1].output.epsilon_static is None
    assert responses[job.jobs[-1].uuid][1].output.supercell_matrix == tuple(
        map(tuple, np.eye(3))
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        atol=1e-8,
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
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
    phonopy_settings = responses[job.jobs[-1].uuid][1].output.phonopy_settings
    assert phonopy_settings.kpoint_density_dos == 7_000
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.entropies,
        [0.0, 1.256496, 8.511348, 15.928285, 22.063785],
        atol=1e-6,
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [0.0, 4.958763, 15.893881, 20.311967, 22.196143],
        atol=1e-6,
    )

    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.internal_energies,
        [5927.157337, 6030.959432, 7141.800004, 8985.865319, 11123.090225],
        atol=1e-6,
    )


# this will test all kpath schemes in combination with primitive cell
@pytest.mark.parametrize(
    "kpath_scheme", ["seekpath", "hinuma", "setyawan_curtarolo", "latimer_munro"]
)
def test_phonon_wf_vasp_only_displacements_kpath(
    mock_vasp, clean_dir, kpath_scheme, si_structure: Structure
):
    # mapping from job name to directory containing test files
    ref_paths = {"phonon static 1/1": "Si_phonons_1/phonon_static_1_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    job = PhononMaker(
        min_length=3.0,
        bulk_relax_maker=None,
        static_energy_maker=None,
        born_maker=None,
        use_symmetrized_structure="primitive",
        kpath_scheme=kpath_scheme,
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
        create_thermal_displacements=True,
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate the outputs
    # print(type(responses))
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [5776.14995034, 5617.74737777, 4725.50269363, 3043.81827626, 694.49078355],
        atol=1e-3,
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
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert_allclose(
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
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        atol=1e-8,
    )
    assert responses[job.jobs[-1].uuid][1].output.code == "vasp"
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonopy_settings,
        PhononComputationalSettings,
    )
    assert responses[job.jobs[-1].uuid][1].output.phonopy_settings.npoints_band == 101
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpath_scheme
        == kpath_scheme
    )
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpoint_density_dos
        == 7_000
    )


# test supply of born charges, epsilon, DFT energy, supercell
def test_phonon_wf_vasp_only_displacements_add_inputs_raises(
    mock_vasp, clean_dir, si_structure: Structure
):
    # mapping from job name to directory containing test files
    ref_paths = {"phonon static 1/1": "Si_phonons_1/phonon_static_1_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    born = [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0.1]],
    ]
    epsilon_static = [
        [5.25, 0, 0],
        [0, 5.25, 0],
        [0, 0, 5.25],
    ]
    total_dft_energy_per_formula_unit = -5

    job = PhononMaker(
        min_length=3,
        bulk_relax_maker=None,
        static_energy_maker=None,
        born_maker=None,
        use_symmetrized_structure="primitive",
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
        create_thermal_displacements=True,
    ).make(
        structure=si_structure,
        total_dft_energy_per_formula_unit=total_dft_energy_per_formula_unit,
        born=born,
        epsilon_static=epsilon_static,
    )
    with pytest.raises(RuntimeError):
        run_locally(job, create_folders=True, ensure_success=True)


# test supply of born charges, epsilon, DFT energy, supercell
def test_phonon_wf_vasp_only_displacements_add_inputs(
    mock_vasp, clean_dir, si_structure: Structure
):
    # mapping from job name to directory containing test files
    ref_paths = {"phonon static 1/1": "Si_phonons_1/phonon_static_1_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    born = [
        [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ]
    epsilon_static = [
        [5.25, 0, 0],
        [0, 5.25, 0],
        [0, 0, 5.25],
    ]
    total_dft_energy_per_formula_unit = -5
    job = PhononMaker(
        min_length=3,
        bulk_relax_maker=None,
        static_energy_maker=None,
        born_maker=None,
        use_symmetrized_structure="primitive",
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
        create_thermal_displacements=True,
    ).make(
        structure=si_structure,
        total_dft_energy_per_formula_unit=total_dft_energy_per_formula_unit,
        born=born,
        epsilon_static=epsilon_static,
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate the outputs
    # print(type(responses))
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [5776.14995034, 5617.74737777, 4725.50269363, 3043.81827626, 694.49078355],
        atol=1e-3,
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
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.force_constants.force_constants[0][0][0][
            0
        ],
        13.032324,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert_allclose(responses[job.jobs[-1].uuid][1].output.born, born)
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.total_dft_energy,
        total_dft_energy_per_formula_unit,
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.epsilon_static, epsilon_static, atol=1e-8
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        atol=1e-8,
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
        == 7_000
    )


# test optional parameters
def test_phonon_wf_vasp_only_displacements_optional_settings(
    mock_vasp, clean_dir, si_structure: Structure
):
    # mapping from job name to directory containing test files
    ref_paths = {"phonon static 1/1": "Si_phonons_1/phonon_static_1_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    job = PhononMaker(
        min_length=3,
        bulk_relax_maker=None,
        static_energy_maker=None,
        born_maker=None,
        use_symmetrized_structure="primitive",
        create_thermal_displacements=False,
        store_force_constants=False,
        prefer_90_degrees=False,
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate the outputs
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [5776.14995034, 5617.74737777, 4725.50269363, 3043.81827626, 694.49078355],
        atol=1e-3,
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.entropies,
        [0, 4.79066818, 13.03470621, 20.37400284, 26.41425489],
        atol=1e-8,
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [0.0, 8.05373626, 15.98005669, 19.98031234, 21.88513476],
        atol=1e-8,
    )

    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.internal_energies,
        [5776.14995034, 6096.81419519, 7332.44393529, 9156.01912756, 11260.1927412],
        atol=1e-8,
    )

    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonon_bandstructure,
        PhononBandStructureSymmLine,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.phonon_dos, PhononDos)
    assert responses[job.jobs[-1].uuid][1].output.thermal_displacement_data is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.structure, Structure)
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert responses[job.jobs[-1].uuid][1].output.force_constants is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert responses[job.jobs[-1].uuid][1].output.total_dft_energy is None
    assert responses[job.jobs[-1].uuid][1].output.born is None
    assert responses[job.jobs[-1].uuid][1].output.epsilon_static is None
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        atol=1e-8,
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
        == 7_000
    )


# test run including all steps of the computation for Si
def test_phonon_wf_vasp_all_steps(mock_vasp, clean_dir, si_structure: Structure):
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
        create_thermal_displacements=True,
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate the outputs
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [5853.74150399, 5692.29089555, 4798.67784919, 3122.48296003, 782.17345333],
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
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.force_constants.force_constants[0][0][0][
            0
        ],
        13.41185599,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert_allclose(responses[job.jobs[-1].uuid][1].output.born, np.zeros((2, 3, 3)))
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.total_dft_energy, -5.74629058
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.epsilon_static,
        ((13.19242034, -0.0, 0.0), (-0.0, 13.19242034, 0.0), (0.0, 0.0, 13.19242034)),
        atol=1e-8,
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
    )
    assert_allclose(
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
        == 7_000
    )


# use a structure where Born charges are actually useful for the computation and change
# the values


# test raises?
# would be good to check if ValueErrors are raised when certain kpath schemes are
# combined with non-standard-primitive structures
# this will test all kpath schemes in combination with primitive cell
@pytest.mark.parametrize(
    "kpath_scheme", ["hinuma", "setyawan_curtarolo", "latimer_munro"]
)
def test_phonon_wf_vasp_only_displacements_kpath_raises_no_cell_change(
    mock_vasp, clean_dir, kpath_scheme, si_structure: Structure
):
    # mapping from job name to directory containing test files
    ref_paths = {"phonon static 1/1": "Si_phonons_1/phonon_static_1_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    with pytest.raises(
        ValueError,
        match=f"You can't use {kpath_scheme=} with the primitive standard "
        "structure, please use seekpath",
    ):
        PhononMaker(
            min_length=3.0,
            bulk_relax_maker=None,
            static_energy_maker=None,
            born_maker=None,
            use_symmetrized_structure=None,
            kpath_scheme=kpath_scheme,
            generate_frequencies_eigenvectors_kwargs={"tstep": 100},
        ).make(si_structure)


@pytest.mark.parametrize(
    "kpath_scheme", ["hinuma", "setyawan_curtarolo", "latimer_munro"]
)
def test_phonon_wf_vasp_only_displacements_kpath_raises(
    mock_vasp, clean_dir, kpath_scheme, si_structure: Structure
):
    # mapping from job name to directory containing test files
    ref_paths = {"phonon static 1/1": "Si_phonons_1/phonon_static_1_1"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {"phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)
    with pytest.raises(
        ValueError,
        match=f"You can't use {kpath_scheme=} with the primitive standard "
        "structure, please use seekpath",
    ):
        PhononMaker(
            min_length=3.0,
            bulk_relax_maker=None,
            static_energy_maker=None,
            born_maker=None,
            use_symmetrized_structure="conventional",
            kpath_scheme=kpath_scheme,
            generate_frequencies_eigenvectors_kwargs={"tstep": 100},
        ).make(si_structure)


def test_phonon_wf_vasp_all_steps_na_cl(mock_vasp, clean_dir):
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

    # validate the outputs
    assert isinstance(responses[phonon_flow.jobs[-1].uuid][1].output, PhononBSDOSDoc)
    assert_allclose(
        responses[phonon_flow.jobs[-1].uuid][1].output.born,
        [
            ((1.1033, 0.0, 0.0), (0.0, 1.1033, -0.0), (0.0, 0.0, 1.1033)),
            ((-1.1033, 0.0, -0.0), (-0.0, -1.1033, 0.0), (-0.0, 0.0, -1.1033)),
        ],
    )

    def test_phonon_wf_vasp_all_steps_na_cl(mock_vasp, clean_dir):
        structure = Structure(
            lattice=[
                [2.30037148, -3.98436029, 0.00000000],
                [2.30037148, 3.98436029, 0.00000000],
                [0.00000000, 0.00000000, 7.28132999],
            ],
            species=["Mg", "Mg", "Mg", "Sb", "Sb"],
            coords=[
                [0.0, 0.0, 0.0],
                [0.33333333, 0.66666666, 0.36832500],
                [0.66666666, 0.33333333, 0.63167500],
                [0.33333333, 0.66666666, 0.77474900],
                [0.66666666, 0.33333333, 0.22525100],
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

        # validate the outputs
        assert isinstance(
            responses[phonon_flow.jobs[-1].uuid][1].output, PhononBSDOSDoc
        )
