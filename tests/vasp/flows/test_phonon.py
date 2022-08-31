import numpy as np
import pytest
from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

from atomate2.vasp.flows.phonons import PhononMaker
from atomate2.vasp.schemas.phonons import (
    PhononBSDOSDoc,
    PhononComputationalSettings,
    PhononJobDirs,
    PhononUUIDs,
    ThermalDisplacementData,
)

# test optional parameters


def test_phonon_wf_only_displacements2(mock_vasp, clean_dir):
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

    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonon_bandstructure,
        PhononBandStructureSymmLine,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.phonon_dos, PhononDos)
    assert (
        getattr(responses[job.jobs[-1].uuid][1].output, "thermal_displacement_data")
        is None
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.structure, Structure)
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.has_imaginary_modes, False
    )
    assert getattr(responses[job.jobs[-1].uuid][1].output, "force_constants") is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert getattr(responses[job.jobs[-1].uuid][1].output, "born") is None
    assert getattr(responses[job.jobs[-1].uuid][1].output, "epsilon_static") is None
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


# test all kpath schemes
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
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.has_imaginary_modes, False
    )
    assert np.isclose(
        responses[job.jobs[-1].uuid][1].output.force_constants[0][0][0][0], 13.032324
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert getattr(responses[job.jobs[-1].uuid][1].output, "born") is None
    assert getattr(responses[job.jobs[-1].uuid][1].output, "epsilon_static") is None
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


# test input born, epsilon, dft energy, supercell


# test run including all steps of the computation
