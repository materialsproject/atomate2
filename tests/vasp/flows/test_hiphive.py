import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

from atomate2.common.schemas.phonons import (
    PhononComputationalSettings,
    PhononJobDirs,
    PhononUUIDs,
)
from atomate2.vasp.flows.hiphive import HiphiveMaker
from atomate2.vasp.schemas.hiphive import LTCDoc


def test_hiphive_wf(mock_vasp, clean_dir):
    from jobflow import run_locally

    mpid = "mp-1265"

    struct = Structure(
        lattice=[
            [2.56829206, -0.0, 1.48280459],
            [0.85609768, 2.4214088, 1.48280459],
            [-1e-08, -0.0, 2.96560719],
        ],
        species=["Mg", "O"],
        coords=[[0, 0, -0], [0.5, 0.5, 0.5]],
    )

    bulk_mod = 151
    cutoffs = [[]]
    n_configs_per_std = 1
    rattle_stds = [0.01, 0.03, 0.08, 0.1]
    min_atoms = 150  # 150 #4
    max_atoms = 600  # 600 #10
    min_length = 18  # 18 #6
    supercell_matrix_kwargs = {
        "min_atoms": min_atoms,
        "max_atoms": max_atoms,
        "min_length": min_length,
        "force_diagonal": True,
    }

    # mapping from job name to directory containing test files
    ref_paths = {
        "phonon static 1/1": "MgO_hiphive/phonon_static_1_1",
        "static": "MgO_hiphive/static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # !!! Generate job
    job = HiphiveMaker().make(
        mpid=mpid,
        structure=struct,
        bulk_modulus=bulk_mod,
        cutoffs=cutoffs,
        n_structures=n_configs_per_std,
        rattle_std=rattle_stds,
        supercell_matrix_kwargs=supercell_matrix_kwargs,
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # !!! validation on the outputs
    assert isinstance(responses[job.jobs[-1].uuid][1].output, LTCDoc)

    assert np.isclose(
        responses[job.jobs[-1].uuid][1].output.lattice_thermal_conductivity,
        47.786,
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
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.has_imaginary_modes, False
    )
    assert responses[job.jobs[-1].uuid][1].output.force_constants is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert np.isclose(
        responses[job.jobs[-1].uuid][1].output.total_relax_energy, -5.74555232
    )
    # assert responses[job.jobs[-1].uuid][1].output.born is None
    # assert responses[job.jobs[-1].uuid][1].output.epsilon_static is None
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
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
            17.63749405576557,
            37.28748895403729,
            59.86453904679404,
            71.42384052118940,
        ],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [
            0.0,
            26.77495057290405,
            40.02624369405739,
            43.12337406416830,
            48.27830408283403,
        ],
    )

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.internal_energy,
        [
            15.12679123890023,
            13.62946902479400,
            9.56024694024794,
            -3.744112780351908,
            -17.679023490572006,
        ],
    )
    assert responses[job.jobs[-1].uuid][1].output.chemsys == "Mg O"
