import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

from atomate2.common.schemas.phonons import (
    PhononBSDOSDoc,
    PhononComputationalSettings,
    PhononJobDirs,
    PhononUUIDs,
)
from atomate2.forcefields.flows.phonons import PhononMaker


def test_phonon_wf(clean_dir):
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # !!! Generate job
    job = PhononMaker(
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
        np.array(responses[job.jobs[-1].uuid][1].output.free_energies) / 1000.0,
        np.array(
            [
                5058.45217527524,
                4907.495751683517,
                3966.5493299635937,
                2157.8178928940474,
                -357.5054580420707,
            ]
        )
        / 1000.0,
        2,
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
    assert responses[job.jobs[-1].uuid][1].output.force_constants is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert np.isclose(
        responses[job.jobs[-1].uuid][1].output.total_dft_energy, -5.372457981109619
    )
    assert responses[job.jobs[-1].uuid][1].output.born is None
    assert responses[job.jobs[-1].uuid][1].output.epsilon_static is None
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
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
            4.783939817386235,
            13.993186953791708,
            21.88641334781562,
            28.19110667148253,
        ],
        3,
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [
            0.0,
            8.860605865667427,
            17.55758943495313,
            21.089039169564796,
            22.625872713428905,
        ],
        3,
    )

    assert np.allclose(
        np.array(responses[job.jobs[-1].uuid][1].output.internal_energies) / 1000.0,
        np.array(
            [
                5058.441587914012,
                5385.880585798466,
                6765.198541655172,
                8723.78588089732,
                10919.019940938391,
            ]
        )
        / 1000.0,
        3,
    )
