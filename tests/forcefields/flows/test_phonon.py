import os
from pathlib import Path

import torch
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
)
from atomate2.forcefields.flows.phonons import PhononMaker


def test_phonon_wf(clean_dir, tmp_path: Path):
    # TODO brittle due to inability to adjust dtypes in CHGNetRelaxMaker
    torch.set_default_dtype(torch.float32)

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    job = PhononMaker(
        use_symmetrized_structure="conventional",
        create_thermal_displacements=False,
        store_force_constants=False,
        prefer_90_degrees=False,
        generate_frequencies_eigenvectors_kwargs={
            "tstep": 100,
            "filename_bs": (filename_bs := f"{tmp_path}/phonon_bs_test.png"),
            "filename_dos": (filename_dos := f"{tmp_path}/phonon_dos_test.pdf"),
        },
    ).make(structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate the outputs
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [5058.4521752, 4907.4957516, 3966.5493299, 2157.8178928, -357.5054580],
        atol=1000,
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
    assert responses[job.jobs[-1].uuid][1].output.force_constants is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.total_dft_energy, -5.37245798, 4
    )
    assert responses[job.jobs[-1].uuid][1].output.born is None
    assert responses[job.jobs[-1].uuid][1].output.epsilon_static is None
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        ((0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),
        atol=1e-8,
    )
    assert responses[job.jobs[-1].uuid][1].output.code == "forcefields"
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
        [0.0, 4.78393981, 13.99318695, 21.88641334, 28.19110667],
        atol=2,
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [0.0, 8.86060586, 17.55758943, 21.08903916, 22.62587271],
        atol=2,
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.internal_energies,
        [5058.44158791, 5385.88058579, 6765.19854165, 8723.78588089, 10919.0199409],
        atol=1000,
    )

    # check phonon plots exist
    assert os.path.isfile(filename_bs)
    assert os.path.isfile(filename_dos)
