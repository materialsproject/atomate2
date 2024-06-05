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


def test_phonon_wf_force_field(clean_dir, si_structure: Structure, tmp_path: Path):
    # TODO brittle due to inability to adjust dtypes in CHGNetRelaxMaker
    torch.set_default_dtype(torch.float32)

    flow = PhononMaker(
        use_symmetrized_structure="conventional",
        create_thermal_displacements=False,
        store_force_constants=False,
        prefer_90_degrees=False,
        generate_frequencies_eigenvectors_kwargs={
            "tstep": 100,
            "filename_bs": (filename_bs := f"{tmp_path}/phonon_bs_test.png"),
            "filename_dos": (filename_dos := f"{tmp_path}/phonon_dos_test.pdf"),
        },
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate the outputs
    ph_bs_dos_doc = responses[flow[-1].uuid][1].output
    assert isinstance(ph_bs_dos_doc, PhononBSDOSDoc)

    assert_allclose(
        ph_bs_dos_doc.free_energies,
        [5058.4521752, 4907.4957516, 3966.5493299, 2157.8178928, -357.5054580],
        atol=1000,
    )

    ph_band_struct = ph_bs_dos_doc.phonon_bandstructure
    assert isinstance(ph_band_struct, PhononBandStructureSymmLine)

    ph_dos = ph_bs_dos_doc.phonon_dos
    assert isinstance(ph_dos, PhononDos)
    assert ph_bs_dos_doc.thermal_displacement_data is None
    assert isinstance(ph_bs_dos_doc.structure, Structure)
    assert_allclose(ph_bs_dos_doc.temperatures, [0, 100, 200, 300, 400])
    assert ph_bs_dos_doc.force_constants is None
    assert isinstance(ph_bs_dos_doc.jobdirs, PhononJobDirs)
    assert isinstance(ph_bs_dos_doc.uuids, PhononUUIDs)
    assert_allclose(ph_bs_dos_doc.total_dft_energy, -5.37245798, 4)
    assert ph_bs_dos_doc.born is None
    assert ph_bs_dos_doc.epsilon_static is None
    assert_allclose(
        ph_bs_dos_doc.supercell_matrix,
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )
    assert_allclose(
        ph_bs_dos_doc.primitive_matrix,
        ((0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),
        atol=1e-8,
    )
    assert ph_bs_dos_doc.code == "forcefields"
    assert isinstance(ph_bs_dos_doc.phonopy_settings, PhononComputationalSettings)
    assert ph_bs_dos_doc.phonopy_settings.npoints_band == 101
    assert ph_bs_dos_doc.phonopy_settings.kpath_scheme == "seekpath"
    assert ph_bs_dos_doc.phonopy_settings.kpoint_density_dos == 7_000
    assert_allclose(
        ph_bs_dos_doc.entropies,
        [0.0, 4.78393981, 13.99318695, 21.88641334, 28.19110667],
        atol=2,
    )
    assert_allclose(
        ph_bs_dos_doc.heat_capacities,
        [0.0, 8.86060586, 17.55758943, 21.08903916, 22.62587271],
        atol=2,
    )
    assert_allclose(
        ph_bs_dos_doc.internal_energies,
        [5058.44158791, 5385.88058579, 6765.19854165, 8723.78588089, 10919.0199409],
        atol=1000,
    )

    # check phonon plots exist
    assert os.path.isfile(filename_bs)
    assert os.path.isfile(filename_dos)
