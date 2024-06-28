from pathlib import Path

import torch
from jobflow import run_locally
from pymatgen.core.structure import Structure

from atomate2.common.schemas.qha import PhononQHADoc
from atomate2.forcefields.flows.qha import CHGNetQhaMaker


def test_qha_dir(clean_dir, si_structure: Structure, tmp_path: Path):
    # TODO brittle due to inability to adjust dtypes in CHGNetRelaxMaker
    torch.set_default_dtype(torch.float32)

    flow = CHGNetQhaMaker(
        number_of_frames=5,
        ignore_imaginary_modes=True,
        phonon_maker_kwargs={
            "min_length": 10,
            "store_force_constants": False,
            "generate_frequencies_eigenvectors_kwargs": {
                "tol_imaginary_modes": 5e-1,
                "tmin": 0,
                "tmax": 1000,
                "tstep": 10,
            },
        },
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # # validate the outputs
    ph_bs_dos_doc = responses[flow[-1].uuid][1].output
    assert isinstance(ph_bs_dos_doc, PhononQHADoc)

    # TODO: add more checks!

    #
    # assert_allclose(
    #     ph_bs_dos_doc.free_energies,
    #     [5058.4521752, 4907.4957516, 3966.5493299, 2157.8178928, -357.5054580],
    #     atol=1000,
    # )
    #
    # ph_band_struct = ph_bs_dos_doc.phonon_bandstructure
    # assert isinstance(ph_band_struct, PhononBandStructureSymmLine)
    #
    # ph_dos = ph_bs_dos_doc.phonon_dos
    # assert isinstance(ph_dos, PhononDos)
    # assert ph_bs_dos_doc.thermal_displacement_data is None
    # assert isinstance(ph_bs_dos_doc.structure, Structure)
    # assert_allclose(ph_bs_dos_doc.temperatures, [0, 100, 200, 300, 400])
    # assert ph_bs_dos_doc.force_constants is None
    # assert isinstance(ph_bs_dos_doc.jobdirs, PhononJobDirs)
    # assert isinstance(ph_bs_dos_doc.uuids, PhononUUIDs)
    # assert_allclose(ph_bs_dos_doc.total_dft_energy, -5.37245798, 4)
    # assert ph_bs_dos_doc.born is None
    # assert ph_bs_dos_doc.epsilon_static is None
    # assert_allclose(
    #     ph_bs_dos_doc.supercell_matrix,
    #     [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    # )
    # assert_allclose(
    #     ph_bs_dos_doc.primitive_matrix,
    #     ((0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),
    #     atol=1e-8,
    # )
    # assert ph_bs_dos_doc.code == "forcefields"
    # assert isinstance(ph_bs_dos_doc.phonopy_settings, PhononComputationalSettings)
    # assert ph_bs_dos_doc.phonopy_settings.npoints_band == 101
    # assert ph_bs_dos_doc.phonopy_settings.kpath_scheme == "seekpath"
    # assert ph_bs_dos_doc.phonopy_settings.kpoint_density_dos == 7_000
    # assert_allclose(
    #     ph_bs_dos_doc.entropies,
    #     [0.0, 4.78393981, 13.99318695, 21.88641334, 28.19110667],
    #     atol=2,
    # )
    # assert_allclose(
    #     ph_bs_dos_doc.heat_capacities,
    #     [0.0, 8.86060586, 17.55758943, 21.08903916, 22.62587271],
    #     atol=2,
    # )
    # assert_allclose(
    #     ph_bs_dos_doc.internal_energies,
    #     [5058.44158791, 5385.88058579, 6765.19854165, 8723.78588089, 10919.0199409],
    #     atol=1000,
    # )
    #
    # # check phonon plots exist
    # assert os.path.isfile(filename_bs)
    # assert os.path.isfile(filename_dos)
