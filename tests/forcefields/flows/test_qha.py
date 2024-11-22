from pathlib import Path

import pytest
from jobflow import run_locally
from numpy.testing import assert_allclose
from pymatgen.core.structure import Structure

from atomate2.common.schemas.qha import PhononQHADoc
from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.flows.qha import CHGNetQhaMaker


def test_qha_dir(clean_dir, si_structure: Structure, tmp_path: Path):
    # TODO brittle due to inability to adjust dtypes in CHGNetRelaxMaker

    flow = CHGNetQhaMaker(
        number_of_frames=5,
        ignore_imaginary_modes=True,
        min_length=10,
        phonon_maker=PhononMaker(
            store_force_constants=False,
            bulk_relax_maker=None,
            generate_frequencies_eigenvectors_kwargs={
                "tol_imaginary_modes": 5e-1,
                "tmin": 0,
                "tmax": 1000,
                "tstep": 10,
            },
        ),
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # # validate the outputs
    ph_bs_dos_doc = responses[flow[-1].uuid][1].output
    assert isinstance(ph_bs_dos_doc, PhononQHADoc)


def test_qha_dir_change_defaults(clean_dir, si_structure: Structure, tmp_path: Path):
    # TODO brittle due to inability to adjust dtypes in CHGNetRelaxMaker

    flow = CHGNetQhaMaker(
        number_of_frames=4,
        ignore_imaginary_modes=True,
        linear_strain=(-0.03, 0.03),
        min_length=10,
        phonon_maker=PhononMaker(
            store_force_constants=False,
            bulk_relax_maker=None,
            generate_frequencies_eigenvectors_kwargs={
                "tol_imaginary_modes": 5e-1,
                "tmin": 0,
                "tmax": 1000,
                "tstep": 10,
            },
        ),
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # print(responses)

    # # validate the outputs
    ph_bs_dos_doc = responses[flow[-1].uuid][1].output
    assert isinstance(ph_bs_dos_doc, PhononQHADoc)
    assert len(ph_bs_dos_doc.volumes) == 5
    assert ph_bs_dos_doc.volumes[0] == pytest.approx(ph_bs_dos_doc.volumes[2] * 0.97**3)
    assert ph_bs_dos_doc.volumes[4] == pytest.approx(ph_bs_dos_doc.volumes[2] * 1.03**3)


def test_qha_dir_manual_supercell(clean_dir, si_structure: Structure, tmp_path: Path):
    # TODO brittle due to inability to adjust dtypes in CHGNetRelaxMaker
    matrix = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    flow = CHGNetQhaMaker(
        number_of_frames=4,
        ignore_imaginary_modes=True,
        min_length=10,
        phonon_maker=PhononMaker(
            store_force_constants=False,
            bulk_relax_maker=None,
            generate_frequencies_eigenvectors_kwargs={
                "tol_imaginary_modes": 5e-1,
                "tmin": 0,
                "tmax": 1000,
                "tstep": 10,
            },
        ),
    ).make(si_structure, supercell_matrix=matrix)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # print(responses)

    # # validate the outputs
    ph_bs_dos_doc = responses[flow[-1].uuid][1].output
    assert isinstance(ph_bs_dos_doc, PhononQHADoc)
    assert_allclose(ph_bs_dos_doc.supercell_matrix, matrix)
