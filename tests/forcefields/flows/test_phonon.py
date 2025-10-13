import os
from pathlib import Path

import pytest
from emmet.core.phonon import (
    CalcMeta,
    PhononBS,
    PhononBSDOSDoc,
    PhononComputationalSettings,
    PhononDOS,
)
from jobflow import run_locally
from numpy.testing import assert_allclose
from pymatgen.core.structure import Structure

from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker


@pytest.mark.parametrize("from_name", [False, True])
def test_phonon_wf_force_field(
    clean_dir, si_structure: Structure, tmp_path: Path, from_name: bool
):
    # TODO brittle due to inability to adjust dtypes in CHGNetRelaxMaker

    phonon_kwargs = dict(
        use_symmetrized_structure="conventional",
        create_thermal_displacements=False,
        store_force_constants=False,
        prefer_90_degrees=False,
        generate_frequencies_eigenvectors_kwargs={
            "tstep": 100,
            "filename_bs": (filename_bs := f"{tmp_path}/phonon_bs_test.png"),
            "filename_dos": (filename_dos := f"{tmp_path}/phonon_dos_test.pdf"),
        },
    )

    if from_name:
        phonon_maker = PhononMaker.from_force_field_name("CHGNet", **phonon_kwargs)
        if phonon_kwargs.get("relax_initial_structure", True):
            assert isinstance(phonon_maker.bulk_relax_maker, ForceFieldRelaxMaker)
            assert "CHGNet" in phonon_maker.bulk_relax_maker.force_field_name

        for attr in ("static_energy_maker", "phonon_displacement_maker"):
            assert "CHGNet" in getattr(phonon_maker, attr).force_field_name

        assert (
            PhononMaker.from_force_field_name(
                "CHGNet", relax_initial_structure=False
            ).bulk_relax_maker
            is None
        )
    else:
        phonon_maker = PhononMaker(**phonon_kwargs)

    flow = phonon_maker.make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate the outputs
    ph_bs_dos_doc = responses[flow[-1].uuid][1].output
    assert isinstance(ph_bs_dos_doc, PhononBSDOSDoc)

    ph_band_struct = ph_bs_dos_doc.phonon_bandstructure
    assert isinstance(ph_band_struct, PhononBS)

    ph_dos = ph_bs_dos_doc.phonon_dos
    assert isinstance(ph_dos, PhononDOS)
    assert ph_bs_dos_doc.thermal_displacement_data is None
    assert isinstance(ph_bs_dos_doc.structure, Structure)
    assert ph_bs_dos_doc.force_constants is None
    assert all(isinstance(cm, CalcMeta) for cm in ph_bs_dos_doc.calc_meta)
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
    assert isinstance(ph_bs_dos_doc.post_process_settings, PhononComputationalSettings)
    assert ph_bs_dos_doc.post_process_settings.npoints_band == 101
    assert ph_bs_dos_doc.post_process_settings.kpath_scheme == "seekpath"
    assert ph_bs_dos_doc.post_process_settings.kpoint_density_dos == 7_000

    ref_vals = {
        "entropy": [0.0, 7.45806197, 24.99582177, 40.53981354, 53.0450785],
        "heat_capacity": [0.0, 15.9212379, 34.32542093, 41.73809612, 44.95600976],
        "internal_energy": [
            10510.17946131,
            11038.76862405,
            13676.21828021,
            17534.72238986,
            21889.29538244,
        ],
        "free_energy": [
            10510.17946131,
            10292.96242722,
            8677.0539271,
            5372.77832663,
            671.26398379,
        ],
    }
    thermo_props = ph_bs_dos_doc.compute_thermo_quantities(
        [0, 100, 200, 300, 400], normalization=None
    )

    assert all(
        thermo_props[k][i] == pytest.approx(val, rel=0.1)
        for k, vals in ref_vals.items()
        for i, val in enumerate(vals)
    )

    # check phonon plots exist
    assert os.path.isfile(filename_bs)
    assert os.path.isfile(filename_dos)
