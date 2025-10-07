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

from atomate2.common.flows.pheasy import BasePhononMaker
from atomate2.common.powerups import add_metadata_to_flow
from atomate2.vasp.flows.pheasy import PhononMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.powerups import update_user_incar_settings


def test_pheasy_wf_vasp(mock_vasp, clean_dir, si_structure: Structure, test_dir):
    # mapping from job name to directory containing test files
    ref_paths = {
        "tight relax 1": "Si_pheasy/tight_relax_1",
        "tight relax 2": "Si_pheasy/tight_relax_2",
        "phonon static 1/2": "Si_pheasy/phonon_static_1_2",
        "phonon static 2/2": "Si_pheasy/phonon_static_2_2",
        "static": "Si_pheasy/static",
        "dielectric": "Si_pheasy/dielectric",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "tight relax 1": {"incar_settings": ["NSW", "ISMEAR", "KSPACING"]},
        "tight relax 2": {"incar_settings": ["NSW", "ISMEAR", "KSPACING"]},
        "phonon static 1/2": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 2/2": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {"incar_settings": ["NSW", "ISMEAR"]},
        "dielectric": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec dulsring the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    si_struct = Structure.from_file(
        test_dir / "vasp/Si_pheasy/tight_relax_1/inputs/POSCAR.gz"
    )

    job = PhononMaker(
        force_diagonal=True,
        min_length=12,
        cal_anhar_fcs=False,
        # use_symmetrized_structure="primitive"
    ).make(structure=si_struct)

    job = update_user_incar_settings(
        job,
        {
            "ENCUT": 600,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "KSPACING": 0.15,
            "ISPIN": 1,
            "EDIFFG": -1e-04,
            "EDIFF": 1e-07,
        },
    )
    job = add_metadata_to_flow(
        flow=job,
        additional_fields={"mp_id": "mp-149", "unit_testing": "yes"},
        class_filter=(BaseVaspMaker, BasePhononMaker, PhononMaker),
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)
    ph_doc = responses[job.jobs[-1].uuid][1].output

    # validate the outputs
    assert isinstance(ph_doc, PhononBSDOSDoc)

    assert isinstance(
        ph_doc.phonon_bandstructure,
        PhononBS,
    )
    assert isinstance(ph_doc.phonon_dos, PhononDOS)
    # assert isinstance(
    #     ph_doc.thermal_displacement_data,
    #     ThermalDisplacementData,
    # )
    assert isinstance(ph_doc.structure, Structure)
    assert ph_doc.has_imaginary_modes is False
    assert isinstance(ph_doc.force_constants, tuple)
    assert all(isinstance(cm, CalcMeta) for cm in ph_doc.calc_meta)
    assert_allclose(ph_doc.total_dft_energy, -5.7466748)
    assert_allclose(
        ph_doc.born,
        [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ],
    )
    assert_allclose(
        ph_doc.epsilon_static,
        (
            (13.31020238, 0.0, -0.000000000000000000000000000000041086505480261033),
            (0.000000000000000000000000000000032869204384208823, 13.31020238, 0.0),
            (
                0.00000000000000000000000000000003697785493223493,
                -0.00000000000000000000000000000005310360021821649,
                13.31020238,
            ),
        ),
        atol=1e-8,
    )
    assert_allclose(
        ph_doc.supercell_matrix,
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )
    assert_allclose(
        ph_doc.primitive_matrix,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        rtol=1e-5,
        atol=1e-10,
    )
    assert ph_doc.code == "vasp"
    assert isinstance(
        ph_doc.post_process_settings,
        PhononComputationalSettings,
    )
    assert ph_doc.post_process_settings.npoints_band == 101
    assert ph_doc.post_process_settings.kpath_scheme == "seekpath"
    assert ph_doc.post_process_settings.kpoint_density_dos == 7000

    assert ph_doc.chemsys == "Si"
