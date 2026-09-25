import numpy as np
import pytest
from emmet.core.phonon import (
    CalcMeta,
    PhononBS,
    PhononBSDOSDoc,
    PhononComputationalSettings,
    PhononDOS,
    ThermalDisplacementData,
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
        create_thermal_displacements=True,
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
    assert isinstance(
        ph_doc.thermal_displacement_data,
        ThermalDisplacementData,
    )
    assert isinstance(ph_doc.structure, Structure)
    assert ph_doc.has_imaginary_modes is False
    assert isinstance(ph_doc.force_constants, list)
    assert all(isinstance(cm, CalcMeta) for cm in ph_doc.calc_meta)
    assert_allclose(ph_doc.total_dft_energy, -2.8733374)
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
            (13.31020238, 0.0, -0.0),
            (0.0, 13.31020238, 0.0),
            (0.0, -0.0, 13.31020238),
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

    # --- check the actual phonon results obtained from the pheasy FC fit ---
    frequencies = np.asarray(ph_doc.phonon_bandstructure.frequencies)

    # acoustic sum rule: the acoustic modes at the Gamma point must vanish
    # (no imaginary modes were found above, so min(frequencies) is Gamma-acoustic)
    assert frequencies.min() == pytest.approx(0.0, abs=0.1)

    # the highest optical branch of diamond-Si is ~15.3 THz with PBE;
    # this guards against unit-conversion or supercell-mapping regressions
    assert 14.0 < frequencies.max() < 17.0

    thermo_props = ph_doc.compute_thermo_quantities(
        [0, 300, 1000], normalization="atoms"
    )
    # exact limits: entropy and heat capacity vanish at 0 K
    assert thermo_props["entropy"][0] == pytest.approx(0.0, abs=1e-8)
    assert thermo_props["heat_capacity"][0] == pytest.approx(0.0, abs=1e-8)
    # zero-point energy of diamond-Si is ~0.06 eV/atom (~5.8 kJ/mol-atom)
    assert thermo_props["free_energy"][0] == pytest.approx(5800, rel=0.15)
    # room-temperature heat capacity of Si in J/(K mol-atom)
    assert thermo_props["heat_capacity"][1] == pytest.approx(20.0, abs=2.5)
    # the high-T limit must approach (but not exceed) Dulong-Petit (3R per atom)
    assert 23.0 < thermo_props["heat_capacity"][2] < 3 * 8.3145


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"cal_anhar_fcs": True, "anhar_max_order": 5}, "anhar_max_order"),
        ({"cal_anhar_fcs": True, "anhar_fit_methods": ["joint"]}, "anhar_fit_methods"),
        ({"cal_anhar_fcs": True, "anhar_fit_methods": []}, "anhar_fit_methods"),
        ({"cal_anhar_fcs": True, "fcs_cutoff_radius": [-1, -2, 10]}, "positive"),
        ({"anhar_alpha_min": -12.0}, "integer below -2"),
        ({"cal_anhar_fcs": True, "num_disp_anhar": -1}, "non-negative integer"),
        (
            {"cal_anhar_fcs": True, "anhar_fit_methods": ["cocktail", "cocktail"]},
            "without repeats",
        ),
        ({"anhar_alpha_min": -1}, "integer below -2"),
        (
            {
                "cal_anhar_fcs": True,
                "anhar_max_order": 4,
                "fcs_cutoff_radius": [-1, 12],
            },
            "positive",
        ),
    ],
)
def test_pheasy_anharmonic_settings_are_checked(settings, match):
    with pytest.raises(ValueError, match=match):
        PhononMaker(**settings)


def test_pheasy_maker_passes_anharmonic_settings(si_structure: Structure):
    """Both pheasy jobs must get the same settings, or the dataset is split wrong."""
    # every value differs from its default, so a dropped keyword is caught
    settings = {
        "cal_anhar_fcs": True,
        "num_displaced_supercells": 5,
        "num_disp_anhar": 30,
        "displacement_anhar": 0.05,
        "anhar_max_order": 4,
        "anhar_fit_methods": ["cocktail", "one-shot"],
        "anhar_alpha_min": -10,
        "fcs_cutoff_radius": [-1, 11, 9],
        "displacement": 0.02,
        "symprec": 1e-4,
        "sym_reduce": False,
    }
    flow = PhononMaker(**settings).make(structure=si_structure)
    jobs = {job.name: job for job in flow.jobs}
    displacements = jobs["generate_phonon_displacements"].function_kwargs
    fit = jobs["generate_frequencies_eigenvectors"].function_kwargs
    for key in (
        "cal_anhar_fcs",
        "num_displaced_supercells",
        "anhar_max_order",
        "anhar_fit_methods",
        "fcs_cutoff_radius",
        "displacement",
        "symprec",
        "sym_reduce",
    ):
        assert displacements[key] == fit[key] == settings[key]
    assert displacements["num_disp_anhar"] == 30
    assert displacements["displacement_anhar"] == 0.05
    assert fit["anhar_alpha_min"] == -10
