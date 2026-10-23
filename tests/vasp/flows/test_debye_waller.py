import numpy as np
import pytest
from jobflow import run_locally
from pymatgen.analysis.diffraction import core as diffraction_core
from pymatgen.core.structure import Structure

from atomate2.common.schemas.debye_waller import DebyeWallerDocument
from atomate2.vasp.flows.phonons import PhononMaker

requires_dwf = pytest.mark.skipif(
    not hasattr(diffraction_core, "get_anisotropic_debye_waller_factors"),
    reason="pymatgen has no anisotropic Debye-Waller factors",
)


@requires_dwf
def test_debye_waller_vasp_flow(si_structure: Structure):
    """The phonon output and the settings go to compute_debye_waller."""
    debye_waller_kwargs = {
        "xrd_kwargs": {"wavelength": "MoKa"},
        "nd_kwargs": {"wavelength": 1.0},
        "tem_kwargs": {"voltage": 300},
    }
    maker = PhononMaker(
        create_debye_waller=True, debye_waller_kwargs=debye_waller_kwargs
    )
    flow = maker.make(si_structure)
    debye_waller = flow.jobs[-1]
    assert debye_waller.name == "compute_debye_waller"

    kwargs = debye_waller.function_kwargs
    assert kwargs["phonon_output"].uuid == flow.output.uuid
    for key, value in debye_waller_kwargs.items():
        assert kwargs[key] == value

    jobs = PhononMaker().make(si_structure).jobs
    assert "compute_debye_waller" not in [job.name for job in jobs]


def test_debye_waller_checks_phonon_maker(si_structure: Structure):
    maker = PhononMaker(create_thermal_displacements=False, create_debye_waller=True)
    with pytest.raises(ValueError, match="create_thermal_displacements=True"):
        maker.make(si_structure)
    maker = PhononMaker(phonon_doc_schema="emmet", create_debye_waller=True)
    with pytest.raises(ValueError, match="phonon_doc_schema='atomate2'"):
        maker.make(si_structure)


def test_debye_waller_checks_pymatgen(monkeypatch, si_structure: Structure):
    monkeypatch.delattr(
        diffraction_core, "get_anisotropic_debye_waller_factors", raising=False
    )
    with pytest.raises(ImportError, match="no anisotropic Debye-Waller factors"):
        PhononMaker(create_debye_waller=True).make(si_structure)
    with pytest.raises(ImportError, match="no anisotropic Debye-Waller factors"):
        DebyeWallerDocument.from_phonon_doc(None)


@requires_dwf
def test_debye_waller_vasp_na_cl(mock_vasp, clean_dir):
    """NaCl from the conventional cell, with the non-analytical correction."""
    structure = Structure(
        lattice=5.691694 * np.eye(3),
        species=["Na"] * 4 + ["Cl"] * 4,
        coords=[
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ],
    )
    ref_paths = {
        "dielectric": "NaCl_phonons/dielectric",
        "phonon static 1/2": "NaCl_phonons/phonon_static_1_2",
        "phonon static 2/2": "NaCl_phonons/phonon_static_2_2",
        "static": "NaCl_phonons/static",
    }
    mock_vasp(
        ref_paths, {name: {"incar_settings": ["NSW", "ISMEAR"]} for name in ref_paths}
    )

    maker = PhononMaker(create_debye_waller=True)
    maker.generate_frequencies_eigenvectors_kwargs = {
        "tstep_thermal_displacements": 300,
        "tmax_thermal_displacements": 300,
    }
    maker.min_length = 3.0
    maker.bulk_relax_maker = None
    flow = maker.make(structure)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    doc = responses[flow.jobs[-1].uuid][1].output

    assert [site.specie.symbol for site in doc.structure] == ["Na", "Cl"]
    u = np.array(doc.thermal_displacement_data.thermal_displacement_matrix)
    assert u[1, :, 0, 0] == pytest.approx([0.026454, 0.020878], rel=1e-3)
