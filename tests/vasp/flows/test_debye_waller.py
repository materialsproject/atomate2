import numpy as np
import pytest
from jobflow import Flow, run_locally
from pymatgen.analysis.diffraction import core as diffraction_core
from pymatgen.core.structure import Structure

from atomate2.common.schemas.debye_waller import DebyeWallerDocument
from atomate2.vasp.flows.debye_waller import DebyeWallerMaker
from atomate2.vasp.flows.phonons import PhononMaker

requires_dwf = pytest.mark.skipif(
    not hasattr(diffraction_core, "get_anisotropic_debye_waller_factors"),
    reason="pymatgen has no anisotropic Debye-Waller factors",
)


@requires_dwf
def test_debye_waller_maker_vasp_flow(si_structure: Structure):
    """The phonon flow output and the settings go to compute_debye_waller."""
    maker = DebyeWallerMaker(
        temperatures=[100, 300],
        mesh=(10, 10, 10),
        freq_min=0.05,
        include_imaginary_modes=True,
        xrd_kwargs={"wavelength": "MoKa"},
        nd_kwargs={"wavelength": 1.0},
        tem_kwargs={"voltage": 300},
    )
    assert maker.phonon_maker.store_force_constants
    assert not maker.phonon_maker.create_thermal_displacements
    flow = maker.make(si_structure)
    phonon_flow, debye_waller = flow.jobs
    assert isinstance(phonon_flow, Flow)
    assert debye_waller.name == "compute_debye_waller"
    assert flow.output.uuid == debye_waller.uuid

    kwargs = debye_waller.function_kwargs
    assert kwargs["phonon_output"].uuid == phonon_flow.output.uuid
    for key in (
        "temperatures",
        "mesh",
        "freq_min",
        "include_imaginary_modes",
        "xrd_kwargs",
        "nd_kwargs",
        "tem_kwargs",
    ):
        assert kwargs[key] == getattr(maker, key)
    assert kwargs["symprec"] == maker.phonon_maker.symprec


def test_debye_waller_maker_checks_phonon_maker():
    with pytest.raises(ValueError, match="store_force_constants=True"):
        DebyeWallerMaker(phonon_maker=PhononMaker())


def test_debye_waller_checks_pymatgen(monkeypatch):
    monkeypatch.delattr(
        diffraction_core, "get_anisotropic_debye_waller_factors", raising=False
    )
    with pytest.raises(ImportError, match="no anisotropic Debye-Waller factors"):
        DebyeWallerMaker()
    with pytest.raises(ImportError, match="no anisotropic Debye-Waller factors"):
        DebyeWallerDocument.from_phonon_doc(None, [300], mesh=(1, 1, 1))


@requires_dwf
def test_debye_waller_maker_vasp_na_cl(mock_vasp, clean_dir):
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

    # the acoustic modes at Gamma are at -0.018 THz here, and must stay out
    maker = DebyeWallerMaker(temperatures=[0, 300], include_imaginary_modes=True)
    maker.phonon_maker.min_length = 3.0
    maker.phonon_maker.bulk_relax_maker = None
    flow = maker.make(structure)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    doc = responses[flow.output.uuid][1].output

    assert doc.mesh == (15, 15, 15)
    assert doc.include_imaginary_modes
    assert [site.specie.symbol for site in doc.structure] == ["Na", "Cl"]
    u = np.array(doc.thermal_displacement_data.thermal_displacement_matrix)
    assert u[1, :, 0, 0] == pytest.approx([0.026007, 0.020425], rel=1e-3)
