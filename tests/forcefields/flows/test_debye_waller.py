"""Tests for the force field Debye-Waller workflow.

The tests use ASE's EMT potential. They need a pymatgen with anisotropic
Debye-Waller factors, which is installed by the debye-waller dependency group,
so they are skipped in the other forcefield CI jobs.
"""

import pytest
from pymatgen.analysis.diffraction import core as diffraction_core

if not hasattr(diffraction_core, "get_anisotropic_debye_waller_factors"):
    pytest.skip(
        "pymatgen has no anisotropic Debye-Waller factors", allow_module_level=True
    )

import json

import numpy as np
from ase.build import bulk
from emmet.core.phonon import PhononBSDOSDoc
from jobflow import run_locally
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.analysis.diffraction.tem import TEMCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.common.schemas.debye_waller import DebyeWallerDocument
from atomate2.forcefields.flows.debye_waller import DebyeWallerMaker

EMT = {"@module": "ase.calculators.emt", "@callable": "EMT"}


def test_debye_waller_maker_emt(clean_dir, memory_jobstore):
    """Run the whole force field workflow with EMT forces on fcc Cu."""
    structure = AseAtomsAdaptor.get_structure(bulk("Cu", "fcc", a=3.61))
    maker = DebyeWallerMaker.from_force_field_name(
        EMT,
        xrd_kwargs={"wavelength": "MoKa"},
        nd_kwargs={"wavelength": 1.0},
        tem_kwargs={"voltage": 100},
    )
    maker.phonon_maker.min_length = 8.0
    # a 7 x 7 x 7 mesh for the one-atom primitive cell
    maker.phonon_maker.generate_frequencies_eigenvectors_kwargs = {
        "kpoint_density_thermal_displacements": 343,
        "tstep_thermal_displacements": 300,
        "tmax_thermal_displacements": 600,
        "freq_min_thermal_displacements": 0.1,
    }

    flow = maker.make(structure)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    doc = responses[flow.output.uuid][1].output
    assert isinstance(doc, DebyeWallerDocument)
    # U is that of the phonon flow
    phonon_doc = responses[flow.jobs[0].output.uuid][1].output
    assert doc.thermal_displacement_data.model_dump() == (
        phonon_doc.thermal_displacement_data.model_dump()
    )
    assert doc.xrd_kwargs == {"wavelength": "MoKa"}
    assert len(doc.structure) == 1

    # the patterns are in the data store, and the document reloads from the store
    stored = memory_jobstore.get_output(flow.output.uuid)
    for key in ("xrd_patterns", "nd_patterns", "tem_pattern_static", "tem_patterns"):
        assert stored[key]["store"] == "data"
    stored = MontyDecoder().process_decoded(
        memory_jobstore.get_output(flow.output.uuid, load=True)
    )
    assert isinstance(stored, DebyeWallerDocument)
    assert json.dumps(stored, cls=MontyEncoder) == json.dumps(doc, cls=MontyEncoder)

    data = doc.thermal_displacement_data
    assert data.temperatures_thermal_displacements == [0, 300, 600]
    assert data.freq_min_thermal_displacements == 0.1
    u = np.array(data.thermal_displacement_matrix)
    assert u.shape == (3, 1, 3, 3)
    # cubic, so U is isotropic, and it grows with temperature
    for u_t in u[:, 0]:
        assert u_t == pytest.approx(u_t[0, 0] * np.eye(3), abs=1e-10)
    assert np.all(np.diff(u[:, 0, 0, 0]) > 0)
    assert u[1, 0, 0, 0] == pytest.approx(0.0053691, rel=1e-4)
    # the reciprocal lattice vectors of the fcc primitive cell are at cos = -1/3
    u_cif = np.array(data.thermal_displacement_matrix_cif)[:, 0]
    assert u_cif == pytest.approx(u[:, 0, :1, :1] * (4 / 3 * np.eye(3) - 1 / 3))

    for static, calculator in (
        (doc.xrd_pattern_static, XRDCalculator("MoKa")),
        (doc.nd_pattern_static, NDCalculator(wavelength=1.0)),
    ):
        reference = calculator.get_pattern(doc.structure, scaled=False)
        assert static.x == pytest.approx(reference.x)
        assert static.y == pytest.approx(reference.y)

    # the structure factor has exp(-2 pi^2 U / d^2), the intensity its square
    for static, patterns in (
        (doc.xrd_pattern_static, doc.xrd_patterns),
        (doc.nd_pattern_static, doc.nd_patterns),
    ):
        d_hkl = np.array(static.d_hkls)
        for u_t, pattern in zip(u[:, 0, 0, 0], patterns, strict=True):
            assert pattern.x == pytest.approx(static.x)
            assert pattern.y / static.y == pytest.approx(
                np.exp(-4 * np.pi**2 * u_t / d_hkl**2)
            )

    tem = TEMCalculator(voltage=100).get_pattern(doc.structure)
    positions = [row["position"] for row in doc.tem_pattern_static]
    assert np.array(positions) == pytest.approx(np.stack(tem["Position"]))
    hkls = [row["hkl"] for row in doc.tem_pattern_static]
    assert hkls == [list(hkl) for hkl in tem["(hkl)"]]
    static = np.array([row["intensity"] for row in doc.tem_pattern_static])
    assert static == pytest.approx(tem["Intensity (norm)"].to_numpy())
    # each TEM pattern is normalized to its strongest spot
    d_hkl = np.array([doc.structure.lattice.d_hkl(hkl) for hkl in hkls])
    for u_t, pattern in zip(u[:, 0, 0, 0], doc.tem_patterns, strict=True):
        assert [row["hkl"] for row in pattern] == hkls
        ratio = np.array([row["intensity"] for row in pattern]) / static
        ratio *= np.exp(4 * np.pi**2 * u_t / d_hkl**2)
        assert ratio == pytest.approx(ratio[0])


def test_debye_waller_maker_defaults():
    """The default phonon maker uses MACE-MP-0 and stores the thermal displacements."""
    maker = DebyeWallerMaker()
    assert maker.phonon_maker.create_thermal_displacements
    assert (
        maker.phonon_maker.phonon_displacement_maker.force_field_name
        == "MLFF.MACE_MP_0"
    )


def test_debye_waller_maker_from_force_field_name():
    """The settings go to the phonon maker, and a given phonon maker is kept."""
    maker = DebyeWallerMaker.from_force_field_name(
        EMT, calculator_kwargs={"asap_cutoff": True}, relax_initial_structure=False
    )
    assert maker.phonon_maker.bulk_relax_maker is None
    calculator_kwargs = maker.phonon_maker.phonon_displacement_maker.calculator_kwargs
    assert calculator_kwargs == {"asap_cutoff": True}
    roundtrip = MontyDecoder().decode(json.dumps(maker, cls=MontyEncoder))
    assert roundtrip.as_dict() == maker.as_dict()
    phonon_maker = maker.phonon_maker
    maker = DebyeWallerMaker.from_force_field_name(EMT, phonon_maker=phonon_maker)
    assert maker.phonon_maker is phonon_maker


def test_debye_waller_document_checks_input():
    phonon_doc = PhononBSDOSDoc(
        structure=AseAtomsAdaptor.get_structure(bulk("Cu", "fcc", a=3.61))
    )
    with pytest.raises(ValueError, match="no thermal displacement matrices"):
        DebyeWallerDocument.from_phonon_doc(phonon_doc)
