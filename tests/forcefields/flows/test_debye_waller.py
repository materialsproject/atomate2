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

import numpy as np
from ase.build import bulk
from ase.calculators import emt
from emmet.core.phonon import PhononBSDOSDoc
from jobflow import run_locally
from phonopy import Phonopy
from phonopy.physical_units import get_physical_units
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.analysis.diffraction.tem import TEMCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from atomate2.common.schemas.debye_waller import DebyeWallerDocument
from atomate2.forcefields.flows.debye_waller import DebyeWallerMaker

EMT = {"@module": "ase.calculators.emt", "@callable": "EMT"}


def test_debye_waller_maker_emt(clean_dir):
    """Run the whole force field workflow with EMT forces on fcc Cu."""
    structure = AseAtomsAdaptor.get_structure(bulk("Cu", "fcc", a=3.61))
    maker = DebyeWallerMaker.from_force_field_name(
        EMT,
        temperatures=[0, 300, 600],
        mesh=(8, 8, 8),
        freq_min=0.1,
        xrd_kwargs={"wavelength": "MoKa"},
        nd_kwargs={"wavelength": 1.0},
        tem_kwargs={"voltage": 100},
    )
    maker.phonon_maker.min_length = 8.0

    flow = maker.make(structure)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    doc = responses[flow.output.uuid][1].output
    assert isinstance(doc, DebyeWallerDocument)
    assert doc.mesh == (8, 8, 8)
    assert not doc.include_imaginary_modes
    assert len(doc.structure) == 1

    data = doc.thermal_displacement_data
    assert data.temperatures_thermal_displacements == [0, 300, 600]
    assert data.freq_min_thermal_displacements == 0.1
    u = np.array(data.thermal_displacement_matrix)
    assert u.shape == (3, 1, 3, 3)
    # cubic, so U is isotropic, and it grows with temperature
    for u_t in u[:, 0]:
        assert u_t == pytest.approx(u_t[0, 0] * np.eye(3), abs=1e-10)
    assert np.all(np.diff(u[:, 0, 0, 0]) > 0)
    assert u[1, 0, 0, 0] == pytest.approx(0.0054614, rel=1e-4)
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
    """The default phonon maker uses MACE-MP-0 and stores the force constants."""
    maker = DebyeWallerMaker()
    assert maker.phonon_maker.store_force_constants
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
    phonon_maker = maker.phonon_maker
    maker = DebyeWallerMaker.from_force_field_name(EMT, phonon_maker=phonon_maker)
    assert maker.phonon_maker is phonon_maker


@pytest.mark.parametrize("freq_min", [1e-6, 0.6, 1.7])
def test_imaginary_modes(freq_min):
    """Imaginary modes are included with the magnitude of their frequency."""
    # bcc Cu is unstable with EMT
    structure = AseAtomsAdaptor.get_structure(bulk("Cu", "bcc", a=2.87, cubic=True))
    phonon = Phonopy(
        get_phonopy_structure(structure),
        supercell_matrix=3 * np.eye(3),
        primitive_matrix=np.eye(3),
    )
    phonon.generate_displacements(distance=0.01)
    forces = []
    for supercell in phonon.supercells_with_displacements:
        atoms = AseAtomsAdaptor.get_atoms(get_pmg_structure(supercell))
        atoms.calc = emt.EMT()
        forces.append(atoms.get_forces())
    phonon.forces = forces
    phonon.produce_force_constants()
    phonon.run_mesh(
        [6, 6, 6], with_eigenvectors=True, is_mesh_symmetry=False, is_gamma_center=True
    )
    frequencies = phonon.mesh.frequencies
    # the acoustic modes at Gamma are at 4e-6 THz. freq_min=0.6 leaves out the
    # imaginary modes at -0.53 THz, and 1.7 also the real modes at 1.6 THz.
    assert frequencies.min() < -0.1

    # an emmet document, which stores the force constants as a list
    phonon_doc = PhononBSDOSDoc(
        structure=structure,
        force_constants=phonon.force_constants.tolist(),
        supercell_matrix=phonon.supercell_matrix.tolist(),
        primitive_matrix=np.eye(3).tolist(),
        code="forcefields",
    )
    temperatures = [0.0, 1.0, 300.0]
    real, both = (
        np.array(
            DebyeWallerDocument.from_phonon_doc(
                phonon_doc,
                temperatures,
                mesh=(6, 6, 6),
                freq_min=freq_min,
                include_imaginary_modes=include,
            ).thermal_displacement_data.thermal_displacement_matrix
        )
        for include in (False, True)
    )

    # sum over all modes with |f| above freq_min, as phonopy does for f > 0,
    # without the acoustic modes at Gamma
    units = get_physical_units()
    abs_freq = np.abs(frequencies)
    eigenvectors = phonon.mesh.eigenvectors
    masses = phonon.primitive.masses
    reference = np.zeros_like(both)
    real_reference = np.zeros_like(both)
    for i_t, temperature in enumerate(temperatures):
        for i_q, i_band in zip(*np.nonzero(abs_freq > freq_min), strict=True):
            if i_q == 0 and i_band < 3:
                continue
            freq = abs_freq[i_q, i_band]
            occupation = 0.0
            if temperature > 1:
                x = freq * units.THzToEv / (units.KB * temperature)
                occupation = 1 / np.expm1(x)
            q2 = units.Hbar * (occupation + 0.5) / (freq * 2 * np.pi)
            q2 *= units.EV / units.AMU * 1e8
            vec = eigenvectors[i_q, :, i_band].reshape(-1, 3)
            for i_site, mass in enumerate(masses):
                term = q2 / mass * np.real(np.outer(vec[i_site], vec[i_site].conj()))
                reference[i_t, i_site] += term
                if frequencies[i_q, i_band] > 0:
                    real_reference[i_t, i_site] += term
    assert real == pytest.approx(real_reference / len(frequencies), rel=1e-6)
    assert both == pytest.approx(reference / len(frequencies), rel=1e-6)


def test_debye_waller_document_checks_input():
    phonon_doc = PhononBSDOSDoc(
        structure=AseAtomsAdaptor.get_structure(bulk("Cu", "fcc", a=3.61))
    )
    with pytest.raises(ValueError, match="must not be negative"):
        DebyeWallerDocument.from_phonon_doc(phonon_doc, [-1.0, 300.0], mesh=(1, 1, 1))
    with pytest.raises(ValueError, match="no force constants"):
        DebyeWallerDocument.from_phonon_doc(phonon_doc, [300.0], mesh=(1, 1, 1))
