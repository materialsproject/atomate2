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
from jobflow import run_locally
from phonopy import Phonopy
from phonopy.physical_units import get_physical_units
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from atomate2.common.schemas.debye_waller import (
    DebyeWallerDocument,
    get_thermal_displacement_matrices,
)
from atomate2.forcefields.flows.debye_waller import DebyeWallerMaker

EMT = {"@module": "ase.calculators.emt", "@callable": "EMT"}


def test_debye_waller_maker_emt(clean_dir):
    """Run the whole force field workflow with EMT forces on fcc Cu."""
    structure = AseAtomsAdaptor.get_structure(bulk("Cu", "fcc", a=3.61))
    maker = DebyeWallerMaker.from_force_field_name(
        EMT,
        temperatures=[0, 300, 600],
        mesh=(8, 8, 8),
        xrd_kwargs={"wavelength": "MoKa"},
    )
    maker.phonon_maker.min_length = 8.0

    flow = maker.make(structure)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    doc = responses[flow.output.uuid][1].output
    assert isinstance(doc, DebyeWallerDocument)
    assert doc.mesh == (8, 8, 8)
    assert len(doc.structure) == 1

    u = np.array(doc.thermal_displacement_data.thermal_displacement_matrix)
    assert u.shape == (3, 1, 3, 3)
    # cubic, so U is isotropic, and it grows with temperature
    for u_t in u[:, 0]:
        assert u_t == pytest.approx(u_t[0, 0] * np.eye(3), abs=1e-10)
    assert np.all(np.diff(u[:, 0, 0, 0]) > 0)
    assert u[1, 0, 0, 0] == pytest.approx(0.0054614, rel=1e-4)

    static = XRDCalculator("MoKa").get_pattern(doc.structure, scaled=False)
    assert doc.xrd_pattern_static.y == pytest.approx(static.y)

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
    assert len(doc.tem_patterns) == 3
    assert len(doc.tem_patterns[0]) == len(doc.tem_pattern_static)
    assert set(doc.tem_pattern_static[0]) == {"position", "hkl", "intensity"}


def test_debye_waller_maker_defaults():
    """The default phonon maker uses MACE-MP-0 and stores the force constants."""
    maker = DebyeWallerMaker()
    assert maker.phonon_maker.store_force_constants
    assert (
        maker.phonon_maker.phonon_displacement_maker.force_field_name
        == "MLFF.MACE_MP_0"
    )


def test_debye_waller_maker_needs_force_constants():
    """The workflow fails early if the force constants are not stored."""
    maker = DebyeWallerMaker.from_force_field_name(EMT)
    maker.phonon_maker.store_force_constants = False
    with pytest.raises(ValueError, match="store_force_constants=True"):
        DebyeWallerMaker(phonon_maker=maker.phonon_maker)


def test_imaginary_modes():
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
    assert frequencies.min() < -0.1

    temperatures = np.array([0.0, 1.0, 300.0])
    real = get_thermal_displacement_matrices(phonon, temperatures)
    both = get_thermal_displacement_matrices(
        phonon, temperatures, include_imaginary_modes=True
    )

    # sum over all modes with |f| above freq_min, as phonopy does for f > 0
    units = get_physical_units()
    abs_freq = np.abs(frequencies)
    eigenvectors = phonon.mesh.eigenvectors
    masses = phonon.primitive.masses
    reference = np.zeros_like(both)
    for i_t, temperature in enumerate(temperatures):
        for i_q, i_band in zip(*np.nonzero(abs_freq > 0.01), strict=True):
            freq = abs_freq[i_q, i_band]
            occupation = 0.0
            if temperature > 1:
                x = freq * units.THzToEv / (units.KB * temperature)
                occupation = 1 / np.expm1(x)
            q2 = units.Hbar * (occupation + 0.5) / (freq * 2 * np.pi)
            q2 *= units.EV / units.AMU * 1e8
            vec = eigenvectors[i_q, :, i_band].reshape(-1, 3)
            for i_site, mass in enumerate(masses):
                reference[i_t, i_site] += (
                    q2 / mass * np.real(np.outer(vec[i_site], vec[i_site].conj()))
                )
    reference /= len(frequencies)
    assert both == pytest.approx(reference, rel=1e-6)
    assert np.all(both[:, 0, 0, 0] > real[:, 0, 0, 0])
