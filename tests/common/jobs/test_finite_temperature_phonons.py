"""Tests of the jobs of the finite-temperature phonon workflow.

The tests of the whole force field workflow are in tests/forcefields/flows.
"""

import gzip
from pathlib import Path

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators import emt
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write
from phonopy import Phonopy
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from atomate2.common.jobs.finite_temperature_phonons import (
    ASE_TRAJECTORY_FILE,
    _assess_trajectory,
    _get_rms_displacement,
    _remove_center_of_mass,
    average_npt_structure,
    fit_finite_temperature_phonons,
    get_md_supercell,
    get_npt_structure,
    select_md_snapshots,
)

TEMPERATURE = 300.0
# a long time step, so that the second half of the synthetic trajectories of 400
# frames lasts 2 ps and the mean positions are checked
TIME_STEP = 10.0
TEST_DIR = Path(__file__).resolve().parents[2] / "test_data"


@pytest.fixture
def cu_supercell():
    """A 2x2x2 supercell of the cubic cell of fcc Cu, 32 atoms."""
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
    return AseAtomsAdaptor.get_structure(atoms)


def _frames(reference, n_frames, sigma, rng, shift=None):
    """Fractional coordinates of frames with Gaussian displacements in Angstrom."""
    disps = rng.normal(0, sigma, size=(n_frames, len(reference), 3))
    if shift is not None:
        disps += shift
    cart = reference.cart_coords + disps
    return np.array([reference.lattice.get_fractional_coords(c) for c in cart])


def _trajectory(reference, rng, sigma=0.05, shift=None, energies=None):
    """A trajectory of 400 frames that starts at the reference, and its energies."""
    frac = _frames(reference, 400, sigma, rng)
    frac[:10] = _frames(reference, 10, 0.01, rng)
    if shift is not None:
        frac[200:] = _frames(reference, 200, sigma, rng, shift=shift)
    if energies is None:
        energies = rng.normal(0, 1e-3, 400) * len(reference)
    return frac, energies


def _verdict(reference, frac, energies, time_step=TIME_STEP):
    return _assess_trajectory(frac, energies, reference, time_step).verdict


def test_assess_trajectory(cu_supercell):
    rng = np.random.default_rng(1)
    n_atoms = len(cu_supercell)

    frac, energies = _trajectory(cu_supercell, rng)
    health = _assess_trajectory(frac, energies, cu_supercell, TIME_STEP)
    assert health.verdict == "stable"
    assert health.n_frames == 400
    # u_vib is sqrt(3) sigma for Gaussian displacements
    assert health.u_vib == pytest.approx(np.sqrt(3) * 0.05, rel=0.05)
    assert health.u_ref**2 == pytest.approx(health.u_shift**2 + health.u_vib**2)
    assert health.nearest_neighbor_distance == pytest.approx(3.61 / np.sqrt(2))

    # a translation of the whole cell is not a displacement
    moved = frac + np.array([0.1, 0.05, 0])
    health_moved = _assess_trajectory(moved, energies, cu_supercell, TIME_STEP)
    assert health_moved.verdict == "stable"
    assert health_moved.u_ref == pytest.approx(health.u_ref)

    # a vibration of 0.25 A per direction is above the Lindemann limit
    frac, energies = _trajectory(cu_supercell, rng, sigma=0.25)
    assert _verdict(cu_supercell, frac, energies) == "melted"

    # the mean positions move in the second half, with a flat energy
    shift = rng.normal(0, 0.4, size=(n_atoms, 3))
    frac, energies = _trajectory(cu_supercell, rng, shift=shift)
    health = _assess_trajectory(frac, energies, cu_supercell, TIME_STEP)
    assert health.verdict == "shifted_or_diffusing"
    assert health.shift_ratio > 1.5
    # in a second half shorter than 1 ps the mean positions are not checked
    assert _verdict(cu_supercell, frac, energies, time_step=1.0) == "stable"

    # the same move with a falling energy is a transformation
    falling = np.concatenate([np.zeros(200), np.linspace(0, -0.1, 200)]) * n_atoms
    frac, energies = _trajectory(cu_supercell, rng, shift=shift, energies=falling)
    assert _verdict(cu_supercell, frac, energies) == "transformed"

    # a falling energy without a move is a transformation too
    frac, energies = _trajectory(cu_supercell, rng, energies=falling)
    assert _verdict(cu_supercell, frac, energies) == "transformed"

    # a rising energy at the reference is disordering
    rising = np.linspace(0, 0.1, 400) * n_atoms
    frac, energies = _trajectory(cu_supercell, rng, energies=rising)
    health = _assess_trajectory(frac, energies, cu_supercell, TIME_STEP)
    assert health.verdict == "disordering"
    assert health.energy_drift == pytest.approx(0.05, rel=0.05)

    # a large move at the very end, in a structure with long bonds, stays below
    # the other limits
    sparse = Structure(Lattice.cubic(8.0), ["Cu"], [[0, 0, 0]]) * (3, 3, 3)
    frac, energies = _trajectory(sparse, rng)
    move = rng.choice([-2.2, 2.2], size=(len(sparse), 3)) / np.sqrt(3)
    frac[360:] = _frames(sparse, 40, 0.05, rng, shift=move)
    health = _assess_trajectory(frac, energies, sparse, TIME_STEP)
    assert health.verdict == "diffusing_or_soft"
    assert health.rms_displacement_end > 1.0

    # large displacements in the first frames mean the atom order is wrong
    frac, energies = _trajectory(cu_supercell, rng)
    frac[:10] = frac[:10][:, rng.permutation(n_atoms)]
    assert _verdict(cu_supercell, frac, energies) == "reference_mismatch"

    # too few frames to compare the quarters
    health = _assess_trajectory(frac[:15], energies[:15], cu_supercell, TIME_STEP)
    assert health.energy_drift is None


def _write_vasp_md(directory, reference, frac, energies, gz=False):
    """Write the INCAR, XDATCAR and OSZICAR of a VASP MD run with POTIM = 2."""
    directory.mkdir()
    lat = reference.lattice.matrix
    lines = ["Cu", "1.0", *[" ".join(f"{x:.10f}" for x in row) for row in lat]]
    lines += ["Cu", str(len(reference))]
    oszicar = []
    for idx, (coords, energy) in enumerate(zip(frac, energies, strict=True), 1):
        lines.append(f"Direct configuration= {idx:5d}")
        lines += [" ".join(f"{x:.10f}" for x in row) for row in coords]
        oszicar += [
            "DAV:   1    -0.1E+02   -0.1E+02   -0.1E+02   100   0.1E+00",
            (
                f"{idx:6d} T=   300. E= -.1E+02 F= {energy:.8E} "
                f"E0= {energy:.8E}  EK= 0.1E+00 SP= 0.0E+00 SK= 0.0E+00"
            ),
        ]
    files = {
        "INCAR": "IBRION = 0\nPOTIM = 2.0\n",
        "XDATCAR": "\n".join(lines) + "\n",
        "OSZICAR": "\n".join(oszicar) + "\n",
    }
    for name, text in files.items():
        if gz:
            with gzip.open(directory / f"{name}.gz", "wt") as file:
                file.write(text)
        else:
            (directory / name).write_text(text)


def test_select_md_snapshots_vasp(cu_supercell):
    rng = np.random.default_rng(2)
    frac = _frames(cu_supercell, 300, 0.05, rng)
    energies = rng.normal(-100, 0.01, 300)
    # two runs with POTIM = 2 fs, the second one gzipped
    _write_vasp_md(Path("md1"), cu_supercell, frac[:150], energies[:150])
    _write_vasp_md(Path("md2"), cu_supercell, frac[150:], energies[150:], gz=True)
    md_dirs = ["host:" + str(Path("md1").resolve()), str(Path("md2").resolve())]

    # 0.1 ps is 50 frames of 2 fs, so 250 frames are left for 10 snapshots
    output = select_md_snapshots.original(md_dirs, "vasp", cu_supercell, 1.0, 0.1, 10)
    structures = output["structures"]
    assert len(structures) == 11
    assert structures[-1] == cu_supercell
    # from the first to the last frame after 0.1 ps
    indices = [50, 78, 105, 133, 161, 188, 216, 244, 271, 299]
    for structure, idx in zip(structures[:-1], indices, strict=True):
        diff = structure.frac_coords - frac[idx]
        assert np.allclose(diff - np.round(diff), 0, atol=1e-8)
    # XDATCAR has the frames after steps 1 to 300
    assert output["snapshot_times"] == pytest.approx(
        [(idx + 1) * 0.002 for idx in indices]
    )
    assert output["md_time"] == pytest.approx(0.6)
    assert output["rms_displacement"] == pytest.approx(np.sqrt(3) * 0.05, rel=0.1)
    assert output["trajectory_health"]["n_frames"] == 300

    with pytest.raises(ValueError, match="fewer than the 300 snapshots"):
        select_md_snapshots.original(md_dirs[:1], "vasp", cu_supercell, 1, 0.1, 300)

    # atoms of another species at the same positions
    other = cu_supercell.copy()
    other.replace_species({"Cu": "Ag"})
    with pytest.raises(ValueError, match="not in the order of the reference"):
        select_md_snapshots.original(md_dirs[:1], "vasp", other, 1, 0.1, 10)

    Path("md4").mkdir()
    for name in ("INCAR", "XDATCAR"):
        (Path("md4") / name).write_text((Path("md1") / name).read_text())
    (Path("md4") / "OSZICAR").write_text("")
    with pytest.raises(ValueError, match=r"0 MD steps in OSZICAR\..*NBLOCK = 1"):
        select_md_snapshots.original(
            [str(Path("md4").resolve())], "vasp", cu_supercell, 1, 0.1, 10
        )


def _write_ase_md(directory, reference, frac, energies, cells=None):
    frames = []
    for idx, (coords, energy) in enumerate(zip(frac, energies, strict=True)):
        atoms = AseAtomsAdaptor.get_atoms(reference)
        if cells is not None:
            atoms.set_cell(cells[idx])
        atoms.set_scaled_positions(coords)
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
        frames.append(atoms)
    directory.mkdir()
    ase_write(directory / ASE_TRAJECTORY_FILE, frames)


def test_select_md_snapshots_forcefields(cu_supercell):
    rng = np.random.default_rng(3)
    frac = _frames(cu_supercell, 151, 0.05, rng)
    energies = rng.normal(0, 0.01, 151)
    # the second run starts from the last frame of the first one
    _write_ase_md(Path("md1"), cu_supercell, frac[:101], energies[:101])
    _write_ase_md(Path("md2"), cu_supercell, frac[100:], energies[100:])

    output = select_md_snapshots.original(
        [str(Path(name).resolve()) for name in ("md1", "md2")],
        "forcefields",
        cu_supercell,
        2.0,
        0.02,
        14,
    )
    # the starting structure of each run is left out, so frame 0 is step 1
    assert output["trajectory_health"]["n_frames"] == 150
    # 10 frames of 2 fs are left out, and 140 frames are left for 14 snapshots
    indices = np.linspace(10, 149, 14).round().astype(int)
    assert output["snapshot_times"] == pytest.approx((indices + 1) * 0.002)
    for structure, idx in zip(output["structures"][:-1], indices, strict=True):
        diff = structure.frac_coords - frac[idx + 1]
        assert np.allclose(diff - np.round(diff), 0, atol=1e-8)

    # the snapshots keep the magnetic moments of the reference
    magmoms = [1.0] * len(cu_supercell)
    magnetic = cu_supercell.copy(site_properties={"magmom": magmoms})
    output = select_md_snapshots.original(
        [str(Path("md1").resolve())], "forcefields", magnetic, 2.0, 0.02, 14
    )
    for structure in output["structures"]:
        assert structure.site_properties == {"magmom": magmoms}

    # a trajectory that leaves the reference gives a warning
    melted = _frames(cu_supercell, 101, 0.3, rng)
    melted[:10] = cu_supercell.frac_coords
    _write_ase_md(Path("md3"), cu_supercell, melted, energies[:101])
    with pytest.warns(UserWarning, match="verdict: melted"):
        output = select_md_snapshots.original(
            [str(Path("md3").resolve())],
            "forcefields",
            cu_supercell,
            2.0,
            0.02,
            14,
        )
    assert output["trajectory_health"]["verdict"] == "melted"


def test_average_npt_structure():
    """Rotated cells of an expanded tetragonal supercell average to its unit cell."""
    structure = Structure(
        Lattice.tetragonal(3.0, 5.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    matrix = np.diag([2, 2, 1])
    supercell = np.diag([1.01, 1.01, 1.03]) @ matrix @ structure.lattice.matrix
    rng = np.random.default_rng(7)
    cells = []
    for _ in range(5):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        cells.append(supercell @ rotation.T)
    averaged = average_npt_structure(np.array(cells), structure, matrix, 1e-3)
    assert np.allclose(averaged.lattice.matrix, np.diag([3.03, 3.03, 5.15]))
    assert np.allclose(averaged.frac_coords, structure.frac_coords)


def test_get_npt_structure(cu_supercell):
    unit_cell = AseAtomsAdaptor.get_structure(bulk("Cu", "fcc", a=3.61, cubic=True))
    rng = np.random.default_rng(4)
    n_frames = 201
    frac = _frames(cu_supercell, n_frames, 0.05, rng)
    # the cell is 2% longer in each direction, with random strains, and it
    # rotates about the z axis
    cells = []
    for angle in rng.uniform(0, 2 * np.pi, n_frames):
        strain = np.eye(3) + rng.normal(0, 0.005, (3, 3))
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
        cells.append(1.02 * cu_supercell.lattice.matrix @ strain @ rotation.T)
    energies = rng.normal(0, 0.01, n_frames)
    _write_ase_md(Path("npt"), cu_supercell, frac, energies, cells=cells)

    args = ("forcefields", unit_cell, (2 * np.eye(3)).tolist(), cu_supercell)
    output = get_npt_structure.original(
        str(Path("npt").resolve()), *args, 2.0, 0.02, 1e-4
    )
    structure = output["structure"]
    # the cell stays cubic and keeps the orientation of the unit cell
    assert np.allclose(
        structure.lattice.matrix, 1.02 * unit_cell.lattice.matrix, atol=5e-3
    )
    assert structure.lattice.abc == pytest.approx([structure.lattice.a] * 3)
    assert np.allclose(structure.frac_coords, unit_cell.frac_coords)
    assert output["trajectory_health"]["verdict"] == "stable"

    with pytest.raises(ValueError, match="none of them after"):
        get_npt_structure.original(str(Path("npt").resolve()), *args, 2.0, 1.0, 1e-4)


def test_get_md_supercell():
    """The supercell has the atom order of phonopy and the magnetic moments."""
    structure = Structure(
        Lattice.cubic(4.17),
        ["O", "Ni", "O", "Ni"],
        [[0.5, 0, 0], [0, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]],
        site_properties={"magmom": [0.0, 2.0, 0.0, -2.0]},
    )
    matrix = np.diag([2, 2, 1])
    supercell = get_md_supercell.original(structure, matrix, 1e-3, "vasp")
    assert len(supercell) == 16
    assert [str(site.specie) for site in supercell] == ["Ni"] * 8 + ["O"] * 8
    # each atom of the sorted unit cell has four images in a row
    assert supercell.site_properties["magmom"] == [2.0] * 4 + [-2.0] * 4 + [0.0] * 8

    supercell = get_md_supercell.original(
        structure.copy().remove_site_property("magmom"),
        np.diag([2, 2, 1]),
        1e-3,
        "vasp",
    )
    assert supercell.site_properties == {}


def test_get_rms_displacement():
    """An Einstein crystal has <u**2> = 3 k_B T / k per atom."""
    structure = AseAtomsAdaptor.get_structure(bulk("Cu", "fcc", a=3.61, cubic=True))
    phonon = Phonopy(get_phonopy_structure(structure), np.diag([2, 2, 2]))
    n_atoms = len(phonon.supercell)
    spring = 2.0  # eV/A^2
    force_constants = np.zeros((n_atoms, n_atoms, 3, 3))
    force_constants[np.arange(n_atoms), np.arange(n_atoms)] = spring * np.eye(3)
    phonon.force_constants = force_constants
    rms = _get_rms_displacement(phonon, TEMPERATURE, 0.1)
    assert rms == pytest.approx(np.sqrt(3 * units.kB * TEMPERATURE / spring))


def test_rms_displacement_matches_mode_sampling():
    """Canonical mode sampling of Cu3Au gives the rms of the force constants.

    Each sample gets a random translation. Removing the displacement of the
    center of mass takes it out again. The unweighted mean of the displacements
    is not zero for two species, so it would not.
    """
    structure = Structure(
        Lattice.cubic(3.75),
        ["Cu", "Cu", "Cu", "Au"],
        [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0, 0]],
    )
    phonon = _emt_phonon(structure, np.diag([2, 2, 2]))
    masses = np.repeat(phonon.supercell.masses, 3)
    n_atoms = len(phonon.supercell)
    dynmat = phonon.force_constants.transpose(0, 2, 1, 3).reshape(
        3 * n_atoms, 3 * n_atoms
    ) / np.sqrt(np.outer(masses, masses))
    eigvals, eigvecs = np.linalg.eigh((dynmat + dynmat.T) / 2)
    keep = eigvals > 1e-6

    rng = np.random.default_rng(6)
    n_samples = 4000
    amplitudes = rng.normal(size=(n_samples, keep.sum())) * np.sqrt(
        units.kB * TEMPERATURE / eigvals[keep]
    )
    disps = (amplitudes @ eigvecs[:, keep].T / np.sqrt(masses)).reshape(
        n_samples, n_atoms, 3
    )
    disps += rng.normal(0, 0.1, size=(n_samples, 1, 3))

    reference = get_pmg_structure(phonon.supercell)
    removed = _remove_center_of_mass(disps, reference)
    rms = np.sqrt(np.mean(np.sum(removed**2, axis=2)))
    assert rms == pytest.approx(
        _get_rms_displacement(phonon, TEMPERATURE, 0.1), rel=0.02
    )
    unweighted = disps - disps.mean(axis=1, keepdims=True)
    assert not np.allclose(unweighted, removed, atol=1e-3)


def _emt_phonon(structure, supercell_matrix):
    """0 K phonopy object of EMT, with its force constants."""
    phonon = Phonopy(
        get_phonopy_structure(structure), supercell_matrix, primitive_matrix="auto"
    )
    phonon.generate_displacements(distance=0.01)
    forces = []
    for cell in phonon.supercells_with_displacements:
        atoms = AseAtomsAdaptor.get_atoms(get_pmg_structure(cell))
        atoms.calc = emt.EMT()
        forces.append(atoms.get_forces())
    phonon.forces = forces
    phonon.produce_force_constants()
    return phonon


def test_fit_recovers_harmonic_force_constants():
    """Forces from known force constants give them back, with the NAC applied."""
    structure = Structure(
        Lattice.cubic(3.75),
        ["Cu", "Cu", "Cu", "Au"],
        [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0, 0]],
    )
    supercell_matrix = np.diag([2, 2, 2]).tolist()
    phonon = _emt_phonon(structure, supercell_matrix)
    supercell = get_pmg_structure(phonon.supercell)
    disps = np.random.default_rng(0).normal(0, 0.03, (20, len(supercell), 3))
    forces = -np.einsum("ijab,mjb->mia", phonon.force_constants, disps)
    snapshots = [
        Structure(
            supercell.lattice,
            supercell.species,
            supercell.cart_coords + disp,
            coords_are_cartesian=True,
        )
        for disp in disps
    ]
    displacement_data = {
        "forces": [*forces, np.zeros((len(supercell), 3))],
        "displaced_structures": [*snapshots, supercell],
        "uuids": ["uuid"] * 21,
        "dirs": ["dir"] * 21,
    }
    snapshot_data = {
        "snapshot_times": [0.0] * 20,
        "md_time": 1.0,
        "rms_displacement": 0.05,
        "trajectory_health": {},
    }
    born = [np.eye(3).tolist()] * 3 + [(-3 * np.eye(3)).tolist()]
    fit_kwargs = {
        "structure": structure,
        "supercell_matrix": supercell_matrix,
        "snapshot_data": snapshot_data,
        "displacement_data": displacement_data,
        "temperature": TEMPERATURE,
        "thermostat": "langevin",
        "md_time_step": 1.0,
        "equilibration_time": 0.5,
        "code": "forcefields",
        "md_code": "forcefields",
        "symprec": 1e-3,
        "rotational_sum_rule": None,
        "epsilon_static": (10 * np.eye(3)).tolist(),
        "npoints_band": 11,
    }
    doc = fit_finite_temperature_phonons.original(born=born, **fit_kwargs)
    assert doc.force_rmse < 0.01 * np.sqrt(np.mean(forces**2))
    force_constants = np.array(doc.force_constants.force_constants)
    assert np.abs(force_constants - phonon.force_constants).max() < 0.05
    assert np.array(doc.born).shape == (4, 3, 3)
    assert doc.phonon_bandstructure.has_nac
    assert not doc.has_imaginary_modes
    assert doc.uuids.displacements_uuids == ["uuid"] * 21
    assert doc.jobdirs.displacements_job_dirs == ["dir"] * 21

    # the Born charges are checked before pheasy runs
    fc_time = Path("FORCE_CONSTANTS").stat().st_mtime_ns
    with pytest.raises(ValueError, match="number of Born charges"):
        fit_finite_temperature_phonons.original(born=born[:3], **fit_kwargs)
    assert Path("FORCE_CONSTANTS").stat().st_mtime_ns == fc_time
