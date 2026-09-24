"""Tests of the finite-temperature phonon workflow.

The end-to-end test runs the force field workflow with ASE's EMT potential, so
it needs no reference data. It lives here and not under tests/forcefields,
because pheasy is only installed in the test-non-ase CI job.
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
from jobflow import run_locally
from phonopy import Phonopy
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from atomate2.common.jobs.finite_temperature_phonons import (
    ASE_TRAJECTORY_FILE,
    _assess_trajectory,
    _get_rms_displacement,
    _remove_center_of_mass,
    fit_finite_temperature_phonons,
    get_md_restart_structure,
    get_md_supercell,
    select_md_snapshots,
)
from atomate2.forcefields.flows.finite_temperature_phonons import (
    ForceFieldFiniteTemperaturePhononMaker,
    _get_force_field_md_maker,
)
from atomate2.forcefields.jobs import ForceFieldStaticMaker
from atomate2.forcefields.md import ForceFieldMDMaker

EMT_CALCULATOR = {"@module": "ase.calculators.emt", "@callable": "EMT"}
TEMPERATURE = 300.0
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


def _verdict(reference, frac, energies):
    return _assess_trajectory(frac, energies, reference, TEMPERATURE).verdict


def test_assess_trajectory(cu_supercell):
    rng = np.random.default_rng(1)
    n_atoms = len(cu_supercell)

    frac, energies = _trajectory(cu_supercell, rng)
    health = _assess_trajectory(frac, energies, cu_supercell, TEMPERATURE)
    assert health.verdict == "stable"
    assert health.is_stable
    assert health.n_frames == 400
    # u_vib is sqrt(3) sigma for Gaussian displacements
    assert health.u_vib == pytest.approx(np.sqrt(3) * 0.05, rel=0.05)
    assert health.u_ref**2 == pytest.approx(health.u_shift**2 + health.u_vib**2)
    assert health.nearest_neighbor_distance == pytest.approx(3.61 / np.sqrt(2))
    assert health.equipartition_rise == pytest.approx(1.5 * units.kB * TEMPERATURE)

    # a translation of the whole cell is not a displacement
    moved = frac + np.array([0.1, 0.05, 0])
    health_moved = _assess_trajectory(moved, energies, cu_supercell, TEMPERATURE)
    assert health_moved.verdict == "stable"
    assert health_moved.u_ref == pytest.approx(health.u_ref)

    # a vibration of 0.2 A per direction is above the Lindemann limit
    frac, energies = _trajectory(cu_supercell, rng, sigma=0.2)
    assert _verdict(cu_supercell, frac, energies) == "melted"

    # the mean positions move in the second half, with a flat energy
    shift = rng.normal(0, 0.4, size=(n_atoms, 3))
    frac, energies = _trajectory(cu_supercell, rng, shift=shift)
    health = _assess_trajectory(frac, energies, cu_supercell, TEMPERATURE)
    assert health.verdict == "shifted_or_diffusing"
    assert health.shift_ratio > 1.5

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
    health = _assess_trajectory(frac, energies, cu_supercell, TEMPERATURE)
    assert health.verdict == "disordering"
    assert health.energy_drift == pytest.approx(0.05, rel=0.05)

    # a large move at the very end, in a structure with long bonds, stays below
    # the other limits
    sparse = Structure(Lattice.cubic(8.0), ["Cu"], [[0, 0, 0]]) * (3, 3, 3)
    frac, energies = _trajectory(sparse, rng)
    move = rng.choice([-2.2, 2.2], size=(len(sparse), 3)) / np.sqrt(3)
    frac[360:] = _frames(sparse, 40, 0.05, rng, shift=move)
    health = _assess_trajectory(frac, energies, sparse, TEMPERATURE)
    assert health.verdict == "diffusing_or_soft"
    assert health.rms_displacement_end > 1.0

    # large displacements in the first frames mean the atom order is wrong
    frac, energies = _trajectory(cu_supercell, rng)
    frac[:10] = frac[:10][:, rng.permutation(n_atoms)]
    assert _verdict(cu_supercell, frac, energies) == "reference_mismatch"

    # too few frames to compare the quarters
    health = _assess_trajectory(frac[:15], energies[:15], cu_supercell, TEMPERATURE)
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
    output = select_md_snapshots.original(
        md_dirs, "vasp", cu_supercell, TEMPERATURE, 1.0, 0.1, 10
    )
    structures = output["structures"]
    assert len(structures) == 11
    assert structures[-1] == cu_supercell
    indices = range(50, 300, 25)
    for structure, idx in zip(structures[:-1], indices, strict=True):
        diff = structure.frac_coords - frac[idx]
        assert np.allclose(diff - np.round(diff), 0, atol=1e-8)
    assert output["snapshot_times"] == pytest.approx([idx * 0.002 for idx in indices])
    assert output["md_time"] == pytest.approx(0.6)
    assert output["rms_displacement"] == pytest.approx(np.sqrt(3) * 0.05, rel=0.1)
    assert output["trajectory_health"]["n_frames"] == 300

    with pytest.raises(ValueError, match="fewer than the 300 snapshots"):
        select_md_snapshots.original(
            md_dirs[:1], "vasp", cu_supercell, 300, 1, 0.1, 300
        )

    # atoms of another species at the same positions
    other = cu_supercell.copy()
    other.replace_species({"Cu": "Ag"})
    with pytest.raises(ValueError, match="not in the order of the reference"):
        select_md_snapshots.original(md_dirs[:1], "vasp", other, 300, 1, 0.1, 10)

    # the same run twice, once with a host name
    with pytest.raises(ValueError, match="its own directory"):
        select_md_snapshots.original(
            [md_dirs[0], str(Path("md1").resolve())],
            "vasp",
            cu_supercell,
            300,
            1,
            0.1,
            10,
        )

    Path("md4").mkdir()
    for name in ("INCAR", "XDATCAR"):
        (Path("md4") / name).write_text((Path("md1") / name).read_text())
    (Path("md4") / "OSZICAR").write_text("")
    with pytest.raises(ValueError, match=r"0 MD steps in OSZICAR\..*NBLOCK = 1"):
        select_md_snapshots.original(
            [str(Path("md4").resolve())], "vasp", cu_supercell, 300, 1, 0.1, 10
        )


def _write_ase_md(directory, reference, frac, energies):
    frames = []
    for coords, energy in zip(frac, energies, strict=True):
        atoms = AseAtomsAdaptor.get_atoms(reference)
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
        300,
        2.0,
        0.02,
        14,
    )
    assert output["trajectory_health"]["n_frames"] == 151
    # 10 frames of 2 fs are left out, and 141 frames are left for 14 snapshots
    indices = range(10, 150, 10)
    assert output["snapshot_times"] == pytest.approx([idx * 0.002 for idx in indices])
    for structure, idx in zip(output["structures"][:-1], indices, strict=True):
        diff = structure.frac_coords - frac[idx]
        assert np.allclose(diff - np.round(diff), 0, atol=1e-8)

    # the snapshots keep the magnetic moments of the reference
    magmoms = [1.0] * len(cu_supercell)
    magnetic = cu_supercell.copy(site_properties={"magmom": magmoms})
    output = select_md_snapshots.original(
        [str(Path("md1").resolve())], "forcefields", magnetic, 300, 2.0, 0.02, 14
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
            300,
            2.0,
            0.02,
            14,
        )
    assert output["trajectory_health"]["verdict"] == "melted"


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


def test_get_md_restart_structure(cu_supercell):
    rng = np.random.default_rng(4)
    velocities = rng.normal(0, 0.01, size=(len(cu_supercell), 3))
    reference = cu_supercell.copy(site_properties={"magmom": [1.0] * 32})

    # a CONTCAR of a VASP MD, with the predictor-corrector block after the
    # velocities
    md_dir = TEST_DIR / "vasp/Si_multi_md/molecular_dynamics_1/outputs"
    si_reference = Structure(
        np.eye(3) * 3.87, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    restart = get_md_restart_structure.original(str(md_dir), "vasp", si_reference)
    assert restart.site_properties["velocities"][0] == pytest.approx(
        [0.40252568e-03, -0.31480439e-02, -0.16045943e-02]
    )
    assert "magmom" not in restart.site_properties

    Path("vasp").mkdir()
    structure = cu_supercell.copy(site_properties={"velocities": velocities})
    structure.to(filename="vasp/CONTCAR", fmt="poscar")
    restart = get_md_restart_structure.original(
        str(Path("vasp").resolve()), "vasp", reference
    )
    assert np.allclose(restart.site_properties["velocities"], velocities, atol=1e-8)
    assert restart.site_properties["magmom"] == [1.0] * 32

    Path("ase").mkdir()
    atoms = AseAtomsAdaptor.get_atoms(cu_supercell)
    atoms.set_velocities(velocities)
    ase_write(Path("ase") / ASE_TRAJECTORY_FILE, [atoms.copy(), atoms])
    restart = get_md_restart_structure.original(
        str(Path("ase").resolve()), "forcefields", reference
    )
    assert np.allclose(restart.site_properties["velocities"], velocities)
    assert np.allclose(restart.cart_coords, cu_supercell.cart_coords)
    assert restart.site_properties["magmom"] == [1.0] * 32

    with pytest.raises(ValueError, match="MD code must be"):
        get_md_restart_structure.original("ase", "lammps", reference)


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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"thermostat": "berendsen"}, "thermostat must be one of"),
        ({"rotational_sum_rule": "X"}, "rotational_sum_rule must be one of"),
        ({"md_runs": 0}, "md_runs must be a positive integer"),
        ({"n_snapshots": 2.5}, "n_snapshots must be a positive integer"),
        ({"alpha_min": -2}, "alpha_min must be an integer below -2"),
        ({"alpha_min": -6.0}, "alpha_min must be an integer below -2"),
        ({"md_time_step": 0}, "must be positive"),
        ({"equilibration_time": 8.0}, "fewer than the 50 snapshots"),
        ({"equilibration_time": -1.0}, "fewer than the 50 snapshots"),
        (
            {"md_time": 0.01, "equilibration_time": 0, "n_snapshots": 5, "md_runs": 20},
            "larger than the 10",
        ),
        ({"md_maker": None}, "md_maker and phonon_displacement_maker must be set"),
        ({"code": "aims"}, "code must be one of"),
        ({"md_code": None}, "md_code must be one of"),
    ],
)
def test_maker_checks(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ForceFieldFiniteTemperaturePhononMaker(**kwargs)


def test_force_field_md_maker(cu_supercell):
    maker = ForceFieldFiniteTemperaturePhononMaker.from_force_field_name(
        EMT_CALCULATOR,
        md_time=1.0,
        md_time_step=2.0,
        equilibration_time=0.5,
        md_runs=3,
        random_seed=5,
    )
    assert maker.get_md_steps() == [167, 167, 166]
    md_maker = maker.get_md_maker(167)
    assert md_maker.n_steps == 167
    assert md_maker.time_step == 2.0
    assert md_maker.temperature == TEMPERATURE
    assert md_maker.mb_velocity_seed == 5
    assert md_maker.dynamics == "nose-hoover-chain"
    assert md_maker.ase_md_kwargs["tchain"] == 1
    # a thermostat period of 40 steps, 80 fs
    period = np.pi * np.sqrt(2) * md_maker.ase_md_kwargs["tdamp"] / units.fs
    assert period == pytest.approx(80.0)
    # the maker of the flow is not changed
    assert maker.md_maker.n_steps == 1000
    assert maker.md_maker.dynamics is None
    # the force field relaxation keeps the symmetry
    assert maker.bulk_relax_maker.fix_symmetry
    assert ForceFieldFiniteTemperaturePhononMaker().bulk_relax_maker.fix_symmetry

    # the MD jobs have a number only when there are several
    flow = maker.make(cu_supercell, supercell_matrix=np.eye(3).tolist())
    md_names = [job.name for job in flow.jobs if "MD" in job.name]
    assert md_names == ["ASE MD 1/3", "ASE MD 2/3", "ASE MD 3/3"]
    maker = ForceFieldFiniteTemperaturePhononMaker(md_time=2.0)
    flow = maker.make(cu_supercell, supercell_matrix=np.eye(3).tolist())
    assert [job.name for job in flow.jobs if "MD" in job.name] == ["ASE MD"]

    langevin = ForceFieldFiniteTemperaturePhononMaker(
        thermostat="langevin"
    ).get_md_maker(10)
    # AseMDMaker sets its default friction when the job runs
    assert langevin.dynamics == "langevin"
    assert langevin.ase_md_kwargs == {}

    maker = ForceFieldFiniteTemperaturePhononMaker(
        md_maker=ForceFieldMDMaker(ase_md_kwargs={"tdamp": 10})
    )
    with pytest.raises(ValueError, match="must not set dynamics or ase_md_kwargs"):
        maker.get_md_maker(10)
    with pytest.raises(TypeError, match="must be a ForceFieldMDMaker"):
        _get_force_field_md_maker(
            ForceFieldFiniteTemperaturePhononMaker(md_maker=ForceFieldStaticMaker()), 10
        )


def _commensurate_frequencies(phonon):
    matrix = np.linalg.inv(phonon.primitive_matrix) @ phonon.supercell_matrix
    phonon.run_qpoints(get_commensurate_points(np.rint(matrix).astype(int)))
    return np.sort(phonon.qpoints.frequencies.ravel())


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
        "thermostat": "nose-hoover",
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

    with pytest.raises(ValueError, match="diagonal supercell matrix"):
        fit_finite_temperature_phonons.original(
            **{**fit_kwargs, "supercell_matrix": [[1, 1, 0], [0, 1, 0], [0, 0, 1]]}
        )
    with pytest.raises(ValueError, match="snapshots followed by the undisplaced"):
        fit_finite_temperature_phonons.original(
            **{**fit_kwargs, "snapshot_data": {"snapshot_times": [0.0] * 19}}
        )
    shuffled = {
        **displacement_data,
        "displaced_structures": [supercell, *snapshots],
    }
    with pytest.raises(ValueError, match="not the undisplaced supercell"):
        fit_finite_temperature_phonons.original(
            **{**fit_kwargs, "displacement_data": shuffled}
        )


def test_finite_temperature_phonon_maker_emt(clean_dir):
    """Run the whole force field workflow with EMT on L1_2 Cu3Au at 300 K."""
    # Au first, so that sorting the atoms by electronegativity reorders them
    structure = Structure(
        Lattice.cubic(3.75),
        ["Au", "Cu", "Cu", "Cu"],
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )
    # a 2x2x2 supercell of the cubic cell, 32 atoms, and 2 ps of MD at 2 fs in
    # two MD jobs
    maker = ForceFieldFiniteTemperaturePhononMaker.from_force_field_name(
        EMT_CALCULATOR,
        min_length=7.0,
        md_time=2.0,
        md_time_step=2.0,
        md_runs=2,
        equilibration_time=0.5,
        n_snapshots=12,
    )
    flow = maker.make(structure)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    doc = responses[flow.output.uuid][1].output

    assert [str(site.specie) for site in doc.structure] == ["Cu", "Cu", "Cu", "Au"]
    assert doc.temperature == TEMPERATURE
    assert doc.thermostat == "nose-hoover"
    assert doc.md_code == doc.code == "forcefields"
    assert doc.md_force_field_name == doc.force_field_name == "ase.calculators.emt.EMT"
    assert doc.force_field_kwargs == {}
    # 1000 steps and the starting structure
    assert doc.md_time == pytest.approx(2.002)
    assert doc.n_snapshots == 12
    assert len(doc.snapshot_times) == 12
    assert min(doc.snapshot_times) == pytest.approx(0.5)
    assert len(doc.md_uuids) == len(doc.md_dirs) == 2
    assert len(doc.uuids.displacements_uuids) == 13
    assert doc.uuids.optimization_run_uuid is not None
    assert doc.uuids.born_run_uuid is None
    assert doc.supercell_matrix == ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    assert doc.trajectory_health.verdict == "stable"
    assert doc.max_residual_force < 1e-8
    assert 0 < doc.force_rmse < 0.2
    assert doc.lasso_alpha > 0
    # Cu3Au is stable at 300 K
    assert not doc.has_imaginary_modes
    assert doc.n_imaginary_modes == 0
    assert doc.min_frequency > -0.1
    # the snapshot amplitude agrees with the one of the fitted force constants
    assert doc.rms_displacement == pytest.approx(
        doc.rms_displacement_from_force_constants, rel=0.25
    )
    force_constants = np.array(doc.force_constants.force_constants)
    assert force_constants.shape == (32, 32, 3, 3)
    fit_dir = Path(doc.jobdirs.taskdoc_run_job_dir)
    for name in ("FORCE_CONSTANTS", "phonopy.yaml", "pheasy_harmonic_fit.log"):
        assert (fit_dir / name).exists()

    # at a fixed volume, the frequencies stay close to the 0 K ones of EMT
    phonon = _emt_phonon(doc.structure, np.array(doc.supercell_matrix))
    freqs_0k = _commensurate_frequencies(phonon)
    phonon.force_constants = force_constants
    freqs = _commensurate_frequencies(phonon)
    top = len(freqs) // 3
    assert freqs[-top:].mean() == pytest.approx(freqs_0k[-top:].mean(), rel=0.15)
