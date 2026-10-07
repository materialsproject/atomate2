"""Tests for the sqs2tdb fit of the CALPHAD workflow."""

import json
import re
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from ase.calculators.emt import EMT
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_pmg_structure
from scipy.constants import R, electron_volt, physical_constants

from atomate2.common.jobs.calphad import (
    _copy_sqs,
    _get_relaxation_strain,
    fit_tdb,
    get_liquid_energy,
    get_sqs_structures,
    get_vibrational_entropy,
)
from atomate2.common.jobs.phonons import _generate_phonon_object
from atomate2.forcefields.flows.phonons import PhononMaker

needs_atat = pytest.mark.skipif(
    shutil.which("sqs2tdb") is None, reason="ATAT sqs2tdb is not installed"
)


def test_get_relaxation_strain():
    """Compare with ATAT checkcell, which gives 0.0791 for these two cells."""
    initial = Structure(Lattice(4 * np.eye(3)), ["Cu"], [[0, 0, 0]])
    relaxed = Structure(
        Lattice([[4.1, 0.2, 0], [0, 3.9, 0], [0.1, 0, 4.3]]), ["Cu"], [[0, 0, 0]]
    )
    assert _get_relaxation_strain(initial, relaxed) == pytest.approx(0.0791, abs=1e-4)
    scaled = Structure(Lattice(4.3 * np.eye(3)), ["Cu"], [[0, 0, 0]])
    assert _get_relaxation_strain(initial, scaled) == pytest.approx(0)


def test_get_liquid_energy():
    """One of two Cu atoms moves by 0.4 A, then the whole cell moves by 1 A."""
    lattice = Lattice(10 * np.eye(3))
    coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    moved = coords + np.array([[0.04, 0, 0], [0, 0, 0]])
    frames = [coords, moved] + [moved + np.array([0, 0.1, 0])] * 3
    steps = [
        SimpleNamespace(structure=Structure(lattice, ["Cu", "Cu"], c), energy=e)
        for c, e in zip(frames, (8.0, 16.0, 24.0, 32.0, 40.0), strict=True)
    ]
    md_output = SimpleNamespace(
        output=SimpleNamespace(ionic_steps=steps), dir_name="md"
    )
    result = get_liquid_energy.original(md_output, "LIQUID", "folder", 0, 8)
    assert result["energy"] == pytest.approx(3.0)
    # five blocks of one frame with 1 to 5 eV per cell
    assert result["energy_standard_error"] == pytest.approx(np.sqrt(0.5))
    # relative to the centre of mass each atom moved 0.2 A
    assert result["mean_squared_displacement"] == pytest.approx(0.04)


def test_get_vibrational_entropy():
    """FCC Cu with EMT, in its 1-atom and its 4-atom cell."""
    maker = PhononMaker()
    a = 3.6
    cells = {
        1: Structure(Lattice((a / 2) * (1 - np.eye(3))), ["Cu"], [[0, 0, 0]]),
        4: Structure(
            Lattice.cubic(a),
            ["Cu"] * 4,
            [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
        ),
    }
    for n_sites, repeats in ((1, 5), (4, 3)):
        supercell = np.diag([repeats] * 3).tolist()
        phonon = _generate_phonon_object(
            cells[n_sites],
            supercell,
            maker.displacement,
            maker.sym_reduce,
            maker.symprec,
            maker.use_symmetrized_structure,
            maker.kpath_scheme,
            maker.code,
        )
        forces = []
        for cell in phonon.supercells_with_displacements:
            atoms = AseAtomsAdaptor.get_atoms(get_pmg_structure(cell))
            atoms.calc = EMT()
            forces.append(atoms.get_forces())
        result = get_vibrational_entropy.original(
            cells[n_sites], supercell, forces, maker
        )
        # the entropy is per SQS cell, 10.60 k_B per atom at 3000 K
        assert result["vibrational_entropy"] == pytest.approx(
            10.60 * n_sites, abs=0.01 * n_sites
        )
        assert result["imaginary_fraction"] == 0


def _get_mixing_calculations(sqs, entropy=None):
    """Mixing energy -0.02 eV/atom and vibrational entropy -0.25 k_B/atom at 50%."""
    end_members = {"Cu": -3.7, "Ni": -5.0}
    calculations = []
    for calc in sqs:
        composition = calc["structure"].composition
        energy = sum(end_members[str(el)] * n for el, n in composition.items())
        if len(composition) == 2:
            energy -= 0.02 * composition.num_atoms
        calculations.append({**calc, "energy": energy})
        if entropy is not None and calc["lattice"] != "LIQUID":
            calculations[-1]["vibrational_entropy"] = composition.num_atoms * (
                entropy - 0.25 * (len(composition) == 2)
            )
    return calculations


@needs_atat
def test_copy_sqs_bump(tmp_path):
    """The pure element end members of CSCL_B2 have the symmetry of BCC_A2."""
    _copy_sqs(("Cu", "Ni"), ("CSCL_B2",), 1, str(tmp_path))
    bumps = sorted(path.parent.name for path in tmp_path.glob("CSCL_B2/*/bump"))
    assert bumps == ["sqs_lev=0_a_Cu=1,b_Cu=1", "sqs_lev=0_a_Ni=1,b_Ni=1"]


@needs_atat
def test_fit_tdb():
    """A mixing energy of -0.02 eV/atom at x = 0.5 gives L0 = -0.08 eV/atom."""
    elements, lattices = ("Cu", "Ni"), ("FCC_A1", "LIQUID")
    sqs = get_sqs_structures.original(elements, lattices, 1)
    assert [(calc["lattice"], calc["folder"]) for calc in sqs] == [
        (lattice, folder)
        for lattice in lattices
        for folder in (
            "sqs_lev=0_a_Cu=1",
            "sqs_lev=0_a_Ni=1",
            "sqs_lev=1_a_Cu=0.5,a_Ni=0.5",
        )
    ]

    calculations = _get_mixing_calculations(sqs)
    terms = {lattice: ["1,0", "2,0"] for lattice in lattices}
    doc = fit_tdb.original(elements, 1, terms, calculations)

    l0 = -0.08 * electron_volt * physical_constants["Avogadro constant"][0]
    for lattice in lattices:
        line = next(
            line for line in doc.tdb.splitlines() if f"L({lattice},CU,NI;0)" in line
        )
        assert float(line.split()[3]) == pytest.approx(l0, abs=0.1)
    # the structures were not relaxed
    strains = [calc.relaxation_strain for calc in doc.calculations]
    assert strains == pytest.approx([0] * 6, abs=1e-12)


@needs_atat
def test_fit_tdb_vibrational_entropy():
    """-0.25 k_B/atom at x = 0.5 gives L0 = -0.08 eV/atom + R T."""
    elements, lattices = ("Cu", "Ni"), ("FCC_A1", "LIQUID")
    sqs = get_sqs_structures.original(elements, lattices, 1)
    calculations = _get_mixing_calculations(sqs, entropy=3.0)
    terms = {lattice: ["1,0", "2,0"] for lattice in lattices}
    doc = fit_tdb.original(elements, 1, terms, calculations)
    line = next(line for line in doc.tdb.splitlines() if "L(FCC_A1,CU,NI;0)" in line)
    assert float(line.split()[3].rpartition("+")[2].removesuffix("*T")) == (
        pytest.approx(R, abs=1e-3)
    )
    line = next(line for line in doc.tdb.splitlines() if "L(LIQUID,CU,NI;0)" in line)
    assert "*T" not in line


@needs_atat
def test_fit_tdb_short_range_order():
    """The CVM term of sqs2tdb scales with |L0| / (12 R) for 12 FCC neighbours."""
    elements, lattices = ("Cu", "Ni"), ("FCC_A1", "LIQUID")
    sqs = get_sqs_structures.original(elements, lattices, 1)
    terms = {lattice: ["1,0", "2,0"] for lattice in lattices}
    calculations = _get_mixing_calculations(sqs)
    doc = fit_tdb.original(elements, 1, terms, calculations, short_range_order=True)
    tdb = doc.tdb.replace("\n    ", "")
    assert "L(FCC_A1,CU,NI;2) 298.15 +3859.4" in tdb
    assert "EXP(-77.363" in tdb
    line = next(line for line in tdb.splitlines() if "L(LIQUID,CU,NI;0)" in line)
    assert "EXP" not in line


@needs_atat
def test_fit_tdb_vibrational_entropy_unstable():
    """CSCL_B2 links to the BCC_A2 end members, whose phonons are unstable."""
    elements = ("Cu", "Ni")
    terms = {
        "FCC_A1": ["1,0", "2,0"],
        "BCC_A2": ["1,0", "2,0"],
        "CSCL_B2": ["1,0:1,0", "2,0:1,0"],
    }
    sqs = get_sqs_structures.original(elements, list(terms), 1)
    calculations = _get_mixing_calculations(sqs, entropy=3.0)
    for calc in calculations:
        calc["imaginary_fraction"] = 0.1 * (calc["lattice"] == "BCC_A2")
    with pytest.warns(UserWarning, match="vibrational entropy of") as record:
        doc = fit_tdb.original(elements, 1, terms, calculations)
    assert sorted(str(warning.message).split()[4] for warning in record) == [
        "BCC_A2",
        "CSCL_B2",
    ]
    fcc, bcc = (
        next(line for line in doc.tdb.splitlines() if f"L({lattice},CU,NI;0)" in line)
        for lattice in ("FCC_A1", "BCC_A2")
    )
    assert "*T" in fcc
    assert "*T" not in bcc


@needs_atat
def test_fit_tdb_links():
    """CSCL_B2 links to BCC_A2 and to other CSCL_B2 folders."""
    elements = ("Cu", "Ni")
    terms = {
        "FCC_A1": ["1,0", "2,0"],
        "BCC_A2": ["1,0", "2,0"],
        "CSCL_B2": ["1,0:1,0", "2,0:1,0"],
    }
    sqs = get_sqs_structures.original(elements, list(terms), 1)
    calculations = [{**calc, "energy": -1.0} for calc in sqs]
    doc = fit_tdb.original(elements, 1, terms, calculations)
    assert "PHASE CSCL_B2" in doc.tdb


@needs_atat
def test_fit_tdb_missing_link():
    """CSCL_B2 links to the BCC_A2 end members, which are not fitted."""
    terms = {"FCC_A1": ["1,0", "2,0"], "CSCL_B2": ["1,0:1,0", "2,0:1,0"]}
    sqs = get_sqs_structures.original(("Cu", "Ni"), list(terms), 1)
    calculations = [{**calc, "energy": -1.0} for calc in sqs]
    with pytest.raises(ValueError, match="links to BCC_A2"):
        fit_tdb.original(("Cu", "Ni"), 1, terms, calculations)


@needs_atat
def test_fit_tdb_warns_unconverged():
    sqs = get_sqs_structures.original(("Cu", "Ni"), ("FCC_A1",), 1)
    calculations = [
        {**calc, "energy": -1.0, "is_force_converged": True} for calc in sqs
    ]
    calculations[0]["is_force_converged"] = False
    with pytest.warns(UserWarning, match="sqs_lev=0_a_Cu=1 did not converge"):
        fit_tdb.original(("Cu", "Ni"), 1, {"FCC_A1": ["1,0", "2,0"]}, calculations)


@needs_atat
def test_fit_tdb_empty_parameter():
    """Level 1 has one mixed composition, too few for L0 and L1."""
    sqs = get_sqs_structures.original(("Cu", "Ni"), ("FCC_A1",), 1)
    calculations = [{**calc, "energy": -1.0} for calc in sqs]
    with pytest.raises(ValueError, match="empty parameter"):
        fit_tdb.original(("Cu", "Ni"), 1, {"FCC_A1": ["1,0", "2,1"]}, calculations)


@needs_atat
def test_fit_tdb_missing_energy():
    sqs = get_sqs_structures.original(("Cu", "Ni"), ("FCC_A1",), 1)
    calculations = [{**calc, "energy": -1.0} for calc in sqs[:-1]]
    with pytest.raises(ValueError, match="No energy for FCC_A1"):
        fit_tdb.original(("Cu", "Ni"), 1, {"FCC_A1": ["1,0", "2,0"]}, calculations)


@needs_atat
def test_fit_tdb_missing_stable_lattice():
    """Re is stable in HCP_A3, which NI4MO_D1A refers to."""
    elements = ("Ni", "Re")
    terms = {"FCC_A1": ["1,0", "2,0"], "NI4MO_D1A": ["1,0:1,0", "2,0:1,0"]}
    sqs = get_sqs_structures.original(elements, list(terms), 1)
    calculations = [{**calc, "energy": -1.0} for calc in sqs]
    with pytest.raises(ValueError, match="ABIN_HCP_A3_RE"):
        fit_tdb.original(elements, 1, terms, calculations)


@needs_atat
@pytest.mark.parametrize(
    ("model", "l0", "l0_per_k"),
    [
        ("MACE-OMAT-0-medium", (7168.0, 15938.0, 366.0), (0.414, -0.357)),
        ("MACE-MATPES-PBE-0", (-2722.8, 2480.6, -17578.7), (2.853, 2.893)),
        ("MACE-MATPES-r2SCAN-0", (3980.0, 14099.9, -13919.9), (-0.116, -0.031)),
        ("GRACE-2L-OMAT", (1357.5, 891.5, -3860.4), (-1.906, 3.358)),
    ],
)
def test_fit_tdb_ni_re(test_dir, model, l0, l0_per_k):
    """Refit Ni-Re from the energies and vibrational entropies of the tutorial."""
    energies = json.loads(
        (test_dir / "common" / "calphad" / "ni_re_energies.json").read_text()
    )
    terms = {
        "FCC_A1": ["1,0", "2,0"],
        "HCP_A3": ["1,0", "2,0"],
        "NI3SN_D019": ["1,0:1,0", "2,0:1,0"],
        "NI4MO_D1A": ["1,0:1,0", "2,0:1,0"],
        "LIQUID": ["1,0", "2,0"],
    }
    doc = fit_tdb.original(("Ni", "Re"), 2, terms, energies[model])
    # the liquid has no vibrational entropy, so no term in T
    slopes = (*l0_per_k, 0.0)
    for lattice, value, slope in zip(
        ("FCC_A1", "HCP_A3", "LIQUID"), l0, slopes, strict=True
    ):
        line = next(
            line for line in doc.tdb.splitlines() if f"L({lattice},NI,RE;0)" in line
        )
        a, b = re.fullmatch(
            r"([-+]?[\d.]+)(?:([-+][\d.]+)\*T)?", line.split()[3]
        ).groups()
        assert float(a) == pytest.approx(value, abs=1)
        assert float(b or 0) == pytest.approx(slope, abs=1e-3)
