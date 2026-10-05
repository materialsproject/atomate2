"""Tests for the sqs2tdb fit of the CALPHAD workflow."""

import shutil

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from scipy.constants import electron_volt, physical_constants

from atomate2.common.jobs.calphad import (
    _get_relaxation_strain,
    fit_tdb,
    get_sqs_structures,
)

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

    end_members = {"Cu": -3.7, "Ni": -5.0}
    calculations = []
    for calc in sqs:
        composition = calc["structure"].composition
        energy = sum(end_members[str(el)] * n for el, n in composition.items())
        if len(composition) == 2:
            energy -= 0.02 * composition.num_atoms
        calculations.append({**calc, "energy": energy})

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
