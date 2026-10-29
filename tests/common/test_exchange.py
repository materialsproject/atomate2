"""Tests for the Heisenberg exchange workflow.

The orderings are toy MnO cells (CsCl type, Mn on a simple cubic lattice) whose
energies come from known exchange constants, so every fit must recover them.
"""

from __future__ import annotations

import pytest
from jobflow import run_locally
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper
from pymatgen.core import Element, Lattice, Structure

from atomate2.common.flows.exchange import ExchangeMaker
from atomate2.common.jobs.exchange import heisenberg_mapping
from atomate2.common.schemas.exchange import ExchangeDocument
from atomate2.common.schemas.magnetism import (
    MagneticOrderingOutput,
    MagneticOrderingRelaxation,
    MagneticOrderingsDocument,
)

LATTICE_CONSTANT = 4.0  # Angstrom
MOMENT = 3.0  # muB on each Mn
CUTOFF = 6.0  # Angstrom: couples the 4 and 5.66 Angstrom Mn-Mn bonds, not 6.93
E0 = -5.0  # eV per Mn
J1 = -2.0  # meV/muB^2, nearest Mn-Mn bond
J2 = 0.3  # meV/muB^2, second-nearest Mn-Mn bond

# Per ordering: the supercell, the spin sign of the Mn at lattice vector n, and
# the spin products -m^2/2 * (n_parallel - n_antiparallel) of J1 (6 bonds) and
# J2 (12 bonds).
ORDERINGS = {
    "FM": ((1, 1, 1), lambda _n: 1, (-27.0, -54.0)),
    "A-AFM": ((2, 1, 1), lambda n: (-1) ** n[0], (-9.0, 18.0)),
    "C-AFM": ((2, 2, 1), lambda n: (-1) ** (n[0] + n[1]), (9.0, 18.0)),
    "G-AFM": ((2, 2, 2), lambda n: (-1) ** sum(n), (27.0, -54.0)),
}


@pytest.fixture
def parent() -> Structure:
    return Structure(
        Lattice.cubic(LATTICE_CONSTANT), ["Mn", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


@pytest.fixture
def orderings(parent) -> list[tuple[Structure, float]]:
    """Each ordering with its energy per atom."""
    orderings = []
    for scaling, spin_sign, (products_j1, products_j2) in ORDERINGS.values():
        structure = parent * scaling
        magmoms = []
        for site in structure:
            lattice_vector = [round(x / LATTICE_CONSTANT) for x in site.coords]
            is_mn = site.specie == Element("Mn")
            magmoms.append(MOMENT * spin_sign(lattice_vector) if is_mn else 0.0)
        structure.add_site_property("magmom", magmoms)

        energy_per_mn = E0 + (J1 * products_j1 + J2 * products_j2) / 1000
        n_mn = structure.composition["Mn"]
        orderings.append((structure, energy_per_mn * n_mn / len(structure)))
    return orderings


def assert_recovers_exchange(ex_params: dict) -> None:
    assert ex_params == pytest.approx(
        {"E0": E0, "J1(Mn1-Mn1)": J1, "J2(Mn1-Mn1)": J2}, abs=1e-6
    )


def test_heisenberg_mapping_recovers_exchange(orderings, parent):
    structures, energies = zip(*orderings, strict=True)
    job = heisenberg_mapping(structures, energies, parent, {"cutoff": CUTOFF})

    response = run_locally(job, create_folders=False, ensure_success=True)
    model = response[job.uuid][1].output

    assert_recovers_exchange(model.ex_params)
    assert model.residual == pytest.approx(0, abs=1e-6)


def test_make_fits_orderings_in_any_order(orderings, parent):
    structures, energies = zip(*reversed(orderings), strict=True)
    maker = ExchangeMaker(heisenberg_settings={"cutoff": CUTOFF}, run_vampire=False)
    flow = maker.make(structures, energies, parent)

    response = run_locally(flow, create_folders=False, ensure_success=True)
    doc = response[flow.output.uuid][1].output

    assert isinstance(doc, ExchangeDocument)
    assert_recovers_exchange(doc.ex_params)
    assert doc.heisenberg_settings == {"cutoff": CUTOFF, "symprec": 0.01}
    assert doc.parent_structure == parent
    assert doc.formula_pretty == "MnO"


def test_make_rejects_structure_without_magmoms(orderings, parent):
    structures, energies = zip(*orderings, strict=True)
    structures[0].remove_site_property("magmom")

    with pytest.raises(ValueError, match="'magmom' site property"):
        ExchangeMaker(run_vampire=False).make(structures, energies, parent)


def test_make_rejects_unequal_lengths(orderings, parent):
    structures, energies = zip(*orderings, strict=True)

    with pytest.raises(ValueError, match="must be equal"):
        ExchangeMaker(run_vampire=False).make(structures, energies[:-1], parent)


def _ordering_doc(orderings, parent, relaxed: bool = False):
    """A MagneticOrderingsDocument that stores the moments apart from the structure."""
    outputs = []
    for structure, energy_per_atom in orderings:
        bare_structure = structure.copy()
        bare_structure.remove_site_property("magmom")
        outputs.append(
            MagneticOrderingOutput(
                structure=bare_structure,
                magmoms=structure.site_properties["magmom"],
                energy_per_atom=energy_per_atom,
                relax_output=MagneticOrderingRelaxation() if relaxed else None,
            )
        )
    return MagneticOrderingsDocument(outputs=outputs, parent_structure=parent)


def test_make_from_ordering_doc_forwards_default_magmoms(orderings, parent):
    maker = ExchangeMaker(
        heisenberg_settings={"cutoff": CUTOFF},
        default_magmoms={"Mn": 5.0},
        run_vampire=False,
    )
    flow = maker.make_from_ordering_doc(_ordering_doc(orderings, parent))

    mapping_settings = flow.jobs[0].function_args[3]
    assert mapping_settings == {"cutoff": CUTOFF, "default_magmoms": {"Mn": 5.0}}
    assert maker.heisenberg_settings == {"cutoff": CUTOFF}

    response = run_locally(flow, create_folders=False, ensure_success=True)
    assert_recovers_exchange(response[flow.output.uuid][1].output.ex_params)


def test_make_from_ordering_doc_warns_on_relaxed_orderings(orderings, parent):
    maker = ExchangeMaker(run_vampire=False)

    with pytest.warns(UserWarning, match="relax_maker=None"):
        maker.make_from_ordering_doc(_ordering_doc(orderings, parent, relaxed=True))


def test_from_model_falls_back_to_lowest_energy_ordering(orderings):
    structures, energies = zip(*orderings, strict=True)
    total_energies = [e * len(s) for s, e in zip(structures, energies, strict=True)]
    with pytest.warns(UserWarning, match="No `parent` cell supplied"):
        mapper = HeisenbergMapper(structures, total_energies, cutoff=CUTOFF)
    model = mapper.get_heisenberg_model()

    doc = ExchangeDocument.from_model(model)

    assert doc.parent_structure == model.structures[0]
    assert doc.heisenberg_settings == {"cutoff": CUTOFF, "symprec": 0.01}
