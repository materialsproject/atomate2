"""Tests for the vendored VampireCaller's input-file writers.

These guard the ``HeisenbergModel`` contract the caller relies on: ``igraph``,
``magnetic_structures[0]`` and ``sublattice_ids[0]`` all share one site indexing,
the magnetic-only cell of the ground-state ordering. ``structures[0]`` is a
different, larger cell (it keeps the non-magnetic ions), so reading the cell from
there silently writes non-magnetic ions into the ``.ucf`` as atoms and
de-synchronises the interaction block's node indices.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.magnetism.heisenberg import HeisenbergModel
from pymatgen.core import Lattice, Structure

from atomate2.vampire import vampire_caller
from atomate2.vampire.vampire_caller import VampireCaller

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def magnetic_structure() -> Structure:
    """Two antiferromagnetically aligned Mn ions on one sublattice."""
    return Structure(
        Lattice.cubic(4.0),
        ["Mn", "Mn"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        site_properties={"magmom": [3.0, -3.0]},
    )


@pytest.fixture
def full_structure() -> Structure:
    """The same ordering with its non-magnetic Al ions retained."""
    return Structure(
        Lattice.cubic(4.0),
        ["Mn", "Mn", "Al", "Al"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0]],
        site_properties={"magmom": [3.0, -3.0, 0.0, 0.0]},
    )


@pytest.fixture
def heisenberg_model(magnetic_structure, full_structure) -> HeisenbergModel:
    """A minimal model carrying only what VampireCaller reads off it."""
    igraph = StructureGraph.from_empty_graph(
        magnetic_structure,
        edge_weight_name="exchange_constant",
        edge_weight_units="meV",
    )
    igraph.add_edge(0, 1, to_jimage=(0, 0, 0), weight=5.0)

    return HeisenbergModel(
        formula="Mn3Al",
        structures=[full_structure],
        magnetic_structures=[magnetic_structure],
        sublattice_ids=[[0, 0]],
        igraph=igraph,
    )


class _FakePopen:
    """Stand in for the vampire-serial subprocess."""

    returncode = 0

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        pass

    def communicate(self):
        return b"", b""


def _run_caller(hm, monkeypatch, tmp_path: Path) -> VampireCaller:
    """Run VampireCaller.__init__ without the vampire-serial binary.

    ``__init__`` is wrapped in ``monty.dev.requires``, which raises unless
    vampire-serial is on PATH; ``__wrapped__`` (set by functools.wraps) reaches
    the real initialiser so the writers stay testable without the binary.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        vampire_caller.VampireCaller,
        "parse_stdout",
        staticmethod(lambda *_: ("{}", 0.0)),
    )

    caller = object.__new__(VampireCaller)
    VampireCaller.__init__.__wrapped__(caller, hm=hm)
    return caller


def _parse_ucf(text: str) -> tuple[list[str], int, list[str]]:
    """Return (atom lines, declared material count, interaction lines)."""
    lines = text.splitlines()
    atoms_at = lines.index("# Atoms num_materials; id cx cy cz mat cat hcat")
    n_atoms, n_mats = (int(value) for value in lines[atoms_at + 1].split())
    inter_at = lines.index("# Interactions")
    n_inter = int(lines[inter_at + 1].split()[0])

    atom_lines = lines[atoms_at + 2 : atoms_at + 2 + n_atoms]
    inter_lines = lines[inter_at + 2 : inter_at + 2 + n_inter]
    assert len(atom_lines) == n_atoms, "declared atom count != atom block length"
    assert len(inter_lines) == n_inter, "declared interaction count != block length"
    return atom_lines, n_mats, inter_lines


def test_ucf_holds_only_magnetic_ions(heisenberg_model, monkeypatch, tmp_path):
    """The .ucf describes the magnetic-only cell, not the full ordering cell."""
    caller = _run_caller(heisenberg_model, monkeypatch, tmp_path)
    atom_lines, _, _ = _parse_ucf((tmp_path / "Mn3Al.ucf").read_text())

    assert len(atom_lines) == len(heisenberg_model.magnetic_structures[0]) == 2
    # The regression this guards: reading the cell from structures[0] would put
    # the two non-magnetic Al ions in the .ucf as atoms.
    assert len(atom_lines) != len(heisenberg_model.structures[0])
    assert caller.structure == heisenberg_model.magnetic_structures[0]


def test_ucf_indices_stay_in_range(heisenberg_model, monkeypatch, tmp_path):
    """Material ids and interaction node indices address existing atoms/materials."""
    _run_caller(heisenberg_model, monkeypatch, tmp_path)
    atom_lines, n_mats, inter_lines = _parse_ucf((tmp_path / "Mn3Al.ucf").read_text())

    mat_ids = [int(line.split()[4]) for line in atom_lines]
    assert set(mat_ids) == set(range(n_mats)), "material ids not a dense 0-based range"

    for line in inter_lines:
        _, i, j = line.split()[:3]
        assert 0 <= int(i) < len(atom_lines)
        assert 0 <= int(j) < len(atom_lines)


def test_mat_file_matches_ucf_materials(heisenberg_model, monkeypatch, tmp_path):
    """Both spin directions of the single sublattice become their own material."""
    _run_caller(heisenberg_model, monkeypatch, tmp_path)
    mat_text = (tmp_path / "Mn3Al.mat").read_text()
    _, n_mats, _ = _parse_ucf((tmp_path / "Mn3Al.ucf").read_text())

    assert int(mat_text.splitlines()[0].split("=")[1]) == n_mats == 2
    # Each material must be pinned to its own unit-cell-category, or Vampire
    # collapses every atom into material 1 and the sublattices vanish.
    categories = {ln for ln in mat_text.splitlines() if "unit-cell-category" in ln}
    assert len(categories) == n_mats


def test_misaligned_model_is_rejected(heisenberg_model, monkeypatch, tmp_path):
    """A model whose igraph does not match its magnetic cell fails loudly."""
    # Exactly the pre-migration bug: the full cell paired with a graph built on
    # the magnetic-only cell.
    heisenberg_model.magnetic_structures = heisenberg_model.structures

    with pytest.raises(ValueError, match="misaligned"):
        _run_caller(heisenberg_model, monkeypatch, tmp_path)
