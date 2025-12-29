from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from atomate2.ase.schemas import AseResult
from atomate2.forcefields.schemas import ForceFieldTaskDocument
from atomate2.forcefields.utils import MLFF

if TYPE_CHECKING:
    from pymatgen.core import Structure


@pytest.mark.parametrize(
    "ase_calculator_name,calculator_meta,warning",
    [
        ("MLFF.CHGNet", None, False),
        ("MLFF.CHGNet", MLFF.CHGNet, False),
        (
            "CHGNetCalculator",
            {"@module": "chgnet.model.dynamics", "@callable": "CHGNetCalculator"},
            False,
        ),
        (
            "CHGNetCalculator",
            None,
            True,
        ),  # Should warn as we cannot get package version
    ],
)
def test_forcefield_task_doc_calculator_meta(
    recwarn,
    ase_calculator_name: str,
    calculator_meta: MLFF | dict | None,
    warning: bool,
    si_structure: Structure,
):
    doc: ForceFieldTaskDocument = ForceFieldTaskDocument.from_ase_compatible_result(
        ase_calculator_name=ase_calculator_name,
        result=AseResult(final_mol_or_struct=si_structure, final_energy=0.0),
        steps=2,
        calculator_meta=calculator_meta,
    )
    assert doc.forcefield_name == ase_calculator_name

    capture_warning = any(
        "Could not determine forcefield version as calculator_meta was not provided."
        in str(w.message)
        for w in recwarn
    )
    assert warning == capture_warning
