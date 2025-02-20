"""Run NEB with ML forcefields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jobflow import job

from atomate2.ase.neb import AseNebMaker
from emmet.core.neb import NebResult
from atomate2.forcefields import MLFF, _get_formatted_ff_name
from atomate2.forcefields.jobs import (
    _DEFAULT_CALCULATOR_KWARGS,
    _FORCEFIELD_DATA_OBJECTS,
)
from atomate2.forcefields.utils import ase_calculator, revert_default_dtype

if TYPE_CHECKING:
    from pathlib import Path

    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


@dataclass
class ForceFieldNebMaker(AseNebMaker):
    """Run NEB with an ML forcefield using ASE."""

    name: str = "Forcefield NEB"
    force_field_name: str | MLFF = MLFF.Forcefield

    def __post_init__(self) -> None:
        """Ensure that force_field_name is correctly assigned."""
        super().__post_init__()
        self.force_field_name = _get_formatted_ff_name(self.force_field_name)

        # Pad calculator_kwargs with default values, but permit user to override them
        self.calculator_kwargs = {
            **_DEFAULT_CALCULATOR_KWARGS.get(
                MLFF(self.force_field_name.split("MLFF.")[-1]), {}
            ),
            **self.calculator_kwargs,
        }

        if not self.neb_doc_kwargs.get("force_field_name"):
            self.neb_doc_kwargs["force_field_name"] = str(self.force_field_name)

    @job(data=_FORCEFIELD_DATA_OBJECTS, schema=NebResult)
    def make(
        self, images: list[Structure], prev_dir: str | Path | None = None
    ) -> NebResult:
        """
        Perform NEB with MLFFs on a set of images.

        Parameters
        ----------
        images: list of pymatgen .Structure
            Structures to perform NEB on.
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
            added to match the method signature of other makers.
        """
        with revert_default_dtype():
            return self.run_ase(images=images, prev_dir=prev_dir)

    @property
    def calculator(self) -> Calculator:
        """ASE calculator, can be overwritten by user."""
        return ase_calculator(
            str(self.force_field_name),  # make mypy happy
            **self.calculator_kwargs,
        )
