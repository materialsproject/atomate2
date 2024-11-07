"""Create NEB jobs with ASE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import job

from atomate2.ase.jobs import _ASE_DATA_OBJECTS, AseMaker
from atomate2.ase.utils import AseNebInterface

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from ase.calculators.calculator import Calculator
    from pymatgen.core import Molecule, Structure

    from atomate2.common.schemas.neb import NebResult


@dataclass
class AseNebMaker(AseMaker):
    """Define scheme for performing ASE NEB calculations."""

    name: str = "ASE NEB maker"
    neb_kwargs: dict = field(default_factory=dict)
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float | None = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    traj_file: str | None = None
    traj_file_fmt: Literal["pmg", "ase", "xdatcar"] = "ase"
    traj_interval: int = 1
    neb_doc_kwargs: dict = field(default_factory=dict)

    def run_ase(
        self,
        images: list[Structure | Molecule],
        prev_dir: str | Path | None = None,
    ) -> NebResult:
        """
        Run an ASE NEB job from a list of images.

        Parameters
        ----------
        images: list of pymatgen .Molecule or .Structure
            pymatgen molecule or structure images
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
                added to match the method signature of other makers.
        """
        return AseNebInterface(
            calculator=self.calculator,
            fix_symmetry=self.fix_symmetry,
            relax_cell=self.relax_cell,
            symprec=self.symprec,
            neb_kwargs=self.neb_kwargs,
            **self.optimizer_kwargs,
        ).run_neb(
            images,
            steps=self.steps,
            traj_file=self.traj_file,
            traj_file_fmt=self.traj_file_fmt,
            interval=self.traj_interval,
            neb_doc_kwargs=self.neb_doc_kwargs,
            **self.relax_kwargs,
        )

    @job(data=_ASE_DATA_OBJECTS, schema=NebResult)
    def make(
        self,
        images: list[Structure | Molecule],
        prev_dir: str | Path | None = None,
    ) -> NebResult:
        """
        Run an ASE NEB job from a list of images.

        Parameters
        ----------
        images: list of pymatgen .Molecule or .Structure
            pymatgen molecule or structure images
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
                added to match the method signature of other makers.
        """
        return self.run_ase(images, prev_dir=prev_dir)


class LennardJonesNebMaker(AseNebMaker):
    """
    Lennard-Jones NEB maker, primarily for testing/debugging.
    """

    name: str = "Lennard-Jones 6-12 NEB"

    @property
    def calculator(self) -> Calculator:
        """Lennard-Jones calculator."""
        from ase.calculators.lj import LennardJones

        return LennardJones(**self.calculator_kwargs)
