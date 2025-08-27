"""Run NEB with ML forcefields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from emmet.core.neb import NebResult
from jobflow import job

from atomate2.ase.neb import AseNebFromEndpointsMaker, AseNebFromImagesMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from atomate2.forcefields.utils import _FORCEFIELD_DATA_OBJECTS, MLFF, ForceFieldMixin

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure
    from typing_extensions import Self


@dataclass
class ForceFieldNebFromImagesMaker(ForceFieldMixin, AseNebFromImagesMaker):
    """Run NEB with an ML forcefield using ASE."""

    name: str = "Forcefield NEB from images"
    force_field_name: str | MLFF = MLFF.Forcefield

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
        return self._run_ase_safe(images=images, prev_dir=prev_dir)


@dataclass
class ForceFieldNebFromEndpointsMaker(ForceFieldMixin, AseNebFromEndpointsMaker):
    """Run NEB with an ML forcefield using ASE."""

    name: str = "Forcefield NEB from endpoints"
    force_field_name: str | MLFF = MLFF.Forcefield

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
        return self._run_ase_safe(images=images, prev_dir=prev_dir)

    @classmethod
    def from_force_field_name(cls, force_field_name: str | MLFF, **kwargs) -> Self:
        """Create a force field NEB job from its name.

        Parameters
        ----------
        force_field_name : str or MLFF
            The name of the forcefield. Should be a valid MLFF member.
        **kwargs
            kwargs to pass to ForceFieldNebFromEndpointsMaker.
        """
        endpoint_relax_maker = ForceFieldRelaxMaker(force_field_name=force_field_name)
        return cls(
            name=f"{force_field_name} NEB from endpoints maker",
            endpoint_relax_maker=endpoint_relax_maker,
            force_field_name=force_field_name,
            **kwargs,
        )
