"""Jobs for Grüneisen-Parameter computations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jobflow import Response, job

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure


@job
def shrink_expand_structure(structure: Structure, perc_vol: float) -> Response:
    """
    Create structures with expanded and reduced volumes.

    structure: optimized pymatgen structure obj
    perc_vol: percentage to shrink and expand the volume
    """
    plus_struct = structure.copy()
    minus_struct = structure.copy()

    plus_struct.scale_lattice(volume=structure.volume * (1 + perc_vol))
    minus_struct.scale_lattice(volume=structure.volume * (1 - perc_vol))

    return Response(output={"plus": plus_struct, "minus": minus_struct})
