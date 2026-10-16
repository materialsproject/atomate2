"""Jobs used in the dielectric workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jobflow import job

from atomate2.common.schemas.dielectric import DielectricDocument

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure


@job(output_schema=DielectricDocument)
def dielectric_analysis(structure: Structure, **kwargs) -> DielectricDocument:
    """
    Collect the results of a dielectric calculation.

    Parameters
    ----------
    structure : .Structure
        The structure of the dielectric calculation.
    **kwargs
        The tensors, passed to :obj:`.DielectricDocument.from_tensors`.

    Returns
    -------
    DielectricDocument
        Document containing the dielectric analysis results.
    """
    return DielectricDocument.from_tensors(structure, **kwargs)
