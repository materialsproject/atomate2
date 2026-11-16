"""Jobs for Debye-Waller factors from phonon thermal displacements."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jobflow import job
from pymatgen.util.due import Doi, due

from atomate2.common.schemas.debye_waller import DebyeWallerDocument

if TYPE_CHECKING:
    from atomate2.common.jobs.gruneisen import PhononDoc


@due.dcite(
    Doi("10.1107/S0108767396005697"),
    description="Atomic displacement parameter nomenclature.",
)
@due.dcite(
    Doi("10.1039/C5CE01219H"),
    description="Anisotropic displacement parameters from phonon calculations.",
)
@due.dcite(
    Doi("10.1088/1361-648X/acd831"),
    description="Implementation strategies in phonopy and phono3py.",
)
@due.dcite(
    Doi("10.7566/JPSJ.92.012001"),
    description="Phonopy and phono3py.",
)
@job(
    output_schema=DebyeWallerDocument,
    data=["xrd_patterns", "nd_patterns", "tem_pattern_static", "tem_patterns"],
)
def compute_debye_waller(
    phonon_output: PhononDoc,
    xrd_kwargs: dict | None = None,
    nd_kwargs: dict | None = None,
    tem_kwargs: dict | None = None,
) -> DebyeWallerDocument:
    """
    Compute the diffraction patterns of a phonon run with Debye-Waller factors.

    Parameters
    ----------
    phonon_output: PhononDoc
        Output document of a phonon flow, run with create_thermal_displacements=True.
    xrd_kwargs: dict or None
        Keyword arguments of pymatgen's XRDCalculator.
    nd_kwargs: dict or None
        Keyword arguments of pymatgen's NDCalculator.
    tem_kwargs: dict or None
        Keyword arguments of pymatgen's TEMCalculator.

    Returns
    -------
    DebyeWallerDocument
    """
    return DebyeWallerDocument.from_phonon_doc(
        phonon_output,
        xrd_kwargs=xrd_kwargs,
        nd_kwargs=nd_kwargs,
        tem_kwargs=tem_kwargs,
    )
