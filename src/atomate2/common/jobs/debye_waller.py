"""Jobs for Debye-Waller factors from phonon thermal displacements."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jobflow import job

from atomate2 import SETTINGS
from atomate2.common.schemas.debye_waller import DebyeWallerDocument

if TYPE_CHECKING:
    from collections.abc import Sequence

    from atomate2.common.jobs.gruneisen import PhononDoc


@job(
    output_schema=DebyeWallerDocument,
    data=["xrd_patterns", "nd_patterns", "tem_patterns"],
)
def compute_debye_waller(
    phonon_output: PhononDoc,
    temperatures: Sequence[float] = tuple(range(0, 1001, 100)),
    mesh: tuple[int, int, int] | float = 7000.0,
    freq_min: float = 0.01,
    include_imaginary_modes: bool = False,
    xrd_kwargs: dict | None = None,
    nd_kwargs: dict | None = None,
    tem_kwargs: dict | None = None,
    symprec: float = SETTINGS.PHONON_SYMPREC,
) -> DebyeWallerDocument:
    """
    Compute the thermal displacements and diffraction patterns of a phonon run.

    Parameters
    ----------
    phonon_output: PhononDoc
        Output document of a phonon flow, run with store_force_constants=True.
    temperatures: Sequence[float]
        Temperatures in K, not negative.
    mesh: tuple[int, int, int] | float
        q-point mesh, or a q-point density used as kppa in pymatgen's
        Kpoints.automatic_density for the primitive cell.
    freq_min: float
        Modes with a frequency of magnitude below this value in THz are left
        out. The three acoustic modes at Gamma are always left out.
    include_imaginary_modes: bool
        Also include the modes below -freq_min, as if their frequency were real
        with the same magnitude.
    xrd_kwargs: dict or None
        Keyword arguments of pymatgen's XRDCalculator.
    nd_kwargs: dict or None
        Keyword arguments of pymatgen's NDCalculator.
    tem_kwargs: dict or None
        Keyword arguments of pymatgen's TEMCalculator.
    symprec: float
        Symmetry precision of the phonon flow.

    Returns
    -------
    DebyeWallerDocument
    """
    return DebyeWallerDocument.from_phonon_doc(
        phonon_output,
        temperatures=temperatures,
        mesh=mesh,
        freq_min=freq_min,
        include_imaginary_modes=include_imaginary_modes,
        xrd_kwargs=xrd_kwargs,
        nd_kwargs=nd_kwargs,
        tem_kwargs=tem_kwargs,
        symprec=symprec,
    )
