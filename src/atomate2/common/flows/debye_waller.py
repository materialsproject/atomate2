"""Flow for Debye-Waller factors from phonon thermal displacements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from pymatgen.util.due import Doi, due

from atomate2.common.jobs.debye_waller import compute_debye_waller
from atomate2.common.schemas.debye_waller import check_pymatgen

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

    from atomate2.common.flows.phonons import BasePhononMaker


@due.dcite(
    Doi("10.1088/1361-648X/acd831"),
    description="Implementation strategies in phonopy and phono3py.",
)
@due.dcite(
    Doi("10.7566/JPSJ.92.012001"),
    description="Phonopy and phono3py.",
)
@dataclass
class BaseDebyeWallerMaker(Maker):
    """
    Maker to calculate Debye-Waller factors from phonon thermal displacements.

    The phonon flow gives the harmonic force constants. phonopy then gives the
    Cartesian thermal displacement matrix U of each site of the primitive cell
    on a q-point mesh. The Debye-Waller factor of a site for the reciprocal
    lattice vector g is exp(-2 pi^2 g^T U g). The X-ray, neutron and electron
    diffraction patterns are computed with pymatgen with these factors at each
    temperature, and once without them.

    The frequencies are not renormalized with temperature. On a coarse q-point
    mesh, U is too small. At finite temperature the error falls only as 1/N
    for an N x N x N mesh, so check its convergence. Imaginary modes are left
    out by default. With include_imaginary_modes=True, they are included as if
    their frequency were real with the same magnitude. This only makes sense
    for small imaginary frequencies.

    This workflow is new and has not been tested widely. It might still change
    in future versions.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    phonon_maker: .BasePhononMaker
        The phonon maker. It must have store_force_constants=True.
    temperatures: list[float]
        Temperatures in K, not negative.
    mesh: tuple[int, int, int] | float
        q-point mesh for the thermal displacements, or a q-point density used
        as kppa in pymatgen's Kpoints.automatic_density for the primitive cell.
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
    """

    name: str = "debye waller"
    phonon_maker: BasePhononMaker = None
    temperatures: list[float] = field(default_factory=lambda: list(range(0, 1001, 100)))
    mesh: tuple[int, int, int] | float = 7000.0
    freq_min: float = 0.01
    include_imaginary_modes: bool = False
    xrd_kwargs: dict | None = None
    nd_kwargs: dict | None = None
    tem_kwargs: dict | None = None

    def __post_init__(self) -> None:
        """Check the phonon maker and pymatgen before any calculation runs."""
        if not self.phonon_maker.store_force_constants:
            raise ValueError(
                "The phonon maker needs store_force_constants=True, since the "
                "thermal displacements are computed from the force constants."
            )
        check_pymatgen()

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Make a flow to calculate the Debye-Waller factors.

        Parameters
        ----------
        structure: Structure
            A pymatgen structure. Start with a structure that is nearly fully
            optimized, as the relaxation settings are strict.
        prev_dir: str or Path or None
            A previous calculation directory to use for copying outputs.
        """
        phonon_flow = self.phonon_maker.make(structure, prev_dir=prev_dir)
        debye_waller = compute_debye_waller(
            phonon_output=phonon_flow.output,
            temperatures=self.temperatures,
            mesh=self.mesh,
            freq_min=self.freq_min,
            include_imaginary_modes=self.include_imaginary_modes,
            xrd_kwargs=self.xrd_kwargs,
            nd_kwargs=self.nd_kwargs,
            tem_kwargs=self.tem_kwargs,
            symprec=self.phonon_maker.symprec,
        )
        return Flow(
            [phonon_flow, debye_waller], output=debye_waller.output, name=self.name
        )
