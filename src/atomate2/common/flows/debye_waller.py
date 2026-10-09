"""Flow for Debye-Waller factors from phonon thermal displacements."""

from __future__ import annotations

from dataclasses import dataclass
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
@dataclass
class BaseDebyeWallerMaker(Maker):
    """
    Maker to calculate Debye-Waller factors from phonon thermal displacements.

    The phonon flow gives the Cartesian thermal displacement matrix U of each
    site of the primitive cell at each temperature, with
    create_thermal_displacements=True. Its q-point mesh and temperatures are set
    with kpoint_density_thermal_displacements and tmin_thermal_displacements,
    tmax_thermal_displacements and tstep_thermal_displacements in the
    generate_frequencies_eigenvectors_kwargs of the phonon maker. The
    Debye-Waller factor of a site for the reciprocal lattice vector g is
    exp(-2 pi^2 g^T U g). The X-ray, neutron and electron diffraction patterns
    are computed with pymatgen with these factors at each temperature, and once
    without them.

    The frequencies are not renormalized with temperature. U is too small on a
    finite q-point mesh, by an error that falls as 1/N for an N x N x N mesh.
    The phonon flow stores no U if its mesh has imaginary modes, and this
    workflow then fails. With exclude_imaginary_modes_thermal_displacements=True
    in the generate_frequencies_eigenvectors_kwargs of the phonon maker, U is
    computed from the real modes only. It is then only an estimate.

    This workflow is new and has not been tested widely. It might still change
    in future versions.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    phonon_maker: .BasePhononMaker
        The phonon maker. It must have create_thermal_displacements=True.
    xrd_kwargs: dict or None
        Keyword arguments of pymatgen's XRDCalculator.
    nd_kwargs: dict or None
        Keyword arguments of pymatgen's NDCalculator.
    tem_kwargs: dict or None
        Keyword arguments of pymatgen's TEMCalculator.
    """

    name: str = "debye waller"
    phonon_maker: BasePhononMaker = None
    xrd_kwargs: dict | None = None
    nd_kwargs: dict | None = None
    tem_kwargs: dict | None = None

    def __post_init__(self) -> None:
        """Check the phonon maker and pymatgen before any calculation runs."""
        if not self.phonon_maker.create_thermal_displacements:
            raise ValueError(
                "The phonon maker needs create_thermal_displacements=True, since the "
                "Debye-Waller factors are computed from its thermal displacements."
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
            xrd_kwargs=self.xrd_kwargs,
            nd_kwargs=self.nd_kwargs,
            tem_kwargs=self.tem_kwargs,
            symprec=self.phonon_maker.symprec,
        )
        return Flow(
            [phonon_flow, debye_waller], output=debye_waller.output, name=self.name
        )
