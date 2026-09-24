"""Define the VASP thermal expansion maker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.cte import BaseCTEMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.flows.elastic import ElasticMaker
from atomate2.vasp.flows.pheasy import PhononMaker
from atomate2.vasp.jobs.core import TightRelaxMaker

if TYPE_CHECKING:
    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class CTEMaker(BaseCTEMaker):
    """
    Maker to calculate the thermal expansion with VASP, pheasy and phono3py.

    A tight double relaxation is performed first. The relaxed structure is then
    passed to the pheasy phonon flow, which fits the second- and third-order
    force constants, and to the elastic flow. Neither flow relaxes the
    structure again. Finally, phono3py gives the mode Grueneisen tensors, and
    the thermal expansion tensor follows from them and the elastic tensor. The
    frequencies are not renormalized with temperature.

    By default, the phonon flow fits the force constants with the one-shot
    method from randomly displaced supercells with 0.03 A displacements, and
    builds the supercells with min_length=12.0. The phonon flow skips the
    static energy calculation, which the thermal expansion does not need. The
    elastic flow is the atomate2 elastic flow without its own
    relaxation. The stress is more sensitive to ENCUT than the forces are, so
    check the ENCUT convergence of the elastic tensor for your material.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    bulk_relax_maker: .BaseVaspMaker or None
        A maker to perform a tight relaxation on the bulk. Set to None to skip
        the relaxation.
    phonon_maker: .PhononMaker
        The pheasy phonon maker. It must have cal_anhar_fcs=True,
        use_symmetrized_structure=None and bulk_relax_maker=None. Its
        anhar_fit_methods set which force constants are used for the thermal
        expansion.
    elastic_maker: .ElasticMaker
        Maker for the elastic tensor. It must have bulk_relax_maker=None.
    temperatures: list[float]
        Temperatures in K.
    mesh: tuple[int, int, int] | float
        q-point mesh for the mode Grueneisen tensors, or a q-point density used
        as kppa in pymatgen's Kpoints.automatic_density for the unit cell.
    tol_imaginary_modes: float
        If a frequency on the mesh is below -tol_imaginary_modes in THz, a
        warning is raised and the thermal expansion of that fit is not computed.
    min_frequency: float
        Modes below this frequency in THz are left out of the thermal expansion.
    """

    name: str = "cte"
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    phonon_maker: PhononMaker = field(
        default_factory=lambda: PhononMaker(
            min_length=12.0,
            bulk_relax_maker=None,
            static_energy_maker=None,
            cal_anhar_fcs=True,
            displacement_anhar=0.03,
            anhar_fit_methods=("one-shot",),
        )
    )
    elastic_maker: ElasticMaker = field(
        default_factory=lambda: ElasticMaker(bulk_relax_maker=None)
    )

    @property
    def prev_calc_dir_argname(self) -> str:
        """Name of the argument that passes the previous calculation directory."""
        return "prev_dir"
