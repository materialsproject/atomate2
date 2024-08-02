"""Flows for calculating elastic constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2 import SETTINGS
from atomate2.common.flows.elastic import BaseElasticMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import TightRelaxMaker
from atomate2.vasp.jobs.elastic import ElasticRelaxMaker

if TYPE_CHECKING:
    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class ElasticMaker(BaseElasticMaker):
    """
    Maker to calculate elastic constants.

    Calculate the elastic constant of a material. Initially, a tight structural
    relaxation is performed to obtain the structure in a state of approximately zero
    stress. Subsequently, perturbations are applied to the lattice vectors and the
    resulting stress tensor is calculated from DFT, while allowing for relaxation of the
    ionic degrees of freedom. Finally, constitutive relations from linear elasticity,
    relating stress and strain, are employed to fit the full 6x6 elastic tensor. From
    this, aggregate properties such as Voigt and Reuss bounds on the bulk and shear
    moduli are derived.

    .. Note::
        It is heavily recommended to symmetrize the structure before passing it to
        this flow. Otherwise, the symmetry reduction routines will not be as
        effective at reducing the total number of deformations needed.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    order : int
        Order of the tensor expansion to be determined. Can be either 2 or 3.
    sym_reduce : bool
        Whether to reduce the number of deformations using symmetry.
    symprec : float
        Symmetry precision to use in the reduction of symmetry.
    bulk_relax_maker : .BaseVaspMaker or None
        A maker to perform a tight relaxation on the bulk. Set to ``None`` to skip the
        bulk relaxation.
    elastic_relax_maker : .BaseVaspMaker
        Maker used to generate elastic relaxations.
    max_failed_deformations: int or float
        Maximum number of deformations allowed to fail to proceed with the fitting
        of the elastic tensor. If an int the absolute number of deformations. If
        a float between 0 an 1 the maximum fraction of deformations. If None any
        number of deformations allowed.
    generate_elastic_deformations_kwargs : dict
        Keyword arguments passed to :obj:`generate_elastic_deformations`.
    fit_elastic_tensor_kwargs : dict
        Keyword arguments passed to :obj:`fit_elastic_tensor`.
    task_document_kwargs : dict
        Additional keyword args passed to :obj:`.ElasticDocument.from_stresses()`.
    """

    name: str = "elastic"
    order: int = 2
    sym_reduce: bool = True
    symprec: float = SETTINGS.SYMPREC
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    elastic_relax_maker: BaseVaspMaker = field(default_factory=ElasticRelaxMaker)
    max_failed_deformations: int | float | None = None
    generate_elastic_deformations_kwargs: dict = field(default_factory=dict)
    fit_elastic_tensor_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    @property
    def prev_calc_dir_argname(self) -> str:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return "prev_dir"
