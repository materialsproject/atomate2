"""Flows for calculating elastic constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2 import SETTINGS
from atomate2.aims.jobs.core import RelaxMaker
from atomate2.common.flows.elastic import BaseElasticMaker

if TYPE_CHECKING:
    from atomate2.aims.jobs.base import BaseAimsMaker


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
    bulk_relax_maker : .BaseAimsMaker or None
        A maker to perform a tight relaxation on the bulk. Set to ``None`` to skip the
        bulk relaxation.
    elastic_relax_maker : .BaseAimsMaker
        Maker used to generate elastic relaxations.
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
    bulk_relax_maker: BaseAimsMaker | None = field(
        default_factory=RelaxMaker.full_relaxation
    )
    elastic_relax_maker: BaseAimsMaker = field(
        default_factory=RelaxMaker.fixed_cell_relaxation
    )
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

    @property
    def stress_sign_correction(self) -> float:
        r"""Correct the sign of the stress tensor.

        This is done because VASP defines the stress tensor to be
            \sigma_ij = -\partial E / \partial n_ij
        and FHI-aims defines it to be
            \sigma_ij = \partial E / \partial n_ij
        """
        return -1.0
