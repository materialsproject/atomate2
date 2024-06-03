"""Flows for calculating elastic constants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker, OnMissing

from atomate2 import SETTINGS
from atomate2.common.jobs.elastic import (
    fit_elastic_tensor,
    generate_elastic_deformations,
    run_elastic_deformations,
)
from atomate2.common.jobs.utils import structure_to_conventional

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.forcefields.jobs import ForceFieldRelaxMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class BaseElasticMaker(Maker, ABC):
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
    bulk_relax_maker : .BaseVaspMaker or .ForceFieldRelaxMaker or None
        A maker to perform a tight relaxation on the bulk. Set to ``None`` to skip the
        bulk relaxation.
    elastic_relax_maker : .BaseVaspMaker or .ForceFieldRelaxMaker
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
    bulk_relax_maker: BaseAimsMaker | BaseVaspMaker | ForceFieldRelaxMaker | None = None
    elastic_relax_maker: BaseAimsMaker | BaseVaspMaker | ForceFieldRelaxMaker = (
        None  # constant volume optimization
    )
    max_failed_deformations: int | float | None = None
    generate_elastic_deformations_kwargs: dict = field(default_factory=dict)
    fit_elastic_tensor_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        equilibrium_stress: Matrix3D = None,
        conventional: bool = False,
    ) -> Flow:
        """
        Make flow to calculate the elastic constant.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure.
        prev_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        equilibrium_stress : tuple of tuple of float
            The equilibrium stress of the (relaxed) structure, if known.
        conventional : bool
            Whether to transform the structure into the conventional cell.
        """
        jobs = []

        if self.bulk_relax_maker is not None:
            # optionally relax the structure
            bulk_kwargs = {}
            if self.prev_calc_dir_argname is not None:
                bulk_kwargs[self.prev_calc_dir_argname] = prev_dir
            bulk = self.bulk_relax_maker.make(structure, **bulk_kwargs)
            jobs.append(bulk)
            structure = bulk.output.structure
            prev_dir = bulk.output.dir_name
            if equilibrium_stress is None:
                equilibrium_stress = bulk.output.output.stress

        if conventional:
            stc = structure_to_conventional(structure, self.symprec)
            jobs.append(stc)
            structure = stc.output

        deformations = generate_elastic_deformations(
            structure,
            order=self.order,
            sym_reduce=self.sym_reduce,
            symprec=self.symprec,
            **self.generate_elastic_deformations_kwargs,
        )

        vasp_deformation_calcs = run_elastic_deformations(
            structure,
            deformations.output,
            elastic_relax_maker=self.elastic_relax_maker,
            prev_dir=prev_dir,
        )
        fit_tensor = fit_elastic_tensor(
            structure,
            vasp_deformation_calcs.output,
            equilibrium_stress=equilibrium_stress,
            order=self.order,
            symprec=self.symprec if self.sym_reduce else None,
            stress_sign_factor=self.stress_sign_correction,
            max_failed_deformations=self.max_failed_deformations,
            **self.fit_elastic_tensor_kwargs,
            **self.task_document_kwargs,
        )

        # allow some of the deformations to fail
        fit_tensor.config.on_missing_references = OnMissing.NONE

        jobs += [deformations, vasp_deformation_calcs, fit_tensor]

        return Flow(
            jobs=jobs,
            output=fit_tensor.output,
            name=self.name,
        )

    @property
    def stress_sign_correction(self) -> float:
        r"""Correct the sign of the stress tensor.

        This is done because VASP defines the stress tensor to be
            \sigma_ij = -\partial E / \partial n_ij
        and FHI-aims defines it to be
            \sigma_ij = \partial E / \partial n_ij
        """
        return 1.0

    @property
    @abstractmethod
    def prev_calc_dir_argname(self) -> str:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.
        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
