"""Flows for calculating magnetic orderings and other magnetism-related tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

from jobflow import Flow, Maker

from atomate2.common.jobs.magnetism import (
    enumerate_magnetic_orderings,
    run_ordering_calculations,
)

if TYPE_CHECKING:
    from pymatgen.core import Element
    from pymatgen.core.structure import Structure

__all__ = ["MagneticOrderingsMaker"]


@dataclass
class MagneticOrderingsMaker(Maker):
    """
    Maker to calculate possible collinear magnetic orderings for a material.

    Given an input structure, possible magnetic orderings will be enumerated and ranked based
    on symmetry up to a maximum number of orderings. Each ordering will be
    relaxed and a higher quality static calculation performed to obtain a total energy.
    The lowest energy ordering is the predicted ground-state collinear ordering. Note:
    to analyze the results of this workflow, use the corresponding builder for
    your DFT code (e.g., atomate2.vasp.builders.magnetism.MagneticOrderingsBuilder).

    This workflow showed decent performance using VASP for a wide range of test materials in a
    benchmark. It was originally implemented in atomate (v1) for VASP as the MagneticOrderingsWF.
    Please refer to the following paper for more information and cite appropriately:

        Horton, M.K., Montoya, J.H., Liu, M. et al. High-throughput prediction of the
        ground-state collinear magnetic order of inorganic materials using Density
        Functional Theory. npj Comput Mater 5, 64 (2019).
        https://doi.org/10.1038/s41524-019-0199-7

    .. Note::
        Good performance of this workflow is ultimately dependent on an appropriate
        choice of Hubbard U, Hund J values and/or the functional. The defaults will work
        well for many transition metal oxides.

    Parameters
    ----------
    name : str
        Name of the flows produced by this Maker.
    static_maker : Maker
        Maker used to peform static calculations for total energy (e.g.,
        atomate2.vasp.jobs.StaticMaker).
    relax_maker : Maker | None
        Maker used to perform relaxations of the enumerated structures (e.g.,
        atomate2.vasp.jobs.RelaxMaker). If None, relaxations will be skipped (i.e., only
        static calculations).
    default_magmoms : dict | None
        Optional default mapping of magnetic elements to their initial magnetic moments
        in µB. Generally these are chosen to be high-spin, since they can relax to a
        low-spin configuration during a DFT electronic configuration. If None, will use
        the default values provided in pymatgen/analysis/magnetism/default_magmoms.yaml.
    strategies : tuple
        Different ordering strategies to use. Choose from ferromagnetic,
        antiferromagnetic, antiferromagnetic_by_motif, ferrimagnetic_by_motif,
        ferrimagnetic_by_species, or nonmagnetic. Here, "motif", means to use a
        different ordering parameter for symmetry inequivalent sites.
    automatic : bool
        If True, will automatically choose sensible strategies.
    truncate_by_symmetry : bool
        If True, will remove very unsymmetrical orderings that are likely physically
        implausible.
    transformation_kwargs : dict | None
        Keyword arguments provided to MagOrderingTransformation in pymatgen.
    """

    static_maker: Maker
    relax_maker: Maker | None = None
    default_magmoms: dict[Element, float] | None = None
    strategies: Sequence[
        Literal[
            "ferromagnetic",
            "antiferromagnetic",
            "antiferromagnetic_by_motif",
            "ferrimagnetic_by_motif",
            "ferrimagnetic_by_species",
            "nonmagnetic",
        ]
    ] = ("ferromagnetic", "antiferromagnetic")
    automatic: bool = True
    truncate_by_symmetry: bool = True
    transformation_kwargs: dict | None = None
    name: str = "magnetic_orderings"

    def __post_init__(self):
        if self.relax_maker is not None:
            static_base_maker_name = self.static_maker.__class__.__mro__[1].__name__
            relax_base_maker_name = self.relax_maker.__class__.__mro__[1].__name__
            assert relax_base_maker_name == static_base_maker_name, (
                "relax and static makers must come from the same base maker (e.g.,"
                " BaseVaspMaker)!"
            )

    @property
    def prev_calc_dir_argname(self):
        """
        Name of the argument that informs the static maker of the previous calculation
        directory. As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        raise NotImplementedError

    def make(
        self,
        structure: Structure,
    ):
        """
        Make a flow to calculate possible ground-state collinear magnetic orderings for
        a given input structure.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.

        Returns
        -------
        flow: Flow
            The magnetic ordering worfklow.
        """
        jobs = []

        orderings = enumerate_magnetic_orderings(
            structure,
            default_magmoms=self.default_magmoms,
            strategies=self.strategies,
            automatic=self.automatic,
            truncate_by_symmetry=self.truncate_by_symmetry,
            transformation_kwargs=self.transformation_kwargs,
        )
        jobs.append(orderings)

        calculations = run_ordering_calculations(
            orderings.output,
            static_maker=self.static_maker,
            relax_maker=self.relax_maker,
            prev_calc_dir_argname=self.prev_calc_dir_argname,
        )

        jobs.append(calculations)

        return Flow(
            jobs=jobs,
            output=calculations.output,
            name=f"{self.name} ({structure.composition.reduced_formula})",
        )
