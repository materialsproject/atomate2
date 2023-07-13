"""VASP-specific flows for calculating magnetic orderings and other magnetism-related tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

from atomate2.common.flows import magnetism as magnetism_flows
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker

if TYPE_CHECKING:
    from pymatgen.core import Element

    from atomate2.vasp.jobs.base import BaseVaspMaker

__all__ = ["MagneticOrderingsMaker"]


@dataclass
class MagneticOrderingsMaker(magnetism_flows.MagneticOrderingsMaker):
    """
    Maker to calculate possible collinear magnetic orderings for a material in VASP.

    Given an input structure, possible magnetic orderings will be enumerated and ranked based
    on symmetry up to a maximum number of orderings. Each ordering will be
    relaxed and a higher quality static calculation performed to obtain a total energy.
    The lowest energy ordering is the predicted ground-state collinear ordering.

    This approach performed decently using VASP for a wide range of test materials in a
    benchmark. It was originally implemented in atomate (v1) for VASP as the
    MagneticOrderingsWF.

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
    static_maker : StaticMaker
        Maker used to peform static calculations for total energy.
    relax_maker : RelaxMaker | None
        Maker used to perform relaxations of the enumerated structures. If None,
        relaxations will be skipped (i.e., only static calculations).
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

    name: str = "magnetic_orderings"
    static_maker: BaseVaspMaker = field(default_factory=StaticMaker)
    relax_maker: BaseVaspMaker | None = field(default_factory=RelaxMaker)
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

    @property
    def prev_calc_dir_argname(self):
        """
        Name of the argument that informs the static maker of the previous calculation
        directory. This only applies if a relax_maker is specified and two calculations are
        performed for each ordering (i.e., relax -> static).
        """
        return "prev_vasp_dir"
