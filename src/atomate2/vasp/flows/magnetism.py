"""VASP-specific flows for magnetism-related calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from atomate2.common.flows.magnetism import (
    MagneticOrderingsMaker as MagneticOrderingsMakerBase,
)
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.schemas.magnetism import MagneticOrderingsDocument
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core.periodic_table import Element

    from atomate2.vasp.jobs.base import BaseVaspMaker

__all__ = ["MagneticOrderingsMaker"]


@dataclass
class MagneticOrderingsMaker(MagneticOrderingsMakerBase):
    """Maker to calculate possible collinear magnetic orderings for a material in VASP.

    Given an input structure, possible magnetic orderings will be enumerated and ranked
    based on symmetry up to a maximum number of orderings. Each ordering will be relaxed
    and a higher quality static calculation performed to obtain a total energy. The
    lowest energy ordering is the predicted ground-state collinear ordering.

    This approach performed decently using VASP for a wide range of test materials in a
    benchmark. It was originally implemented in atomate (v1) for VASP as the
    MagneticOrderingsWF.

    Please refer to the following paper for more information and cite appropriately:

        Horton, M.K., Montoya, J.H., Liu, M. et al. High-throughput prediction of the
        ground-state collinear magnetic order of inorganic materials using Density
        Functional Theory. npj Computational Materials 5, 64 (2019).
        https://doi.org/10.1038/s41524-019-0199-7

    .. Note::
        Good performance of this workflow is ultimately dependent on an appropriate
        choice of Hubbard U, Hund J values and/or the functional. The defaults will work
        well for many transition metal oxides.

    Parameters
    ----------
    name : str
        Name of the flows produced by this Maker.
    static_maker : BaseVaspMaker
        Maker used to perform static calculations for total energy. Defaults to a
        StaticMaker with stricter settings (EDIFF=1e-7).
    relax_maker : BaseVaspMaker | None
        Maker used to perform relaxations of the enumerated structures. By default, the
        RelaxMaker is provided (preferred). If None, relaxations will be skipped and
        only static calculations will be performed.
    default_magmoms : dict | None
        Optional default mapping of magnetic elements to their initial magnetic moments
        in µB. Generally these are chosen to be high-spin, since they can relax to a
        low-spin configuration during a DFT electronic configuration. If None, will use
        the default values provided in pymatgen/analysis/magnetism/default_magmoms.yaml.
    strategies : tuple[str]
        Different ordering strategies to use. Choose from ferromagnetic,
        antiferromagnetic, antiferromagnetic_by_motif, ferrimagnetic_by_motif,
        ferrimagnetic_by_species, or nonmagnetic. Here, "motif", means to use a
        different ordering parameter for symmetry inequivalent sites.
    automatic : bool
        If True, will automatically choose sensible strategies. Defaults to True.
    truncate_by_symmetry : bool
        If True, will remove very unsymmetrical orderings that are likely physically
        implausible. Defaults to True.
    transformation_kwargs : dict | None
        Keyword arguments provided to MagOrderingTransformation in pymatgen. Defaults to
        None.
    """

    name: str = "magnetic_orderings"
    static_maker: BaseVaspMaker = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(user_incar_settings={"EDIFF": 1e-7})
        )
    )
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
    def prev_calc_dir_argname(self) -> str:
        """Argument name informing static maker of previous calculation directory.

        This only applies if a relax_maker is specified and two calculations
        are performed for each ordering (i.e., relax -> static).

        This is necessary because different DFT codes have different previous directory
        argnames. TODO: remove this patch fix when prev_dir is implemented.
        """
        return "prev_vasp_dir"

    @staticmethod
    def _build_doc_fn(tasks):
        """Wrap the function MagneticOrderingsDocument.from_tasks."""
        return MagneticOrderingsDocument.from_tasks(tasks)
