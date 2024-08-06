"""Flows for calculating magnetic orderings and other magnetism-related tasks."""

from __future__ import annotations

import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from jobflow import Flow, Maker
from pymatgen.core import Element

from atomate2.common.jobs.magnetism import (
    enumerate_magnetic_orderings,
    postprocess_orderings,
    run_ordering_calculations,
)
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core.structure import Structure


__all__ = ["MagneticOrderingsMaker"]


@dataclass
class MagneticOrderingsMaker(Maker, ABC):
    """Maker to calculate possible collinear magnetic orderings for a material.

    Given an input structure, possible magnetic orderings will be enumerated and ranked
    based on symmetry up to a maximum number of orderings. Each ordering will be
    optionally relaxed and a higher quality static calculation performed to obtain a
    total energy. The lowest energy ordering is the predicted ground-state collinear
    ordering.

    This Maker can be trivially implemented for your DFT code of choice by
    utilizing the appropriate static/relax makers. However, for postprocessing to
    work correctly, one must ensure that the calculation outputs can be processed into a
    MagneticOrderingsDocument. As the TaskDoc class is currently defined only for
    VASP, one should ensure that the task document returned during their DFT runs
    contains the necessary parameters (e.g., TaskDoc.input.magnetic_moments).
    This warning will be removed once a universal TaskDoc is implemented (Issue #741).

    This workflow was benchmarked with VASP for a wide range of test materials and
    originally implemented in atomate (v1) for VASP as the MagneticOrderingsWF. Please
    refer to the following paper for more information and cite appropriately:

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
    static_maker : Maker
        Maker used to perform static calculations for total energy. VASP is selected as
        the default DFT code (atomate2.vasp.jobs.StaticMaker).
    relax_maker : Maker | None
        Maker used to perform relaxations of the enumerated structures. VASP is selected
        as the default DFT code (atomate2.vasp.jobs.RelaxMaker). If this field is None,
        relaxations will be skipped (i.e., only static calculations are performed).
    default_magmoms : dict | None
        Optional default mapping of magnetic elements to their initial magnetic moments
        in μB. Generally these are chosen to be high-spin, since they can relax to a
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
    static_maker: Maker = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(user_incar_settings={"EDIFF": 1e-7})
        )
    )
    relax_maker: Maker | None = field(default_factory=RelaxMaker)
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

    def __post_init__(self) -> None:
        """Ensure that the static and relax makers come from the same base maker.

        This ensures that the same DFT code is used for both calculations.
        """
        if self.relax_maker is None:
            warnings.warn(
                (
                    "No relax_maker provided, relaxations will be skipped. Please be"
                    " sure that this is intended!"
                ),
                stacklevel=2,
            )
        else:
            static_base_maker_name = self.static_maker.__class__.__mro__[1].__name__
            relax_base_maker_name = self.relax_maker.__class__.__mro__[1].__name__
            if relax_base_maker_name != static_base_maker_name:
                warnings.warn(
                    "The provided static and relax makers do not use the "
                    "same DFT code! Please check the base maker used.",
                    stacklevel=2,
                )

    def make(
        self,
        structure: Structure,
    ) -> Flow:
        """Make a flow to calculate collinear magnetic orderings for a given structure.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.

        Returns
        -------
        flow: Flow
            The magnetic ordering workflow.
        """
        jobs = []

        if Element("Co") in structure.elements:
            warnings.warn(
                (
                    "Co detected in structure! Please consider testing both low-spin"
                    " and high-spin configurations. The current default for Co (without"
                    " oxidation state) is high-spin. Refer to the defaults in"
                    " pymatgen/analysis/magnetism/default_magmoms.yaml for more"
                    " information. "
                ),
                stacklevel=2,
            )

        orderings = enumerate_magnetic_orderings(
            structure,
            default_magmoms=self.default_magmoms,
            strategies=self.strategies,
            automatic=self.automatic,
            truncate_by_symmetry=self.truncate_by_symmetry,
            transformation_kwargs=self.transformation_kwargs,
        )

        calculations = run_ordering_calculations(
            orderings.output,  # pylint: disable=no-member
            static_maker=self.static_maker,
            relax_maker=self.relax_maker,
        )

        postprocessing = postprocess_orderings(calculations.output)
        jobs = [orderings, calculations, postprocessing]

        return Flow(
            jobs=jobs,
            output=postprocessing.output,
            name=f"{self.name} ({structure.composition.reduced_formula})",
        )
