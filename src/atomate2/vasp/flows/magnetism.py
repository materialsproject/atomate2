"""Flows for calculating magnetic orderings and other magnetism-related tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Literal

from jobflow import Flow, Maker, Response, Job
from pymatgen.core import Element
from pymatgen.core.structure import Structure

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import TightRelaxMaker, StaticMaker

__all__ = ["MagneticOrderingsMaker"]

from atomate2.common.jobs.magnetism import generate_magnetic_orderings, run_ordering_calculations, analyze_orderings


@dataclass
class MagneticOrderingsMaker(Maker):
    """
    Maker to calculate possible collinear magnetic orderings for a material.

    Given an input material, possible orderings will be enumerated and ranked based
    on symmetry up to a certain maximum number of orderings. Each ordering will be
    relaxed and a higher quality static calculation performed to obtain a total energy.
    The lowest energy ordering is the predicted stable ordering.

    This approach was found to have decent performance for a wide range of test
    materials and was benchmarked in DOI: 10.1038/s41524-019-0199-7

    This Maker is based on the MagneticOrderingsWF in a previous version of atomate:

    https://github.com/hackingmaterials/atomate/blob/main/atomate/vasp/workflows/base/magnetism.py

    .. Note::
        Good performance of this workflow is ultimately dependent on an appropriate
        choice of Hubbard U, Hund J values and/or functional. The defaults will work
        well for many transition metal oxides.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    default_magmoms : Optional[Dict]
        mapping of magnetic elements to their initial magnetic moments in µB,
        generally these are chosen to be high-spin since they can relax to a
        low-spin configuration during a DFT electronic configuration
    strategies : tuple
        different ordering strategies to use, choose from ferromagnetic, antiferromagnetic,
        antiferromagnetic_by_motif, ferrimagnetic_by_motif and
        ferrimagnetic_by_species (here, "motif", means to use a different ordering
        parameter for symmetry inequivalent sites)
    automatic : bool
        if True, will automatically choose sensible strategies
    truncate_by_symmetry : bool
        if True, will remove very unsymmetrical orderings that are likely physically implausible
    enumerator_kwargs : Optional[Dict]
        Keyword arguments for MagneticStructureEnumerator in pymatgen
    relax_maker : .BaseVaspMaker or None
        A maker to perform the relaxation on the bulk. Set to ``None`` to skip the
        relaxation.
    static_maker : .BaseVaspMaker
        Maker used to peform static calculations for total energy.
    """

    name: str = "magnetic_ordering"
    default_magmoms: Optional[Dict[Element, float]] = None
    strategies: Sequence[
        Literal["ferromagnetic", "antiferromagnetic", "antiferromagnetic_by_motif", ""]
    ] = ("ferromagnetic", "antiferromagnetic")
    automatic: bool = False
    truncate_by_symmetry: bool = True
    enumerator_kwargs: Optional[Dict] = None
    relax_maker: Optional[BaseVaspMaker] = field(
        default_factory=lambda: DoubleRelaxMaker(relax_maker=TightRelaxMaker())
    )
    static_maker: BaseVaspMaker = field(default_factory=lambda: StaticMaker)

    def make(
        self,
        structure: Structure,
        # TODO: likely don't need prev_vasp_dir?
        # prev_vasp_dir: str | Path | None = None,
    ):
        """
        Make flow to calculate a possible ground-state magnetic orderings.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure.
        """
        jobs = []

        orderings: Job = generate_magnetic_orderings(
            structure,
            default_magmoms=self.default_magmoms,
            strategies=self.strategies,
            # TODO: remove this argument and subsume into `strategies`?
            automatic=self.automatic,
            truncate_by_symmetry=self.truncate_by_symmetry,
            enumerator_kwargs=self.enumerator_kwargs
        )
        jobs.append(orderings)

        # TODO: allow some orderings to fail
        if self.relax_maker is not None:
            # optionally relax the structure
            relaxation_calcs: Response = run_ordering_calculations(
                orderings.output,
                maker=self.relax_maker
            )
            jobs.append(relaxation_calcs)

        static_calcs: Response = run_ordering_calculations(
            relaxation_calcs.output if self.relax_maker else orderings.output,
            maker=self.static_maker
        )

        analyze_orderings: Response = analyze_orderings(
            static_calcs.output
        )

        flow = Flow(jobs=jobs, output=analyze_orderings.output, name=self.name)
        return flow
