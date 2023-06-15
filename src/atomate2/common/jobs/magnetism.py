"""Jobs used for enumeration, calculation, and analysis of magnetic orderings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional, Sequence

from jobflow import Flow, Maker, Response, job
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator

if TYPE_CHECKING:
    from pymatgen.core import Element
    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)

__all__ = [
    "enumerate_magnetic_orderings",
    "run_ordering_calculations",
    "analyze_orderings",
]


@job
def enumerate_magnetic_orderings(
    structure: Structure,
    default_magmoms: dict[str, float] | None = None,
    strategies: Sequence[
        Literal[
            "ferromagnetic",
            "antiferromagnetic",
            "antiferromagnetic_by_motif",
            "ferrimagnetic_by_motif",
            "ferrimagnetic_by_species",
            "nonmagnetic",
        ]
    ] = ("ferromagnetic", "antiferromagnetic"),
    automatic: bool = True,
    truncate_by_symmetry: bool = True,
    transformation_kwargs: dict | None = None,
) -> list[Structure]:
    """
    Enumerate possible collinear magnetic orderings for a given structure.

    This method is a wrapper around pymatgen's `MagneticStructureEnumerator`. Please see
    that class's documentation for more details.

    Parameters
    ----------
    structure: input structure
    default_magmoms: Optional default mapping of magnetic elements to their initial magnetic moments
        in µB. Generally these are chosen to be high-spin, since they can relax to a
        low-spin configuration during a DFT electronic configuration. If None, will use
        the default values provided in pymatgen/analysis/magnetism/default_magmoms.yaml.
    strategies: different ordering strategies to use, choose from:
        ferromagnetic, antiferromagnetic, antiferromagnetic_by_motif,
        ferrimagnetic_by_motif and ferrimagnetic_by_species (here, "motif",
        means to use a different ordering parameter for symmetry inequivalent
        sites)
    automatic: if True, will automatically choose sensible strategies
    truncate_by_symmetry: if True, will remove very unsymmetrical
        orderings that are likely physically implausible
    transformation_kwargs: keyword arguments to pass to
        MagOrderingTransformation, to change automatic cell size limits, etc.

    Returns
    -------
    Tuple:
        Ordered structures

    """
    enumerator = MagneticStructureEnumerator(
        structure,
        default_magmoms=default_magmoms,
        strategies=strategies,
        automatic=automatic,
        truncate_by_symmetry=truncate_by_symmetry,
        transformation_kwargs=transformation_kwargs,
    )

    return enumerator.ordered_structures


@job
def run_ordering_calculations(
    orderings: list[Structure],
    maker: Maker,
):
    """
    Run calculations for a list of enumerated orderings. This job will automatically
    replace itself with calculations.

    Parameters
    ----------
    orderings : List[Structure]
        A list of pymatgen structures.
    maker : .BaseVaspMaker
        A VaspMaker to use to calculate the energies of the orderings.

    Returns
    -------
    Response:
        A response with a flow of the calculations.
    """

    jobs = []
    outputs = []
    for idx, ordering in enumerate(orderings):
        job = maker.make(ordering)  # , prev_vasp_dir=prev_vasp_dir
        job.append_name(f" {i + 1}/{len(orderings)}")
        jobs.append(job)

        # extract the outputs we want
        output = {
            # TODO: check output format here
            "energy": job.output.output.energy,
            "uuid": job.output.uuid,
            "job_dir": job.output.dir_name,
        }

        outputs.append(output)

    flow = Flow(jobs, outputs)
    return Response(replace=flow)


@job  # (output_schema=...)
def analyze_orderings(*args):
    raise NotImplementedError
