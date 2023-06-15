"""Jobs used for enumeration, calculation, and analysis of magnetic orderings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional, Sequence

from jobflow import Flow, Maker, Response, job
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)

__all__ = [
    "enumerate_magnetic_orderings",
    "run_ordering_calculations",
    "analyze_orderings",
]


@job("enumerate orderings")
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
) -> tuple[list[Structure], list[str]]:
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
    Tuple[List[Structure], List[str]]:
        Ordered structures, origins (e.g., "fm", "afm")

    """
    enumerator = MagneticStructureEnumerator(
        structure,
        default_magmoms=default_magmoms,
        strategies=strategies,
        automatic=automatic,
        truncate_by_symmetry=truncate_by_symmetry,
        transformation_kwargs=transformation_kwargs,
    )

    return enumerator.ordered_structures, enumerator.ordered_structure_origins


@job(name="run orderings")
def run_ordering_calculations(
    orderings: tuple[Sequence[Structure], Sequence[str]],
    maker: Maker,
    prev_calc_dir_argname: str,
    prev_calc_dirs: Sequence[str] | Sequence[Path] | None = None,
):
    """
    Run calculations for a list of enumerated orderings. This job will automatically
    replace itself with calculations. These can either be static or relax calculations.

    Parameters
    ----------
    orderings : List[Structure]
        A list of pymatgen structures.
    maker : .Maker
        A Maker to use to calculate the energies of the orderings.

    Returns
    -------
    Response:
        A response with a flow of the calculations.
    """

    jobs, outputs = [], []
    for idx, (ordering, prev_calc_dir) in enumerate(zip(orderings, prev_calc_dirs)):
        struct, origin = ordering
        job = maker.make(struct, **{prev_calc_dir_argname: prev_calc_dir})
        job.append_name(f" {idx + 1}/{len(orderings)} ({origin})")

        output = {
            "uuid": job.output.uuid,
            "energy": job.output.output.energy,
            "job_dir": job.output.dir_name,
        }
        jobs.append(job)
        outputs.append(output)

    flow = Flow(jobs, outputs)
    return Response(replace=flow)


@job(name="analyze orderings")  # (output_schema=...)
def analyze_orderings(*args):
    raise NotImplementedError
