"""Jobs used for enumeration/calculation of collinear magnetic orderings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Sequence

from jobflow import Flow, Maker, Response, job
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)

__all__ = [
    "enumerate_magnetic_orderings",
    "run_ordering_calculations",
]


@job(name="enumerate orderings")
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
    default_magmoms: Optional default mapping of magnetic elements to their initial
        magnetic moments in µB. Generally these are chosen to be high-spin, since they
        can relax to a low-spin configuration during a DFT electronic configuration. If
        None, will use the default values provided in
        pymatgen/analysis/magnetism/default_magmoms.yaml.
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
    static_maker: Maker,
    relax_maker: Maker | None = None,
    prev_calc_dir_argname: str | None = None,
):
    """
    Run calculations for a list of enumerated orderings. This job will automatically
    replace itself with calculations.

    Parameters
    ----------
    orderings : tuple[Sequence[Structure], Sequence[str]]
        A tuple containing a sequence of ordered structures and another sequence of
        strings indicating the origin of each structure (e.g., "fm", "afm").
    static_maker : .Maker
        A Maker to use to calculate the energies of the orderings. This is required.
    relax_maker : .Maker | None
        An optional Maker to use to relax the structures before calculating energies.
    prev_calc_dir_argname : str | None
        The name of the argument to pass to the static_maker to indicate the previous
        calculation directory if relax_maker is not None (e.g., for VASP:
        "prev_vasp_dir").

    Returns
    -------
    Response:
        Replaces the job with a Flow that will run all calculations.
    """
    jobs = []
    num_orderings = len(orderings[0])
    for idx, (struct, origin) in enumerate(zip(*orderings)):
        name = f"{idx + 1}/{num_orderings} ({origin})"

        parent_structure = struct.copy()
        parent_structure.remove_spin()
        metadata = {"parent_structure": parent_structure, "ordering": origin}

        parent_uuid = None
        kwargs = {}
        if relax_maker is not None:
            relax_job = relax_maker.make(struct, **kwargs)
            relax_job.append_name(" " + name)
            relax_job.metadata.update(metadata)

            kwargs[prev_calc_dir_argname] = relax_job.output.dir_name
            kwargs["run_vasp_kwargs"] = {"vasp_job_kwargs": {"copy_magmom": True}}

            struct = relax_job.output.structure
            parent_uuid = relax_job.output.uuid
            jobs.append(relax_job)

        metadata["parent_uuid"] = parent_uuid
        static_job = static_maker.make(struct, **kwargs)
        static_job.append_name(" " + name)
        static_job.metadata.update(metadata)

        jobs.append(static_job)

    flow = Flow(jobs)
    return Response(replace=flow)
