"""Jobs used for enumeration/calculation of collinear magnetic orderings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from jobflow import Flow, Maker, Response, job
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator

from atomate2.common.schemas.magnetism import MagneticOrderingsDocument

if TYPE_CHECKING:
    from collections.abc import Sequence

    from emmet.core.tasks import TaskDoc
    from pymatgen.core.structure import Structure


logger = logging.getLogger(__name__)


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
    """Enumerate possible collinear magnetic orderings for a given structure.

    This method is a wrapper around pymatgen's `MagneticStructureEnumerator`. Please see
    the corresponding documentation in pymatgen for more detailed descriptions.

    Parameters
    ----------
    structure : Structure
        Input structure
    default_magmoms : dict[str, float]
        Optional default mapping of magnetic elements to their initial magnetic moments
        in μB. Generally these are chosen to be high-spin, since they can relax to a
        low-spin configuration during a DFT electronic configuration. If None, will use
        the default values provided in pymatgen/analysis/magnetism/default_magmoms.yaml.
    strategies : Sequence[Literal["ferromagnetic", "antiferromagnetic", ...]]
        Different ordering strategies to use, choose from: ferromagnetic,
        antiferromagnetic, antiferromagnetic_by_motif, ferrimagnetic_by_motif and
        ferrimagnetic_by_species (here, "motif", means to use a different ordering
        parameter for symmetry inequivalent sites). Defaults to
        ("ferromagnetic", "antiferromagnetic").
    automatic : bool
        If True, will automatically choose sensible strategies
    truncate_by_symmetry: : bool
        If True, will remove very unsymmetrical orderings that are likely physically
        implausible
    transformation_kwargs : dict
        Keyword arguments to pass to MagOrderingTransformation, to change automatic cell
        size limits, etc.

    Returns
    -------
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
    relax_maker: Maker | None,
) -> Response:
    """Run calculations for a list of enumerated orderings.

    This job will automatically replace itself with calculations.

    Parameters
    ----------
    orderings : tuple[Sequence[Structure], Sequence[str]]
        A tuple containing a sequence of ordered structures and another sequence of
        strings indicating the origin of each structure (e.g., "fm", "afm").
    static_maker : .Maker
        A Maker to use to generate jobs for calculating the energies of the orderings.
        This is required.
    relax_maker : .Maker | None
        An optional Maker to use to generate jobs for relaxing the structures (before
        static calculations).

    Returns
    -------
    Replaces the job with a Flow that will run all calculations.
    """
    jobs = []
    num_orderings = len(orderings[0])
    for idx, (struct, origin) in enumerate(zip(*orderings, strict=True)):
        name = f"{idx + 1}/{num_orderings} ({origin})"

        parent_structure = struct.copy()
        parent_structure.remove_spin()
        metadata = {"parent_structure": parent_structure, "ordering": origin}

        parent_uuid = None
        static_job_kwargs = {}  # previous calc dir only
        if relax_maker is not None:
            relax_job = relax_maker.make(struct)
            relax_job.append_name(" " + name)
            relax_job.metadata.update(metadata)  # type: ignore[union-attr]
            jobs.append(relax_job)

            structure = relax_job.output.structure
            parent_uuid = relax_job.output.uuid
            static_job_kwargs["prev_dir"] = relax_job.output.dir_name

        static_job = static_maker.make(structure, **static_job_kwargs)
        static_job.append_name(" " + name)

        metadata["parent_uuid"] = parent_uuid
        static_job.metadata.update(metadata)  # type: ignore[union-attr]

        jobs.append(static_job)

    flow = Flow(
        jobs,
        output=[(j.output, j.metadata, j.uuid) for j in jobs],  # type: ignore[union-attr]
    )
    return Response(replace=flow)


@job(name="postprocess orderings")
def postprocess_orderings(
    tasks_metadata_uuids: list[tuple[TaskDoc, dict, str]],
) -> MagneticOrderingsDocument:
    """Identify ground state ordering and build summary document.

    This job performs the same analysis as that performed by the
    MagneticOrderingsBuilder. It is provided here for convenience and runs automatically
    at the conclusion of a successful MagneticOrderingsFlow.

    Parameters
    ----------
    tasks_metadata_uuids : list[tuple[TaskDoc, dict, str]]
        A list of tuples containing the output task document, metadata, and uuid for
        each job. This format is used to construct a dict that mimics how job data is
        stored and provided to the builder input.
    build_doc_fn : Callable[[list[dict]], MagneticOrderingsDocument]
        A function to build the summary document from a list of dicts.

    Returns
    -------
    MagneticOrderingsDocument
        A summary document containing the ground state ordering and other information.
    """
    tasks = [
        {"output": output, "metadata": metadata, "uuid": uuid}  # mimic builder input
        for output, metadata, uuid in tasks_metadata_uuids
    ]

    return MagneticOrderingsDocument.from_tasks(tasks)
