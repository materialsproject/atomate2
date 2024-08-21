"""Jobs for electrode analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, NamedTuple

from emmet.core.electrode import InsertionElectrodeDoc
from emmet.core.mpid import MPID
from emmet.core.structure_group import StructureGroupDoc
from jobflow import Flow, Maker, Response, job
from pymatgen.analysis.defects.generators import ChargeInterstitialGenerator
from pymatgen.entries.computed_entries import ComputedStructureEntry
from ulid import ULID

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.alchemy import ElementLike
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.io.vasp.outputs import VolumetricData


logger = logging.getLogger(__name__)

__author__ = "Jimmy Shen"
__email__ = "jmmshn@gmail.com"


class RelaxJobSummary(NamedTuple):
    """A summary of a relaxation job."""

    structure: Structure
    entry: ComputedEntry
    dir_name: str
    uuid: str


@job
def get_stable_inserted_results(
    structure: Structure,
    inserted_element: ElementLike,
    structure_matcher: StructureMatcher,
    static_maker: Maker,
    relax_maker: Maker,
    get_charge_density: Callable,
    insertions_per_step: int = 4,
    n_steps: int | None = None,
    n_inserted: int = 0,
) -> Response:
    """Attempt ion insertion.

    The basic unit for cation insertion is:
        [get_stable_inserted_structure]:
            (static) -> N x (chgcar analysis -> relax) -> (return best structure)

    Parameters
    ----------
    structure:
        The structure to insert into.
    inserted_species:
        The species to insert.
    structure_matcher:
        The structure matcher to use to determine if additional
        insertion is needed.
    static_maker:
        A maker to perform static calculations.
    relax_maker:
        A maker to perform relaxation calculations.
    get_charge_density:
        A function to get the charge density from a previous calculation.
        Whether to use the AECCAR0 and AECCAR2 files for the charge density.
        This is often necessary since the CHGCAR file has spurious effects near the
        core which often breaks the min-filter algorithms used to identify the local
        minima.
    insertions_per_step:
        The maximum number of ion insertion sites to attempt.
    n_steps:
        The maximum number of steps to perform.
    n_inserted:
        The number of ions inserted so far, used to help assign a unique name to the
        different jobs.
    """
    if structure is None:
        return []
    if n_steps is not None and n_steps <= 0:
        return []
    # append job name
    add_name = f"{n_inserted}"

    static_job = static_maker.make(structure=structure)
    chg_job = get_charge_density_job(static_job.output.dir_name, get_charge_density)
    insertion_job = get_inserted_structures(
        chg_job.output,
        inserted_species=inserted_element,
        insertions_per_step=insertions_per_step,
    )
    relax_jobs = get_relaxed_job_summaries(
        structures=insertion_job.output, relax_maker=relax_maker, append_name=add_name
    )

    min_en_job = get_min_energy_summary(
        relaxed_summaries=relax_jobs.output,
        ref_structure=structure,
        structure_matcher=structure_matcher,
    )
    nn_step = n_steps - 1 if n_steps is not None else None
    next_step = get_stable_inserted_results(
        structure=min_en_job.output[0],
        inserted_element=inserted_element,
        structure_matcher=structure_matcher,
        static_maker=static_maker,
        relax_maker=relax_maker,
        get_charge_density=get_charge_density,
        insertions_per_step=insertions_per_step,
        n_steps=nn_step,
        n_inserted=n_inserted + 1,
    )

    for job_ in [static_job, chg_job, insertion_job, min_en_job, relax_jobs, next_step]:
        job_.append_name(f" {add_name}")
    combine_job = get_computed_entries(next_step.output, min_en_job.output)
    replace_flow = Flow(
        jobs=[
            static_job,
            chg_job,
            insertion_job,
            relax_jobs,
            min_en_job,
            next_step,
            combine_job,
        ],
        output=combine_job.output,
    )
    return Response(replace=replace_flow)


@job
def get_computed_entries(
    multi: list[ComputedEntry], single: RelaxJobSummary | None
) -> list[ComputedEntry]:
    """Add a single new entry to a list of entries.

    Parameters
    ----------
    multi: The list of entries.
    single: Possible tuple containing the new entry

    Returns
    -------
        The list of entries with the new entry added.
    """
    if single is None:
        return multi
    # keep the [1] for now, if jobflow supports NamedTuple, we can do this much cleaner
    s_ = RelaxJobSummary._make(single)

    # Ensure that the entry_id is an acceptable MPID
    try:
        entry_id = MPID(s_.uuid)
    except ValueError:
        entry_id = ULID()
    s_.entry.entry_id = str(entry_id)

    # Store UUID for provenance
    s_.entry.data["UUID"] = s_.uuid

    ent = ComputedStructureEntry(
        structure=s_.structure,
        energy=s_.entry.energy,
        parameters=s_.entry.parameters,
        data=s_.entry.data,
        entry_id=s_.entry.entry_id,
    )
    return [*multi, ent]


@job(output_schema=StructureGroupDoc)
def get_structure_group_doc(
    computed_entries: list[ComputedEntry], ignored_species: str
) -> Response:
    """Take in `ComputedEntry` and return a `StructureGroupDoc`."""
    for ient in computed_entries:
        ient.data["material_id"] = ient.entry_id
    return StructureGroupDoc.from_grouped_entries(
        computed_entries, ignored_specie=ignored_species
    )


@job(output_schema=InsertionElectrodeDoc)
def get_insertion_electrode_doc(
    computed_entries: ComputedEntry, working_ion_entry: ComputedEntry
) -> Response:
    """Return a `InsertionElectrodeDoc`."""
    for ient in computed_entries:
        ient.data["material_id"] = ient.entry_id
    return InsertionElectrodeDoc.from_entries(
        computed_entries, working_ion_entry, battery_id=None
    )


@job
def get_inserted_structures(
    chg: VolumetricData,
    inserted_species: ElementLike,
    insertions_per_step: int = 4,
    charge_insertion_generator: ChargeInterstitialGenerator | None = None,
) -> list[Structure]:
    """Get the inserted structures.

    Parameters
    ----------
    chg: The charge density.
    inserted_species: The species to insert.
    insertions_per_step: The maximum number of ion insertion sites to attempt.
    charge_insertion_generator: The charge insertion generator to use,
        tolerances should be set here.


    Returns
    -------
        The inserted structures.
    """
    if charge_insertion_generator is None:
        charge_insertion_generator = ChargeInterstitialGenerator()
    gen = charge_insertion_generator.generate(chg, insert_species=[inserted_species])
    inserted_structures = [defect.defect_structure for defect in gen]
    return inserted_structures[:insertions_per_step]


@job
def get_relaxed_job_summaries(
    structures: list[Structure],
    relax_maker: Maker,
    append_name: str = "",
) -> Response:
    """Spawn relaxation jobs.

    Parameters
    ----------
    structures: The structures to relax.
    relax_maker: The maker to use to spawn relaxation jobs.

    Returns
    -------
        The relaxation jobs.
    """
    relax_jobs = []
    outputs = []
    for ii, structure in enumerate(structures):
        job_ = relax_maker.make(structure=structure)
        relax_jobs.append(job_)
        job_.append_name(f" {append_name} ({ii})")
        d_ = {
            "structure": job_.output.structure,
            "entry": job_.output.entry,
            "dir_name": job_.output.dir_name,
            "uuid": job_.output.uuid,
        }
        outputs.append(RelaxJobSummary(**d_))

    replace_flow = Flow(relax_jobs, output=outputs)
    return Response(replace=replace_flow, output=outputs)


@job
def get_min_energy_summary(
    relaxed_summaries: list[RelaxJobSummary],
    ref_structure: Structure,
    structure_matcher: StructureMatcher,
) -> Response:
    """Get the structure with the lowest energy.

    Parameters
    ----------
    structures: The structures to compare.
    ref_structure: The reference structure to compare to.
    structure_matcher: The structure matcher to use to compare structures.

    Returns
    -------
        The structure with the lowest energy.
    """
    # Since the outputs parser will see a NamedTuple and immediately convert it to
    # a list We have to convert the list of lists to a list of NamedTuples
    relaxed_summaries = list(map(RelaxJobSummary._make, relaxed_summaries))
    topotactic_summaries = [
        summary
        for summary in relaxed_summaries
        if structure_matcher.fit(ref_structure, summary.structure)
    ]

    if len(topotactic_summaries) == 0:
        return None

    return min(topotactic_summaries, key=lambda x: x.entry.energy_per_atom)


@job
def get_charge_density_job(
    prev_dir: Path | str,
    get_charge_density: Callable,
) -> VolumetricData:
    """Get the charge density from a task document.

    Parameters
    ----------
    prev_dir: The previous directory where the static calculation was performed.
    get_charge_density: A function to get the charge density from a task document.

    Returns
    -------
        The charge density.
    """
    return get_charge_density(prev_dir)
