"""Jobs used for enumeration, calculation, and analysis of magnetic orderings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional, Sequence

from jobflow import Flow, Maker, Response, job
from pymatgen.analysis.magnetism.analyzer import (
    CollinearMagneticStructureAnalyzer,
    MagneticStructureEnumerator,
)

from atomate2.common.schemas.magnetism import (
    MagnetismDocument,
    MagnetismInput,
    MagnetismOutput,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)

__all__ = [
    "enumerate_magnetic_orderings",
    "run_ordering_calculations",
    "analyze_ordering_calculations",
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
    prev_calc_dir_argname: str | None = None,
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
        kwargs = {prev_calc_dir_argname: prev_calc_dir} if prev_calc_dir_argname else {}
        job = maker.make(struct, **kwargs)
        job.append_name(f" {idx + 1}/{len(orderings)} ({origin})")
        jobs.append(job)

        output = MagnetismOutput.from_task_document(job.output)
        outputs.append(output)

    flow = Flow(jobs, output=outputs)
    return Response(replace=flow)


@job(name="analyze orderings")
def analyze_ordering_calculations(
    outputs: Sequence[MagnetismOutput],
    parent_structure: Structure,
) -> list[MagnetismDocument]:
    """
    Analyze the results of magnetic orderings calculations. This job will
    process the output documents of the calculations and return new documnets
    with relevant parameters (such as whether the ordering changed, the total
    magnetization, whether the particular ordering is the ground state, etc.)

    Parameters
    ----------
    data : tuple
        A tuple of the results of the calculations.

    Returns
    -------
    MagnetismDocument:
        A document containing the results of the analysis.
    """

    formula = parent_structure.formula
    formula_pretty = parent_structure.composition.reduced_formula

    energies = [calc.energy_per_atom for calc in outputs]
    ground_state_energy = min(energies)

    possible_ground_state_idxs = [i for i in energies if i == ground_state_energy]
    if len(possible_ground_state_idxs) > 1:
        logger.warning(
            f"Multiple identical energies exist, duplicate calculations for {formula}?"
        )
    idx = possible_ground_state_idxs[0]
    ground_state_energy = energies[idx]
    ground_state_uuid = outputs[idx].uuid

    docs = []
    for output in outputs:
        doc = {}
        doc["formula"] = formula
        doc["formula_pretty"] = formula_pretty
        doc["structure"] = output["structure"]
        doc["ground_state_energy"] = ground_state_energy
        doc["ground_state_uuid"] = ground_state_uuid
        # input_structure = Structure.from_dict(optimize_task["input"]["structure"])
        # input_magmoms = optimize_task["input"]["incar"]["MAGMOM"]
        # input_structure.add_site_property("magmom", input_magmoms)

        # final_structure = Structure.from_dict(d["output"]["structure"])

        # # picking a fairly large threshold so that default 0.6 µB magmoms don't
        # # cause problems with analysis, this is obviously not appropriate for
        # # some magnetic structures with small magnetic moments (e.g. CuO)
        # input_analyzer = CollinearMagneticStructureAnalyzer(
        #     input_structure, threshold=0.61
        # )
        # final_analyzer = CollinearMagneticStructureAnalyzer(
        #     final_structure, threshold=0.61
        # )

        # if d["task_id"] == ground_state_task_id:
        #     stable = True
        #     decomposes_to = None
        # else:
        #     stable = False
        #     decomposes_to = ground_state_task_id
        # energy_above_ground_state_per_atom = (
        #     d["output"]["energy_per_atom"] - ground_state_energy
        # )

        # # tells us the order in which structure was guessed
        # # 1 is FM, then AFM..., -1 means it was entered manually
        # # useful to give us statistics about how many orderings
        # # we actually need to calculate
        # task_label = d["task_label"].split(" ")
        # ordering_index = task_label.index("ordering")
        # ordering_index = int(task_label[ordering_index + 1])
        # if self.get("origins", None):
        #     ordering_origin = self["origins"][ordering_index]
        # else:
        #     ordering_origin = None

        # final_magmoms = final_structure.site_properties["magmom"]
        # magmoms = {"vasp": final_magmoms}
        # if self["perform_bader"]:
        #     # if bader has already been run during task ingestion,
        #     # use existing analysis
        #     if "bader" in d:
        #         magmoms["bader"] = d["bader"]["magmom"]
        #     # else try to run it
        #     else:
        #         try:
        #             dir_name = d["dir_name"]
        #             # strip hostname if present, implicitly assumes
        #             # ToDB task has access to appropriate dir
        #             if ":" in dir_name:
        #                 dir_name = dir_name.split(":")[1]
        #             magmoms["bader"] = bader_analysis_from_path(dir_name)["magmom"]
        #             # prefer bader magmoms if we have them
        #             final_magmoms = magmoms["bader"]
        #         except Exception as e:
        #             magmoms["bader"] = f"Bader analysis failed: {e}"

        # input_order_check = [0 if abs(m) < 0.61 else m for m in input_magmoms]
        # final_order_check = [0 if abs(m) < 0.61 else m for m in final_magmoms]
        # ordering_changed = not np.array_equal(
        #     np.sign(input_order_check), np.sign(final_order_check)
        # )

        # symmetry_changed = (
        #     final_structure.get_space_group_info()[0]
        #     != input_structure.get_space_group_info()[0]
        # )

        # total_magnetization = abs(
        #     d["calcs_reversed"][0]["output"]["outcar"]["total_magnetization"]
        # )
        # num_formula_units = sum(
        #     d["calcs_reversed"][0]["composition_unit_cell"].values()
        # ) / sum(d["calcs_reversed"][0]["composition_reduced"].values())
        # total_magnetization_per_formula_unit = total_magnetization / num_formula_units
        # total_magnetization_per_unit_volume = (
        #     total_magnetization / final_structure.volume
        # )

        # summary = {
        #     "formula": formula,
        #     "formula_pretty": formula_pretty,
        #     "parent_structure": self["parent_structure"].as_dict(),
        #     "wf_meta": d["wf_meta"],  # book-keeping
        #     "task_id": d["task_id"],
        #     "structure": final_structure.as_dict(),
        #     "magmoms": magmoms,
        #     "input": {
        #         "structure": input_structure.as_dict(),
        #         "ordering": input_analyzer.ordering.value,
        #         "symmetry": input_structure.get_space_group_info()[0],
        #         "index": ordering_index,
        #         "origin": ordering_origin,
        #         "input_index": self.get("input_index", None),
        #     },
        #     "total_magnetization": total_magnetization,
        #     "total_magnetization_per_formula_unit": total_magnetization_per_formula_unit,
        #     "total_magnetization_per_unit_volume": total_magnetization_per_unit_volume,
        #     "ordering": final_analyzer.ordering.value,
        #     "ordering_changed": ordering_changed,
        #     "symmetry": final_structure.get_space_group_info()[0],
        #     "symmetry_changed": symmetry_changed,
        #     "energy_per_atom": d["output"]["energy_per_atom"],
        #     "stable": stable,
        #     "decomposes_to": decomposes_to,
        #     "energy_above_ground_state_per_atom": energy_above_ground_state_per_atom,
        #     "energy_diff_relax_static": energy_diff_relax_static,
        #     "created_at": datetime.utcnow(),
        # }
        docs.append(MagnetismDocument(**doc))
    return docs
