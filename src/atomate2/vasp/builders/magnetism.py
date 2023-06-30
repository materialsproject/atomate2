"""Module defining elastic document builder."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
from maggma.builders import Builder

from atomate2 import SETTINGS
from atomate2.common.schemas.magnetism import MagnetismDocument

if TYPE_CHECKING:
    from maggma.core import Store


# class MagneticOrderingsBuilder(Builder):
#     """
#     Builder to analyze the results of magnetic orderings calculations in VASP. This job
#     will process the output documents of the calculations and return new documents
#     with relevant parameters (such as the total magnetization, whether the ordering
#     changed, whether the particular ordering is the ground state, etc.)

#     Parameters
#     ----------
#     tasks : .Store
#         Store of task documents.
#     magnetic_orderings : .Store
#         Store for magnetic ordering documents.
#     query : dict
#         Dictionary query to limit tasks to be analyzed.
#     structure_match_tol : float
#         Numerical tolerance for structure equivalence.
#     **kwargs
#         Keyword arguments that will be passed to the Builder init.
#     """

#     def __init__(
#         self,
#         tasks: Store,
#         magnetic_orderings: Store,
#         query: dict = None,
#         structure_match_tol: float = 1e-5,
#         **kwargs,
#     ):
#         self.tasks = tasks
#         self.magnetic_orderings = magnetic_orderings
#         self.query = query if query else {}
#         self.structure_match_tol = structure_match_tol
#         self.kwargs = kwargs

#         super().__init__(sources=[tasks], targets=[magnetic_orderings], **kwargs)

#     def ensure_indexes(self):
#         """Ensure indices on the tasks and elasticity collections."""
#         self.tasks.ensure_index("output.formula_pretty")
#         self.tasks.ensure_index("last_updated")
#         self.elasticity.ensure_index("fitting_data.uuids.0")
#         self.elasticity.ensure_index("last_updated")

#     def get_items(self):
#         """
#         Get all items to process into elastic documents.

#         Yields
#         ------
#         list of dict
#             A list of deformation tasks aggregated by formula and containing the
#             required data to generate elasticity documents.
#         """
#         self.logger.info("Magnetic orderings builder started")
#         self.logger.debug("Adding/ensuring indices...")
#         self.ensure_indexes()

#         q = dict(self.query)
#         q.update(
#             {
#                 "output.transformations.history.0.@class": "DeformationTransformation",
#                 "output.orig_inputs.NSW": {"$gt": 1},
#                 "output.orig_inputs.ISIF": {"$gt": 2},
#             }
#         )
#         return_props = [
#             "uuid",
#             "output.transformations",
#             "output.output.stress",
#             "output.formula_pretty",
#             "output.dir_name",
#         ]

#         self.logger.info("Starting aggregation")
#         nformulas = len(self.tasks.distinct("output.formula_pretty", criteria=q))
#         results = self.tasks.groupby(
#             "output.formula_pretty", criteria=q, properties=return_props
#         )
#         self.logger.info("Aggregation complete")

#         for n, (keys, docs) in enumerate(results):
#             self.logger.debug(
#                 f"Getting {keys['output']['formula_pretty']} ({n + 1} of {nformulas})"
#             )
#             yield docs

#     def process_item(self, tasks: list[dict]) -> list[ElasticDocument]:
#         """
#         Process deformation tasks into elasticity documents.

#         The deformation tasks will be grouped based on their parent structure (i.e., the
#         structure before the deformation was applied).

#         Parameters
#         ----------
#         tasks : list of dict
#             A list of deformation task, all with the same formula.

#         Returns
#         -------
#         list of .ElasticDocument
#             A list of elastic documents for each unique parent structure.
#         """
#         self.logger.debug(f"Processing {tasks[0]['output']['formula_pretty']}")

#         if not tasks:
#             return []

#         # group deformations by parent structure
#         grouped = _group_deformations(tasks, self.structure_match_tol)

#         elastic_docs = []
#         for tasks in grouped:
#             elastic_doc = _get_elastic_document(
#                 tasks, self.symprec, self.fitting_method
#             )
#             elastic_docs.append(elastic_doc)

#         return elastic_docs

#     def update_targets(self, items: list[ElasticDocument]):
#         """
#         Insert new elastic documents into the elasticity store.

#         Parameters
#         ----------
#         items : list of .ElasticDocument
#             A list of elasticity documents.
#         """
#         items = chain.from_iterable(filter(bool, items))  # type: ignore

#         if len(items) > 0:
#             self.logger.info(f"Updating {len(items)} elastic documents")
#             self.elasticity.update(items, key="fitting_data.uuids.0")
#         else:
#             self.logger.info("No items to update")

#     formula = parent_structure.formula
#     formula_pretty = parent_structure.composition.reduced_formula

#     energies = [calc.energy_per_atom for calc in outputs]
#     ground_state_energy = min(energies)

#     # TODO: make this check with a tolerance
#     possible_ground_state_idxs = [i for i in energies if i == ground_state_energy]
#     if len(possible_ground_state_idxs) > 1:
#         logger.warning(
#             f"Multiple identical energies exist, duplicate calculations for {formula}?"
#         )
#     idx = possible_ground_state_idxs[0]
#     ground_state_energy = energies[idx]
#     ground_state_uuid = outputs[idx].uuid

#     docs = []
#     for output in outputs:
#         doc = {}
#         doc["formula"] = formula
#         doc["formula_pretty"] = formula_pretty
#         doc["parent_structure"] = output["structure"]
#         doc["ground_state_energy"] = ground_state_energy
#         doc["ground_state_uuid"] = ground_state_uuid
#         doc["parent_structure"] = parent_structure

#         input_analyzer = CollinearMagneticStructureAnalyzer(
#             input_structure, threshold=0.61
#         )

#         final_analyzer = CollinearMagneticStructureAnalyzer(
#             final_structure, threshold=0.61
#         )

#         if d["task_id"] == ground_state_task_id:
#             stable = True
#             decomposes_to = None
#         else:
#             stable = False
#             decomposes_to = ground_state_task_id
#         energy_above_ground_state_per_atom = (
#             d["output"]["energy_per_atom"] - ground_state_energy
#         )

#         # tells us the order in which structure was guessed
#         # 1 is FM, then AFM..., -1 means it was entered manually
#         # useful to give us statistics about how many orderings
#         # we actually need to calculate
#         task_label = d["task_label"].split(" ")
#         ordering_index = task_label.index("ordering")
#         ordering_index = int(task_label[ordering_index + 1])
#         if self.get("origins", None):
#             ordering_origin = self["origins"][ordering_index]
#         else:
#             ordering_origin = None

#         final_magmoms = final_structure.site_properties["magmom"]
#         magmoms = {"vasp": final_magmoms}

#         if self["perform_bader"]:
#             # if bader has already been run during task ingestion,
#             # use existing analysis
#             if "bader" in d:
#                 magmoms["bader"] = d["bader"]["magmom"]
#             # else try to run it
#             else:
#                 try:
#                     dir_name = d["dir_name"]
#                     # strip hostname if present, implicitly assumes
#                     # ToDB task has access to appropriate dir
#                     if ":" in dir_name:
#                         dir_name = dir_name.split(":")[1]
#                     magmoms["bader"] = bader_analysis_from_path(dir_name)["magmom"]
#                     # prefer bader magmoms if we have them
#                     final_magmoms = magmoms["bader"]
#                 except Exception as e:
#                     magmoms["bader"] = f"Bader analysis failed: {e}"

#         input_order_check = [0 if abs(m) < 0.61 else m for m in input_magmoms]
#         final_order_check = [0 if abs(m) < 0.61 else m for m in final_magmoms]
#         ordering_changed = not np.array_equal(
#             np.sign(input_order_check), np.sign(final_order_check)
#         )

#         symmetry_changed = (
#             final_structure.get_space_group_info()[0]
#             != input_structure.get_space_group_info()[0]
#         )

#         total_magnetization = abs(
#             d["calcs_reversed"][0]["output"]["outcar"]["total_magnetization"]
#         )
#         num_formula_units = sum(
#             d["calcs_reversed"][0]["composition_unit_cell"].values()
#         ) / sum(d["calcs_reversed"][0]["composition_reduced"].values())
#         total_magnetization_per_formula_unit = total_magnetization / num_formula_units
#         total_magnetization_per_unit_volume = (
#             total_magnetization / final_structure.volume
#         )

#         summary = {
#             "formula": formula,
#             "formula_pretty": formula_pretty,
#             "parent_structure": self["parent_structure"].as_dict(),
#             "wf_meta": d["wf_meta"],  # book-keeping
#             "task_id": d["task_id"],
#             "structure": final_structure.as_dict(),
#             "magmoms": magmoms,
#             "input": {
#                 "structure": input_structure.as_dict(),
#                 "ordering": input_analyzer.ordering.value,
#                 "symmetry": input_structure.get_space_group_info()[0],
#                 "index": ordering_index,
#                 "origin": ordering_origin,
#                 "input_index": self.get("input_index", None),
#             },
#             "total_magnetization": total_magnetization,
#             "total_magnetization_per_formula_unit": total_magnetization_per_formula_unit,
#             "total_magnetization_per_unit_volume": total_magnetization_per_unit_volume,
#             "ordering": final_analyzer.ordering.value,
#             "ordering_changed": ordering_changed,
#             "symmetry": final_structure.get_space_group_info()[0],
#             "symmetry_changed": symmetry_changed,
#             "energy_per_atom": d["output"]["energy_per_atom"],
#             "stable": stable,
#             "decomposes_to": decomposes_to,
#             "energy_above_ground_state_per_atom": energy_above_ground_state_per_atom,
#             "energy_diff_relax_static": energy_diff_relax_static,
#             "created_at": datetime.utcnow(),
#         }
#         docs.append(MagnetismDocument(**doc))
#     return docs
