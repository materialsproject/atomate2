from datetime import datetime
from itertools import groupby
from typing import Dict, Iterator, List, Literal, Optional

import numpy as np
from emmet.core.electronic_structure import ElectronicStructureDoc
from emmet.core.material import MaterialsDoc
from maggma.builders import Builder
from maggma.stores import Store
from maggma.utils import grouper
from monty.json import MontyDecoder, jsanitize
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.cp2k.inputs import Cp2kInput
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.cp2k.schemas.calc_types import TaskType
from atomate2.cp2k.schemas.defect import DefectDoc, DefectiveMaterialDoc
from atomate2.settings import Atomate2Settings

__author__ = "Nicholas Winner <nwinner@berkeley.edu>"


class DefectBuilder(Builder):
    """
    The DefectBuilder collects task documents performed on structures containing a single point defect.
    The builder is intended to group tasks corresponding to the same defect (species including charge state),
    find the best ones, and perform finite-size defect corrections to create a defect document. These
    defect documents can then be assembled into defect phase diagrams using the DefectThermoBuilder.

    In order to make the build process easier, an entry must exist inside of the task doc that identifies it
    as a point defect calculation. Currently this is the Pymatgen defect object keyed by "defect". In the future,
    this may be changed to having a defect transformation in the transformation history.

    The process is as follows:

        1.) Find all documents containing the defect query.
        2.) Find all documents that do not contain the defect query, and which have DOS and dielectric data already
            calculated. These are the candidate bulk tasks.
        3.) For each candidate defect task, attempt to match to a candidate bulk task of the same number of sites
            (+/- 1) with the required properties for analysis. Reject defects that do not have a corresponding
            bulk calculation.
        4.) Convert (defect, bulk task) doc pairs to DefectDocs
        5.) Post-process and validate defect document
        6.) Update the defect store
    """

    # TODO how to incorporate into settings?
    DEFAULT_ALLOWED_DFCT_TASKS = [
        TaskType.Structure_Optimization.value,
    ]

    DEFAULT_ALLOWED_BULK_TASKS = [
        TaskType.Structure_Optimization.value,
        TaskType.Static.value,
    ]

    def __init__(
        self,
        tasks: Store,
        defects: Store,
        dielectric: Store,
        electronic_structure: Store,
        materials: Store,
        electrostatic_potentials: Store,
        task_validation: Optional[Store] = None,
        query: Optional[Dict] = None,
        bulk_query: Optional[Dict] = None,
        allowed_dfct_types: Optional[List[str]] = DEFAULT_ALLOWED_DFCT_TASKS,
        allowed_bulk_types: Optional[List[str]] = DEFAULT_ALLOWED_BULK_TASKS,
        task_schema: Literal[
            "cp2k"
        ] = "cp2k",  # TODO cp2k specific right now, but this will go in common eventually
        settings: Dict | None = None,
        **kwargs,
    ):
        """
        Args:
            tasks: Store of task documents
            defects: Store of defect documents to generate
            dielectric: Store of dielectric data
            electronic_structure: Store of electronic structure data
            materials: Store of materials documents
            electrostatic_potentials: Store of electrostatic potential data. These
                are generally stored in seperately from the tasks on GridFS due to their size.
            task_validation: Store of task validation documents. If true, then only tasks that have passed
                validation will be considered.
            query: dictionary to limit tasks to be analyzed. NOT the same as the defect_query property
            allowed_task_types: list of task_types that can be processed
            settings: EmmetBuildSettings object
        """

        self.tasks = tasks
        self.defects = defects
        self.materials = materials
        self.dielectric = dielectric
        self.electronic_structure = electronic_structure
        self.electrostatic_potentials = electrostatic_potentials
        self.task_validation = task_validation
        self._allowed_dfct_types = (
            allowed_dfct_types  # TODO How to incorporate into getitems?
        )
        self._allowed_bulk_types = (
            allowed_bulk_types  # TODO How to incorporate into getitems?
        )

        settings = settings if settings else {}
        self.settings = Atomate2Settings(**settings)  # TODO don't think this is right
        self.query = query if query else {}
        self.bulk_query = bulk_query if bulk_query else {}
        self.timestamp = None
        self._mpid_map = {}
        self.task_schema = task_schema
        self.kwargs = kwargs

        # TODO Long term, schemas should be part of the matching and grouping process so that a builder can be run on a mixture
        self.query.update(
            {
                "output.@module": f"atomate2.{self.task_schema}.schemas.task",
                "output.@class": "TaskDocument",
            }
        )
        self.bulk_query.update(
            {
                "output.@module": f"atomate2.{self.task_schema}.schemas.task",
                "output.@class": "TaskDocument",
            }
        )
        self._defect_query = "output.additional_json.info.defect"

        self._required_defect_properties = [
            self._defect_query,
            self.tasks.key,
            "output.output.energy",
            "output.output.structure",
            "output.input",
            "output.nsites",
            "output.cp2k_objects.v_hartree",
        ]

        self._required_bulk_properties = [
            self.tasks.key,
            "output.output.energy",
            "output.output.structure",
            "output.input",
            "output.cp2k_objects.v_hartree",
            "output.output.vbm",
        ]

        self._optional_defect_properties = []
        self._optional_bulk_properties = []

        sources = [
            tasks,
            dielectric,
            electronic_structure,
            materials,
            electrostatic_potentials,
        ]
        if self.task_validation:
            sources.append(self.task_validation)
        super().__init__(sources=sources, targets=[defects], **kwargs)

    @property
    def defect_query(self) -> str:
        """
        The standard query for defect tasks.
        """
        return self._defect_query

    # TODO Hartree pot should be required but only for charged defects
    @property
    def required_defect_properties(self) -> List:
        """
        Properties essential to processing a defect task.
        """
        return self._required_defect_properties

    @property
    def required_bulk_properties(self) -> List:
        """
        Properties essential to processing a bulk task.
        """
        return self._required_bulk_properties

    @property
    def optional_defect_properties(self) -> List:
        """
        Properties that are optional for processing a defect task.
        """
        return self._optional_defect_properties

    @property
    def optional_bulk_properties(self) -> List:
        """
        Properties that are optional for bulk tasks.
        """
        return self._optional_bulk_properties

    @property
    def mpid_map(self) -> Dict:
        return self._mpid_map

    @property
    def allowed_dfct_types(self) -> set:
        return {TaskType(t) for t in self._allowed_dfct_types}

    @property
    def allowed_bulk_types(self) -> set:
        return {TaskType(t) for t in self._allowed_bulk_types}

    def ensure_indexes(self):
        """
        Ensures indicies on the tasks and materials collections
        """

        # Basic search index for tasks
        self.tasks.ensure_index(self.tasks.key)
        self.tasks.ensure_index("output.last_updated")
        self.tasks.ensure_index("output.state")
        self.tasks.ensure_index("output.formula_pretty")  # TODO is necessary?

        # Search index for materials
        self.materials.ensure_index("material_id")
        self.materials.ensure_index("last_updated")
        self.materials.ensure_index("task_ids")

        # Search index for defects
        self.defects.ensure_index("material_id")
        self.defects.ensure_index("last_updated")
        self.defects.ensure_index("task_ids")

        if self.task_validation:
            self.task_validation.ensure_index("task_id")
            self.task_validation.ensure_index("valid")

    def prechunk(self, number_splits: int) -> Iterator[Dict]:

        tag_query = {}
        if len(self.settings.BUILD_TAGS) > 0 and len(self.settings.EXCLUDED_TAGS) > 0:
            tag_query["$and"] = [
                {"tags": {"$in": self.settings.BUILD_TAGS}},
                {"tags": {"$nin": self.settings.EXCLUDED_TAGS}},
            ]
        elif len(self.settings.BUILD_TAGS) > 0:
            tag_query["tags"] = {"$in": self.settings.BUILD_TAGS}

        # Get defect tasks
        temp_query = self.query.copy()
        temp_query.update(tag_query)
        temp_query.update(
            {d: {"$exists": True, "$ne": None} for d in self.required_defect_properties}
        )
        temp_query.update({self.defect_query: {"$exists": True}, "state": "successful"})
        defect_tasks = {
            doc[self.tasks.key]
            for doc in self.tasks.query(
                criteria=temp_query, properties=[self.tasks.key]
            )
        }

        # Get bulk tasks
        temp_query = self.bulk_query.copy()
        temp_query.update(tag_query)
        temp_query.update({d: {"$exists": True} for d in self.required_bulk_properties})
        temp_query.update(
            {self.defect_query: {"$exists": False}, "state": "successful"}
        )
        bulk_tasks = {
            doc[self.tasks.key]
            for doc in self.tasks.query(
                criteria=temp_query, properties=[self.tasks.key]
            )
        }

        N = np.ceil(len(defect_tasks) / number_splits)
        for task_chunk in grouper(defect_tasks, N):
            yield {"query": {"task_id": {"$in": task_chunk + list(bulk_tasks)}}}

    def get_items(self) -> Iterator[List[Dict]]:
        """
        Gets all items to process into defect documents.
        This does no datetime checking; relying on on whether
        task_ids are included in the Defect Collection.

        The procedure is as follows:

            1. Get all tasks with standard "defect" query tag
            2. Filter all tasks by skipping tasks which are already in the Defect Store
            3. Get all tasks that could be used as bulk
            4. Filter all bulks which do not have corresponding Dielectric and
               ElectronicStructure data (if a band gap exists for that task).
            5. Group defect tasks by defect matching
            6. Given defect object in a group, bundle them with bulk tasks
               identified with structure matching
            7. Yield the item bundles

        Returns:
            Iterator of (defect documents, task bundles)

                The defect document is an existing defect doc to be updated with new data, or None

                task bundles bundle are all the tasks that correspond to the same defect and all possible
                bulk tasks that could be matched to them.
        """

        self.logger.info("Defect builder started")
        self.logger.info(
            f"Allowed defect types: {[task_type.value for task_type in self.allowed_dfct_types]}"
        )
        self.logger.info(
            f"Allowed bulk types: {[task_type.value for task_type in self.allowed_bulk_types]}"
        )

        self.logger.info("Setting indexes")
        self.ensure_indexes()

        # Save timestamp to mark buildtime for material documents
        self.timestamp = datetime.utcnow()

        self.logger.info("Finding tasks to process")

        ##### Get defect tasks #####
        temp_query = self.query.copy()
        temp_query.update(
            {d: {"$exists": True, "$ne": None} for d in self.required_defect_properties}
        )
        temp_query.update(
            {self.defect_query: {"$exists": True}, "output.state": "successful"}
        )
        defect_tasks = {
            doc[self.tasks.key]
            for doc in self.tasks.query(
                criteria=temp_query, properties=[self.tasks.key]
            )
        }

        # TODO Seems slow
        not_allowed = {
            doc[self.tasks.key]
            for doc in self.tasks.query(
                criteria={self.tasks.key: {"$in": list(defect_tasks)}},
                properties=["output.calcs_reversed"],
            )
            if TaskType(doc["output"]["calcs_reversed"][0]["task_type"])
            not in self.allowed_dfct_types
        }
        if not_allowed:
            self.logger.debug(
                f"{len(not_allowed)} defect tasks dropped. Not allowed TaskType"
            )
        defect_tasks = defect_tasks - not_allowed

        ##### Get bulk tasks #####
        temp_query = self.bulk_query.copy()
        temp_query.update(
            {d: {"$exists": True, "$ne": None} for d in self.required_bulk_properties}
        )
        temp_query.update(
            {self.defect_query: {"$exists": False}, "output.state": "successful"}
        )
        bulk_tasks = {
            doc[self.tasks.key]
            for doc in self.tasks.query(
                criteria=temp_query, properties=[self.tasks.key]
            )
        }

        # TODO seems slow
        not_allowed = {
            doc[self.tasks.key]
            for doc in self.tasks.query(
                criteria={self.tasks.key: {"$in": list(bulk_tasks)}},
                properties=["output.calcs_reversed"],
            )
            if TaskType(doc["output"]["calcs_reversed"][0]["task_type"])
            not in self.allowed_bulk_types
        }
        if not_allowed:
            self.logger.debug(
                f"{len(not_allowed)} bulk tasks dropped. Not allowed TaskType"
            )
        bulk_tasks = bulk_tasks - not_allowed

        # TODO Not the same validation behavior as material builders?
        # If validation store exists, find tasks that are invalid and remove them
        if self.task_validation:
            validated = {
                doc[self.tasks.key]
                for doc in self.task_validation.query({}, [self.task_validation.key])
            }

            defect_tasks = defect_tasks.intersection(validated)
            bulk_tasks = bulk_tasks.intersection(validated)

            invalid_ids = {
                doc[self.tasks.key]
                for doc in self.task_validation.query(
                    {"is_valid": False}, [self.task_validation.key]
                )
            }
            self.logger.info(f"Removing {len(invalid_ids)} invalid tasks")
            defect_tasks = defect_tasks - invalid_ids
            bulk_tasks = bulk_tasks - invalid_ids

        processed_defect_tasks = {
            t_id
            for d in self.defects.query({}, ["task_ids"])
            for t_id in d.get("task_ids", [])
        }
        all_tasks = defect_tasks | bulk_tasks

        self.logger.debug(f"All tasks: {len(all_tasks)}")
        self.logger.debug(f"Bulk tasks before filter: {len(bulk_tasks)}")
        bulk_tasks = set(filter(self.__preprocess_bulk, bulk_tasks))
        self.logger.debug(f"Bulk tasks after filter: {len(bulk_tasks)}")
        self.logger.debug(f"All defect tasks: {len(defect_tasks)}")
        unprocessed_defect_tasks = defect_tasks - processed_defect_tasks

        if not unprocessed_defect_tasks:
            self.logger.info("No unprocessed defect tasks. Exiting")
            return
        elif not bulk_tasks:
            self.logger.info("No compatible bulk calculations. Exiting.")
            return

        self.logger.info(
            f"Found {len(unprocessed_defect_tasks)} unprocessed defect tasks"
        )
        self.logger.info(
            f"Found {len(bulk_tasks)} bulk tasks with dielectric properties"
        )

        # Set total for builder bars to have a total
        self.total = len(unprocessed_defect_tasks)

        # yield list of defects that are of the same type, matched to an appropriate bulk calc
        self.logger.info(f"Starting defect matching.")

        for defect, defect_task_group in self.__filter_and_group_tasks(
            unprocessed_defect_tasks
        ):
            task_ids = self.__match_defects_to_bulks(bulk_tasks, defect_task_group)
            if not task_ids:
                continue
            doc = self.__get_defect_doc(defect)
            if doc:
                self.logger.info(f"DOC IS {doc.defect.__repr__()}")
            item_bundle = self.__get_item_bundle(task_ids)
            m = next(iter(task_ids.values()))[1]
            material_id = self.mpid_map[m]
            yield doc, item_bundle, material_id, defect_task_group

    def process_item(self, items):
        """
        Process a group of defect tasks that correspond to the same defect into a single defect
        document. If the DefectDoc already exists, then update it and return it. If it does not,
        create a new DefectDoc

        Args:
            items: (DefectDoc or None, [(defect task dict, bulk task dict, dielectric dict), ... ]

        returns: the defect document as a dictionary
        """
        defect_doc, item_bundle, material_id, task_ids = items
        self.logger.info(
            f"Processing group of {len(item_bundle)} defects into DefectDoc"
        )
        if item_bundle:
            for _, (defect_task, bulk_task, dielectric) in item_bundle.items():
                if not defect_doc:
                    defect_doc = DefectDoc.from_tasks(
                        defect_task=defect_task,
                        bulk_task=bulk_task,
                        dielectric=dielectric,
                        query=self.defect_query,
                        key=self.tasks.key,
                        material_id=material_id,
                    )
                else:
                    defect_doc.update_one(
                        defect_task,
                        bulk_task,
                        dielectric,
                        query=self.defect_query,
                        key=self.tasks.key,
                    )  # TODO Atomate2Store wrapper
                defect_doc.task_ids = list(
                    set(task_ids + defect_doc.task_ids)
                )  # TODO should I store the bulk id too?
            return jsanitize(
                defect_doc.dict(), allow_bson=True, enum_values=True, strict=True
            )
        return {}

    def update_targets(self, items):
        """
        Inserts the new task_types into the task_types collection
        """

        items = [item for item in items if item]

        if len(items) > 0:
            self.logger.info(f"Updating {len(items)} defects")
            for item in items:
                item.update({"_bt": self.timestamp})
                self.defects.remove_docs(
                    {
                        "task_ids": item["task_ids"],
                    }
                )
            self.defects.update(items, key="task_ids")
        else:
            self.logger.info("No items to update")

    def __filter_and_group_tasks(self, tasks):
        """
        Groups defect tasks. Tasks are grouped according to the reduced representation
        of the defect, and so tasks with different settings (e.g. supercell size, functional)
        will be grouped together.

        Args:
            tasks: task_ids (according to self.tasks.key) for unprocessed defects

        returns:
            [ (defect, [task_ids] ), ...] where task_ids correspond to the same defect
        """

        props = [self.defect_query, self.tasks.key, "output.structure"]

        self.logger.debug(f"Finding equivalent tasks for {len(tasks)} defects")

        sm = StructureMatcher(allow_subset=False)  # TODO build settings
        defects = [
            {
                self.tasks.key: t[self.tasks.key],
                "defect": self.__get_defect_from_task(t),
                "structure": Structure.from_dict(t["output"]["structure"]),
            }
            for t in self.tasks.query(
                criteria={self.tasks.key: {"$in": list(tasks)}}, properties=props
            )
        ]
        for d in defects:
            # TODO remove oxidation state because spins/oxidation cause errors in comparison.
            #  but they shouldnt if those props are close in value
            d["structure"].remove_oxidation_states()
            d["defect"].user_charges = [d["structure"].charge]

        def key(x):
            s = x["defect"].structure
            return get_sg(s), s.composition.reduced_composition

        def are_equal(x, y):
            """To decide if defects are equal."""
            if x["structure"].charge != y["structure"].charge:
                return False
            if x["defect"] == y["defect"]:
                return True
            return False

        sorted_s_list = sorted(enumerate(defects), key=lambda x: key(x[1]))
        all_groups = []

        # For each pre-grouped list of structures, perform actual matching.
        for k, g in groupby(sorted_s_list, key=lambda x: key(x[1])):
            unmatched = list(g)
            while len(unmatched) > 0:
                i, refs = unmatched.pop(0)
                matches = [i]
                inds = list(
                    filter(
                        lambda j: are_equal(refs, unmatched[j][1]),
                        list(range(len(unmatched))),
                    )
                )
                matches.extend([unmatched[i][0] for i in inds])
                unmatched = [
                    unmatched[i] for i in range(len(unmatched)) if i not in inds
                ]
                all_groups.append(
                    (
                        defects[i]["defect"],
                        [defects[i][self.tasks.key] for i in matches],
                    )
                )

        self.logger.debug(f"{len(all_groups)} groups")
        return all_groups

    def __get_defect_from_task(self, task):
        """
        Using the defect_query property, retrieve a pymatgen defect object from the task document
        """
        defect = unpack(self.defect_query, task)
        return MontyDecoder().process_decoded(defect)

    def __get_defect_doc(self, defect):
        """
        Given a defect, find the DefectDoc corresponding to it in the defects store if it exists

        returns: DefectDoc or None
        """
        material_id = self._get_mpid(defect.structure)
        docs = [
            DefectDoc(**doc)
            for doc in self.defects.query(
                criteria={"material_id": material_id}, properties=None
            )
        ]
        for doc in docs:
            if self.__defect_match(defect, doc.defect):
                return doc
        return None

    def __defect_match(self, x, y):
        """Match two defects, including there charges"""
        sm = StructureMatcher()
        if x.user_charges[0] != y.user_charges[0]:
            return False

        # Elem. changes needed to distinguish ghost vacancies
        if x.element_changes == y.element_changes and sm.fit(
            x.defect_structure, y.defect_structure
        ):
            return True

        return False

    # TODO should move to returning dielectric doc or continue returning the total diel tensor?
    def __get_dielectric(self, key):
        """
        Given a bulk task's task_id, find the material_id, and then use it to query the dielectric store
        and retrieve the total dielectric tensor for defect analysis. If no dielectric exists, as would
        be the case for metallic systems, return None.
        """
        for diel in self.dielectric.query(
            criteria={"material_id": key}, properties=["total"]
        ):
            return diel["total"]
        return None

    # TODO retrieving the electrostatic potential is by far the most expesive part of the builder. Any way to reduce?
    def __get_item_bundle(self, task_ids):
        """
        Gets a group of items that can be processed together into a defect document.

        Args:
            bulk_tasks: possible bulk tasks to match to defects
            defect_task_group: group of equivalent defects (defined by PointDefectComparator)

        returns: dict {run type: (defect task dict, bulk_task_dict, dielectric dict)}
        """
        return {
            rt: (
                self.tasks.query_one(criteria={self.tasks.key: pairs[0]}, load=True),
                self.tasks.query_one(criteria={self.tasks.key: pairs[1]}, load=True),
                self.__get_dielectric(self._mpid_map[pairs[1]]),
            )
            for rt, pairs in task_ids.items()
        }

    def _get_mpid(self, structure):
        """
        Given a structure, determine if an equivalent structure exists, with a material_id,
        in the materials store.

        Args:
            structure: Candidate structure

        returns: material_id, if one exists, else None
        """
        sga = SpacegroupAnalyzer(
            structure, symprec=self.settings.SYMPREC
        )  # TODO Add angle tolerance
        mats = self.materials.query(
            criteria={
                "chemsys": structure.composition.chemical_system,
            },
            properties=["structure", "material_id"],
        )
        # TODO coudl more than one material match true?
        sm = StructureMatcher(
            primitive_cell=True, comparator=ElementComparator()
        )  # TODO add tolerances
        for m in mats:
            if sm.fit(structure, Structure.from_dict(m["structure"])):
                return m["material_id"]
        return None

    def __match_defects_to_bulks(self, bulk_ids, defect_ids) -> list[tuple]:
        """
        Given task_ids of bulk and defect tasks, match the defects to a bulk task that has
        commensurate:
            - Composition
            - Number of sites
            - Symmetry
        """
        self.logger.debug(f"Finding bulk/defect task combinations.")
        self.logger.debug(f"Bulk tasks: {bulk_ids}")
        self.logger.debug(f"Defect tasks: {defect_ids}")

        # TODO mongo projection on array doesn't work (see above)
        props = [
            self.tasks.key,
            self.defect_query,
            "output.input",
            "output.nsites",
            "output.output.structure",
            "output.output.energy",
            "output.calcs_reversed",
        ]
        defects = list(
            self.tasks.query(
                criteria={self.tasks.key: {"$in": list(defect_ids)}}, properties=props
            )
        )
        ps = self.__get_pristine_supercell(defects[0])
        ps.remove_oxidation_states()  # TODO might cause problems
        bulks = list(
            self.tasks.query(
                criteria={
                    self.tasks.key: {"$in": list(bulk_ids)},
                    "output.formula_pretty": jsanitize(ps.composition.reduced_formula),
                },
                properties=props,
            )
        )

        pairs = [
            (defect, bulk)
            for bulk in bulks
            for defect in defects
            if self.__are_bulk_and_defect_commensurate(bulk, defect)
        ]
        self.logger.debug(f"Found {len(pairs)} commensurate bulk/defect pairs")

        def key(x):
            return -x[0]["output"]["nsites"], x[0]["output"]["output"]["energy"]

        def _run_type(x):
            return x[0]["output"]["calcs_reversed"][0]["run_type"]

        rt_pairs = {}
        for rt, group in groupby(pairs, key=_run_type):
            rt_pairs[rt] = [
                (defect[self.tasks.key], bulk[self.tasks.key])
                for defect, bulk in sorted(list(group), key=key)
            ]

        # Return only the first (best) pair for each rt
        return {rt: lst[0] for rt, lst in rt_pairs.items()}

    # TODO Checking for same dft settings (e.g. OT/diag) is a little cumbersome.
    # Maybe, in future, task doc can be defined to have OT/diag as part of input summary
    # for fast querying
    def __are_bulk_and_defect_commensurate(self, b, d):
        """
        Check if a bulk and defect task are commensurate.

        Checks for:
            1. Same run type.
            2. Same pristine structures with no supercell reduction
            3. Compatible DFT settings
        """
        # TODO add settings
        sm = StructureMatcher(
            ltol=1e-3,
            stol=0.1,
            angle_tol=1,
            primitive_cell=False,
            scale=True,
            attempt_supercell=False,
            allow_subset=False,
            comparator=ElementComparator(),
        )
        rtb = b.get("output").get("input").get("xc").split("+U")[0]
        rtd = d.get("output").get("input").get("xc").split("+U")[0]
        baux = {
            dat["element"]: dat.get("auxiliary_basis")
            for dat in b["output"]["input"]["atomic_kind_info"]["atomic_kinds"].values()
        }
        daux = {
            dat["element"]: dat.get("auxiliary_basis")
            for dat in d["output"]["input"]["atomic_kind_info"]["atomic_kinds"].values()
        }

        if rtb == rtd:
            if sm.fit(
                self.__get_pristine_supercell(d), self.__get_pristine_supercell(b)
            ):
                cib = Cp2kInput.from_dict(
                    b["output"]["calcs_reversed"][0]["input"]["cp2k_input"]
                )
                cid = Cp2kInput.from_dict(
                    d["output"]["calcs_reversed"][0]["input"]["cp2k_input"]
                )
                bis_ot = cib.check("force_eval/dft/scf/ot")
                dis_ot = cid.check("force_eval/dft/scf/ot")
                if (bis_ot and dis_ot) or (not bis_ot and not dis_ot):
                    for el in baux:
                        if baux[el] != daux[el]:
                            return False
                    return True
        return False

    def __preprocess_bulk(self, task):
        """
        Given a TaskDoc that could be a bulk for defect analysis, check to see if it can be used. Bulk
        tasks must have:

            (1) Correspond to an existing material_id in the materials store
            (2) If the bulk is not a metal, then the dielectric tensor must exist in the dielectric store
            (3) If bulk is not a metal, electronic structure document must exist in the store

        """
        self.logger.debug(f"Preprocessing bulk task {task}")
        t = next(
            self.tasks.query(
                criteria={self.tasks.key: task},
                properties=["output.output.structure", "mpid"],
            )
        )

        struc = Structure.from_dict(
            t.get("output").get("output").get("structure")
        )  # TODO specific to atomate2
        mpid = self._get_mpid(struc)
        if not mpid:
            self.logger.debug(f"No material id found for bulk task {task}")
            return False
        self._mpid_map[task] = mpid
        self.logger.debug(f"Material ID: {mpid}")

        elec = self.electronic_structure.query_one(
            properties=["band_gap"], criteria={self.electronic_structure.key: mpid}
        )
        if not elec:
            self.logger.debug(f"Electronic structure data not found for {mpid}")
            return False

        # TODO right now pulling dos from electronic structure, should just pull summary document
        if elec["band_gap"] > 0:
            diel = self.__get_dielectric(mpid)
            if not diel:
                self.logger.info(
                    f"Task {task} for {mpid} ({struc.composition.reduced_formula}) requires "
                    f"dielectric properties, but none found in dielectric store"
                )
                return False

        return True

    def __get_pristine_supercell(self, task):
        """
        Given a task document for a defect calculation, retrieve the un-defective, pristine supercell.
            - If defect transform exists, the following transform's input will be returned
            - If no follow up transform exists, the calculation input will be returned

        If defect cannot be found in task, return the input structure.

        scale_matrix = np.array(scaling_matrix, int)
        if scale_matrix.shape != (3, 3):
            scale_matrix = np.array(scale_matrix * np.eye(3), int)
        new_lattice = Lattice(np.dot(scale_matrix, self._lattice.matrix))
        """
        d = unpack(query=self.defect_query, d=task)
        out_structure = MontyDecoder().process_decoded(
            task["output"]["output"]["structure"]
        )
        if d:
            defect = MontyDecoder().process_decoded(d)
            s = defect.structure.copy()
            sc_mat = out_structure.lattice.matrix.dot(np.linalg.inv(s.lattice.matrix))
            s.make_supercell(sc_mat.round())
            return s
        else:
            return out_structure


class DefectiveMaterialBuilder(Builder):

    """
    This builder creates collections of the DefectThermoDoc object.

        (1) Find all DefectDocs that correspond to the same bulk material
            given by material_id
        (2) Create a new DefectThermoDoc for all of those documents
        (3) Insert/Update the defect_thermos store with the new documents
    """

    def __init__(
        self,
        defects: Store,
        defect_thermos: Store,
        materials: Store,
        query: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Args:
            defects: Store of defect documents (generated by DefectBuilder)
            defect_thermos: Store of DefectThermoDocs to generate.
            materials: Store of MaterialDocs to construct phase diagram
            electronic_structures: Store of DOS objects
            query: dictionary to limit tasks to be analyzed
        """

        self.defects = defects
        self.defect_thermos = defect_thermos
        self.materials = materials

        self.query = query if query else {}
        self.timestamp = None
        self.kwargs = kwargs

        super().__init__(
            sources=[defects, materials], targets=[defect_thermos], **kwargs
        )

    def ensure_indexes(self):
        """
        Ensures indicies on the collections
        """

        # Basic search index for tasks
        self.defects.ensure_index("material_id")
        self.defects.ensure_index("defect_id")

        # Search index for materials
        self.defect_thermos.ensure_index("material_id")

    # TODO need to only process new tasks. Fast builder so currently is OK for small collections
    def get_items(self) -> Iterator[List[Dict]]:
        """
        Gets items to process into DefectThermoDocs.

        returns:
            iterator yielding tuples containing:
                - group of DefectDocs belonging to the same bulk material as indexed by material_id,
                - materials in the chemsys of the bulk material for constructing phase diagram
                - Dos of the bulk material for constructing phase diagrams/getting doping

        """

        self.logger.info("Defect thermo builder started")
        self.logger.info("Setting indexes")
        self.ensure_indexes()

        # Save timestamp to mark build time for defect thermo documents
        self.timestamp = datetime.utcnow()

        # Get all tasks
        self.logger.info("Finding tasks to process")
        temp_query = dict(self.query)
        temp_query["state"] = "successful"

        # unprocessed_defect_tasks = all_tasks - processed_defect_tasks

        all_docs = [doc for doc in self.defects.query(self.query)]

        self.logger.debug(f"Found {len(all_docs)} defect docs to process")

        def filterfunc(x):
            if not self.materials.query_one(
                criteria={"material_id": x["material_id"]}, properties=None
            ):
                self.logger.debug(
                    f"No material with MPID={x['material_id']} in the material store"
                )
                return False
            return True
            defect = MontyDecoder().process_decoded(x["defect"])
            for el in defect.element_changes:
                if el not in self.thermo:
                    self.logger.debug(f"No entry for {el} in Thermo Store")
                    return False

            return True

        for key, group in groupby(
            filter(filterfunc, sorted(all_docs, key=lambda x: x["material_id"])),
            key=lambda x: x["material_id"],
        ):
            try:
                yield list(group)
            except LookupError as exception:
                raise exception

    def process_item(self, defects):
        """
        Process a group of defects belonging to the same material into a defect thermo doc
        """
        defect_docs = [DefectDoc(**d) for d in defects]
        self.logger.info(f"Processing {len(defect_docs)} defects")
        defect_thermo_doc = DefectiveMaterialDoc.from_docs(
            defect_docs, material_id=defect_docs[0].material_id
        )
        return defect_thermo_doc.dict()

    def update_targets(self, items):
        """
        Inserts the new DefectThermoDocs into the defect_thermos store
        """
        items = [item for item in items if item]
        for item in items:
            item.update({"_bt": self.timestamp})

        if len(items) > 0:
            self.logger.info(f"Updating {len(items)} defect thermo docs")
            self.defect_thermos.update(
                docs=jsanitize(items, allow_bson=True, enum_values=True, strict=True),
                key=self.defect_thermos.key,
            )
        else:
            self.logger.info("No items to update")

    def __get_electronic_structure(self, material_id):
        """
        Gets the electronic structure of the bulk material
        """
        self.logger.info(f"Getting electronic structure for {material_id}")

        # TODO This is updated to return the whole query because a.t.m. the
        # DOS part of the electronic builder isn't working, so I'm using
        # this to pull direct from the store of dos objects with no processing.
        dosdoc = self.electronic_structures.query_one(
            criteria={self.electronic_structures.key: material_id},
            properties=None,
        )
        t_id = ElectronicStructureDoc(**dosdoc).dos.total["1"].task_id
        dos = self.dos.query_one(
            criteria={"task_id": int(t_id)}, properties=None
        )  # TODO MPID str/int issues
        return dos

    def __get_materials(self, key) -> List:
        """
        Given a group of DefectDocs, use the bulk material_id to get materials in the chemsys from the
        materials store.
        """
        bulk = self.materials.query_one(criteria={"material_id": key}, properties=None)
        if not bulk:
            raise LookupError(
                f"The bulk material ({key}) for these defects cannot be found in the materials store"
            )
        return MaterialsDoc(**bulk)

    def __get_thermos(self, composition) -> List:
        return list(
            self.thermo.query(criteria={"elements": {"$size": 1}}, properties=None)
        )


class DefectValidator(Builder):
    def __init__(
        self,
        tasks: Store,
        defect_validation: Store,
        chunk_size: int = 1000,
        defect_query="output.additional_json.info.defect",
    ):
        self.tasks = tasks
        self.defect_validation = defect_validation
        self.chunk_size = chunk_size
        self.defect_query = defect_query
        super().__init__(
            sources=tasks, targets=defect_validation, chunk_size=chunk_size
        )

    def get_items(self):
        self.logger.info("Getting tasks")
        tids = list(
            self.tasks.query(
                criteria={self.defect_query: {"$exists": True}},
                properties=[self.tasks.key],
            )
        )
        self.logger.info(f"{len(tids)} to process")
        yield from self.tasks.query()

    def process_item(self, item):
        from atomate2.cp2k.schemas.defect import DefectValidation

        tid = item[self.tasks.key]
        return jsanitize(
            DefectValidation.process_task(item, tid).dict(),
            allow_bson=True,
            enum_values=True,
            strict=True,
        )

    def update_targets(self, items: List):
        """
        Inserts the new task_types into the task_types collection
        """
        items = [item for item in items if item]
        if len(items) > 0:
            self.logger.info(f"Updating {len(items)} defects")
            self.defect_validation.update(items, key=self.defect_validation.key)
        else:
            self.logger.info("No items to update")
        return super().update_targets(items)


def unpack(query, d):
    """
    Unpack a mongo-style query into dictionary retrieval
    """
    if not d:
        return None
    if not query:
        return d
    if isinstance(d, List):
        return unpack(query[1:], d.__getitem__(int(query.pop(0))))
    if isinstance(query, str):
        for seperator in [".", ":", "->"]:
            tmp = query.split(seperator)
            if len(tmp) > 1:
                return unpack(query.split("."), d)
    return unpack(query[1:], d.__getitem__(query.pop(0)))


# TODO SHOULD GO IN COMMON
def get_sg(struc, symprec=0.01) -> int:
    """helper function to get spacegroup with a loose tolerance"""
    try:
        return struc.get_space_group_info(symprec=symprec)[1]
    except Exception:
        return -1
