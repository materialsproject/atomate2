"""
This module contains 3 builders for elastic properties.

1.  The ElasticAnalysisBuilder builds individual documents
    corresponding to aggregated tasks corresponding to a
    single input structure, e. g. all of the tasks in one
    workflow.  This is where elastic tensors are fitted.
2.  The ElasticAggregateBuilder aggregates elastic documents
    that all correspond to the same structure and also
    assigns a "state" based on the elastic document's validity.
3.  The ElasticCopyBuilder is a simple copy builder that
    transfers all of the aggregated tasks with "successful" states
    into a production collection.  It's not strictly necessary,
    but is currently in use in the production pipeline.
"""
import logging
import warnings
from datetime import datetime
from itertools import chain

from monty.json import jsanitize
from monty.serialization import loadfn
from maggma.builders import Builder


logger = logging.getLogger(__name__)


class ElasticBuilder(Builder):
    """
      The Materials Builder matches VASP task documents by structure similarity into materials
      document. The purpose of this builder is group calculations and determine the best structure.
      All other properties are derived from other builders.
      The process is as follows:
          1.) Find all documents with the same formula
          2.) Select only task documents for the task_types we can select properties from
          3.) Aggregate task documents based on strucutre similarity
          4.) Create a MaterialDoc from the group of task documents
          5.) Validate material document

    Args:
        tasks: Store of task documents
        elasticity: Store of materials documents to generate
        query: dictionary to limit tasks to be analyzed
    """

    def __init__(
        self,
        tasks: Store,
        elasticity: Store,
        query: Optional[Dict] = None,
        **kwargs,
    ):

        self.tasks = tasks
        self.elasticity = elasticity
        self.query = query if query else {}
        self.kwargs = kwargs

        super().__init__(sources=[tasks], targets=[elasticity], **kwargs)

    def ensure_indexes(self):
        """
        Ensures indicies on the tasks and materials collections
        """

        # Basic search index for tasks
        self.tasks.ensure_index("nsites")
        self.tasks.ensure_index("formula_pretty")
        self.tasks.ensure_index("last_updated")

        # Search index for elastic documents
        self.elasticity.ensure_index("optimization_task_id")
        self.elasticity.ensure_index("last_updated")

    def get_items(self):
        """
        Gets all items to process into elastic documents

        Returns:
            generator of tasks aggregated by formula with relevant data
            projection to process into elasticity documents
        """

        self.logger.info("Elastic Builder Started")
        self.logger.debug("Adding indices")
        self.ensure_indexes()

        # Get only successful elastic deformation tasks with parent structure
        q = dict(self.query)
        q["state"] = "successful"
        q.update({"task_label": {
            "$regex": "[(elastic deformation)(structure optimization)]"}})

        return_props = ['output', 'input', 'completed_at', 'transmuter',
                        'task_id', 'task_label', 'formula_pretty', 'dir_name']

        formulas = self.tasks.distinct('formula_pretty', criteria=q)

        self.logger.info("Starting aggregation")
        cmd_cursor = self.tasks.groupby("formula_pretty", criteria=q,
                                        properties=return_props)
        self.logger.info("Aggregation complete")
        self.total = len(formulas)

        for n, doc in enumerate(cmd_cursor):
            # TODO: refactor for task sets without structure opt
            logger.debug("Getting formula {}, {} of {}".format(
                doc['_id']['formula_pretty'], n, len(formulas)))
            yield doc['docs']

    def process_item(self, item):
        """
        Process the tasks and materials into an elasticity collection

        Args:
            item: a dictionary of documents keyed by materials id

        Returns:
            an elasticity document
        """

        all_docs = []
        tasks = item
        if not item:
            return all_docs
        logger.debug("Processing formula {}".format(tasks[0]['formula_pretty']))

        # Group tasks by optimization with corresponding lattice
        grouped = group_deformations_by_optimization_task(tasks)
        elastic_docs = []
        for opt_task, defo_tasks in grouped:
            # Catch the warnings, just for convenience
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                elastic_doc = get_elastic_analysis(opt_task, defo_tasks)
                if elastic_doc:
                    elastic_docs.append(elastic_doc)
        return elastic_docs

    def update_targets(self, items):
        """
        Inserts the new elasticity documents into the elasticity collection

        Args:
            items ([dict]): list of elasticity docs
        """
        items = filter(bool, items)
        items = chain.from_iterable(items)
        items = [jsanitize(doc, strict=True) for doc in items]

        self.logger.info("Updating {} elastic documents".format(len(items)))

        self.elasticity.update(items, key='optimization_dir_name')
