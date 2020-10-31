from pydantic import BaseModel
from typing import List, Optional, Type

from atomate2.common.store import AtomateStore
from atomate2.vasp.models.calculation import VaspObject
from atomate2.vasp.models.task import VaspTaskDoc


class VaspTaskStore(AtomateStore):
    """
    Store intended to allow pushing and pulling VASP tasks into multiple stores.

    Document data (i.e., standard information about the calculation) gets stored in a
    mongoDB database whereas large calculation data (band structures, charge densities
    etc) gets stored in a blob store, such as a GridFSStore or Amazon S3Store.
    """

    def __init__(
        self,
        *args,
        save: Optional[VaspObject, List[VaspObject]] = tuple(VaspObject),
        load: Optional[VaspObject, List[VaspObject]] = None,
        docs_collection_name: str = "tasks",
        document_model: Type[BaseModel] = VaspTaskDoc,
        **kwargs,
    ):
        """
        Create a store for querying and updating VASP tasks.

        Vasp task documents are stored in the "vasp_tasks" collection.

        By default, any VASP Objects (band structure, density of states, charge density)
        in VASP tasks will be automatically saved when the task is updated. However
        VASP objects will not be loaded unless specified.

        Args:
            *args: Positional arguments that will be passed to the AtomateStore
                constructor. See the AtomateStore.__init__ function for more details.
            save: Which VASP objects to store in the data store. By default
                all VASP objects are stored.
            load: Which VASP objects to load from the data store when querying the
                task document. By default no VASP objects are loaded.
            docs_collection_name: The collection name in which to store VASP task
                documents.
            document_model: The document model for VASP task documents. This will
                enforce that documents have the correct schema.
            **kwargs: Additional keyword arguments that are passed to the AtomateStore
                constructor. See the AtomateStore.__init__ function for more details.
        """
        super(VaspTaskStore, self).__init__(
            *args,
            **kwargs,
            save=save,
            load=load,
            document_model=document_model,
            docs_collection_name=docs_collection_name,
        )
