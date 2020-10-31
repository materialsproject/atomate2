from enum import Enum
from itertools import groupby

from pathlib import Path

from monty.dev import deprecated
from bson import ObjectId
from pydantic import BaseModel
from pydash import get, has, set_
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    Any,
    Hashable,
    TypeVar,
    Type,
)

from atomate2.utils.utils import find_in_dictionary, update_in_dictionary
from maggma.core import Sort, Store

from maggma.stores import GridFSStore, MongoStore, MongoURIStore, S3Store
from pymatgen import loadfn


T = TypeVar("T", bound="AtomateStore")


class AtomateStore(Store):
    """
    Special store intended to allow pushing and pulling documents into multiple stores.
    """

    def __init__(
        self,
        docs_store: MongoStore,
        data_store: Store,
        save: Union[Hashable, List[Hashable]] = None,
        load: Union[Hashable, List[Hashable]] = None,
        document_model: Type[BaseModel] = None,
        docs_collection_name: Optional[str] = None,
    ):
        """
        Initialize an AtomateStore.

        Args:
            docs_store: MongoStore or MongoURIStore store for basic documents.
            data_store: Maggma store for large data objects such as charge densities and
                band structures.
            save: List of keys to save in the data store when uploading documents.
            load: List of keys to load from the data store when querying.
            document_model: Pydantic document model to cast returned documents to.
                This allows enforcing a schema for valid document responses.
            docs_collection_name: Override the collection name of the docs_store.
        """
        self.docs_store = docs_store
        self.data_store = data_store
        self.document_model = document_model
        self.docs_collection_name = docs_collection_name
        self.docs_store.collection_name = docs_collection_name

        if not save:
            save = []
        elif not isinstance(save, (list, tuple)):
            save = (save,)
        self.save = save

        if not load:
            load = []
        elif not isinstance(load, (list, tuple)):
            load = (load,)
        self.load = load

        kwargs = {
            k: getattr(docs_store, k)
            for k in ("key", "last_updated_field", "last_updated_type")
        }
        super(AtomateStore, self).__init__(**kwargs)

    def name(self) -> str:
        """Return a string representing this data source."""
        return f"AtomateStore-{self.docs_store.name}"

    def connect(self, force_reset: bool = False):
        """
        Connect to the source data.

        Args:
            force_reset: Whether to reset the connection or not.
        """
        self.docs_store.connect(force_reset=force_reset)
        self.data_store.connect(force_reset=force_reset)

    def close(self):
        """Closes any connections."""
        self.docs_store.close()
        self.data_store.close()

    def count(self, criteria: Optional[Dict] = None) -> int:
        """
        Counts the number of documents matching the query criteria.

        Args:
            criteria: PyMongo filter for documents to count in.
        """
        return self.docs_store.count(criteria=criteria)

    def query(
        self,
        criteria: Optional[Dict] = None,
        properties: Union[Dict, List, None] = None,
        sort: Optional[Dict[str, Union[Sort, int]]] = None,
        skip: int = 0,
        limit: int = 0,
        load: Optional[Union[Hashable, List[Hashable]]] = None,
    ) -> Optional[Iterator[Union[Dict, BaseModel]]]:
        """
        Queries the Atomate Store for documents.

        Args:
            criteria: PyMongo filter for documents to search in.
            properties: Properties to return in grouped documents.
            sort: Dictionary of sort order for fields. Keys are field names and values
                are 1 for ascending or -1 for descending.
            skip: Number documents to skip.
            limit: Limit on total number of documents returned.
            load: List of keys to load from the data store.

        Returns:
            The documents. If self.document_model has been set, the documents will be
            cast to the document model.
        """
        if load is None:
            load = self.load
        elif not isinstance(load, (tuple, list)):
            load = [load]
        load = [o.value if isinstance(o, Enum) else o for o in load]

        docs = self.docs_store.query(
            criteria=criteria, properties=properties, sort=sort, skip=skip, limit=limit
        )

        for doc in docs:
            if load:
                object_info = find_in_dictionary(doc, load)

                data_criteria = {"_id": {"$in": list(object_info.values())}}
                objects = self.data_store.query(
                    criteria=data_criteria, properties=["_id", "data"]
                )
                object_map = {o["_id"]: o["data"] for o in objects}

                to_insert = {loc: object_map[oid] for loc, oid in object_info.items()}
                update_in_dictionary(doc, to_insert)

            if self.document_model:
                doc = self.document_model(**doc)

            yield doc

    def query_one(
        self,
        criteria: Optional[Dict] = None,
        properties: Union[Dict, List, None] = None,
        sort: Optional[Dict[str, Union[Sort, int]]] = None,
        load: Optional[Union[Hashable, List[Hashable]]] = None,
    ) -> Optional[Union[Dict, BaseModel]]:
        """
        Queries the Store for a single document.

        Args:
            criteria: PyMongo filter for documents to search.
            properties: Properties to return in the document.
            sort: Dictionary of sort order for fields. Keys are field names and values
                are 1 for ascending or -1 for descending.
            load: List of keys to load from the data store.

        Returns:
            The document. If self.document_model has been set, the document will be
            cast to the document model.
        """
        docs = self.query(
            criteria=criteria, properties=properties, load=load, sort=sort, limit=1
        )
        return next(docs, None)

    def update(
        self,
        docs: Union[List[Dict], Dict, BaseModel, List[BaseModel]],
        key: Union[List, str, None] = None,
        save: Optional[Union[Hashable, List[Hashable]]] = None,
    ):
        """
        Update documents into the Store

        Args:
            docs: The document or list of documents to update.
            key: Field name(s) to determine uniqueness for a document, can be a list of
                multiple fields, a single field, or None if the Store's key field is to
                be used.
            save: List of keys to save in the data store when uploading documents.
        """
        if save is None:
            save = self.save
        elif not isinstance(save, (tuple, list)):
            save = [save]
        save = [o.value if isinstance(o, Enum) else o for o in save]

        if not isinstance(docs, list):
            docs = [docs]

        blob_data = []
        dict_docs = []
        for doc in docs:
            if isinstance(doc, BaseModel):
                doc = doc.dict()
            doc["_id"] = ObjectId
            dict_docs.append(doc)

            if save:
                # first extract all objects from the doc and replace them with ObjectIDs
                object_info = find_in_dictionary(doc, save)
                object_ids = {k: ObjectId() for k in object_info.keys()}
                update_in_dictionary(doc, object_ids)

                # Now format blob data for saving in the data_store
                for loc, data in object_info.items():
                    blob = {"data": data, "_id": object_ids[loc], "doc_id": doc["_id"]}

                    # Create the metadata for data blobs, this could be an option to set
                    # metadata dynamically?
                    if "task_id" in doc:
                        blob["metadata"] = {"task_id": doc["task_id"]}
                    blob_data.append(blob)

        self.docs_store.update(dict_docs, key=key)

        if save:
            tids = all(
                ["metadata" in b and "task_id" in b["metadata"] for b in blob_data]
            )
            blob_keys = ["_id", "metadata.task_id"] if tids else ["_id"]
            self.data_store.update(blob_data, key=blob_keys)

    def ensure_index(
        self, key: str, unique: bool = False, docs_store: bool = True
    ) -> bool:
        """
        Tries to create an index on and return true if it succeeded.

        Args:
            key: Single key to index.
            unique: Whether or not this index contains only unique keys.
            docs_store: Whether the index is created in the docs_store. If False, the
                index will be created in the data_store.

        Returns:
            Whether the index exists/was created correctly.
        """
        if docs_store:
            return self.docs_store.ensure_index(key, unique=unique)
        else:
            return self.data_store.ensure_index(key, unique=unique)

    def groupby(
        self,
        keys: Union[List[str], str],
        criteria: Optional[Dict] = None,
        properties: Union[Dict, List, None] = None,
        sort: Optional[Dict[str, Union[Sort, int]]] = None,
        skip: int = 0,
        limit: int = 0,
        load: Optional[Union[Hashable, List[Hashable]]] = None,
    ) -> Iterator[Tuple[Dict, List[Dict]]]:
        """
        Simple grouping function that will group documents by keys.

        Args:
            keys: Fields to group documents.
            criteria: PyMongo filter for documents to search in.
            properties: Properties to return in grouped documents
            sort: Dictionary of sort order for fields. Keys are field names and values
                are 1 for ascending or -1 for descending.
            skip: Number documents to skip.
            limit: Limit on total number of documents returned.
            load: List of keys to load from the data store.

        Returns:
            Generator returning tuples of (dict, list of docs).
        """
        keys = keys if isinstance(keys, list) else [keys]

        if isinstance(properties, dict):
            # make sure all keys are in properties...
            properties.update(dict(zip(keys, [1] * len(keys))))

        docs = self.query(
            properties=properties,
            criteria=criteria,
            sort=sort,
            skip=skip,
            limit=limit,
            load=load,
        )
        data = [doc for doc in docs if all(has(doc, k) for k in keys)]

        def grouping_keys(doc):
            return tuple(get(doc, k) for k in keys)

        for vals, group in groupby(sorted(data, key=grouping_keys), key=grouping_keys):
            doc = {}
            for k, v in zip(keys, vals):
                set_(doc, k, get(v, k))
            yield doc, list(group)

    def remove_docs(self, criteria: Dict):
        """
        Remove docs matching the criteria.

        Args:
            criteria: Criteria for documents to remove.
        """
        docs = self.query(criteria, properties=["_id"])
        doc_ids = [doc["_id"] for doc in docs]

        self.docs_store.remove_docs(criteria)
        self.data_store.remove_docs({"doc_id": {"$in": doc_ids}})

    @property
    @deprecated(message="This will be removed in the future")
    def collection(self):
        return self.docs_store.collection_name

    @classmethod
    def from_db_file(
        cls: Type[T], db_file: Union[str, Path], admin: Optional[bool] = True, **kwargs
    ) -> T:
        """
        Create AtomateStore from a database file.

        Multiple options are supported for the database file. The file should be in
        json or yaml format.

        The simplest format is a monty dumped version of the store, generated using:

        ```python
        from monty.serialization import dumpfn
        dumpfn("atomate_store.json", atomate_store)
        ```

        Alternatively, the format can be a dictionary containing the docs_store and
        data_store keys, with the values as the dictionary representation of those
        stores.

        Alternatively, the old atomate file format is supported. Here the file should
        contain the keys:
        - host (str): The hostname of the database.
        - port (int): The port used to access the database.
        - collection (str): The collection in which to store documents.
        - authsource (str, optional): Authorization source for connecting to the
          database.
        - host_uri (str, optional): URI string specifying the database and login
          details. If this is specified, any other authentication information will be
          ignored.
        - admin_user (str, optional): The username for an account with admin privileges.
        - admin_password (str, optional): The password for an account with admin
          privileges.
        - readonly_user (str, optional): The username for an account with read
          privileges only.
        - readonly_password (str, optional): The password for an account with read
          privileges only.
        - data_store_kwargs (dict, optional): Keyword arguments that determine where
          to store large calculation data (band structures, density of states, charge
          density etc). Leaving this argument blank indicates that GridFS will be used.
          Alternatively, the following stores and keyword arguments are supported:
        - docs_store_kwargs: Additional keyword arguments that are passed to the
          MongoStore (and, optionally, GridFSStore if no data_store_kwargs are given).

          **S3 Store kwargs**
          - bucket (str): The S3 bucket where the data is stored.
          - s3_profile (str): The S3 profile that contains the login information
            typically found at ~/.aws/credentials.
          - compress (bool): Whether compression is used.
          - endpoint_url (str): The URL used to access the S3 store.

        - data_store_prefix (str, optional): The prefix for the collection used
          for the datastore.

        Args:
            db_file: Path to the file containing the credentials.
            admin: Whether to use the admin user (only applicable to old atomate
                style login information).
            **kwargs: Additional keyword arguments that get passed to the AtomateStore
                constructor.

        Returns:
            An AtomateStore.
        """
        credentials = loadfn(db_file)

        if isinstance(credentials, AtomateStore):
            return credentials

        if "docs_store" in credentials and "data_store" in credentials:
            docs_store = credentials["docs_store"]
            data_store = credentials["data_store"]
        elif "docs_store" in credentials or "data_store" in credentials:
            raise ValueError(
                "Both or neither of docs_store and data_store must be specified."
            )
        else:
            docs_store = _get_docs_store(credentials, admin)
            data_store = _get_data_store(credentials, admin)

        return cls(docs_store, data_store, **kwargs)


def _get_docs_store(credentials: Dict[str, Any], admin: bool) -> MongoStore:
    """
    Get the docs store from mongodb credentials.

    See AtomateStore.from_db_file for supported mongo connection arguments.
    """

    collection_name = credentials.get("collection")
    return _get_mongo_like_store(collection_name, credentials, admin)


def _get_data_store(credentials: Dict[str, Any], admin: bool) -> Store:
    """
    Get the data store from database credentials.

    See AtomateStore.from_db_file for supported store types and connection arguments.
    """
    data_store_prefix = credentials.get("data_store_prefix", "atomate")
    collection_name = f"{data_store_prefix}_datastore"

    if "data_store_kwargs" not in credentials:
        # Use GridFS to store data
        if "host_uri" in credentials:
            raise ValueError("GridFS from URI specification not supported.")
        else:
            auth = _get_mongo_auth(credentials, admin)
            auth.update(credentials.get("mongo_store_kwargs", {}))
            return GridFSStore(
                credentials["database"], collection_name, compression=True, **auth
            )

    data_store_kwargs = credentials.get("data_store_kwargs")
    if "bucket" in data_store_kwargs:
        # Store is a S3 bucket
        index_collection_name = f"{collection_name}_index"
        index_store = _get_mongo_like_store(index_collection_name, credentials, admin)
        index_store.key = "fs_id"

        return S3Store(
            index=index_store, sub_dir=collection_name, key="fs_id", **data_store_kwargs
        )
    raise ValueError("Unsupported data store")


def _get_mongo_like_store(
    collection_name: str, credentials: Dict[str, Any], admin: bool
) -> Union[MongoStore, MongoURIStore]:
    """Get either a MongoStore or MongoURIStore from a collection and credentials."""
    mongo_store_kwargs = credentials.get("mongo_store_kwargs", {})

    if "host_uri" in credentials:
        return MongoURIStore(
            credentials["host_uri"],
            credentials.get("database", None),
            collection_name,
            **mongo_store_kwargs,
        )

    auth = _get_mongo_auth(credentials, admin)
    auth.update(mongo_store_kwargs)
    return MongoStore(credentials["database"], collection_name, **auth)


def _get_mongo_auth(credentials: Dict[str, Any], admin: bool) -> Dict[str, Any]:
    """Get mongo authentication kwargs from the credentials specification."""
    auth = {}
    if admin and "admin_user" not in credentials:
        raise ValueError(
            "Trying to use admin credentials, but no admin credentials are defined. "
            "Use admin=False if only read_only credentials are available."
        )

    if admin:
        auth["user"] = credentials.get("admin_user", "")
        auth["password"] = credentials.get("admin_password", "")
    else:
        auth["user"] = credentials.get("readonly_user", "")
        auth["password"] = credentials.get("readonly_password", "")

    # this way, we won't override the MongoStore defaults
    for key in ("host", "port"):
        if key in credentials:
            auth[key] = credentials["key"]

    auth["authsource"] = credentials.get("authsource", credentials["database"])
    return auth
