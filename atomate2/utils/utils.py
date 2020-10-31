import socket

from pathlib import Path

from typing import Union, Dict, Hashable, Any, Tuple, List


def get_uri(dir_name: Union[str, Path]) -> str:
    """
    Returns the URI path for a directory. This allows files hosted on different file
    servers to have distinct locations.

    Args:
        dir_name: A directory name.

    Returns:
        Full URI path, e.g., fileserver.host.com:/full/path/of/dir_name
    """
    fullpath = Path(dir_name).absolute()
    hostname = socket.gethostname()
    try:
        hostname = socket.gethostbyaddr(hostname)[0]
    except socket.gaierror:
        pass
    return "{}:{}".format(hostname, fullpath)


def find_in_dictionary(
    d: Dict[Hashable, Any], keys: Union[Hashable, List[Hashable]]
) -> Dict[Tuple, Any]:
    """
    Find the route to and values of keys in a dictionary.

    This function works on nested dictionaries and those containing lists or tuples.

    For example:

    ```python
    d = {
        "a": [0, {"b": 1, "x": 2}],
        "c": {
            "d": {"x": 3}
        }
    }
    find_in_dictionary(d, ["b", "x"])

    # returns: {('a', 1, 'x'): 2, ('a', 1, 'b'): 1, ('c', 'd', 'x'): 3}
    ```

    Args:
        d: A dictionary.
        keys: A key or list of keys to find.

    Returns:
        A dictionary mapping the route to the keys and the value at that route.
    """
    if not isinstance(keys, list):
        keys = [keys]
    found_items = {}

    def _lookup(obj, path=None):
        if path is None:
            path = ()

        if isinstance(obj, dict):
            for key in keys:
                if key in obj:
                    found_items[path + (key,)] = obj[key]

            for k, v in obj.items():
                _lookup(v, path + (k,))

        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _lookup(v, path + (i,))

    _lookup(d)
    return found_items


def update_in_dictionary(d: Dict[Hashable, Any], updates: Dict[Tuple, Any]):
    """
    Update a dictionary (in place) at specific locations with a new values.

    This function works on nested dictionaries and those containing lists or tuples.

    For example:

    ```python
    d = {
        "a": [0, {"b": 1, "x": 2}],
        "c": {
            "d": {"x": 3}
        }
    }
    update_in_dictionary(d, {('a', 1, 'x'): 100, ('c', 'd', 'x'): 100})

    # d = {
    #     "a": [0, {"b": 1, "x": 100}],
    #     "c": {
    #         "d": {"x": 100}
    #     }
    # }
    ```

    Args:
        d: A dictionary to update.
        updates: The updates to perform, as a dictionary of {location: update}.
    """
    for loc, update in updates.items():
        pos = d
        for idx in loc[:-1]:
            pos = pos[idx]
        pos[loc[-1]] = update
