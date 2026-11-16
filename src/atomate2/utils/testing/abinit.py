"""Utilities for testing ABINIT calculations.

This module defines the format of the abiget.json.gz reference files that mock
the abipy abiget_* helpers: while the abinit tests replay them from reference
data (see tests/abinit/conftest.py), the abinit-generate-reference CLI command
captures them during a real run (see atomate2.cli.dev.abinit_generate_reference).
Both call sites share the helpers in this module, so the format has a single
definition.

The reader uses monty's loadfn (which handles .gz transparently); the writer
uses a deterministic gzip JSON format (mtime=0, sort_keys=True) to ensure
byte-identical files across record runs.
"""

from __future__ import annotations

import gzip
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping


logger = logging.getLogger(__name__)

# The abipy AbinitInput methods whose results are recorded to the abiget.json.gz
# reference files. These helpers otherwise run the real abinit binary from
# within jobs that call them (e.g. generate_perts).
ABIGET_METHOD_NAMES = (
    "abiget_irred_ddeperts",
    "abiget_irred_dteperts",
    "abiget_irred_phperts",
    "abiget_ibz",
)


def iter_abiget_methods() -> list[str]:
    """
    List the abiget_* methods of abipy's AbinitInput.

    Class-level introspection (inherited methods included): the abiget_ prefix
    defines the namespace of the AbinitInput methods that build and run their
    own abinit task, and is the contract for what is recorded and mocked.
    """
    from abipy.abio.inputs import AbinitInput

    return [
        name
        for name in dir(AbinitInput)
        if name.startswith("abiget_") and callable(getattr(AbinitInput, name))
    ]


def _make_unregistered_abiget_stub(method_name: str, hint: str) -> Callable[..., Any]:
    """
    Create the stub installed for an abiget method outside ABIGET_METHOD_NAMES.

    Parameters
    ----------
    method_name
        Name of the unregistered method.
    hint
        Consumer-specific remediation hint included in the error message.
    """

    def _unregistered_stub(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"AbinitInput.{method_name} is not registered in ABIGET_METHOD_NAMES "
            f"and is therefore neither recorded nor mocked. {hint}"
        )

    return _unregistered_stub


@contextmanager
def patch_abiget_methods(
    wrappers: Mapping[str, Callable[..., Any]], unregistered_hint: str
) -> Iterator[None]:
    """
    Patch all abiget_* methods of abipy's AbinitInput for the duration of the context.

    Methods registered in ABIGET_METHOD_NAMES get the caller-supplied wrapper;
    any other abiget_* method gets a stub that raises, so that an unregistered
    call fails loudly instead of running the real abinit inside a recorded or
    mocked run. All original methods are restored on exit.

    Parameters
    ----------
    wrappers
        Wrapper to install, keyed by method name, for each method in
        ABIGET_METHOD_NAMES.
    unregistered_hint
        Remediation hint embedded in the error raised by the stub installed
        for unregistered abiget_* methods.
    """
    from abipy.abio.inputs import AbinitInput

    original_methods = {
        name: getattr(AbinitInput, name) for name in iter_abiget_methods()
    }
    try:
        for name in original_methods:
            if name in ABIGET_METHOD_NAMES:
                setattr(AbinitInput, name, wrappers[name])
            else:
                setattr(
                    AbinitInput,
                    name,
                    _make_unregistered_abiget_stub(name, unregistered_hint),
                )
        yield
    finally:
        for name, original in original_methods.items():
            setattr(AbinitInput, name, original)


def canon_value(value: Any) -> Any:
    """
    Convert numpy values to plain Python objects with a version-stable repr.

    Plain conversion is required so that reference keys are identical across
    numpy versions (repr of numpy scalars changed with numpy 2.0).
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def abiget_call_key(method_name: str, args: tuple, kwargs: dict) -> str:
    """
    Get the canonical key of the distinguishing call arguments for abiget methods.

    workdir and manager never take part in the key, as they only define where
    the real abinit task runs.

    The per-method keys are:
    - "abiget_irred_phperts": the q-point, e.g. "(0.5, 0.0, 0.0)"
    - "abiget_ibz": comma-joined "name=value" pairs of the call kwargs,
      sorted by name, e.g. "kptopt=3, shiftk=(0, 0, 0)"
    - "abiget_irred_ddeperts" and "abiget_irred_dteperts": "default" (called
      once, without distinguishing arguments)

    Parameters
    ----------
    method_name
        Name of the abipy AbinitInput method being called.
    args
        Positional call arguments (qpt can be passed positionally to
        abiget_irred_phperts).
    kwargs
        Keyword call arguments.
    """
    if method_name == "abiget_irred_phperts":
        qpt = kwargs.get("qpt", args[0] if args else None)
        if qpt is None:
            raise ValueError(
                "The q-point argument is required to compute the reference "
                f"key of {method_name}."
            )
        return str(tuple(canon_value(q) for q in qpt))
    if method_name == "abiget_ibz":
        return ", ".join(
            f"{k}={canon_value(kwargs[k])}"
            for k in sorted(kwargs)
            if k not in ("workdir", "manager")
        )
    return "default"


def abiget_results_equal(method_name: str, a: Any, b: Any) -> bool:
    """
    Compare two canonically serialized abiget results for equality.

    Parameters
    ----------
    method_name
        Name of the abipy AbinitInput method that produced the results.
    a, b
        Serialized results in the reference file format.
    """
    if method_name == "abiget_ibz":
        return np.array_equal(
            np.asarray(a["points"]), np.asarray(b["points"])
        ) and np.array_equal(np.asarray(a["weights"]), np.asarray(b["weights"]))
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def dump_abiget_result(method_name: str, result: Any) -> object:
    """
    Convert the return value of an abipy abiget helper to JSON-safe data.

    The load counterpart load_abiget_result mirrors these transformations.

    Parameters
    ----------
    method_name
        Name of the abipy AbinitInput method that produced the result.
    result
        Return value of the abipy method.
    """
    if method_name == "abiget_ibz":
        # abiget_ibz returns a namedtuple of numpy arrays: store the arrays as
        # plain lists, the load path rebuilds the namedtuple from them.
        return {
            "points": np.asarray(result.points).tolist(),
            "weights": np.asarray(result.weights).tolist(),
        }
    # The abiget_irred_*perts methods return lists of monty AttrDict (a dict
    # subclass): store them as plain dicts, the load path restores them as
    # AttrDicts so that attribute access (e.g. pert.i1pert) keeps working.
    return [dict(pert) for pert in result]


def load_abiget_result(method_name: str, data: Any) -> Any:
    """
    Rebuild the return value of an abipy abiget helper from reference data.

    Parameters
    ----------
    method_name
        Name of the abipy AbinitInput method that produced the result.
    data
        JSON-safe data produced by dump_abiget_result.
    """
    if method_name == "abiget_ibz":
        from typing import NamedTuple

        class Ibz(NamedTuple):
            points: np.ndarray
            weights: np.ndarray

        return Ibz(points=np.array(data["points"]), weights=np.array(data["weights"]))
    from monty.collections import AttrDict

    return [AttrDict(**pert) for pert in data]


def dump_abiget_gz(data: dict, filename: Path) -> None:
    """
    Write abiget reference data as deterministic gzip JSON (zeroed header).

    Python's gzip writer embeds the current mtime in the gzip header, so two
    record runs with identical data would otherwise produce different bytes.
    Together with sort_keys=True this makes two record runs byte-identical.
    """
    payload = json.dumps(data, sort_keys=True).encode()
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_bytes(gzip.compress(payload, mtime=0))


# No custom load_abiget_gz loader: use monty's loadfn, which handles .gz transparently.


class AbigetRecorder:
    """
    Record abipy abiget_* results into deterministic abiget.json.gz files.

    The recorder tracks the reference files already (re-)written during the
    current run (a test fixture or a CLI reference run): the first record call
    per file resets it, so re-recording replaces rather than appends.
    """

    def __init__(self) -> None:
        self._recorded_files: set[str] = set()

    def reset(self) -> None:
        """Forget the files recorded so far, so the next record call resets them."""
        self._recorded_files.clear()

    def record(
        self, ref_file: Path, method_name: str, key: str, result_data: Any
    ) -> None:
        """
        Merge the serialized result of an abiget call into a reference file.

        A duplicate call with the same method and key is legitimate only if
        the abipy method is deterministic: identical results are recorded once
        (with an info log), different results raise instead of silently
        overwriting.

        Parameters
        ----------
        ref_file
            Path to the abiget.json.gz reference file.
        method_name
            Name of the abipy AbinitInput method that was called.
        key
            Canonical call key (see abiget_call_key).
        result_data
            Serialized result (see dump_abiget_result).
        """
        ref_file = Path(ref_file)
        if str(ref_file) in self._recorded_files and ref_file.exists():
            from monty.serialization import loadfn

            data = loadfn(ref_file)
        else:
            # First record call for this file in the current run: reset it.
            data = {}
        if key in data.get(method_name, {}):
            # Duplicate call with the same distinguishing arguments.
            if not abiget_results_equal(
                method_name, result_data, data[method_name][key]
            ):
                raise ValueError(
                    f"Inconsistent duplicate call to {method_name} with key "
                    f"{key!r}: recorded {data[method_name][key]} differs "
                    f"from the newly computed {result_data}."
                )
            logger.info(
                f"duplicate call to {method_name} with key {key}, "
                "identical result - recorded once"
            )
            return
        data.setdefault(method_name, {})[key] = result_data
        dump_abiget_gz(data, ref_file)
        self._recorded_files.add(str(ref_file))
