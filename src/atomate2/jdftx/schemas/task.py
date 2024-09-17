# mypy: ignore-errors

"""Core definition of a JDFTx Task Document."""

import logging
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

from custodian.jdftx.jobs import JDFTxJob  # Waiting on Sophie's PR
from emmet.core.structure import StructureMetadata
from monty.serialization import loadfn
from pydantic import BaseModel, Field

from atomate2.jdftx.schemas.calculation import (
    Calculation,
    CalculationInput,
    CalculationOutput,
)
from atomate2.jdftx.schemas.enums import CalcType, JDFTxStatus, TaskType
from atomate2.jdftx.sets.base import FILE_NAMES
from atomate2.utils.datetime import datetime_str

__author__ = "Cooper Tezak <cooper.tezak@colorado.edu>"

logger = logging.getLogger(__name__)
_T = TypeVar("_T", bound="TaskDoc")
# _DERIVATIVE_FILES = ("GRAD", "HESS")


class CustodianDoc(BaseModel):
    corrections: Optional[list[Any]] = Field(
        None,
        title="Custodian Corrections",
        description="list of custodian correction data for calculation.",
    )

    job: Optional[Union[dict[str, Any], JDFTxJob]] = Field(
        None,
        title="Custodian Job Data",
        description="Job data logged by custodian.",
    )


class TaskDoc(StructureMetadata):
    """Calculation-level details about JDFTx calculations."""

    dir_name: Optional[Union[str, Path]] = Field(
        None, description="The directory for this JDFTx task"
    )

    task_type: Optional[Union[CalcType, TaskType]] = Field(
        None, description="the type of JDFTx calculation"
    )

    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp for this task document was last updated",
    )

    calc_inputs: Optional[CalculationInput] = Field(
        {}, description="JDFTx calculation inputs"
    )

    calc_outputs: Optional[CalculationOutput] = Field(
        None,
        description="JDFTx calculation outputs",
    )

    state: Optional[JDFTxStatus] = Field(
        None, description="State of this JDFTx calculation"
    )

    # implemented in VASP and Qchem. Do we need this?
    # it keeps a list of all calculations in a given task.
    # calcs_reversed: Optional[list[Calculation]] = Field(
    # None,
    # title="Calcs reversed data",
    # description="Detailed data for each JDFTx calculation contributing to the task document.",
    # )

    @classmethod
    def from_directory(
        cls: type[_T],
        dir_name: Union[Path, str],
        additional_fields: dict[str, Any] = None,
        **jdftx_calculation_kwargs,
    ) -> _T:
        """
        Create a task document from a directory containing JDFTx files.

        Parameters
        ----------
        dir_name
            The path to the folder containing the calculation outputs.
        store_additional_json
            Whether to store additional json files in the calculation directory.
        additional_fields
            dictionary of additional fields to add to output document.
        **qchem_calculation_kwargs
            Additional parsing options that will be passed to the
            :obj:`.Calculation.from_qchem_files` function.

        Returns
        -------
        TaskDoc
            A task document for the JDFTx calculation
        """
        logger.info(f"Getting task doc in: {dir_name}")

        additional_fields = {} if additional_fields is None else additional_fields
        dir_name = Path(dir_name)
        calc_doc = Calculation.from_files(
            dir_name=dir_name,
            jdftxinput_file=FILE_NAMES["in"],
            jdftxoutput_file=FILE_NAMES["out"],
        )

        doc = cls.from_structure(
            meta_structure=calc_doc.output.structure,
            dir_name=dir_name,
            calc_outputs=calc_doc.output,
            calc_inputs=calc_doc.input,
            # task_type=
            # state=_get_state()
        )

        doc = doc.model_copy(update=additional_fields)
        return doc


def get_uri(dir_name: Union[str, Path]) -> str:
    """
    Return the URI path for a directory.

    This allows files hosted on different file servers to have distinct locations.

    Parameters
    ----------
    dir_name : str or Path
        A directory name.

    Returns
    -------
    str
        Full URI path, e.g., "fileserver.host.com:/full/payj/of/fir_name".
    """
    import socket

    fullpath = Path(dir_name).absolute()
    hostname = socket.gethostname()
    try:
        hostname = socket.gethostbyaddr(hostname)[0]
    except (socket.gaierror, socket.herror):
        pass
    return f"{hostname}:{fullpath}"


def _parse_custodian(dir_name: Path) -> Optional[dict]:
    """
    Parse custodian.json file.

    Calculations done using custodian have a custodian.json file which tracks the makers
    performed and any errors detected and fixed.

    Parameters
    ----------
    dir_name
        Path to calculation directory.

    Returns
    -------
    Optional[dict]
        The information parsed from custodian.json file.
    """
    filenames = tuple(dir_name.glob("custodian.json*"))
    if len(filenames) >= 1:
        return loadfn(filenames[0], cls=None)
    return None


# TODO currently doesn't work b/c has_jdftx_completed method is not implemented
def _get_state(calc: Calculation) -> JDFTxStatus:
    """Get state from calculation document of JDFTx task."""
    if calc.has_jdftx_completed:
        return JDFTxStatus.SUCCESS
    return JDFTxStatus.FAILED
