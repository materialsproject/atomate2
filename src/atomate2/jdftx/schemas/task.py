# mypy: ignore-errors

"""Core definition of a JDFTx Task Document."""

import logging
from pathlib import Path
from typing import Any, Optional, TypeVar, Union, type

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
        store_additional_json: bool = True,
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
            jdftxinput_file="inputs.in",
            jdftxoutput_file="output.out",
        )
        # task_files = _find_qchem_files(dir_name)

        # if len(task_files) == 0:
        #     raise FileNotFoundError("No JDFTx files found!")

        ### all logic for calcs_reversed ###
        # critic2 = {}
        # custom_smd = {}
        # calcs_reversed = []
        # for task_name, files in task_files.items():
        #     if task_name == "orig":
        #         continue
        #     else:
        #         calc_doc = Calculation.from_qchem_files(
        #             dir_name,
        #             task_name,
        #             **files,
        #             **qchem_calculation_kwargs,
        #         )
        #         calcs_reversed.append(calc_doc)
        # all_qchem_objects.append(qchem_objects)

        # lists need to be reversed so that newest calc is the first calc, all_qchem_objects are also reversed to match
        # calcs_reversed.reverse()

        # all_qchem_objects.reverse()

        ### Custodian stuff ###
        # custodian = _parse_custodian(dir_name)
        # additional_json = None
        # if store_additional_json:
        #     additional_json = _parse_additional_json(dir_name)
        #     for key, _ in additional_json.items():
        #         if key == "processed_critic2":
        #             critic2["processed"] = additional_json["processed_critic2"]
        #         elif key == "cpreport":
        #             critic2["cp"] = additional_json["cpreport"]
        #         elif key == "YT":
        #             critic2["yt"] = additional_json["yt"]
        #         elif key == "bonding":
        #             critic2["bonding"] = additional_json["bonding"]
        #         elif key == "solvent_data":
        #             custom_smd = additional_json["solvent_data"]

        # orig_inputs = (
        #     CalculationInput.from_qcinput(_parse_orig_inputs(dir_name))
        #     if _parse_orig_inputs(dir_name)
        #     else {}
        # )

        # dir_name = get_uri(dir_name)  # convert to full path

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
