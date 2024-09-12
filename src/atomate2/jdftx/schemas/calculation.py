"""Core definitions of a JDFTx calculation document."""

# mypy: ignore-errors

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure

from atomate2.jdftx.io.jdftxinfile import JDFTXInfile
from atomate2.jdftx.io.jdftxoutfile import JDFTXOutfile
from atomate2.jdftx.schemas.enums import TaskType

__author__ = "Cooper Tezak <cote3804@colorado.edu>"
logger = logging.getLogger(__name__)



class CalculationInput(BaseModel):
    """Document defining JDFTx calculation inputs."""

    # TODO Break out parameters into more explicit dataclass
    # fields.
    # Waiting on parsers to be finished.

    structure: Structure = Field(
        None, description="input structure to JDFTx calcualtion"
    )

    parameters: dict = Field(None, description="input tags in JDFTx in file")

    @classmethod
    def from_jdftxinput(cls, jdftxinput: JDFTXInfile) -> "CalculationInput":
        """
        Create a JDFTx InputDoc schema from a JDFTXInfile object.

        Parameters
        ----------
        jdftxinput
            A JDFTXInfile object.

        Returns
        -------
        CalculationInput
            The input document.
        """
        return cls(
            structure=jdftxinput.structure,
            parameters=jdftxinput.as_dict(),
        )


class CalculationOutput(BaseModel):
    """Document defining JDFTx calculation outputs."""

    # TODO Break out jdftxoutput into more dataclass fields instead of lumping
    # everything into parameters.
    # Waiting on parsers to be finished.

    structure: Optional[Structure] = Field(
        None,
        description="optimized geometry of the structure after calculation",
    )
    parameters: Optional[dict] = Field(
        None,
        description="Calculation input parameters",
    )

    @classmethod
    def from_jdftxoutput(cls, jdftxoutput: JDFTXOutfile) -> "CalculationOutput":
        """
        Create a JDFTx output document from a JDFTXOutfile object.

        Parameters
        ----------
        jdftxoutput
            A JDFTXOutfile object.

        Returns
        -------
        CalculationOutput
            The output document.
        """
        optimized_structure = jdftxoutput.structure
        # jstrucs = jdftxoutput.jstrucs
        # TODO write method on JoutStructures to get trajectory
        # and handle optional storing of trajectory
        jsettings_fluid = jdftxoutput.jsettings_fluid
        jsettings_electronic = jdftxoutput.jsettings_electronic
        jsettings_lattice = jdftxoutput.jsettings_lattice
        jsettings_ionic = jdftxoutput.jsettings_ionic
        return cls(
            structure=optimized_structure,
            **asdict(jsettings_fluid),
            **asdict(jsettings_electronic),
            **asdict(jsettings_lattice),
            **asdict(jsettings_ionic),
        )


class Calculation(BaseModel):
    """Full JDFTx calculation inputs and outputs."""

    dir_name: str = Field(None, description="The directory for this JDFTx calculation")
    input: CalculationInput = Field(
        None, description="JDFTx input settings for the calculation"
    )
    output: CalculationOutput = Field(
        None, description="The JDFTx calculation output document"
    )

    # TODO implement these after parser is complete
    # task_name: str = Field(
    #     None,
    #     description="Name of task given by custodian (e.g. opt1, opt2, freq1, freq2)",
    # )
    # task_type: TaskType = Field(
    #     None,
    #     description="Calculation task type like Single Point,
    #       Geometry Optimization, Frequency...",
    # )
    # calc_type: Union[CalcType, str] = Field(
    #     None,
    #     description="Combination of calc type and task type",
    # )
    # completed_at: str = Field(
    #     None, description="Timestamp for when the calculation was completed"
    # )
    # has_jdftx_completed: Union[JDFTxStatus, bool] = Field(
    #     None, description="Whether JDFTx calculated the calculation successfully"
    # )
    # We'll only need this if we are using Custodian
    # to do error handling and calculation resubmission, which
    # will create additional files and paths
    # output_file_paths: dict[str, Union[str, Path, dict[s tr, Path]]] = Field(
    #     None,
    #     description="Paths (relative to dir_name) of the
    #       JDFTx output files associated with this calculation",
    # )

    @classmethod
    def from_files(
        cls,
        dir_name: Union[Path, str],
        jdftxinput_file: Union[Path, str],
        jdftxoutput_file: Union[Path, str],
        jdftxinput_kwargs: Optional[dict] = None,
        jdftxoutput_kwargs: Optional[dict] = None,
        # task_name  # do we need task names? These are created by Custodian
    ) -> "Calculation":
        """
        Create a QChem calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name
            The directory containing the JDFTx calculation outputs.
        jdftxinput_file
            Path to the JDFTx in file relative to dir_name.
        jdftxoutput_file
            Path to the JDFTx out file relative to dir_name.
        jdftxinput_kwargs
            Additional keyword arguments that will be passed to the
            :obj:`.JDFTXInFile.from_file` method
        jdftxoutput_kwargs
            Additional keyword arguments that will be passed to the
            :obj:`.JDFTXOutFile.from_file` method

        Returns
        -------
        Calculation
            A JDFTx calculation document.
        """
        dir_name = Path(dir_name)
        jdftxinput_file = dir_name / jdftxinput_file
        jdftxoutput_file = dir_name / jdftxoutput_file

        jdftxinput_kwargs = jdftxinput_kwargs if jdftxinput_kwargs else {}
        jdftxinput = JDFTXInfile.from_file(jdftxinput_file, **jdftxinput_kwargs)

        jdftxoutput_kwargs = jdftxoutput_kwargs if jdftxoutput_kwargs else {}
        jdftxoutput = JDFTXOutfile.from_file(jdftxoutput_file, **jdftxoutput_kwargs)

        # completed_at = str(datetime.fromtimestamp(qcoutput_file.stat().st_mtime))
        # TODO parse times from JDFTx out file and implement them here

        input_doc = CalculationInput.from_jdftxinput(jdftxinput)
        output_doc = CalculationOutput.from_jdftxoutput(jdftxoutput)

        # TODO implement the get method on the output parser.
        # has_jdftx_completed = (
        #     JDFTxStatus.SUCCESS
        #     if jdftxoutput.get("completed")
        #     else JDFTxStatus.FAILED
        # )

        return cls(
            dir_name=str(dir_name),
            input=input_doc,
            output=output_doc,
            # TODO implement these methods if we want them
            # jdftx_version=qcoutput.data["version"],
            # has_jdftx_completed=has_qchem_completed,
            # completed_at=completed_at,
            # task_type=task_type(input_doc),
            # calc_type=calc_type(input_doc, validate_lot=validate_lot),
            # task_name=
        )


# def solvent(
#     parameters: CalculationInput,
#     validate_lot: bool = True,
#     custom_smd: Optional[str] = None,
# ) -> str:
#     """
#     Return the solvent used for this calculation.

#     Args:
#         parameters: dict of Q-Chem input parameters
#         custom_smd: (Optional) string representing SMD parameters for a
#         non-standard solvent
#     """
# TODO adapt this for JDFTx


def task_type(
    parameters: CalculationInput,
) -> TaskType:
    """Return TaskType for JDFTx calculation."""
    if parameters.job_type == "sp":
        return TaskType("Single Point")
    if parameters.job_type == "opt":
        return TaskType("Geometry Optimization")
    if parameters.job_type == "freq":
        return TaskType("Frequency")
    if parameters.job_type == "md":
        return TaskType("Molecular Dynamics")

    return TaskType("Unknown")


# def calc_type(
#     parameters: CalculationInput,
#     validate_lot: bool = True,
#     special_run_type: Optional[str] = None,
# ) -> CalcType:
#     """
#     Determine the calc type.

#     Args:
#         parameters: CalculationInput parameters
#     """
#     pass
# TODO implement this
