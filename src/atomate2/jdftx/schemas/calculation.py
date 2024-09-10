"""Core definitions of a JDFTx calculation document."""

# mypy: ignore-errors

import logging
import re
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from emmet.core.qchem.calc_types import CalcType, LevelOfTheory, TaskType
from emmet.core.qchem.calc_types.calc_types import BASIS_SETS, FUNCTIONALS


# from emmet.core.qchem.calc_types.em_utils import (
#     level_of_theory,
#     task_type,
#     calc_type,
# )
from atomate2.jdftx.schemas.task import JDFTxStatus
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.qchem.inputs import QCInput
from pymatgen.io.qchem.outputs import QCOutput

from atomate2.jdftx.io.JDFTXInfile import JDFTXInfile, JDFTXStructure
from atomate2.jdftx.io.JDFTXOutfile import JDFTXOutfile

functional_synonyms = {
    "b97mv": "b97m-v",
    "b97mrv": "b97m-rv",
    "wb97xd": "wb97x-d",
    "wb97xd3": "wb97x-d3",
    "wb97xv": "wb97x-v",
    "wb97mv": "wb97m-v",
}

smd_synonyms = {
    "DIELECTRIC=7,230;N=1,410;ALPHA=0,000;BETA=0,859;GAMMA=36,830;PHI=0,000;PSI=0,000": "diglyme",
    "DIELECTRIC=18,500;N=1,415;ALPHA=0,000;BETA=0,735;GAMMA=20,200;PHI=0,000;PSI=0,000": "3:7 EC:EMC",
}

__author__ = "Rishabh D. Guha <rdguha@lbl.gov>"
logger = logging.getLogger(__name__)

# class QChemObject(ValueEnum):
# Not sure but can we have something like GRAD and HESS
# as QChem data objects


class CalculationInput(BaseModel):
    """
    Document defining JDFTx calculation inputs.
    """

    structure: Structure = Field(
        None, description="input structure to JDFTx calcualtion"
    )

    parameters: Dict = Field(
        None, description="input tags in JDFTx in file"
    )


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

    structure: Optional[Structure] = Field(
        None,
        description="optimized geometry of the structure after calculation",
    )
    parameters: Optional[Dict] = Field( #TODO currently (I think) redundant with structure in these parameters
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
        electronic_output = jdftxoutput.electronic_output

        return cls(
            structure=optimized_structure,
            Ecomponents=jdftxoutput.Ecomponents,
            **electronic_output,
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

    #TODO implement these after parser is complete
    # task_name: str = Field(
    #     None,
    #     description="Name of task given by custodian (e.g. opt1, opt2, freq1, freq2)",
    # )
    # task_type: TaskType = Field(
    #     None,
    #     description="Calculation task type like Single Point, Geometry Optimization. Frequency...",
    # )
    # calc_type: Union[CalcType, str] = Field(
    #     None,
    #     description="Combination dict of LOT + TaskType: B97-D/6-31g*/VACUUM Geometry Optimization",
    # )
    # completed_at: str = Field(
    #     None, description="Timestamp for when the calculation was completed"
    # )
    # has_jdftx_completed: Union[JDFTxStatus, bool] = Field(
    #     None, description="Whether JDFTx calculated the calculation successfully"
    # )
    # We'll only need this if we are using Custodian to do error handling and calculation resubmission
    # output_file_paths: Dict[str, Union[str, Path, Dict[s tr, Path]]] = Field(
    #     None,
    #     description="Paths (relative to dir_name) of the QChem output files associated with this calculation",
    # )

    @classmethod
    def from_files(
        cls,
        dir_name: Union[Path, str],
        jdftxinput_file: Union[Path, str],
        jdftxoutput_file: Union[Path, str],
        jdftxinput_kwargs: Optional[Dict] = None,
        jdftxoutput_kwargs: Optional[Dict] = None,
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
            Additional keyword arguments that will be passed to the :obj:`.JDFTXInFile.from_file` method
        jdftxoutput_kwargs
            Additional keyword arguments that will be passed to the :obj:`.JDFTXOutFile.from_file` method

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

            #TODO implement these methods if we want them
            # jdftx_version=qcoutput.data["version"],
            # has_jdftx_completed=has_qchem_completed,
            # completed_at=completed_at,
            # task_type=task_type(input_doc),
            # calc_type=calc_type(input_doc, validate_lot=validate_lot),
            # task_name=
        )

def level_of_theory(
    parameters: CalculationInput, validate_lot: bool = True
) -> LevelOfTheory:
    """

    Returns the level of theory for a calculation,
    based on the input parameters given to Q-Chem

    Args:
        parameters: Dict of Q-Chem input parameters

    """
    funct_raw = parameters.rem.get("method")
    basis_raw = parameters.rem.get("basis")

    if funct_raw is None or basis_raw is None:
        raise ValueError(
            'Method and basis must be included in "rem" section ' "of parameters!"
        )

    disp_corr = parameters.rem.get("dft_d")

    if disp_corr is None:
        funct_lower = funct_raw.lower()
        funct_lower = functional_synonyms.get(funct_lower, funct_lower)
    else:
        # Replace Q-Chem terms for D3 tails with more common expressions
        disp_corr = disp_corr.replace("_bj", "(bj)").replace("_zero", "(0)")
        funct_lower = f"{funct_raw}-{disp_corr}"

    basis_lower = basis_raw.lower()

    solvent_method = parameters.rem.get("solvent_method", "").lower()

    if solvent_method == "":
        solvation = "VACUUM"
    elif solvent_method in ["pcm", "cosmo"]:
        solvation = "PCM"
    # TODO: Add this once added into pymatgen and atomate
    # elif solvent_method == "isosvp":
    #     if parameters.get("svp", {}).get("idefesr", 0):
    #         solvation = "CMIRS"
    #     else:
    #         solvation = "ISOSVP"
    elif solvent_method == "smd":
        solvation = "SMD"
    else:
        raise ValueError(f"Unexpected implicit solvent method {solvent_method}!")

    if validate_lot:
        functional = [f for f in FUNCTIONALS if f.lower() == funct_lower]
        if not functional:
            raise ValueError(f"Unexpected functional {funct_lower}!")

        functional = functional[0]

        basis = [b for b in BASIS_SETS if b.lower() == basis_lower]
        if not basis:
            raise ValueError(f"Unexpected basis set {basis_lower}!")

        basis = basis[0]

        lot = f"{functional}/{basis}/{solvation}"

        return LevelOfTheory(lot)
    else:
        warnings.warn(
            "User has turned the validate flag off."
            "This can have downstream effects if the chosen functional and basis "
            "is not in the available sets of MP employed functionals and the user"
            "wants to include the TaskDoc in the MP infrastructure."
            "Users should ignore this warning if their objective is just to create TaskDocs",
            UserWarning,
            stacklevel=2,
        )
        functional = funct_lower
        basis = basis_lower
        lot = f"{functional}/{basis}/{solvation}"

        return lot


def solvent(
    parameters: CalculationInput,
    validate_lot: bool = True,
    custom_smd: Optional[str] = None,
) -> str:
    """
    Returns the solvent used for this calculation.

    Args:
        parameters: Dict of Q-Chem input parameters
        custom_smd: (Optional) string representing SMD parameters for a
        non-standard solvent
    """
    lot = level_of_theory(parameters, validate_lot=validate_lot)
    if validate_lot:
        solvation = lot.value.split("/")[-1]
    else:
        solvation = lot.split("/")[-1]

    if solvation == "PCM":
        # dielectric = float(parameters.get("solvent", {}).get("dielectric", 78.39))
        # dielectric = float(parameters.get("solvent", {}))
        # dielectric = getattr(parameters, "solvent", None)
        # dielectric_string = f"{dielectric.get('dielectric', '0.0'):.2f}".replace(".", ",")
        dielectric_string = getattr(parameters, "solvent", None)
        return f"DIELECTRIC= {dielectric_string}"
    # TODO: Add this once added into pymatgen and atomate
    # elif solvation == "ISOSVP":
    #     dielectric = float(parameters.get("svp", {}).get("dielst", 78.39))
    #     rho = float(parameters.get("svp", {}).get("rhoiso", 0.001))
    #     return f"DIELECTRIC={round(dielectric, 2)},RHO={round(rho, 4)}"
    # elif solvation == "CMIRS":
    #     dielectric = float(parameters.get("svp", {}).get("dielst", 78.39))
    #     rho = float(parameters.get("svp", {}).get("rhoiso", 0.001))
    #     a = parameters.get("pcm_nonels", {}).get("a")
    #     b = parameters.get("pcm_nonels", {}).get("b")
    #     c = parameters.get("pcm_nonels", {}).get("c")
    #     d = parameters.get("pcm_nonels", {}).get("d")
    #     solvrho = parameters.get("pcm_nonels", {}).get("solvrho")
    #     gamma = parameters.get("pcm_nonels", {}).get("gamma")
    #
    #     string = f"DIELECTRIC={round(dielectric, 2)},RHO={round(rho, 4)}"
    #     for name, (piece, digits) in {"A": (a, 6), "B": (b, 6), "C": (c, 1), "D": (d, 3),
    #                                   "SOLVRHO": (solvrho, 2), "GAMMA": (gamma, 1)}.items():
    #         if piece is None:
    #             piecestring = "NONE"
    #         else:
    #             piecestring = f"{name}={round(float(piece), digits)}"
    #         string += "," + piecestring
    #     return string
    elif solvation == "SMD":
        solvent = parameters.smx.get("solvent", "water")
        if solvent == "other":
            if custom_smd is None:
                raise ValueError(
                    "SMD calculation with solvent=other requires custom_smd!"
                )

            names = ["DIELECTRIC", "N", "ALPHA", "BETA", "GAMMA", "PHI", "PSI"]
            numbers = [float(x) for x in custom_smd.split(",")]

            string = ""
            for name, number in zip(names, numbers):
                string += f"{name}={number:.3f};"
            return string.rstrip(",").rstrip(";").replace(".", ",")
        else:
            return f"SOLVENT={solvent.upper()}"
    else:
        return "NONE"


def lot_solvent_string(
    parameters: CalculationInput,
    validate_lot: bool = True,
    custom_smd: Optional[str] = None,
) -> str:
    """
    Returns a string representation of the level of theory and solvent used for this calculation.

    Args:
        parameters: Dict of Q-Chem input parameters
        custom_smd: (Optional) string representing SMD parameters for a
        non-standard solvent
    """
    if validate_lot:
        lot = level_of_theory(parameters, validate_lot=validate_lot).value
    else:
        lot = level_of_theory(parameters, validate_lot=validate_lot)
    solv = solvent(parameters, custom_smd=custom_smd, validate_lot=validate_lot)
    return f"{lot}({solv})"


def task_type(
    parameters: CalculationInput, special_run_type: Optional[str] = None
) -> TaskType:
    if special_run_type == "frequency_flattener":
        return TaskType("Frequency Flattening Geometry Optimization")
    elif special_run_type == "ts_frequency_flattener":
        return TaskType("Frequency Flattening Transition State Geometry Optimization")

    if parameters.job_type == "sp":
        return TaskType("Single Point")
    elif parameters.job_type == "force":
        return TaskType("Force")
    elif parameters.job_type == "opt":
        return TaskType("Geometry Optimization")
    elif parameters.job_type == "ts":
        return TaskType("Transition State Geometry Optimization")
    elif parameters.job_type == "freq":
        return TaskType("Frequency Analysis")

    return TaskType("Unknown")


def calc_type(
    parameters: CalculationInput,
    validate_lot: bool = True,
    special_run_type: Optional[str] = None,
) -> CalcType:
    """
    Determines the calc type

    Args:
        parameters: CalculationInput parameters
    """
    tt = task_type(parameters, special_run_type=special_run_type).value
    if validate_lot:
        rt = level_of_theory(parameters, validate_lot=validate_lot).value
        return CalcType(f"{rt} {tt}")
    else:
        rt = level_of_theory(parameters, validate_lot=validate_lot)
        return str(f"{rt} {tt}")
