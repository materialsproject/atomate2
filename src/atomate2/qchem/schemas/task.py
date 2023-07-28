"""

Need to implement a Task Document equivalent. Possibly import from emmet
"""

""" Core definition of a Q-Chem Task Document """
import logging

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from pathlib import Path

from pydantic import BaseModel, Field
from pymatgen.core.structure import Molecule
from collections import OrderedDict

from emmet.core.structure import MoleculeMetadata
from emmet.core.task import BaseTaskDocument
from jobflow.utils import ValueEnum

from atomate2 import SETTINGS, __version__
from atomate2.qchem.schemas.calc_types import (
    LevelOfTheory,
    CalcType,
    TaskType,
    calc_type,
    level_of_theory,
    task_type,
    solvent,
    lot_solvent_string,
)
from atomate2.qchem.schemas.calculation import (
    Calculation,
    QChemObject,
    Status,
)

__author__ = "Evan Spotte-Smith <ewcspottesmith@lbl.gov>"

logger = logging.getLogger(__name__)
_T = TypeVar("_T", bound="TaskDocument")
_GRAD_HESS_FILES = ("GRAD", "HESS")


class QChemStatus(ValueEnum):
    """
    Q-Chem Calculation State
    """

    SUCCESS = "successful"
    FAILED = "unsuccessful"


class InputSummary(BaseModel):
    """Summary of inputs for a QChem calculation"""

    molecule: Molecule = Field(None, description="The input molecule geometry")

    rem: Dict[str, Any] = Field(
        None, description="A dictionary of all the input parameters of the QChem input file"
    )

    pcm: Dict[str, Any] = Field(
        None, description="A dictionary of the pcm solvent section defining its behavior"
    )

    solvent: Dict[str, Any] = Field(
        None, description="A dictionary defining the solvent parameters used with PCM"
    )

    smx: Dict[str, Any] = Field(
        None, description="A dictionary defining solvent parameters used with the SMD solvent method"
    )

    geom_opt: Dict[str, Any] = Field(
        None, description="A dictionary of input parameters for the geom_opt section of the QChem input file"
    )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "molecule": self.molecule,
            "rem": self.rem,
            "pcm": self.pcm,
            "solvent": self.solvent,
            "smx": self.smx,
            "geom_opt": self.geom_opt,
        }


class OutputSummary(BaseModel):
    """
    Summary of an output for a Q-Chem calculation
    """

    initial_molecule: Molecule = Field(None, description="Input Molecule object")
    optimized_molecule: Molecule = Field(None, description="Optimized Molecule object")

    final_energy: float = Field(
        None, description="Final electronic energy for the calculation (units: Hartree)"
    )
    enthalpy: float = Field(
        None, description="Total enthalpy of the molecule (units: kcal/mol)"
    )
    entropy: float = Field(
        None, description="Total entropy of the molecule (units: cal/mol-K"
    )

    mulliken: List[Any] = Field(
        None, description="Mulliken atomic partial charges and partial spins"
    )
    resp: List[float] = Field(
        None,
        description="Restrained Electrostatic Potential (RESP) atomic partial charges",
    )
    nbo: Dict[str, Any] = Field(
        None, description="Natural Bonding Orbital (NBO) output"
    )

    frequencies: List[float] = Field(
        None, description="Vibrational frequencies of the molecule (units: cm^-1)"
    )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "initial_molecule": self.initial_molecule,
            "optimized_molecule": self.optimized_molecule,
            "final_energy": self.final_energy,
            "enthalpy": self.enthalpy,
            "entropy": self.entropy,
            "mulliken": self.mulliken,
            "resp": self.resp,
            "nbo": self.nbo,
            "frequencies": self.frequencies,
        }


class TaskDocument(BaseTaskDocument, MoleculeMetadata):
    """
    Definition of a Q-Chem task document
    """

    calc_code = "Q-Chem"
    completed = True

    is_valid: bool = Field(
        True, description="Whether this task document passed validation or not"
    )
    state: QChemStatus = Field(None, description="State of this calculation")

    cputime: float = Field(None, description="The system CPU time in seconds")
    walltime: float = Field(None, description="The real elapsed time in seconds")

    calcs_reversed: List[Dict] = Field(
        [], description="The 'raw' calculation docs used to assembled this task"
    )

    orig: Dict[str, Any] = Field(
        {}, description="Summary of the original Q-Chem inputs"
    )
    input = Field(InputSummary())
    output = Field(OutputSummary())

    critic2: Dict[str, Any] = Field(
        None, description="Output from Critic2 critical point analysis code"
    )
    custom_smd: str = Field(
        None, description="Parameter string for SMD implicit solvent model"
    )

    special_run_type: str = Field(
        None, description="Special workflow name (if applicable)"
    )

    # TODO - type of `tags` field seems to differ among task databases
    # sometimes List, sometimes Dict
    # left as Any here to ensure tags don't cause validation to fail.
    tags: Any = Field(None, description="Metadata tags")

    warnings: Dict[str, bool] = Field(
        None, description="Any warnings related to this task document"
    )

    @property
    def level_of_theory(self) -> LevelOfTheory:
        return level_of_theory(self.orig)

    @property
    def solvent(self) -> str:
        return solvent(self.orig, custom_smd=self.custom_smd)

    @property
    def lot_solvent(self) -> str:
        return lot_solvent_string(self.orig, custom_smd=self.custom_smd)

    @property
    def task_type(self) -> TaskType:
        return task_type(self.orig, special_run_type=self.special_run_type)

    @property
    def calc_type(self) -> CalcType:
        return calc_type(self.special_run_type, self.orig)

    @property
    def entry(self) -> Dict[str, Any]:

        if self.output.optimized_molecule is not None:
            mol = self.output.optimized_molecule
        else:
            mol = self.output.initial_molecule

        if self.charge is None:
            charge = mol.charge
        else:
            charge = self.charge

        if self.spin_multiplicity is None:
            spin = mol.spin_multiplicity
        else:
            spin = self.spin_multiplicity

        entry_dict = {
            "entry_id": self.task_id,
            "task_id": self.task_id,
            "charge": charge,
            "spin_multiplicity": spin,
            "level_of_theory": self.level_of_theory,
            "solvent": self.solvent,
            "lot_solvent": self.lot_solvent,
            "custom_smd": self.custom_smd,
            "task_type": self.task_type,
            "calc_type": self.calc_type,
            "molecule": mol,
            "composition": mol.composition,
            "formula": mol.composition.alphabetical_formula,
            "energy": self.output.final_energy,
            "output": self.output.as_dict(),
            "critic2": self.critic2,
            "orig": self.orig,
            "tags": self.tags,
            "last_updated": self.last_updated,
        }

        return entry_dict


    @classmethod
    def from_directory(
        cls: Type[_T],
        dir_name: Union[Path, str],
        store_additional_json: bool = SETTINGS.QCHEM_STORE_ADDITIONAL_JSON,
        additional_fields: Dict[str, Any] = None,
        **qchem_calculation_kwargs,
    ) -> _T:
        """
        Create a task document from a directory containing QChem files.

        Parameters
        ----------
        dir_name
            The path to the folder containing the calculation outputs.
        stor_additional_json
            Whether to store additional json files found in the calculation directory.
        additional_fields
            Dictionary of additional fields to add to output document.
        **qchem_calculation_kwargs
            Additional parsing options that will be passed to the
            :obj: `.Calculation.from_qchem_files` function. This is to do
        
        Returns
        -------
        QChem Task Document
        """
        logger.info(f"Getting task doc in: {dir_name}")

        additional_fields = {} if additional_fields is None else additional_fields
        dir_name = Path(dir_name)
        task_files = _find_qchem_files(dir_name) #have to implement this method

        if len(task_files) == 0:
            raise FileNotFoundError("No QChem files found!")
        
        calcs_reversed = []
        all_qchem_objects = []
        for task_name, files in task_files.items():
            calc_doc, qchem_objects = Calculation.from_qchem_files(
                dir_name, task_name, **files, **qchem_calculation_kwargs
            )

def _find_qchem_files(
        path: Union[str, Path],
        grad_hess_files: Tuple[str, ...] = _GRAD_HESS_FILES,
) -> Dict[str, Any]:
    """
    Find QChem files in a directory.

    Only files in folders with names matching a task name (mol.qout.*) will be returned.

    Parameters
    ----------
    path
        Path to a directory to search.
    
    Returns
    -------
    dict[str, Any]
        The filenames of the calculation outputs for each QChem task, given as a ordered
        dictionary of::
            {
                task_name:{
                    "qchem_out_file": qout_filename,
                    "qchem_log_file": qclog_filename,   
                }
                ...
            }
    """
    #task_names = ["mol"] + [f""]
    path = Path(path)
    task_file = OrderedDict()

    # def _get_task_files(files, suffix=""):
    #     qchem_files = {}
    #     grad_hess_files = []
    #     for file in files:
    #         if file.match()
        