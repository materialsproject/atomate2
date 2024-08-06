# mypy: ignore-errors

""" Core definition of a Q-Chem Task Document """
from typing import Any, Dict, List, Optional, Callable

from pydantic import BaseModel, Field
from pymatgen.core.structure import Molecule

from emmet.core.structure import MoleculeMetadata
from emmet.core.task import BaseTaskDocument
from emmet.core.utils import ValueEnum
from emmet.core.qchem.calc_types import (
    LevelOfTheory,
    CalcType,
    TaskType,
    calc_type,
    level_of_theory,
    task_type,
    solvent,
    lot_solvent_string,
)


__author__ = "Evan Spotte-Smith <ewcspottesmith@lbl.gov>"


class JDFTxStatus(ValueEnum):
    """
    JDFTx Calculation State
    """

    SUCCESS = "successful"
    FAILED = "unsuccessful"


class OutputSummary(BaseModel):
    """
    Summary of an output for a Q-Chem calculation
    """

    initial_molecule: Optional[Molecule] = Field(
        None, description="Input Molecule object"
    )
    optimized_molecule: Optional[Molecule] = Field(
        None, description="Optimized Molecule object"
    )

    final_energy: Optional[float] = Field(
        None, description="Final electronic energy for the calculation (units: Hartree)"
    )
    enthalpy: Optional[float] = Field(
        None, description="Total enthalpy of the molecule (units: kcal/mol)"
    )
    entropy: Optional[float] = Field(
        None, description="Total entropy of the molecule (units: cal/mol-K"
    )

    mulliken: Optional[List[Any]] = Field(
        None, description="Mulliken atomic partial charges and partial spins"
    )
    resp: Optional[List[float]] = Field(
        None,
        description="Restrained Electrostatic Potential (RESP) atomic partial charges",
    )
    nbo: Optional[Dict[str, Any]] = Field(
        None, description="Natural Bonding Orbital (NBO) output"
    )

    frequencies: Optional[List[float]] = Field(
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

    calc_code: str = "Q-Chem"
    completed: bool = True

    is_valid: bool = Field(
        True, description="Whether this task document passed validation or not"
    )
    state: Optional[QChemStatus] = Field(None, description="State of this calculation")

    cputime: Optional[float] = Field(None, description="The system CPU time in seconds")
    walltime: Optional[float] = Field(
        None, description="The real elapsed time in seconds"
    )

    calcs_reversed: List[Dict] = Field(
        [], description="The 'raw' calculation docs used to assembled this task"
    )

    orig: Dict[str, Any] = Field(
        {}, description="Summary of the original Q-Chem inputs"
    )
    output: OutputSummary = Field(OutputSummary())

    critic2: Optional[Dict[str, Any]] = Field(
        None, description="Output from Critic2 critical point analysis code"
    )
    custom_smd: Optional[str] = Field(
        None, description="Parameter string for SMD implicit solvent model"
    )

    special_run_type: Optional[str] = Field(
        None, description="Special workflow name (if applicable)"
    )

    smiles: Optional[str] = Field(
        None,
        description="Simplified molecular-input line-entry system (SMILES) string for the molecule involved "
        "in this calculation.",
    )

    species_hash: Optional[str] = Field(
        None,
        description="Weisfeiler Lehman (WL) graph hash using the atom species as the graph "
        "node attribute.",
    )
    coord_hash: Optional[str] = Field(
        None,
        description="Weisfeiler Lehman (WL) graph hash using the atom coordinates as the graph "
        "node attribute.",
    )

    # TODO - type of `tags` field seems to differ among task databases
    # sometimes List, sometimes Dict
    # left as Any here to ensure tags don't cause validation to fail.
    tags: Optional[Any] = Field(None, description="Metadata tags")

    warnings: Optional[Dict[str, bool]] = Field(
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
            charge = int(mol.charge)
        else:
            charge = int(self.charge)

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
            "species_hash": self.species_hash,
            "coord_hash": self.coord_hash,
        }

        return entry_dict


def filter_task_type(
    entries: List[Dict[str, Any]],
    task_type: TaskType,
    sort_by: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    Filter (and sort) TaskDocument entries based on task type

    :param entries: List of TaskDocument entry dicts
    :param TaskType: TaskType to accept
    :param sort_by: Function used to sort (default None)
    :return: Filtered (sorted) list of entries
    """

    filtered = [f for f in entries if f["task_type"] == task_type]

    if sort_by is not None:
        return sorted(filtered, key=sort_by)
    else:
        return filtered