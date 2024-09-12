from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import List, Optional

from monty.io import zopen

from atomate2.jdftx.io.jdftxoutfileslice import JDFTXOutfileSlice
from atomate2.jdftx.io.jminsettings import (
    JMinSettingsElectronic,
    JMinSettingsFluid,
    JMinSettingsIonic,
    JMinSettingsLattice,
)
from atomate2.jdftx.io.joutstructures import JOutStructures


class ClassPrintFormatter:
    def __str__(self) -> str:
        """Generic means of printing class to command line in readable format"""
        return (
            str(self.__class__)
            + "\n"
            + "\n".join(
                str(item) + " = " + str(self.__dict__[item])
                for item in sorted(self.__dict__)
            )
        )


def check_file_exists(func):
    """Check if file exists (and continue normally) or raise an exception if it does not"""

    @wraps(func)
    def wrapper(filename):
        filename = Path(filename)
        if not filename.is_file():
            raise OSError(f"'{filename}' file doesn't exist!")
        return func(filename)

    return wrapper


@check_file_exists
def read_file(file_name: str) -> list[str]:
    """
    Read file into a list of str

    Parameters
    ----------
    filename: Path or str
        name of file to read

    Returns
    -------
    text: list[str]
        list of strings from file
    """
    with zopen(file_name, "r") as f:
        text = f.readlines()
    return text


def get_start_lines(
    text: list[str],
    start_key: Optional[str] = "*************** JDFTx",
    add_end: Optional[bool] = False,
) -> list[int]:
    """
    Get the line numbers corresponding to the beginning of seperate JDFTx calculations
    (in case of multiple calculations appending the same out file)

    Args:
        text: output of read_file for out file
    """
    start_lines = []
    for i, line in enumerate(text):
        if start_key in line:
            start_lines.append(i)
    if add_end:
        start_lines.append(i)
    return start_lines


def read_outfile_slices(file_name: str) -> list[list[str]]:
    """
    Read slice of out file into a list of str

    Parameters
    ----------
    filename: Path or str
        name of file to read
    out_slice_idx: int
        index of slice to read from file

    Returns
    -------
    texts: list[list[str]]
        list of out file slices (individual calls of JDFTx)
    """
    _text = read_file(file_name)
    start_lines = get_start_lines(_text, add_end=True)
    texts = []
    for i in range(len(start_lines) - 1):
        text = _text[start_lines[i] : start_lines[i + 1]]
        texts.append(text)
    return texts


@dataclass
class JDFTXOutfile(JDFTXOutfileSlice):
    """
    A class to read and process a JDFTx out file
    """

    slices: List[JDFTXOutfileSlice] = field(default_factory=list)
    #####
    prefix: str = None

    jstrucs: JOutStructures = None
    jsettings_fluid: JMinSettingsFluid = None
    jsettings_electronic: JMinSettingsElectronic = None
    jsettings_lattice: JMinSettingsLattice = None
    jsettings_ionic: JMinSettingsIonic = None

    xc_func: str = None

    lattice_initial: list[list[float]] = None
    lattice_final: list[list[float]] = None
    lattice: list[list[float]] = None
    a: float = None
    b: float = None
    c: float = None

    fftgrid: list[int] = None
    geom_opt: bool = None
    geom_opt_type: str = None

    # grouping fields related to electronic parameters.
    # Used by the get_electronic_output() method
    _electronic_output = [
        "EFermi",
        "Egap",
        "Emin",
        "Emax",
        "HOMO",
        "LUMO",
        "HOMO_filling",
        "LUMO_filling",
        "is_metal",
    ]
    EFermi: float = None
    Egap: float = None
    Emin: float = None
    Emax: float = None
    HOMO: float = None
    LUMO: float = None
    HOMO_filling: float = None
    LUMO_filling: float = None
    is_metal: bool = None
    etype: str = None

    broadening_type: str = None
    broadening: float = None
    kgrid: list = None
    truncation_type: str = None
    truncation_radius: float = None
    pwcut: float = None
    rhocut: float = None

    pp_type: str = None
    total_electrons: float = None
    semicore_electrons: int = None
    valence_electrons: float = None
    total_electrons_uncharged: int = None
    semicore_electrons_uncharged: int = None
    valence_electrons_uncharged: int = None
    Nbands: int = None

    atom_elements: list = None
    atom_elements_int: list = None
    atom_types: list = None
    spintype: str = None
    Nspin: int = None
    Nat: int = None
    atom_coords_initial: list[list[float]] = None
    atom_coords_final: list[list[float]] = None
    atom_coords: list[list[float]] = None

    has_solvation: bool = False
    fluid: str = None

    @classmethod
    def from_file(cls, file_path: str):
        texts = read_outfile_slices(file_path)
        slices = []
        for text in texts:
            slices.append(JDFTXOutfileSlice.from_out_slice(text))
        instance = cls.from_out_slice(texts[-1])
        instance.slices = slices
        return instance

    def __getitem__(self, key: int | str):
        if type(key) is int:
            return self.slices[key]
        if type(key) is str:
            return getattr(self, key)

    def __len__(self):
        return len(self.slices)
