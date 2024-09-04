import math
import os
from dataclasses import dataclass
from functools import wraps

import numpy as np
import scipy.constants as const
from data import atom_valence_electrons

HA2EV = 2.0 * const.value("Rydberg constant times hc in eV")
ANG2BOHR = 1 / (const.value("Bohr radius") * 10**10)


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
        if not os.path.isfile(filename):
            raise OSError("'" + filename + "' file doesn't exist!")
        return func(filename)

    return wrapper


@check_file_exists
def read_file(file_name: str) -> list[str]:
    """
    Read file into a list of str

    Args:
        filename: name of file to read
    """
    with open(file_name) as f:
        text = f.readlines()
    return text


def find_key(key_input, tempfile):
    # finds line where key occurs in stored input, last instance
    key_input = str(key_input)
    line = len(tempfile)  # default to end
    for i in range(len(tempfile)):
        if key_input in tempfile[i]:
            line = i
    return line


def find_first_range_key(
    key_input, tempfile, startline=0, endline=-1, skip_pound=False
):
    # finds all lines that exactly begin with key
    key_input = str(key_input)
    startlen = len(key_input)
    L = []

    if endline == -1:
        endline = len(tempfile)
    for i in range(startline, endline):
        line = tempfile[i]
        if skip_pound == True:
            for j in range(10):  # repeat to make sure no really weird formatting
                line = line.lstrip()
                line = line.lstrip("#")
        line = line[0:startlen]
        if line == key_input:
            L.append(i)
    if not L:
        L = [len(tempfile)]
    return L


def find_all_key(key_input, tempfile, startline=0):
    # DEPRECATED: NEED TO REMOVE INSTANCES OF THIS FUNCTION AND SWITCH WITH find_first_range_key
    # finds all lines where key occurs in in lines
    L = []  # default
    key_input = str(key_input)
    for i in range(startline, len(tempfile)):
        if key_input in tempfile[i]:
            L.append(i)
    return L


@dataclass
class JDFTXOutfile(ClassPrintFormatter):
    """
    A class to read and process a JDFTx out file

    Attributes
    ----------
        see JDFTx documentation for tag info and typing
    """

    prefix: str = None

    lattice_initial: list[list[float]] = None
    lattice_final: list[list[float]] = None
    lattice: list[list[float]] = None
    a: float = None
    b: float = None
    c: float = None

    fftgrid: list[int] = None

    EFermi: float = None
    Egap: float = None
    Emin: float = None
    Emax: float = None
    HOMO: float = None
    LUMO: float = None
    HOMO_filling: float = None
    LUMO_filling: float = None
    is_metal: bool = None

    broadening_type: str = None
    broadening: float = None
    kgrid: list = None
    truncation_type: str = None
    truncation_radius: float = None
    pwcut: float = None
    fluid: str = None

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

    Ecomponents: dict = field(default_factory=dict)
    is_gc: bool = False # is it a grand canonical calculation

    @classmethod
    def from_file(cls, file_name: str):
        """
        Read file into class object

        Args:
            file_name: file to read
        """
        instance = cls()

        text = read_file(file_name)

        line = find_key("dump-name", text)
        dumpname = text[line].split()[1]
        prefix = dumpname.split(".")[0]
        instance.prefix = prefix

        line = find_key("spintype ", text)
        spintype = text[line].split()[1]
        if spintype == "no-spin":
            spintype = None
            Nspin = 1
        elif spintype == "z-spin":
            Nspin = 2
        else:
            raise NotImplementedError("have not considered this spin yet")
        instance.spintype = spintype
        instance.Nspin = Nspin

        line = find_key("elec-smearing ", text)
        if line != len(text):
            broadening_type = text[line].split()[1]
            broadening = float(text[line].split()[2])
        else:
            broadening_type = None
            broadening = 0
        instance.broadening_type = broadening_type
        instance.broadening = broadening * HA2EV

        line = find_key("kpoint-folding ", text)
        instance.kgrid = [int(x) for x in text[line].split()[1:4]]

        maptypes = {
            "Periodic": None,
            "Slab": "slab",
            "Cylindrical": "wire",
            "Wire": "wire",
            "Spherical": "spherical",
            "Isolated": "box",
        }
        line = find_key("coulomb-interaction", text)
        if line != len(text):
            truncation_type = text[line].split()[1]
            truncation_type = maptypes[truncation_type]
            direc = None
            if len(text[line].split()) == 3:
                direc = text[line].split()[2]
            if truncation_type == "slab" and direc != "001":
                raise ValueError("BGW slab Coulomb truncation must be along z!")
            if truncation_type == "wire" and direc != "001":
                raise ValueError("BGW wire Coulomb truncation must be periodic in z!")

            if truncation_type == "error":
                raise ValueError("Problem with this truncation!")
            if truncation_type == "spherical":
                line = find_key("Initialized spherical truncation of radius", text)
                instance.truncation_radius = float(text[line].split()[5]) / ANG2BOHR
            instance.truncation_type = truncation_type
        else:
            instance.truncation_type = None

        line = find_key("elec-cutoff ", text)
        instance.pwcut = float(text[line].split()[1]) * HA2EV

        line = find_all_key("Chosen fftbox size", text)[0]
        fftgrid = [int(x) for x in text[line].split()[6:9]]
        instance.fftgrid = fftgrid

        line = find_key("kpoint-reduce-inversion", text)
        if line == len(text):
            raise ValueError(
                "kpoint-reduce-inversion must = no in single point DFT runs so kgrid without time-reversal symmetry is used (BGW requirement)"
            )
        if text[line].split()[1] != "no":
            raise ValueError(
                "kpoint-reduce-inversion must = no in single point DFT runs so kgrid without time-reversal symmetry is used (BGW requirement)"
            )

        line = find_key("Dumping 'jdft.eigStats' ...", text)
        if line == len(text):
            raise ValueError(
                'Must run DFT job with "dump End EigStats" to get summary gap information!'
            )
        instance.Emin = float(text[line + 1].split()[1]) * HA2EV
        instance.HOMO = float(text[line + 2].split()[1]) * HA2EV
        instance.EFermi = float(text[line + 3].split()[2]) * HA2EV
        instance.LUMO = float(text[line + 4].split()[1]) * HA2EV
        instance.Emax = float(text[line + 5].split()[1]) * HA2EV
        instance.Egap = float(text[line + 6].split()[2]) * HA2EV
        if instance.broadening_type is not None:
            instance.HOMO_filling = (2 / instance.Nspin) * cls.calculate_filling(
                instance.broadening_type,
                instance.broadening,
                instance.HOMO,
                instance.EFermi,
            )
            instance.LUMO_filling = (2 / instance.Nspin) * cls.calculate_filling(
                instance.broadening_type,
                instance.broadening,
                instance.LUMO,
                instance.EFermi,
            )
        else:
            instance.HOMO_filling = 2 / instance.Nspin
            instance.LUMO_filling = 0
        instance.is_metal = instance._determine_is_metal()

        line = find_first_range_key("fluid ", text)
        instance.fluid = text[line[0]].split()[1]
        if instance.fluid == "None":
            instance.fluid = None

        line = find_all_key("nElectrons", text)
        if len(line) > 1:
            idx = 4
        else:
            idx = 1  # nElectrons was not printed in scf iterations then
        instance.total_electrons = float(text[line[-1]].split()[idx])
        instance.Nbands = int(math.ceil(instance.total_electrons))

        startline = find_key("Input parsed successfully", text)
        endline = find_key("---------- Initializing the Grid ----------", text)
        lines = find_first_range_key("ion ", text, startline=startline, endline=endline)
        atom_elements = [text[x].split()[1] for x in lines]
        instance.Nat = len(atom_elements)
        atom_coords = [text[x].split()[2:5] for x in lines]
        instance.atom_coords_initial = np.array(atom_coords, dtype=float)
        atom_types = []
        for x in atom_elements:
            if x not in atom_types:
                atom_types.append(x)
        instance.atom_elements = atom_elements
        mapping_dict = dict(zip(atom_types, range(1, len(atom_types) + 1)))
        instance.atom_elements_int = [mapping_dict[x] for x in instance.atom_elements]
        instance.atom_types = atom_types
        line = find_key("# Ionic positions in", text) + 1
        coords = np.array(
            [text[i].split()[2:5] for i in range(line, line + instance.Nat)],
            dtype=float,
        )
        instance.atom_coords_final = coords
        instance.atom_coords = instance.atom_coords_final.copy()

        startline = find_key("---------- Setting up pseudopotentials ----------", text)
        endline = find_first_range_key("Initialized ", text, startline=startline)[0]
        lines = find_all_key("valence electrons", text)
        try:
            atom_total_elec = [int(float(text[x].split()[0])) for x in lines]
            pp_type = "SG15"
        except:
            pp_type = "GBRV"
            raise ValueError(
                "SG15 valence electron reading failed, make sure right pseudopotentials were used!"
            )
        total_elec_dict = dict(zip(instance.atom_types, atom_total_elec))
        instance.pp_type = pp_type

        element_total_electrons = np.array(
            [total_elec_dict[x] for x in instance.atom_elements]
        )
        element_valence_electrons = np.array(
            [atom_valence_electrons[x] for x in instance.atom_elements]
        )
        element_semicore_electrons = element_total_electrons - element_valence_electrons

        instance.total_electrons_uncharged = np.sum(element_total_electrons)
        instance.valence_electrons_uncharged = np.sum(element_valence_electrons)
        instance.semicore_electrons_uncharged = np.sum(element_semicore_electrons)

        instance.semicore_electrons = instance.semicore_electrons_uncharged
        instance.valence_electrons = (
            instance.total_electrons - instance.semicore_electrons
        )  # accounts for if system is charged

        lines = find_all_key("R =", text)
        line = lines[0]
        lattice_initial = (
            np.array(
                [x.split()[1:4] for x in text[(line + 1) : (line + 4)]], dtype=float
            ).T
            / ANG2BOHR
        )
        instance.lattice_initial = lattice_initial.copy()

        templines = find_all_key("LatticeMinimize", text)
        if len(templines) > 0:
            line = templines[-1]
            lattice_final = (
                np.array(
                    [x.split()[1:4] for x in text[(line + 1) : (line + 4)]], dtype=float
                ).T
                / ANG2BOHR
            )
            instance.lattice_final = lattice_final.copy()
            instance.lattice = lattice_final.copy()
        else:
            instance.lattice = lattice_initial.copy()
        instance.a, instance.b, instance.c = np.sum(instance.lattice**2, axis=1) ** 0.5

        instance.has_solvation = instance.check_solvation()

        # Cooper added
        line = find_key("# Energy components:", text)
        instance.is_gc = key_exists('target-mu', text)
        instance.Ecomponents = instance.read_ecomponents(line, text)
        return instance
    
    @property
    def structure(self):
        latt = self.lattice
        coords = self.atom_coords_final
        elements = self.atom_elements
        structure = Structure(
            lattice=latt,
            species=elements,
            coords=coords
        )
        return structure

    def calculate_filling(broadening_type, broadening, eig, EFermi):
        # most broadening implementations do not have the denominator factor of 2, but JDFTx does currently
        #   remove if use this for other code outfile reading
        x = (eig - EFermi) / (2.0 * broadening)
        if broadening_type == "Fermi":
            filling = 0.5 * (1 - np.tanh(x))
        elif broadening_type == "Gauss":
            filling = 0.5 * (1 - math.erf(x))
        elif broadening_type == "MP1":
            filling = 0.5 * (1 - math.erf(x)) - x * np.exp(-1 * x**2) / (2 * np.pi**0.5)
        elif broadening_type == "Cold":
            filling = (
                0.5 * (1 - math.erf(x + 0.5**0.5))
                + np.exp(-1 * (x + 0.5**0.5) ** 2) / (2 * np.pi) ** 0.5
            )
        else:
            raise NotImplementedError("Have not added other broadening types")

        return filling

    def _determine_is_metal(self):
        TOL_PARTIAL = 0.01
        if (
            self.HOMO_filling / (2 / self.Nspin) > (1 - TOL_PARTIAL)
            and self.LUMO_filling / (2 / self.Nspin) < TOL_PARTIAL
        ):
            return False
        return True

    def check_solvation(self):
        return self.fluid is not None

    def write():
        # don't need a write method since will never do that
        return NotImplementedError("There is no need to write a JDFTx out file")
