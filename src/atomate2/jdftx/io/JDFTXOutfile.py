import os
from functools import wraps
import math
import numpy as np
from dataclasses import dataclass, field
import scipy.constants as const
from atomate2.jdftx.io.data import atom_valence_electrons
from pymatgen.core import Structure
from pymatgen.core.trajectory import Trajectory
from typing import List
from pymatgen.core.units import Ha_to_eV, ang_to_bohr


#Ha_to_eV = 2.0 * const.value('Rydberg constant times hc in eV')
# ang_to_bohr = 1 / (const.value('Bohr radius') * 10**10)

class ClassPrintFormatter():
    def __str__(self) -> str:
        '''generic means of printing class to command line in readable format'''
        return str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in sorted(self.__dict__)))

def check_file_exists(func):
    '''Check if file exists (and continue normally) or raise an exception if it does not'''
    @wraps(func)
    def wrapper(filename):
        if not os.path.isfile(filename):
            raise OSError('\'' + filename + '\' file doesn\'t exist!')
        return func(filename)
    return wrapper

@check_file_exists
def read_file(file_name: str) -> list[str]:
        '''
        Read file into a list of str

        Args:
            filename: name of file to read
        '''
        with open(file_name, 'r') as f:
            text = f.readlines()
        return text

def find_key(key_input, tempfile):
    '''
    Finds last instance of key in output file. 

    Parameters
    ----------
    key_input: str
        key string to match
    tempfile: List[str]
        output from readlines() function in read_file method
    '''
    key_input = str(key_input)
    line = None
    for i in range(0,len(tempfile)):
        if key_input in tempfile[i]:
            line = i
    return line


def find_first_range_key(key_input, tempfile, startline=0, endline=-1, skip_pound = False):
    #finds all lines that exactly begin with key
    key_input = str(key_input)
    startlen = len(key_input)
    L = []

    if endline == -1:
        endline = len(tempfile)
    for i in range(startline,endline):
        line = tempfile[i]
        if skip_pound == True:
            for j in range(10):  #repeat to make sure no really weird formatting
                line = line.lstrip()
                line = line.lstrip('#')
        line = line[0:startlen]
        if line == key_input:
            L.append(i)
    if not L:
        L = [len(tempfile)]
    return L

def key_exists(key_input, tempfile):
    line = find_key(key_input, tempfile)
    if line == None:
        return False
    else:
        return True

def find_all_key(key_input, tempfile, startline = 0):
    #DEPRECATED: NEED TO REMOVE INSTANCES OF THIS FUNCTION AND SWITCH WITH find_first_range_key
    #finds all lines where key occurs in in lines
    L = []     #default
    key_input = str(key_input)
    for i in range(startline,len(tempfile)):
        if key_input in tempfile[i]:
            L.append(i)
    return L

@dataclass
class JDFTXOutfile(ClassPrintFormatter):
    '''
    A class to read and process a JDFTx out file

    Attributes:
        see JDFTx documentation for tag info and typing
    '''

    prefix: str = None

    lattice_initial: list[list[float]] = None
    lattice_final: list[list[float]] = None
    lattice: list[list[float]] = None
    a: float = None
    b: float = None
    c: float = None

    fftgrid: list[int] = None

    # grouping fields related to electronic parameters.
    # Used by the get_electronic_output() method
    _electronic_output = [ 
    "EFermi", "Egap", "Emin", "Emax", "HOMO",
    "LUMO", "HOMO_filling", "LUMO_filling", "is_metal"
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

    broadening_type: str = None
    broadening: float = None
    kgrid: list = None
    truncation_type: str = None
    truncation_radius: float = None
    pwcut: float = None

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

    #@ Cooper added @#
    Ecomponents: dict = field(default_factory=dict)
    is_gc: bool = False # is it a grand canonical calculation
    trajectory_positions: list[list[list[float]]] = None
    trajectory_lattice: list[list[list[float]]] = None
    trajectory_forces: list[list[list[float]]] = None
    trajectory_ecomponents: list[dict] = None
    is_converged: bool = None #TODO implement this

    @classmethod
    def _get_prefix(cls, text: str) -> str:
        '''
        Get output prefix from the out file

        Args:
            text: output of read_file for out file
        '''
        line = find_key('dump-name', text)
        dumpname = text[line].split()[1]
        prefix = dumpname.split('.')[0]
        return prefix
    
    @classmethod
    def _set_spinvars(cls, text: str) -> tuple[str, int]:
        '''
        Set spintype and Nspin from out file text for instance

        Args:
            text: output of read_file for out file
        '''
        line = find_key('spintype ', text)
        spintype = text[line].split()[1]
        if spintype == 'no-spin':
            spintype = None
            Nspin = 1
        elif spintype == 'z-spin':
            Nspin = 2
        else:
            raise NotImplementedError('have not considered this spin yet')
        return spintype, Nspin
    
    @classmethod
    def _get_broadeningvars(cls, text:str) -> tuple[str, float]:
        '''
        Get broadening type and value from out file text

        Args:
            text: output of read_file for out file
        '''
        line = find_key('elec-smearing ', text)
        if line != len(text):
            broadening_type = text[line].split()[1]
            broadening = float(text[line].split()[2]) * Ha_to_eV
        else:
            broadening_type = None
            broadening = 0
        return broadening_type, broadening
    
    @classmethod
    def _get_truncationvars(cls, text:str) -> tuple[str, float]:
        '''
        Get truncation type and value from out file text

        Args:
            text: output of read_file for out file
        '''
        maptypes = {'Periodic': None, 'Slab': 'slab', 'Cylindrical': 'wire', 'Wire': 'wire',
                    'Spherical': 'spherical', 'Isolated': 'box'}
        line = find_key('coulomb-interaction', text)
        truncation_type = None
        truncation_radius = None
        if line != len(text):
            truncation_type = text[line].split()[1]
            truncation_type = maptypes[truncation_type]
            direc = None
            if len(text[line].split()) == 3:
                direc = text[line].split()[2]
            if truncation_type == 'slab' and direc != '001':
                raise ValueError('BGW slab Coulomb truncation must be along z!')
            if truncation_type == 'wire' and direc != '001':
                raise ValueError('BGW wire Coulomb truncation must be periodic in z!')
            if truncation_type == 'error':
                raise ValueError('Problem with this truncation!')
            if truncation_type == 'spherical':
                line = find_key('Initialized spherical truncation of radius', text)
                truncation_radius = float(text[line].split()[5]) / ang_to_bohr
        return truncation_type, truncation_radius
    
    @classmethod
    def _get_elec_cutoff(cls, text:str) -> float:
        '''
        Get the electron cutoff from the out file text

        Args:
            text: output of read_file for out file
        '''
        line = find_key('elec-cutoff ', text)
        pwcut = float(text[line].split()[1]) * Ha_to_eV
        return pwcut

    @classmethod
    def _get_fftgrid(cls, text:str) -> list[int]:
        '''
        Get the FFT grid from the out file text

        Args:
            text: output of read_file for out file
        '''
        line = find_key('Chosen fftbox size', text)
        fftgrid = [int(x) for x in text[line].split()[6:9]]
        return fftgrid

    @classmethod
    def from_file(cls, file_name: str):
        '''
        Read file into class object

        Args:
            file_name: file to read
        '''
        instance = cls()

        text = read_file(file_name)

        instance.prefix = cls._get_prefix(text)

        spintype, Nspin = cls._set_spinvars(text)
        instance.spintype = spintype
        instance.Nspin = Nspin

        broadening_type, broadening = cls._get_broadeningvars(text)
        instance.broadening_type = broadening_type
        instance.broadening = broadening

        line = find_key('kpoint-folding ', text)
        instance.kgrid = [int(x) for x in text[line].split()[1:4]]

        truncation_type, truncation_radius = cls._get_truncationvars(text)
        instance.truncation_type = truncation_type
        instance.truncation_radius = truncation_radius

        instance.pwcut = cls.truncation_type(text)

        instance.fftgrid = cls._get_fftgrid(text)

        # Are these needed for DFT calcs?
        # Ben: idk
        # line = find_key('kpoint-reduce-inversion', text)
        # if line == len(text):
        #     raise ValueError('kpoint-reduce-inversion must = no in single point DFT runs so kgrid without time-reversal symmetry is used (BGW requirement)')
        # if text[line].split()[1] != 'no':
        #     raise ValueError('kpoint-reduce-inversion must = no in single point DFT runs so kgrid without time-reversal symmetry is used (BGW requirement)')

        line = find_key('Dumping \'eigStats\' ...', text)
        if line == len(text):
            raise ValueError('Must run DFT job with "dump End EigStats" to get summary gap information!')
        instance.Emin = float(text[line+1].split()[1]) * Ha_to_eV
        instance.HOMO = float(text[line+2].split()[1]) * Ha_to_eV
        instance.EFermi = float(text[line+3].split()[2]) * Ha_to_eV
        instance.LUMO = float(text[line+4].split()[1]) * Ha_to_eV
        instance.Emax = float(text[line+5].split()[1]) * Ha_to_eV
        instance.Egap = float(text[line+6].split()[2]) * Ha_to_eV
        if instance.broadening_type is not None:
            instance.HOMO_filling = (2 / instance.Nspin) * cls.calculate_filling(instance.broadening_type, instance.broadening, instance.HOMO, instance.EFermi)
            instance.LUMO_filling = (2 / instance.Nspin) * cls.calculate_filling(instance.broadening_type, instance.broadening, instance.LUMO, instance.EFermi)
        else:
            instance.HOMO_filling = (2 / instance.Nspin)
            instance.LUMO_filling = 0
        instance.is_metal = instance._determine_is_metal()

        line = find_first_range_key('fluid ', text)
        instance.fluid = text[line[0]].split()[1]
        if instance.fluid == 'None':
            instance.fluid = None

        line = find_all_key('nElectrons', text)
        if len(line) > 1:
            idx = 4
        else:
            idx = 1  #nElectrons was not printed in scf iterations then
        instance.total_electrons = float(text[line[-1]].split()[idx])
        instance.Nbands = int(math.ceil(instance.total_electrons))

        startline = find_key('Input parsed successfully', text)
        endline = find_key('---------- Initializing the Grid ----------', text)
        lines = find_first_range_key('ion ', text, startline = startline, endline = endline)
        atom_elements = [text[x].split()[1] for x in lines]
        instance.Nat = len(atom_elements)
        atom_coords = [text[x].split()[2:5] for x in lines]
        instance.atom_coords_initial = np.array(atom_coords, dtype = float)
        atom_types = []
        for x in atom_elements:
            if not x in atom_types:
                atom_types.append(x)
        instance.atom_elements = atom_elements
        mapping_dict = dict(zip(atom_types, range(1, len(atom_types) + 1)))
        instance.atom_elements_int = [mapping_dict[x] for x in instance.atom_elements]
        instance.atom_types = atom_types
        line = find_key('# Ionic positions in', text) + 1
        coords = np.array([text[i].split()[2:5] for i in range(line, line + instance.Nat)], dtype = float)
        instance.atom_coords_final = coords
        instance.atom_coords = instance.atom_coords_final.copy()

        startline = find_key('---------- Setting up pseudopotentials ----------', text)
        endline = find_first_range_key('Initialized ', text, startline = startline)[0]
        lines = find_all_key('valence electrons', text)
        try:
            atom_total_elec = [int(float(text[x].split()[0])) for x in lines]
            pp_type = 'SG15'
        except:
            pp_type = 'GBRV'
            raise ValueError('SG15 valence electron reading failed, make sure right pseudopotentials were used!')
        total_elec_dict = dict(zip(instance.atom_types, atom_total_elec))
        instance.pp_type = pp_type

        element_total_electrons = np.array([total_elec_dict[x] for x in instance.atom_elements])
        element_valence_electrons = np.array([atom_valence_electrons[x] for x in instance.atom_elements])
        element_semicore_electrons = element_total_electrons - element_valence_electrons

        instance.total_electrons_uncharged = np.sum(element_total_electrons)
        instance.valence_electrons_uncharged = np.sum(element_valence_electrons)
        instance.semicore_electrons_uncharged = np.sum(element_semicore_electrons)

        instance.semicore_electrons = instance.semicore_electrons_uncharged
        instance.valence_electrons = instance.total_electrons - instance.semicore_electrons  #accounts for if system is charged

        lines = find_all_key('R =', text)
        line = lines[0]
        lattice_initial = np.array([x.split()[1:4] for x in text[(line + 1):(line + 4)]], dtype = float).T / ang_to_bohr
        instance.lattice_initial = lattice_initial.copy()

        templines = find_all_key('LatticeMinimize', text)
        if len(templines) > 0:
            line = templines[-1]
            lattice_final = np.array([x.split()[1:4] for x in text[(line + 1):(line + 4)]], dtype = float).T / ang_to_bohr
            instance.lattice_final = lattice_final.copy()
            instance.lattice = lattice_final.copy()
        else:
            instance.lattice = lattice_initial.copy()
        instance.a, instance.b, instance.c = np.sum(instance.lattice**2, axis = 1)**0.5

        instance.has_solvation = instance.check_solvation()

        #@ Cooper added @#
        line = find_key("# Energy components:", text)
        instance.is_gc = key_exists('target-mu', text)
        instance.Ecomponents = instance.read_ecomponents(line, text)
        instance._build_trajectory(templines)

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
        #most broadening implementations do not have the denominator factor of 2, but JDFTx does currently
        #   remove if use this for other code outfile reading
        x = (eig - EFermi) / (2.0 * broadening)
        if broadening_type == 'Fermi':
            filling = 0.5 * (1 - np.tanh(x))
        elif broadening_type == 'Gauss':
            filling = 0.5 * (1 - math.erf(x))
        elif broadening_type == 'MP1':
            filling = 0.5 * (1 - math.erf(x)) - x * np.exp(-1 * x**2) / (2 * np.pi**0.5)
        elif broadening_type == 'Cold':
            filling = 0.5* (1 - math.erf(x + 0.5**0.5)) + np.exp(-1 * (x + 0.5**0.5)**2) / (2 * np.pi)**0.5
        else:
            raise NotImplementedError('Have not added other broadening types')

        return filling

    def _determine_is_metal(self):
        TOL_PARTIAL = 0.01
        if self.HOMO_filling / (2 / self.Nspin) > (1 - TOL_PARTIAL) and self.LUMO_filling / (2 / self.Nspin) < TOL_PARTIAL:
            return False
        return True

    def check_solvation(self):
        return self.fluid is not None

    def write():
        #don't need a write method since will never do that
        return NotImplementedError('There is no need to write a JDFTx out file')

    def _build_trajectory(self, text):
        '''
        Builds the trajectory lists and sets the instance attributes.
        
        '''
        # Needs to handle LatticeMinimize and IonicMinimize steps in one run
        # can do this by checking if lattice vectors block is present and if
        # so adding it to the lists. If it isn't present, copy the last 
        # lattice from the list.
        # initialize lattice list with starting lattice and remove it
        # from the list after iterating through all the optimization steps
        trajectory_positions = []
        trajectory_lattice = [self.lattice_initial]
        trajectory_forces = []
        trajectory_ecomponents = []

        ion_lines = find_first_range_key('# Ionic positions in', text)
        force_lines = find_first_range_key('# Forces in', text)
        ecomp_lines = find_first_range_key('# Energy components:', text)
        print(ion_lines, force_lines, ecomp_lines)
        for iline, ion_line, force_line, ecomp_line in enumerate(zip(ion_lines, force_lines, ecomp_lines)):
            coords = np.array([text[i].split()[2:5] for i in range(ion_line + 1, ion_line + self.Nat + 1)], dtype = float)
            forces = np.array([text[i].split()[2:5] for i in range(force_line + 1, force_line + self.Nat + 1)], dtype = float)
            ecomp = self.read_ecomponents(ecomp_line, text)
            lattice_lines = find_first_range_key('# Lattice vectors:', text, startline=ion_line, endline=ion_lines[iline-1])
            if len(lattice_lines) == 0: # if no lattice lines found, append last lattice
                trajectory_lattice.append(trajectory_lattice[-1])
            else:
                line = lattice_lines[0]
                trajectory_lattice.append(np.array([x.split()[1:4] for x in text[(line + 1):(line + 4)]], dtype = float).T / ang_to_bohr)
            trajectory_positions.append(coords)
            trajectory_forces.append(forces)
            trajectory_ecomponents.append(ecomp)
        trajectory_lattice = trajectory_lattice[1:] # remove starting lattice

        self.trajectory_positions = trajectory_positions
        self.trajectory_lattice = trajectory_lattice
        self.trajectory_forces = trajectory_forces
        self.trajectory_ecomponents = trajectory_ecomponents

    @property
    def trajectory(self):
        '''
        Returns a pymatgen trajectory object
        '''
        # structures = []
        # for coords, lattice 
        # traj = Trajectory.from_structures

    def read_ecomponents(self, line:int, text:str):
        Ecomponents = {}
        if self.is_gc == True:
            final_E_type = "G"
        else:
            final_E_type = "F"
        for tmp_line in text[line+1:]:
            chars = tmp_line.strip().split()
            if tmp_line.startswith("--"):
                continue
            E_type = chars[0]
            Energy = float(chars[-1]) * Ha_to_eV
            Ecomponents.update({E_type:Energy})
            if E_type == final_E_type:
                return Ecomponents
    
    @property
    def electronic_output(self) -> dict:
        '''
        Return a dictionary with all relevant electronic information.
        Returns values corresponding to these keys in _electronic_output
        field.
        '''
        dct = {}
        for field in self.__dataclass_fields__:
            if field in self._electronic_output:
                value = getattr(self, field)
                dct[field] = value
        return dct

    def to_dict(self) -> dict:
        # convert dataclass to dictionary representation
        dct = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            dct[field] = value
        return dct