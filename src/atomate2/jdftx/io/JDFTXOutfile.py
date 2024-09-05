import os
from functools import wraps
import math
from ase import Atom, Atoms
import numpy as np
from dataclasses import dataclass, field
import scipy.constants as const
from atomate2.jdftx.io.data import atom_valence_electrons
from pymatgen.core import Structure
from pymatgen.core.trajectory import Trajectory
from typing import List, Optional
from pymatgen.core.units import Ha_to_eV, ang_to_bohr, bohr_to_ang


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

    def _get_start_lines(text:str, start_key: Optional[str]="*************** JDFTx"):
        '''
        Get the line numbers corresponding to the beginning of seperate JDFTx calculations
        (in case of multiple calculations appending the same out file)

        Args:
            text: output of read_file for out file
        '''
        start_lines = []
        for i, line in enumerate(text):
            if start_key in line:
                start_lines.append(i)
        return start_lines

    def _get_prefix(text: str) -> str:
        '''
        Get output prefix from the out file

        Args:
            text: output of read_file for out file
        '''
        prefix = None
        line = find_key('dump-name', text)
        dumpname = text[line].split()[1]
        if "." in dumpname:
            prefix = dumpname.split('.')[0]
        return prefix
    
    def _get_spinvars(text: str) -> tuple[str, int]:
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
    
    def _get_broadeningvars(text:str) -> tuple[str, float]:
        '''
        Get broadening type and value from out file text

        Args:
            text: output of read_file for out file
        '''
        line = find_key('elec-smearing ', text)
        if not line is None:
            broadening_type = text[line].split()[1]
            broadening = float(text[line].split()[2]) * Ha_to_eV
        else:
            broadening_type = None
            broadening = 0
        return broadening_type, broadening
    
    def _get_truncationvars(text:str) -> tuple[str, float]:
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
        if not line is None:
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
    
    def _get_elec_cutoff(text:str) -> float:
        '''
        Get the electron cutoff from the out file text

        Args:
            text: output of read_file for out file
        '''
        line = find_key('elec-cutoff ', text)
        pwcut = float(text[line].split()[1]) * Ha_to_eV
        return pwcut

    def _get_fftgrid(text:str) -> list[int]:
        '''
        Get the FFT grid from the out file text

        Args:
            text: output of read_file for out file
        '''
        line = find_key('Chosen fftbox size', text)
        fftgrid = [int(x) for x in text[line].split()[6:9]]
        return fftgrid

    def _get_kgrid(text:str) -> list[int]:
        '''
        Get the kpoint grid from the out file text

        Args:
            text: output of read_file for out file
        '''
        line = find_key('kpoint-folding ', text)
        kgrid = [int(x) for x in text[line].split()[1:4]]
        return kgrid
    
    @classmethod
    def _get_eigstats_varsdict(cls, text:str, prefix:str | None) -> dict[str, float]:
        varsdict = {}
        _prefix = ""
        if not prefix is None:
            _prefix = f"{prefix}."
        line = find_key(f'Dumping \'{_prefix}eigStats\' ...', text)
        if line is None:
            raise ValueError('Must run DFT job with "dump End EigStats" to get summary gap information!')
        varsdict["Emin"] = float(text[line+1].split()[1]) * Ha_to_eV
        varsdict["HOMO"] = float(text[line+2].split()[1]) * Ha_to_eV
        varsdict["EFermi"] = float(text[line+3].split()[2]) * Ha_to_eV
        varsdict["LUMO"] = float(text[line+4].split()[1]) * Ha_to_eV
        varsdict["Emax"] = float(text[line+5].split()[1]) * Ha_to_eV
        varsdict["Egap"] = float(text[line+6].split()[2]) * Ha_to_eV
        return varsdict
    
    def _set_eigvars(self, text:str) -> None:
        eigstats = self._get_eigstats_varsdict(text, self.prefix)
        self.Emin = eigstats["Emin"]
        self.HOMO = eigstats["HOMO"]
        self.EFermi = eigstats["EFermi"]
        self.LUMO = eigstats["LUMO"]
        self.Emax = eigstats["Emax"]
        self.Egap = eigstats["Egap"]
    

    def _get_pp_type(self, text:str) -> str:
        '''
        '''
        skey = "Reading pseudopotential file"
        line = find_key(skey, text)
        ppfile_example = text[line].split(skey)[1].split(":")[0].strip("'")
        pptype = None
        readable = ["GBRV", "SG15"]
        for _pptype in readable:
            if _pptype in ppfile_example:
                if not pptype is None:
                    if ppfile_example.index(pptype) < ppfile_example.index(_pptype):
                        pptype = _pptype
                    else:
                        pass
                else:
                    pptype = _pptype
        if pptype is None:
            raise ValueError(f"Could not determine pseudopotential type from file name {ppfile_example}")
        return pptype
    
    def _set_pseudo_vars(self, text:str) -> None:
        '''
        '''
        self.pp_type = self._get_pp_type(text)
        if self.pp_type == "SG15":
            self._set_pseudo_vars_SG15(text)
        elif self.pp_type == "GBRV":
            self._set_pseudo_vars_GBRV(text)
    
    def _set_pseudo_vars_SG15(self, text:str) -> None:
        '''
        '''
        startline = find_key('---------- Setting up pseudopotentials ----------', text)
        endline = find_first_range_key('Initialized ', text, startline = startline)[0]
        lines = find_all_key('valence electrons', text)
        lines = [x for x in lines if x < endline and x > startline]
        atom_total_elec = [int(float(text[x].split()[0])) for x in lines]
        total_elec_dict = dict(zip(self.atom_types, atom_total_elec))
        element_total_electrons = np.array([total_elec_dict[x] for x in self.atom_elements])
        element_valence_electrons = np.array([atom_valence_electrons[x] for x in self.atom_elements])
        element_semicore_electrons = element_total_electrons - element_valence_electrons
        self.total_electrons_uncharged = np.sum(element_total_electrons)
        self.valence_electrons_uncharged = np.sum(element_valence_electrons)
        self.semicore_electrons_uncharged = np.sum(element_semicore_electrons)
        self.semicore_electrons = self.semicore_electrons_uncharged
        self.valence_electrons = self.total_electrons - self.semicore_electrons  #accounts for if system is charged


    def _set_pseudo_vars_GBRV(self, text:str) -> None:
        ''' TODO: implement this method
        '''
        self.total_electrons_uncharged = None
        self.valence_electrons_uncharged = None
        self.semicore_electrons_uncharged = None
        self.semicore_electrons = None
        self.valence_electrons = None

    
    

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

        spintype, Nspin = cls._get_spinvars(text)
        instance.spintype = spintype
        instance.Nspin = Nspin

        broadening_type, broadening = cls._get_broadeningvars(text)
        instance.broadening_type = broadening_type
        instance.broadening = broadening

        instance.kgrid = cls._get_kgrid(text)

        truncation_type, truncation_radius = cls._get_truncationvars(text)
        instance.truncation_type = truncation_type
        instance.truncation_radius = truncation_radius

        instance.pwcut = cls._get_elec_cutoff(text)

        instance.fftgrid = cls._get_fftgrid(text)

        # Are these needed for DFT calcs?
        # Ben: idk
        # line = find_key('kpoint-reduce-inversion', text)
        # if line == len(text):
        #     raise ValueError('kpoint-reduce-inversion must = no in single point DFT runs so kgrid without time-reversal symmetry is used (BGW requirement)')
        # if text[line].split()[1] != 'no':
        #     raise ValueError('kpoint-reduce-inversion must = no in single point DFT runs so kgrid without time-reversal symmetry is used (BGW requirement)')

        instance._set_eigvars(text)
        print(f"Egap: {instance.Egap}")
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
        instance._set_pseudo_vars(text)

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

    def _build_lattice2(self, text: list[str]) -> None:
        atoms_list = get_atoms_list_from_out(text)
        

    @property
    def trajectory(self):
        '''
        Returns a pymatgen trajectory object
        '''
        # structures = []
        # for coords, lattice 
        # traj = Trajectory.from_structures

    def read_ecomponents(self, line:int, text:str) -> dict:
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


def get_input_coord_vars_from_outfile(text: list[str]):
    start_line = get_start_line(text)
    names = []
    posns = []
    R = np.zeros([3,3])
    lat_row = 0
    active_lattice = False
    for i, line in enumerate(text):
        if i > start_line:
            tokens = line.split()
            if len(tokens) > 0:
                if tokens[0] == "ion":
                    names.append(tokens[1])
                    posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                elif tokens[0] == "lattice":
                    active_lattice = True
                elif active_lattice:
                    if lat_row < 3:
                        R[lat_row, :] = [float(x) for x in tokens[:3]]
                        lat_row += 1
                    else:
                        active_lattice = False
                elif "Initializing the Grid" in line:
                    break
    if not len(names) > 0:
        raise ValueError("No ion names found")
    if len(names) != len(posns):
        raise ValueError("Unequal ion positions/names found")
    if np.sum(R) == 0:
        raise ValueError("No lattice matrix found")
    return names, posns, R


def get_atoms_from_outfile_data(names: list[str], posns: np.ndarray, R: np.ndarray, charges: Optional[np.ndarray] = None, E: Optional[float] = 0, momenta=Optional[np.ndarray] = None):
    atoms = Atoms()
    posns *= bohr_to_ang
    R = R.T*bohr_to_ang
    atoms.cell = R
    if charges is None:
        charges = np.zeros(len(names))
    if momenta is None:
        momenta = np.zeros([len(names), 3])
    for i in range(len(names)):
        atoms.append(Atom(names[i], posns[i], charge=charges[i], momentum=momenta[i]))
    atoms.E = E
    return atoms


def get_atoms_list_from_out_reset_vars(nAtoms=100, _def=100):
    ''' Remnant of having 0 creativity for elegant coding solutions
    This should be replaced at some point, possibly with a new class object for collecting
    data required to build an Atoms object
    '''
    R = np.zeros([3, 3])
    posns = []
    names = []
    chargeDir = {}
    active_lattice = False
    lat_row = 0
    active_posns = False
    log_vars = False
    coords = None
    new_posn = False
    active_lowdin = False
    idxMap = {}
    j = 0
    E = 0
    if nAtoms is None:
        nAtoms = _def
    charges = np.zeros(nAtoms, dtype=float)
    forces = []
    active_forces = False
    coords_forces = None
    return R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges, forces, active_forces, coords_forces


def get_atoms_list_from_out_slice(text: list[str], i_start: int, i_end: int) -> list[Atoms]:
    ''' Gives a list of atoms objects corresponding to individual lattice/ionization steps
    for a slice of the out file bounded by i_start and i_end
    (corresponding to an individual call of JDFTx)
    '''
    charge_key = "oxidation-state"
    opts = []
    nAtoms = None
    R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars()
    for i, line in enumerate(text):
        if i > i_start and i < i_end:
            if new_posn:
                if "Lowdin population analysis " in line:
                    active_lowdin = True
                elif "R =" in line:
                    active_lattice = True
                elif "# Forces in" in line:
                    active_forces = True
                    coords_forces = line.split()[3]
                elif line.find('# Ionic positions in') >= 0:
                    coords = line.split()[4]
                    active_posns = True
                elif active_lattice:
                    if lat_row < 3:
                        R[lat_row, :] = [float(x) for x in line.split()[1:-1]]
                        lat_row += 1
                    else:
                        active_lattice = False
                        lat_row = 0
                elif active_posns:
                    tokens = line.split()
                    if len(tokens) and tokens[0] == 'ion':
                        names.append(tokens[1])
                        posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                        if tokens[1] not in idxMap:
                            idxMap[tokens[1]] = []
                        idxMap[tokens[1]].append(j)
                        j += 1
                    else:
                        posns = np.array(posns)
                        active_posns = False
                        nAtoms = len(names)
                        if len(charges) < nAtoms:
                            charges = np.zeros(nAtoms)
                ##########
                elif active_forces:
                    tokens = line.split()
                    if len(tokens) and tokens[0] == 'force':
                        forces.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                    else:
                        forces = np.array(forces)
                        active_forces = False
                ##########
                elif "Minimize: Iter:" in line:
                    if "F: " in line:
                        E = float(line[line.index("F: "):].split(' ')[1])
                    elif "G: " in line:
                        E = float(line[line.index("G: "):].split(' ')[1])
                elif active_lowdin:
                    if charge_key in line:
                        look = line.rstrip('\n')[line.index(charge_key):].split(' ')
                        symbol = str(look[1])
                        line_charges = [float(val) for val in look[2:]]
                        chargeDir[symbol] = line_charges
                        for atom in list(chargeDir.keys()):
                            for k, idx in enumerate(idxMap[atom]):
                                charges[idx] += chargeDir[atom][k]
                    elif "#" not in line:
                        active_lowdin = False
                        log_vars = True
                elif log_vars:
                    if np.sum(R) == 0.0:
                        R = get_input_coord_vars_from_outfile(text)[2]
                    if coords != 'cartesian':
                        posns = np.dot(posns, R)
                    if len(forces) == 0:
                        forces = np.zeros([nAtoms, 3])
                    if coords_forces.lower() != 'cartesian':
                        forces = np.dot(forces, R)
                    opts.append(get_atoms_from_outfile_data(names, posns, R, charges=charges, E=E, momenta=forces))
                    R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
                        new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars(
                        nAtoms=nAtoms)
            elif "Computing DFT-D3 correction:" in line:
                new_posn = True
    return opts


def get_start_lines(text: list[str], add_end=False) -> list[int]:
    start_lines = []
    end_line = 0
    for i, line in enumerate(text):
        if "JDFTx 1." in line:
            start_lines.append(i)
        end_line = i
    if add_end:
        start_lines.append(end_line)
    return start_lines


def get_start_line(text: list[str]) -> int:
    start_line = get_start_lines(text, add_end=False)[-1]
    return start_line


def get_atoms_list_from_out(text: list[str]) -> list[Atoms]:
    start_lines = get_start_lines(text, add_end=True)
    atoms_list = []
    for i in range(len(start_lines) - 1):
        atoms_list += get_atoms_list_from_out_slice(text, start_lines[i], start_lines[i+1])
    return atoms_list


def is_done(text: list[str]) -> bool:
    start_line = get_start_line(text)
    done = False
    for i, line in enumerate(text):
        if i > start_line:
            if "Minimize: Iter:" in line:
                done = False
            elif "Minimize: Converged" in line:
                done = True
    return done


def get_initial_lattice(text: list[str], start:int) -> np.ndarray:
    start_key = "lattice  \\"
    active = False
    R = np.zeros([3, 3])
    lat_row = 0
    for i, line in enumerate(text):
        if i > start:
            if active:
                if lat_row < 3:
                    R[lat_row, :] = [float(x) for x in line.split()[0:3]]
                    lat_row += 1
                else:
                    active = False
                    lat_row = 0
            elif start_key in line:
                active = True
    return R