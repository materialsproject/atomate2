import os
from functools import wraps
import math
from ase import Atom, Atoms
from jdftx.io.JMinSettings import JMinSettings, JMinSettingsElectronic, JMinSettingsFluid, JMinSettingsIonic, JMinSettingsLattice
import numpy as np
from dataclasses import dataclass, field
import scipy.constants as const
from atomate2.jdftx.io.data import atom_valence_electrons
from jdftx.io.JStructures import JStructures
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

    Parameters
    ----------
    filename: Path or str
        name of file to read

    Returns
    -------
    text: list[str]
        list of strings from file
    '''
    with open(file_name, 'r') as f:
        text = f.readlines()
    return text


@check_file_exists
def read_outfile(file_name: str, out_slice_idx: int = -1) -> list[str]:
    '''
    Read slice of out file into a list of str

    Parameters
    ----------
    filename: Path or str
        name of file to read
    out_slice_idx: int
        index of slice to read from file

    Returns
    -------
    text: list[str]
        list of strings from file
    '''
    with open(file_name, 'r') as f:
        _text = f.readlines()
    start_lines = get_start_lines(text, add_end=True)
    text = _text[start_lines[out_slice_idx]:start_lines[out_slice_idx+1]]
    return text

def get_start_lines(text: list[str], start_key: Optional[str]="*************** JDFTx", add_end: Optional[bool]=False) -> list[int]:
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
    if add_end:
        start_lines.append(i)
    return start_lines


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


def find_first_range_key(key_input: str, tempfile: list[str], startline: int=0, endline: int=-1, skip_pound:bool = False) -> list[int]:
    '''
    Find all lines that exactly begin with key_input in a range of lines

    Parameters
    ----------
    key_input: str
        key string to match
    tempfile: List[str]
        output from readlines() function in read_file method
    startline: int
        line to start searching from
    endline: int
        line to stop searching at
    skip_pound: bool
        whether to skip lines that begin with a pound sign

    Returns
    -------
    L: list[int]
        list of line numbers where key_input occurs
    
    '''
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
    # Ben: I don't think this is deprecated by find_first_range_key, since this function
    # doesn't require the key to be at the beginning of the line
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

    jstrucs: JStructures = None
    jsettings_fluid: JMinSettingsFluid = None
    jsettings_electronic: JMinSettingsElectronic = None
    jsettings_lattice: JMinSettingsLattice = None
    jsettings_ionic: JMinSettingsIonic = None

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

    # def _get_start_lines(self, text:str, start_key: Optional[str]="*************** JDFTx", add_end: Optional[bool]=False) -> list[int]:
    #     '''
    #     Get the line numbers corresponding to the beginning of seperate JDFTx calculations
    #     (in case of multiple calculations appending the same out file)

    #     Args:
    #         text: output of read_file for out file
    #     '''
    #     start_lines = []
    #     for i, line in enumerate(text):
    #         if start_key in line:
    #             start_lines.append(i)
    #     if add_end:
    #         start_lines.append(i)
    #     return start_lines

    def _get_prefix(text: list[str]) -> str:
        '''
        Get output prefix from the out file

        Parameters
        ----------
            text: list[str]
                output of read_file for out file

        Returns
        -------
            prefix: str
                prefix of dump files for JDFTx calculation
        '''
        prefix = None
        line = find_key('dump-name', text)
        dumpname = text[line].split()[1]
        if "." in dumpname:
            prefix = dumpname.split('.')[0]
        return prefix
    
    def _get_spinvars(text: list[str]) -> tuple[str, int]:
        '''
        Set spintype and Nspin from out file text for instance

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
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
    
    def _get_broadeningvars(text:list[str]) -> tuple[str, float]:
        '''
        Get broadening type and value from out file text

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        '''
        line = find_key('elec-smearing ', text)
        if not line is None:
            broadening_type = text[line].split()[1]
            broadening = float(text[line].split()[2]) * Ha_to_eV
        else:
            broadening_type = None
            broadening = 0
        return broadening_type, broadening
    
    def _get_truncationvars(text:list[str]) -> tuple[str, float]:
        '''
        Get truncation type and value from out file text

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
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
    
    def _get_elec_cutoff(text:list[str]) -> float:
        '''
        Get the electron cutoff from the out file text

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        '''
        line = find_key('elec-cutoff ', text)
        pwcut = float(text[line].split()[1]) * Ha_to_eV
        return pwcut

    def _get_fftgrid(text:list[str]) -> list[int]:
        '''
        Get the FFT grid from the out file text

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        '''
        line = find_key('Chosen fftbox size', text)
        fftgrid = [int(x) for x in text[line].split()[6:9]]
        return fftgrid

    def _get_kgrid(text:list[str]) -> list[int]:
        '''
        Get the kpoint grid from the out file text

        Parameters
        ----------
            text: list[str]
                output of read_file for out file
        '''
        line = find_key('kpoint-folding ', text)
        kgrid = [int(x) for x in text[line].split()[1:4]]
        return kgrid
    
    def _get_eigstats_varsdict(self, text:list[str], prefix:str | None) -> dict[str, float]:
        '''
        Get the eigenvalue statistics from the out file text
        
        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        prefix: str
            prefix for the eigStats section in the out file
        
        Returns
        -------
        varsdict: dict[str, float]
            dictionary of eigenvalue statistics
        '''
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
    
    def _set_eigvars(self, text:list[str]) -> None:
        '''
        Set the eigenvalue statistics variables
        
        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        '''
        eigstats = self._get_eigstats_varsdict(text, self.prefix)
        self.Emin = eigstats["Emin"]
        self.HOMO = eigstats["HOMO"]
        self.EFermi = eigstats["EFermi"]
        self.LUMO = eigstats["LUMO"]
        self.Emax = eigstats["Emax"]
        self.Egap = eigstats["Egap"]
    

    def _get_pp_type(self, text:list[str]) -> str:
        '''
        Get the pseudopotential type used in calculation

        Parameters
        ----------
        text: list[str]
            output of read_file for out file

        Returns
        ----------
        pptype: str
            Pseudopotential library used
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
    
    
    def _set_pseudo_vars(self, text:list[str]) -> None:
        '''
        Set the pseudopotential variables   

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        '''
        self.pp_type = self._get_pp_type(text)
        if self.pp_type == "SG15":
            self._set_pseudo_vars_SG15(text)
        elif self.pp_type == "GBRV":
            self._set_pseudo_vars_GBRV(text)
    
    def _set_pseudo_vars_SG15(self, text:list[str]) -> None:
        '''
        Set the pseudopotential variables for SG15 pseudopotentials

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
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


    def _set_pseudo_vars_GBRV(self, text:list[str]) -> None:
        ''' TODO: implement this method
        '''
        self.total_electrons_uncharged = None
        self.valence_electrons_uncharged = None
        self.semicore_electrons_uncharged = None
        self.semicore_electrons = None
        self.valence_electrons = None


    def _collect_settings_lines(self, text:list[str], start_key:str) -> list[int]:
        '''
        Collect the lines of settings from the out file text

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        start_key: str
            key to start collecting settings lines

        Returns
        -------
        lines: list[int]
            list of line numbers where settings occur
        '''
        started = False
        lines = []
        for i, line in enumerate(text):
            if started:
                if line.strip().split()[-1].strip() == "\\":
                    lines.append(i)
                else:
                    started = False
            elif start_key in line:
                started = True
                #lines.append(i) # we DONT want to do this
            elif len(lines):
                break
        return lines

    def _create_settings_dict(self, text:list[str], start_key:str) -> dict:
        '''
        Create a dictionary of settings from the out file text

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        start_key: str
            key to start collecting settings lines

        Returns
        -------
        settings_dict: dict
            dictionary of settings
        '''
        lines = self._collect_settings_lines(text, start_key)
        settings_dict = {}
        for line in lines:
            line_text_list = text[line].strip().split()
            key = line_text_list[0]
            value = line_text_list[1]
            settings_dict[key] = value
        return settings_dict
    
    def _get_settings_object(self, text:list[str], settings_class: JMinSettings) -> JMinSettings:
        '''
        Get the settings object from the out file text
        
        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        settings_class: JMinSettings
            settings class to create object from
            
        Returns
        -------
        settings_obj: JMinSettings
            settings object
        '''
        settings_dict = self._create_settings_dict(text, settings_class.start_key)
        if len(settings_dict):
            settings_obj = settings_class(**settings_dict)
        else:
            settings_obj = None
        return settings_obj
    

    def _set_min_settings(self, text:list[str]) -> None:
        '''
        Set the settings objects from the out file text

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        '''
        self.jsettings_fluid = self._get_settings_object(text, JMinSettingsFluid)
        self.jsettings_electronic = self._get_settings_object(text, JMinSettingsElectronic)
        self.jsettings_lattice = self._get_settings_object(text, JMinSettingsLattice)
        self.jsettings_ionic = self._get_settings_object(text, JMinSettingsIonic)

    def _set_geomopt_vars(self, text:list[str]) -> None:
        ''' 
        Set vars geom_opt and geom_opt_type for initializing self.jstrucs

        Parameters
        ----------
            text: list[str]
                output of read_file for out file
        '''
        if self.jsettings_ionic is None or self.jsettings_lattice is None:
            self._set_min_settings(text)
        #
        if self.jsettings_ionic is None or self.jsettings_lattice is None:
            raise ValueError("Unknown issue in setting settings objects")
        else:
            if self.jsettings_lattice.nIterations > 0:
                self.geom_opt = True
                self.geom_opt_type = "lattice"
            elif self.jsettings_ionic.nIterations > 0:
                self.geom_opt = True
                self.geom_opt_type = "ionic"
            else:
                self.geom_opt = False
                self.geom_opt_type = "single point"


    def _set_jstrucs(self, text:list[str]) -> None:
        '''
        Set the JStructures object from the out file text

        Parameters
        ----------
            text: list[str]
                output of read_file for out file
        '''
        self.jstrucs = JStructures.from_out_slice(text, iter_type=self.geom_opt_type)


    def _set_orb_fillings(self) -> None:
        '''
        Calculate and set HOMO and LUMO fillings
        '''
        if self.broadening_type is not None:
            self.HOMO_filling = (2 / self.Nspin) * self.calculate_filling(self.broadening_type, self.broadening, self.HOMO, self.EFermi)
            self.LUMO_filling = (2 / self.Nspin) * self.calculate_filling(self.broadening_type, self.broadening, self.LUMO, self.EFermi)
        else:
            self.HOMO_filling = (2 / self.Nspin)
            self.LUMO_filling = 0


    def _set_fluid(self, text: list[str]) -> None: # Is this redundant to the fluid settings?
        '''
        Set the fluid class variable
        
        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        '''
        line = find_first_range_key('fluid ', text)
        self.fluid = text[line[0]].split()[1]
        if self.fluid == 'None':
            self.fluid = None


    def _set_total_electrons(self, text:str) -> None:
        '''
        Set the total_Electrons class variable

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        '''
        lines = find_all_key('nElectrons', text)
        if len(lines) > 1:
            idx = 4
        else:
            idx = 1  #nElectrons was not printed in scf iterations then
        self.total_electrons = float(text[lines[-1]].split()[idx])

    def _set_Nbands(self, text: list[str]) -> None:
        '''
        Set the Nbands class variable

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        '''
        lines = self.find_all_key('elec-n-bands', text)
        line = lines[0]
        nbands = int(text[line].strip().split()[-1].strip())
        self.Nbands = nbands

    def _set_atom_vars(self, text: list[str]) -> None:
        '''
        Set the atom variables
        
        Parameters
        ----------
        text: list[str]
            output of read_file for out file'''
        startline = find_key('Input parsed successfully', text)
        endline = find_key('---------- Initializing the Grid ----------', text)
        lines = find_first_range_key('ion ', text, startline = startline, endline = endline)
        atom_elements = [text[x].split()[1] for x in lines]
        self.Nat = len(atom_elements)
        atom_coords = [text[x].split()[2:5] for x in lines]
        self.atom_coords_initial = np.array(atom_coords, dtype = float)
        atom_types = []
        for x in atom_elements:
            if not x in atom_types:
                atom_types.append(x)
        self.atom_elements = atom_elements
        mapping_dict = dict(zip(atom_types, range(1, len(atom_types) + 1)))
        self.atom_elements_int = [mapping_dict[x] for x in self.atom_elements]
        self.atom_types = atom_types
        line = find_key('# Ionic positions in', text) + 1
        coords = np.array([text[i].split()[2:5] for i in range(line, line + self.Nat)], dtype = float)
        self.atom_coords_final = coords
        self.atom_coords = self.atom_coords_final.copy()

    def _set_lattice_vars(self, text: list[str]) -> None:
        '''
        Set the lattice variables
        
        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        '''
        lines = find_all_key('R =', text)
        line = lines[0]
        lattice_initial = np.array([x.split()[1:4] for x in text[(line + 1):(line + 4)]], dtype = float).T / ang_to_bohr
        self.lattice_initial = lattice_initial.copy()
        templines = find_all_key('LatticeMinimize', text)
        if len(templines) > 0:
            line = templines[-1]
            lattice_final = np.array([x.split()[1:4] for x in text[(line + 1):(line + 4)]], dtype = float).T / ang_to_bohr
            self.lattice_final = lattice_final.copy()
            self.lattice = lattice_final.copy()
        else:
            self.lattice = lattice_initial.copy()
        self.a, self.b, self.c = np.sum(self.lattice**2, axis = 1)**0.5


    def _set_ecomponents(self, text: list[str]) -> None:
        '''
        Set the energy components dictionary
        
        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        '''
        line = find_key("# Energy components:", text)
        self.Ecomponents = self._read_ecomponents(line, text)








    
    

    @classmethod
    def from_file(cls, file_name: str):
        '''
        Read file into class object

        Args:
            file_name: file to read
        '''
        instance = cls()

        #text = read_file(file_name)
        text = read_outfile(file_name)
        instance._set_min_settings(text)
        instance._set_geomopt_vars(text)
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
        instance._set_orb_fillings()
        instance.is_metal = instance._determine_is_metal()
        instance._set_fluid(text)
        instance._set_total_electrons(text)
        instance._set_Nbands()
        instance._set_atom_vars(text)
        instance._set_pseudo_vars(text)
        instance._set_lattice_vars(text)
        instance.has_solvation = instance.check_solvation()

        instance._set_jstrucs(text)

        #@ Cooper added @#
        instance.is_gc = key_exists('target-mu', text)
        instance._set_ecomponents(text)
        # instance._build_trajectory(templines)

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
    

    def _determine_is_metal(self) -> bool:
        '''
        Determine if the system is a metal based on the fillings of HOMO and LUMO

        Returns
        --------
        is_metal: bool
            True if system is metallic
        '''
        TOL_PARTIAL = 0.01
        is_metal = True
        if self.HOMO_filling / (2 / self.Nspin) > (1 - TOL_PARTIAL) and self.LUMO_filling / (2 / self.Nspin) < TOL_PARTIAL:
            is_metal = False
        return is_metal

    def check_solvation(self) -> bool:
        '''
        Check if calculation used implicit solvation
        
        Returns
        --------
        has_solvation: bool
            True if calculation used implicit solvation
        '''
        has_solvation = self.fluid is not None
        return has_solvation

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
            ecomp = self._read_ecomponents(ecomp_line, text)
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

    def _read_ecomponents(self, line:int, text:str) -> dict:
        '''
        Read the energy components from the out file text
        
        Parameters
        ----------
        line: int
            line number where energy components are found
        text: list[str]
            output of read_file for out file
        
        Returns
        -------
        Ecomponents: dict
            dictionary of energy components
        '''
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
