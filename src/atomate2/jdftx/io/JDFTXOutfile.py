import os
from functools import wraps
import math
from ase import Atom, Atoms
from atomate2.jdftx.io.JMinSettings import JMinSettings, JMinSettingsElectronic, JMinSettingsFluid, JMinSettingsIonic, JMinSettingsLattice
from atomate2.jdftx.io.JDFTXOutfileSlice import JDFTXOutfileSlice
import numpy as np
from dataclasses import dataclass, field
import scipy.constants as const
from atomate2.jdftx.io.data import atom_valence_electrons
from atomate2.jdftx.io.JStructures import JStructures
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

@check_file_exists
def read_outfile_slices(file_name: str) -> list[list[str]]:
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
    texts: list[list[str]]
        list of out file slices (individual calls of JDFTx)
    '''
    with open(file_name, 'r') as f:
        _text = f.readlines()
    start_lines = get_start_lines(_text, add_end=True)
    texts = []
    for i in range(len(start_lines)-1):
        text = _text[start_lines[i]:start_lines[i+1]]
        texts.append(text)
    return texts

@dataclass
class JDFTXOutfile(List[JDFTXOutfileSlice], ClassPrintFormatter):
    '''
    A class to read and process a JDFTx out file
    '''

    @classmethod
    def from_file(cls, file_path: str):
        texts = read_outfile_slices(file_path)
        instance = cls()
        for text in texts:
            instance.append(JDFTXOutfileSlice.from_out_slice(text))
        pass

    def __getattr__(self, name):
        if self:
            return getattr(self[-1], name)
        raise AttributeError(f"'JDFTXOutfile' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self.__annotations__:
            super().__setattr__(name, value)
        elif self:
            setattr(self[-1], name, value)
        else:
            raise AttributeError(f"'JDFTXOutfile' object has no attribute '{name}'")
