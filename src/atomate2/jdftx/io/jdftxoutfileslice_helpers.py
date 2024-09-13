""" Module containing helper functions for parsing JDFTx output files.

This module contains helper functions for parsing JDFTx output files.
"""

from typing import Optional


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


def find_key_first(key_input, tempfile):
    """
    Finds first instance of key in output file.

    Parameters
    ----------
    key_input: str
        key string to match
    tempfile: List[str]
        output from readlines() function in read_file method
    """
    key_input = str(key_input)
    line = None
    for i in range(len(tempfile)):
        if key_input in tempfile[i]:
            line = i
            break
    return line


def find_key(key_input, tempfile):
    """
    Finds last instance of key in output file.

    Parameters
    ----------
    key_input: str
        key string to match
    tempfile: List[str]
        output from readlines() function in read_file method
    """
    key_input = str(key_input)
    line = None
    lines = find_all_key(key_input, tempfile)
    if len(lines):
        line = lines[-1]
    return line


def find_first_range_key(
    key_input: str,
    tempfile: list[str],
    startline: int = 0,
    endline: int = -1,
    skip_pound: bool = False,
) -> list[int]:
    """
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

    """
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


def key_exists(key_input: str, tempfile: list[str]):
    """ Check if key_input exists in tempfile.

    Search through tempfile for key_input. Return True if found,
    False otherwise.

    Parameters
    ----------
    key_input: str
        key string to match
    tempfile: List[str]
        output from readlines() function in read_file method

    Returns
    -------
    bool
        True if key_input exists in tempfile, False otherwise
    """
    line = find_key(key_input, tempfile)
    if line == None:
        return False
    return True


def find_all_key(key_input: str, tempfile: list[str], startline: int = 0):
    """ Find all lines containing key_input.
    
    Search through tempfile for all lines containing key_input. Returns a list
    of line numbers.
    
    Parameters
    ----------
    key_input: str
        key string to match
    tempfile: List[str]
        output from readlines() function in read_file method
    startline: int
        line to start searching from
        
    Returns
    -------
    line_list: list[int]
        list of line numbers where key_input occurs
    """
    line_list = []  # default
    for i in range(startline, len(tempfile)):
        if key_input in tempfile[i]:
            line_list.append(i)
    return line_list