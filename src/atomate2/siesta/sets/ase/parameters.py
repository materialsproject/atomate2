# fmt: off
import logging

from ase.calculators.calculator import Parameters
from ase.units import Ry, eV

logger = logging.getLogger(__name__)
"""
2017.04 - Pedro Brandimarte: changes for python 2-3 compatible
"""
meV = 0.001 * eV


class PAOBasisBlock(Parameters):
    """
    Representing a block in PAO.Basis for one species.
    """

    def __init__(self, block):
        """
        Parameters
        ----------
            -block : String. A block defining the basis set of a single
                     species using the format of a PAO.Basis block.
                     The initial label should be left out since it is
                     determined programatically.
                     Example1: 2 nodes 1.0
                               n=2 0 2 E 50.0 2.5
                               3.50 3.50
                               0.95 1.00
                               1 1 P 2
                               3.50
                     Example2: 1
                               0 2 S 0.2
                               5.00 0.00
                     See siesta manual for details.
        """
        assert isinstance(block, str)
        Parameters.__init__(self, block=block)

    def script(self, label):
        """
        Write the fdf script for the block.

        Parameters
        ----------
            -label : The label to insert in front of the block.
        """
        return label + " " + self["block"]


class Species(Parameters):
    """
    Parameters for specifying the behaviour for a single species in the
    calculation. If the tag argument is set to an integer then atoms with
    the specified element and tag will be a separate species.

    Pseudopotential and basis set can be specified. Additionally the species
    can be set be a ghost species, meaning that they will not be considered
    atoms, but the corresponding basis set will be used.
    """

    def __init__(self,
                 symbol,
                 basis_set="DZP",
                 pseudopotential=None,
                 tag=None,
                 ghost=False,
                 excess_charge=None):
        kwargs = locals()
        kwargs.pop("self")
        Parameters.__init__(self, **kwargs)



class SiestaParameters(Parameters):
    """
    Parameter class for SIESTA calculator, extending ASE Parameters.

    Attributes
    ----------
        label (str): Base name for input/output files.
        mesh_cutoff (float): Mesh cutoff energy in eV for grid points.
        energy_shift (float): Confining energy for basis set generation in eV.
        kpts: Tuple of k-points for Brillouin zone sampling.
        xc (str): Exchange-correlation functional (e.g., 'LDA', 'GGA').
        basis_set (str): Basis set type (e.g., 'SZ', 'DZP').
        spin (str): Spin configuration ('non-polarized', 'collinear', etc.).
        species: List of species objects for basis and pseudopotential specification.
        pseudo_qualifier (str): Qualifier for pseudopotential file names.
        pseudo_path (str): Directory path for pseudopotential files.
        symlink_pseudos (bool): Whether to symlink or copy pseudopotentials.
        atoms (Atoms): ASE Atoms object for the system.
        restart (str): Path to restart file (if any).
        fdf_arguments (dict): Additional FDF arguments for SIESTA input.
        atomic_coord_format (str): Format for atomic coordinates ('xyz' or 'zmatrix').
        bandpath: Band path for band structure calculations.
        structure_fdf (str): Name of separate structure file to generate (default: 'structure.fdf').
                            The main FDF will use %include to reference this file.
                            Set to None to inline structure in main FDF file.
    """

    def __init__(
            self,
            label="siesta",
            mesh_cutoff=200 * Ry,
            energy_shift=100 * meV,
            kpts=None,
            xc="LDA",
            basis_set="DZP",
            spin="non-polarized",
            species=(),
            pseudo_qualifier=None,
            pseudo_path=None,
            symlink_pseudos=None,
            atoms=None,
            restart=None,
            fdf_arguments=None,
            atomic_coord_format="xyz",
            bandpath=None,
            structure_fdf="structure.fdf",
            ):
        logger.info("SiestaParameters.__init__()")
        kwargs = locals()
        kwargs.pop("self")
        Parameters.__init__(self, **kwargs)
