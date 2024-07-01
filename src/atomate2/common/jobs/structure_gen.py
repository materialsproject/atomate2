### TODO:AmorphousMaker class from original MPMorph github repo. Plans to be integrated into atomate2

import os
import shutil
from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar


class AmorphousMaker:
    def __init__(
        self,
        el_num_dict: dict,
        box_scale: Union[float, List[float]],
        tol: float = 2.0,
        packmol_path: str = "packmol",
        clean: bool = True,
        xyz_paths: List = None,
        time_seed: bool = True,
    ):
        """
        Class for generating initial constrained-random packed structures for the
        simulation of amorphous or liquid structures. This is a wrapper for "packmol" package.
        Only works for cubic boxes for now.
        Args:
            el_num_dict (dict): dictionary of number of atoms of each species. If
                number of molecules is specified, an xyz file with the same name
                needs to be provided as xyz_paths.
                e.g. {"V":22, "Li":10, "O":75, "B":10}
                e.g. {"H2O": 20}
            box_scale (float) or (numpy array): all lattice vectors are multiplied with this.
                e.g. if one scalar value is given, it is the edge length of a cubic
                simulation box (e.g. if np.array([1.2, 0.9, 1.0]) is given, the unit
                lattice vectors will be multiplied with this.
            tol (float): tolerance factor for how close the atoms can get (angstroms).
                e.g. tol = 2.0 angstroms
            packmol_path (str): path to the packmol executable
            clean (bool): whether the intermedite files generated are deleted.
            xyz_paths (list): list of paths (str) to xyz files corresponding to
                molecules, if given so in el_num_dict. File names must match the
                molecule formula.
            time_seed (bool): whether to generate a random seed based on system time
        """
        self.el_num_dict = el_num_dict
        self.box_scale = box_scale
        self.tol = tol
        self._lattice = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self._structure = None
        self._el_dict = None
        self.packmol_path = packmol_path
        self.clean = clean
        self.xyz_paths = xyz_paths
        self.time_seed = time_seed
        if self.xyz_paths:
            assert len(self.xyz_paths) == len(self.el_num_dict.keys())
            self.clean = False

    def __repr__(self):
        return (
            "AmorphousMaker: generates constrained-random packed initial structure for"
            " MD using packmol."
        )

    @property
    def box(self):
        """
        Returns: box vectors scaled with box_scale
        """
        return (np.array(self._lattice) * self.box_scale).tolist()

    @property
    def random_packed_structure(self):
        """
        Returns: A constrained-random packed Structure object
        """
        self._el_dict = self.call_packmol()
        self._structure = self.get_structure(self._el_dict, self.box)
        return self._structure

    def call_packmol(self):
        """
        Returns
        -------
            A dict of coordinates of atoms for each element type
            e.g. {'V': [[4.969925, 8.409291, 5.462153], [9.338829, 9.638388, 9.179811], ...]
                  'Li': [[5.244308, 8.918049, 1.014577], [2.832759, 3.605796, 2.330589], ...]}
        """
        # this ensures periodic boundaries don't cause problems
        pm_l = self.tol / 2
        pm_h = self.box_scale - self.tol / 2
        try:
            len(pm_h)
        except:
            pm_h = [pm_h for i in range(3)]

        with open("packmol.input", "w") as f:
            f.write(
                "tolerance " + str(self.tol) + "\nfiletype xyz\noutput mixture.xyz\n"
            )
            for el in self.el_num_dict:
                f.write(
                    "structure "
                    + el
                    + ".xyz\n"
                    + "  number "
                    + str(self.el_num_dict[el])
                    + "\n  inside box"
                    + 3 * (" " + str(pm_l))
                    + (" " + str(pm_h[0]))
                    + (" " + str(pm_h[1]))
                    + (" " + str(pm_h[2]))
                    + "\nend structure\n\n"
                )

            if self.time_seed:
                f.write("seed -1\n")

            if self.xyz_paths:
                for path in self.xyz_paths:
                    try:
                        shutil.copy2(path, "./")
                    except:
                        pass
            else:
                for el in self.el_num_dict.keys():
                    with open(el + ".xyz", "w") as f:
                        f.write("1\ncomment\n" + el + " 0.0 0.0 0.0\n")

        try:
            os.system(self.packmol_path + " < packmol.input")
        except:
            raise OSError("packmol cannot be found!")
        if self.clean:
            for el in self.el_num_dict.keys():
                os.system("rm " + el + ".xyz")
            os.system("rm packmol.input")
        return self.xyz_to_dict("mixture.xyz")

    def xyz_to_dict(self, filename: str):
        """
        This is a generic xyz to dictionary converter.
        Used to get the structure from packmol output.
        """
        with open(filename) as f:
            lines = f.readlines()
            N = int(lines[0].rstrip("\n"))
            el_dict: dict[str, list] = {}
            for line in lines[2:]:
                l = line.rstrip("\n").split()
                if l[0] in el_dict:
                    el_dict[l[0]].append([float(i) for i in l[1:]])
                else:
                    el_dict[l[0]] = [[float(i) for i in l[1:]]]
        if sum([len(x) for x in el_dict.values()]) != N:
            raise ValueError("Inconsistent number of atoms")
        self._el_dict = OrderedDict(el_dict)  # type:ignore[assignment]
        if self.clean:
            os.system("rm " + filename)
        return self._el_dict

    @staticmethod
    def get_structure(el_dict: dict, lattice: List[List]):
        """
        Args:
            el_dict (dict): coordinates of atoms for each element type
            e.g. {'V': [[4.969925, 8.409291, 5.462153], [9.338829, 9.638388, 9.179811], ...]
                  'Li': [[5.244308, 8.918049, 1.014577], [2.832759, 3.605796, 2.330589], ...]}
            lattice (list): is the lattice in the form of [[x1,x2,x3],[y1,y2,y3],[z1,z2,z3]]
        Returns: pymatgen Structure
        """
        species = []
        coords = []
        for el in el_dict:
            for atom in el_dict[el]:
                species.append(el)
                coords.append(atom)
        return Structure(lattice, species, coords, coords_are_cartesian=True)

    def get_poscar(self):
        return Poscar(self.random_packed_structure)

    @staticmethod
    def xyzdict_to_poscar(el_dict: dict, lattice: List[List], filepath: str = "POSCAR"):
        """
        Generates XYZ file from element coordinate dictionary and lattice
        Args:
            el_dict (dict): coordinates of atoms for each element type
            e.g. {'V': [[4.969925, 8.409291, 5.462153], [9.338829, 9.638388, 9.179811], ...]
                  'Li': [[5.244308, 8.918049, 1.014577], [2.832759, 3.605796, 2.330589], ...]}
            lattice (list): is the lattice in the form of [[x1,x2,x3],[y1,y2,y3],[z1,z2,z3]]
            filepath (str): path to POSCAR to be generated
        Returns:
            writes a POSCAR file
        """
        with open(filepath, "w") as f:
            f.write("Parsed form XYZ file\n")
            f.write("1.0\n")
            for vec in lattice:
                f.write(" ".join([str(v) for v in vec]) + "\n")
            el_dict = OrderedDict(el_dict)
            for key in el_dict.keys():
                f.write(key + " ")
            f.write("\n")
            for key in el_dict.keys():
                f.write(str(len(el_dict[key])) + " ")
            f.write("\nCartesian\n")
            for key in el_dict.keys():
                for atom in el_dict[key]:
                    f.write(" ".join([str(i) for i in atom]) + "\n")


def get_random_packed(
    composition: Union[Composition, str],
    add_specie=None,
    target_atoms: int = 100,
    vol_per_atom: float = None,
    vol_exp: float = 1.0,
    modify_species: dict = None,
    use_time_seed: bool = True,
    mpr: Optional[MPRester] = None,
):
    """
    Helper method to use the AmorphousMaker to generate a randomly packed unit cell. If the volume (per atom) of the
    unit cell is not provided, the volume per atom will be predicted using structures from the materials project.

    :param composition: Formula or composition of the desired unit cell
    :param add_specie:
    :param target_atoms: Desired number of atoms in the unit cell. Generated unit cell will have at least this many
    atoms, although may be greater if the number of atoms in the specified formula/composition is evenly
    divide target_atoms.
    :param vol_per_atom: Volume of the unit cell in cubic Angstroms per atom (A^3/atom)
    :param vol_exp: Factor to multiply the volume by.
    :param modify_species:
    :param use_time_seed: Whether or not to use a time based random seed. Defaults to true
    :param mpr: custom MPRester object if additional specifications needed (such as API or endpoint)
    :return:
    """
    if isinstance(composition, str):
        composition = Composition(composition)
    if isinstance(add_specie, str):
        add_specie = Composition(add_specie)

    if vol_per_atom is None:
        if mpr is None:
            mpr = MPRester()

        comp_entries = mpr.get_entries(composition.reduced_formula, inc_structure=True)
        if len(comp_entries) > 0:
            vols = np.min(
                [
                    entry.structure.volume / entry.structure.num_sites
                    for entry in comp_entries
                ]
            )
        else:
            # Find all Materials project entries containing the elements in the
            # desired composition to estimate starting volume.
            _entries = mpr.get_entries_in_chemsys(
                [str(el) for el in composition.elements], inc_structure=True
            )
            entries = []
            for entry in _entries:
                if set(entry.structure.composition.elements) == set(
                    composition.elements
                ):
                    entries.append(entry)
                if len(entry.structure.composition.elements) >= 2:
                    entries.append(entry)

            vols = [
                entry.structure.volume / entry.structure.num_sites for entry in entries
            ]
        vol_per_atom = np.mean(vols)

    # Find total composition of atoms in the unit cell
    formula, factor = composition.get_integer_formula_and_factor()
    integer_composition = Composition(formula)
    full_cell_composition = integer_composition * np.ceil(
        target_atoms / integer_composition.num_atoms
    )
    if add_specie is not None:
        full_cell_composition += add_specie

    # Generate dict of elements and amounts for AmorphousMaker
    structure = {}
    for el in full_cell_composition:
        structure[str(el)] = int(full_cell_composition.element_composition.get(el))

    if modify_species is not None:
        for i, v in modify_species.items():
            structure[i] += v
    # use packmol to get a random configured structure
    packmol_path = os.environ["PACKMOL_PATH"]
    amorphous_maker_params = {
        "box_scale": (vol_per_atom * full_cell_composition.num_atoms * vol_exp)
        ** (1 / 3),
        "packmol_path": packmol_path,
        "xyz_paths": None,
        "time_seed": use_time_seed,
    }

    glass = AmorphousMaker(structure, **amorphous_maker_params)
    structure = glass.random_packed_structure
    return structure
