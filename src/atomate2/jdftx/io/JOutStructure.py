from dataclasses import dataclass

import numpy as np
from pymatgen.core.structure import Lattice, Structure
from pymatgen.core.units import Ha_to_eV, bohr_to_ang

from atomate2.jdftx.io.JEiters import JEiters


@dataclass
class JOutStructure(Structure):
    """
    A mutant of the pymatgen Structure class for flexiblity in holding JDFTx optimization data
    """

    iter_type: str = None
    etype: str = None
    eiter_type: str = None
    emin_flag: str = None
    Ecomponents: dict = None
    elecMinData: JEiters = None
    stress: np.ndarray = None
    strain: np.ndarray = None
    iter: int = None
    E: float = None
    grad_K: float = None
    alpha: float = None
    linmin: float = None
    t_s: float = None
    geom_converged: bool = False
    geom_converged_reason: str = None
    line_types = [
        "emin",
        "lattice",
        "strain",
        "stress",
        "posns",
        "forces",
        "ecomp",
        "lowdin",
        "opt",
    ]

    def __init__(
        self,
        lattice: np.ndarray,
        species: list[str],
        coords: list[np.ndarray],
        site_properties: dict[str, list],
    ):
        super().__init__(
            lattice=lattice,
            species=species,
            coords=coords,
            site_properties=site_properties,
        )

    @classmethod
    def from_text_slice(
        cls,
        text_slice: list[str],
        eiter_type: str = "ElecMinimize",
        iter_type: str = "IonicMinimize",
        emin_flag: str = "---- Electronic minimization -------",
    ):
        """
        Create a JAtoms object from a slice of an out file's text corresponding
        to a single step of a native JDFTx optimization

        Parameters
        ----------
        text_slice: list[str]
            A slice of text from a JDFTx out file corresponding to a single optimization step / SCF cycle
        eiter_type: str
            The type of electronic minimization step
        iter_type: str
            The type of optimization step
        emin_flag: str
            The flag that indicates the start of a log message for a JDFTx optimization step
        """
        # instance = super.__init__(lattice=np.eye(3), species=[], coords=[], site_properties={})
        instance = cls(lattice=np.eye(3), species=[], coords=[], site_properties={})
        if iter_type not in ["IonicMinimize", "LatticeMinimize"]:
            iter_type = instance.correct_iter_type(iter_type)
        instance.eiter_type = eiter_type
        instance.iter_type = iter_type
        instance.emin_flag = emin_flag
        line_collections = instance.init_line_collections()
        for i, line in enumerate(text_slice):
            read_line = False
            for line_type in line_collections:
                sdict = line_collections[line_type]
                if sdict["collecting"]:
                    lines, collecting, collected = instance.collect_generic_line(
                        line, sdict["lines"]
                    )
                    sdict["lines"] = lines
                    sdict["collecting"] = collecting
                    sdict["collected"] = collected
                    read_line = True
                    break
            if not read_line:
                for line_type in line_collections:
                    if not line_collections[line_type]["collected"]:
                        if instance.is_generic_start_line(line, line_type):
                            line_collections[line_type]["collecting"] = True
                            line_collections[line_type]["lines"].append(line)
                            break

        # Ecomponents needs to be parsed before emin to set etype
        instance.parse_ecomp_lines(line_collections["ecomp"]["lines"])
        instance.parse_emin_lines(line_collections["emin"]["lines"])
        # Lattice must be parsed before posns/forces incase of direct coordinates
        instance.parse_lattice_lines(line_collections["lattice"]["lines"])
        instance.parse_posns_lines(line_collections["posns"]["lines"])
        instance.parse_forces_lines(line_collections["forces"]["lines"])
        # Strain and stress can be parsed in any order
        instance.parse_strain_lines(line_collections["strain"]["lines"])
        instance.parse_stress_lines(line_collections["stress"]["lines"])
        # Lowdin must be parsed after posns
        instance.parse_lowdin_lines(line_collections["lowdin"]["lines"])
        # Opt line must be parsed after ecomp
        instance.parse_opt_lines(line_collections["opt"]["lines"])

        return instance

    def correct_iter_type(self, iter_type: str | None) -> str | None:
        """
        Corrects the iter_type string to match the JDFTx convention

        Parameters
        ----------
        iter_type:
            The type of optimization step

        Returns
        -------
        iter_type: str | None
            The corrected type of optimization step
        """
        if iter_type is not None:
            if "lattice" in iter_type.lower():
                iter_type = "LatticeMinimize"
            elif "ionic" in iter_type.lower():
                iter_type = "IonicMinimize"
            else:
                iter_type = None
        return iter_type

    def init_line_collections(self) -> dict:
        # TODO: Move line_collections to be used as a class variable
        """
        Initializes a dictionary of line collections for each type of line in a JDFTx out file

        Returns
        -------
        line_collections: dict
            A dictionary of line collections for each type of line in a JDFTx out file
        """
        line_collections = {}
        for line_type in self.line_types:
            line_collections[line_type] = {
                "lines": [],
                "collecting": False,
                "collected": False,
            }
        return line_collections

    def is_emin_start_line(self, line_text: str) -> bool:
        """
        Returns True if the line_text is the start of a log message for a JDFTx optimization step

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx optimization step
        """
        is_line = self.emin_flag in line_text
        return is_line

    def parse_emin_lines(self, emin_lines: list[str]) -> None:
        """
        Parses the lines of text corresponding to the electronic minimization data of a JDFTx out file

        Parameters
        ----------
        emin_lines: list[str]
            A list of lines of text from a JDFTx out file containing the electronic minimization data
        """
        if len(emin_lines):
            self.elecMinData = JEiters.from_text_slice(
                emin_lines, iter_type=self.eiter_type, etype=self.etype
            )

    def is_lattice_start_line(self, line_text: str) -> bool:
        """
        Returns True if the line_text is the start of a log message for a JDFTx optimization step

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx optimization step
        """
        is_line = "# Lattice vectors:" in line_text
        return is_line

    def parse_lattice_lines(self, lattice_lines: list[str]) -> None:
        """
        Parses the lines of text corresponding to the lattice vectors of a JDFTx out file

        Parameters
        ----------
        lattice_lines: list[str]
            A list of lines of text from a JDFTx out file containing the lattice vectors
        """
        R = None
        if len(lattice_lines):
            R = self._bracket_num_list_str_of_3x3_to_nparray(lattice_lines, i_start=2)
            R = R.T * bohr_to_ang
            self.lattice = Lattice(R)

    def is_strain_start_line(self, line_text: str) -> bool:
        """
        Returns True if the line_text is the start of a log message for a JDFTx optimization step

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx optimization step
        """
        is_line = "# Strain tensor in" in line_text
        return is_line

    def parse_strain_lines(self, strain_lines: list[str]) -> None:
        """
        Parses the lines of text corresponding to the strain tensor of a JDFTx out file

        Parameters
        ----------
        strain_lines: list[str]
            A list of lines of text from a JDFTx out file containing the strain tensor
        """
        ST = None
        if len(strain_lines):
            ST = self._bracket_num_list_str_of_3x3_to_nparray(strain_lines, i_start=1)
            ST = ST.T * 1  # Conversion factor?
        self.strain = ST

    def is_stress_start_line(self, line_text: str) -> bool:
        """
        Returns True if the line_text is the start of a log message for a JDFTx optimization step

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx optimization step
        """
        is_line = "# Stress tensor in" in line_text
        return is_line

    def parse_stress_lines(self, stress_lines: list[str]) -> None:
        """
        Parses the lines of text corresponding to the stress tensor of a JDFTx out file

        Parameters
        ----------
        stress_lines: list[str]
            A list of lines of text from a JDFTx out file containing the stress tensor
        """
        ST = None
        if len(stress_lines):
            ST = self._bracket_num_list_str_of_3x3_to_nparray(stress_lines, i_start=1)
            ST = ST.T * 1  # Conversion factor?
        self.stress = ST

    def is_posns_start_line(self, line_text: str) -> bool:
        """
        Returns True if the line_text is the start of a log message for a JDFTx optimization step

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file containing the positions of atoms

        Returns
        -------
            is_line: bool
                True if the line_text is the start of a log message for a JDFTx optimization step
        """
        is_line = "# Ionic positions" in line_text
        return is_line

    def parse_posns_lines(self, posns_lines: list[str]) -> None:
        """
        Parses the lines of text corresponding to the positions of a JDFTx out file

        Parameters
        ----------
        posns_lines: list[str]
            A list of lines of text from a JDFTx out file
        """
        nAtoms = len(posns_lines) - 1
        coords_type = posns_lines[0].split("positions in")[1].strip().split()[0].strip()
        posns = []
        names = []
        for i in range(nAtoms):
            line = posns_lines[i + 1]
            name = line.split()[1].strip()
            posn = np.array([float(x.strip()) for x in line.split()[2:5]])
            names.append(name)
            posns.append(posn)
        posns = np.array(posns)
        if coords_type.lower() != "cartesian":
            posns = np.dot(posns, self.lattice.matrix)
        else:
            posns *= bohr_to_ang
        for i in range(nAtoms):
            self.append(species=names[i], coords=posns[i], coords_are_cartesian=True)

    def is_forces_start_line(self, line_text: str) -> bool:
        """
        Returns True if the line_text is the start of a log message for a JDFTx optimization step

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx optimization step
        """
        is_line = "# Forces in" in line_text
        return is_line

    def parse_forces_lines(self, forces_lines: list[str]) -> None:
        """
        Parses the lines of text corresponding to the forces of a JDFTx out file

        Parameters
        ----------
        forces_lines: list[str]
            A list of lines of text from a JDFTx out file containing the forces
        """
        nAtoms = len(forces_lines) - 1
        coords_type = forces_lines[0].split("Forces in")[1].strip().split()[0].strip()
        forces = []
        for i in range(nAtoms):
            line = forces_lines[i + 1]
            force = np.array([float(x.strip()) for x in line.split()[2:5]])
            forces.append(force)
        forces = np.array(forces)
        if coords_type.lower() != "cartesian":
            forces = np.dot(
                forces, self.lattice.matrix
            )  # TODO: Double check this conversion
            # (since self.cell is in Ang, and I need the forces in eV/ang, how
            # would you convert forces from direct coordinates into cartesian?)
        else:
            forces *= 1 / bohr_to_ang
        forces *= Ha_to_eV
        self.forces = forces

    def is_ecomp_start_line(self, line_text: str) -> bool:
        """
        Returns True if the line_text is the start of a log message for a JDFTx optimization step

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx optimization step
        """
        is_line = "# Energy components" in line_text
        return is_line

    def parse_ecomp_lines(self, ecomp_lines: list[str]) -> None:
        """
        Parses the lines of text corresponding to the energy components of a JDFTx out file

        Parameters
        ----------
        ecomp_lines: list[str]
            A list of lines of text from a JDFTx out file
        """
        self.Ecomponents = {}
        for line in ecomp_lines:
            if " = " in line:
                lsplit = line.split(" = ")
                key = lsplit[0].strip()
                val = float(lsplit[1].strip())
                self.Ecomponents[key] = val * Ha_to_eV
        if self.etype is None:
            self.etype = key

    def is_lowdin_start_line(self, line_text: str) -> bool:
        """
        Returns True if the line_text is the start of a Lowdin population analysis in a JDFTx out file

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a Lowdin population analysis in a JDFTx out file
        """
        is_line = "#--- Lowdin population analysis ---" in line_text
        return is_line

    def parse_lowdin_lines(self, lowdin_lines: list[str]) -> None:
        """
        Parses the lines of text corresponding to a Lowdin population analysis in a JDFTx out file

        Parameters
        ----------
        lowdin_lines: list[str]
            A list of lines of text from a JDFTx out file
        """
        charges_dict = {}
        moments_dict = {}
        for line in lowdin_lines:
            if self.is_charges_line(line):
                charges_dict = self.parse_lowdin_line(line, charges_dict)
            elif self.is_moments_line(line):
                moments_dict = self.parse_lowdin_line(line, moments_dict)
        names = [s.name for s in self.species]
        charges = None
        moments = None
        if len(charges_dict):
            charges = np.zeros(len(names))
            for el in charges_dict:
                idcs = [i for i in range(len(names)) if names[i] == el]
                for i, idx in enumerate(idcs):
                    charges[idx] += charges_dict[el][i]
        if len(moments_dict):
            moments = np.zeros(len(names))
            for el in moments_dict:
                idcs = [i for i in range(len(names)) if names[i] == el]
                for i, idx in enumerate(idcs):
                    moments[idx] += moments_dict[el][i]
        self.charges = charges
        self.magnetic_moments = moments

    def is_charges_line(self, line_text: str) -> bool:
        """
        Returns True if the line_text is a line of text from a JDFTx out file corresponding to a Lowdin population analysis

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is a line of text from a JDFTx out file corresponding to a Lowdin population
        """
        is_line = "oxidation-state" in line_text
        return is_line

    def is_moments_line(self, line_text: str) -> bool:
        """
        Returns True if the line_text is a line of text from a JDFTx out file corresponding to a Lowdin population analysis

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is a line of text from a JDFTx out file corresponding to a Lowdin population
        """
        is_line = "magnetic-moments" in line_text
        return is_line

    def parse_lowdin_line(
        self, lowdin_line: str, lowdin_dict: dict[str, float]
    ) -> dict[str, float]:
        """
        Parses a line of text from a JDFTx out file corresponding to a Lowdin population analysis

        Parameters
        ----------
        lowdin_line: str
            A line of text from a JDFTx out file
        lowdin_dict: dict[str, float]
            A dictionary of Lowdin population analysis data

        Returns
        -------
        lowdin_dict: dict[str, float]
            A dictionary of Lowdin population analysis data
        """
        tokens = [v.strip() for v in lowdin_line.strip().split()]
        name = tokens[2]
        vals = [float(x) for x in tokens[3:]]
        lowdin_dict[name] = vals
        return lowdin_dict

    def is_opt_start_line(self, line_text: str) -> bool:
        """
        Returns True if the line_text is the start of a log message for a JDFTx optimization step

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx optimization step
        """
        is_line = f"{self.iter_type}:" in line_text and "Iter:" in line_text
        return is_line

    def is_opt_conv_line(self, line_text: str) -> bool:
        """
        Returns True if the line_text is the end of a JDFTx optimization step

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is the end of a JDFTx optimization step
        """
        is_line = f"{self.iter_type}: Converged" in line_text

    def parse_opt_lines(self, opt_lines: list[str]) -> None:
        """
        Parses the lines of text corresponding to the optimization step of a JDFTx out file

        Parameters
        ----------
        opt_lines: list[str]
            A list of lines of text from a JDFTx out file
        """
        if len(opt_lines):
            for line in opt_lines:
                if self.is_opt_start_line(line):
                    iter = int(self._get_colon_var_t1(line, "Iter:"))
                    self.iter = iter
                    E = self._get_colon_var_t1(line, f"{self.etype}:")
                    self.E = E * Ha_to_eV
                    grad_K = self._get_colon_var_t1(line, "|grad|_K: ")
                    self.grad_K = grad_K
                    alpha = self._get_colon_var_t1(line, "alpha: ")
                    self.alpha = alpha
                    linmin = self._get_colon_var_t1(line, "linmin: ")
                    self.linmin = linmin
                    t_s = self._get_colon_var_t1(line, "t[s]: ")
                    self.t_s = t_s
                elif self.is_opt_conv_line(line):
                    self.geom_converged = True
                    self.geom_converged_reason = (
                        line.split("(")[1].split(")")[0].strip()
                    )

    def is_generic_start_line(self, line_text: str, line_type: str) -> bool:
        # I am choosing to map line_type to a function this way because
        # I've had horrible experiences with storing functions in dictionaries
        # in the past
        """
        Returns True if the line_text is the start of a section of the JDFTx out file
        corresponding to the line_type

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file
        line_type: str
            The type of line to check for

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a section of the JDFTx out file
        """
        if line_type == "lowdin":
            return self.is_lowdin_start_line(line_text)
        if line_type == "opt":
            return self.is_opt_start_line(line_text)
        if line_type == "ecomp":
            return self.is_ecomp_start_line(line_text)
        if line_type == "forces":
            return self.is_forces_start_line(line_text)
        if line_type == "posns":
            return self.is_posns_start_line(line_text)
        if line_type == "stress":
            return self.is_stress_start_line(line_text)
        if line_type == "strain":
            return self.is_strain_start_line(line_text)
        if line_type == "lattice":
            return self.is_lattice_start_line(line_text)
        if line_type == "emin":
            return self.is_emin_start_line(line_text)
        raise ValueError(f"Unrecognized line type {line_type}")

    def collect_generic_line(
        self, line_text: str, generic_lines: list[str]
    ) -> tuple[list[str], bool, bool]:
        """
        Collects a line of text into a list of lines if the line is not empty, and otherwise
        updates the collecting and collected flags

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file
        generic_lines: list[str]
            A list of lines of text of the same type

        Returns
        -------
        generic_lines: list[str]
            A list of lines of text of the same type
        collecting: bool
            True if the line_text is not empty
        collected: bool
            True if the line_text is empty (end of section)
        """
        collecting = True
        collected = False
        if not len(line_text.strip()):
            collecting = False
            collected = True
        else:
            generic_lines.append(line_text)
        return generic_lines, collecting, collected

    def _bracket_num_list_str_of_3_to_nparray(self, line: str) -> np.ndarray:
        """
        Converts a string of the form "[ x y z ]" to a 3x1 numpy array

        Parameters
        ----------
        line: str
            A string of the form "[ x y z ]"
        """
        return np.array([float(x) for x in line.split()[1:-1]])

    def _bracket_num_list_str_of_3x3_to_nparray(
        self, lines: list[str], i_start=0
    ) -> np.ndarray:
        """
        Converts a list of strings of the form "[ x y z ]" to a 3x3 numpy array

        Parameters
        ----------
        lines: list[str]
            A list of strings of the form "[ x y z ]"
        i_start: int
            The index of the first line in lines

        Returns
        -------
        out: np.ndarray
            A 3x3 numpy array
        """
        out = np.zeros([3, 3])
        for i in range(3):
            out[i, :] += self._bracket_num_list_str_of_3_to_nparray(lines[i + i_start])
        return out

    def _get_colon_var_t1(self, linetext: str, lkey: str) -> float | None:
        """
        Reads a float from an elec minimization line assuming value appears as
        "... lkey value ..."

        Parameters
        ----------
        linetext: str
            A line of text from a JDFTx out file
        lkey: str
            A string that appears before the float value in linetext
        """
        colon_var = None
        if lkey in linetext:
            colon_var = float(linetext.split(lkey)[1].strip().split(" ")[0])
        return colon_var
