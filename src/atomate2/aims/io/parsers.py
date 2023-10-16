"""AIMS output parser, taken from ASE with modifications."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from ase import Atom
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.constraints import FixAtoms, FixCartesian
from ase.io import ParseError
from ase.utils import lazymethod, lazyproperty, reader

from atomate2.aims.utils.msonable_atoms import MSONableAtoms
from atomate2.aims.utils.units import ev_per_A3_to_kbar

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Molecule, Structure

LINE_NOT_FOUND = object()


class AimsParseError(Exception):
    """Exception raised if an error occurs when parsing an Aims output file."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


# Read aims.out files
scalar_property_to_line_key = {
    "free_energy": ["| Electronic free energy"],
    "number_of_iterations": ["| Number of self-consistency cycles"],
    "magnetic_moment": ["N_up - N_down"],
    "n_atoms": ["| Number of atoms"],
    "n_bands": [
        "Number of Kohn-Sham states",
        "Reducing total number of  Kohn-Sham states",
        "Reducing total number of Kohn-Sham states",
    ],
    "n_electrons": ["The structure contains"],
    "n_kpts": ["| Number of k-points"],
    "n_spins": ["| Number of spin channels"],
    "electronic_temp": ["Occupation type:"],
    "fermi_energy": ["| Chemical potential (Fermi level)"],
}


class AimsOutChunk:
    """Base class for AimsOutChunks."""

    def __init__(self, lines: list[str]):
        """Construct the AimsOutChunk.

        Parameters
        ----------
        lines: list[str]
            The set of lines from the output file the encompasses either
            a single structure within a trajectory or
            general information about the calculation (header)
        """
        self.lines = lines

    def reverse_search_for(self, keys: list[str], line_start: int = 0):
        """Find the last time one of the keys appears in self.lines.

        Parameters
        ----------
        keys: list[str]
            The key strings to search for in self.lines
        line_start: int
            The lowest index to search for in self.lines

        Returns
        -------
        int
            The last time one of the keys appears in self.lines
        """
        for ll, line in enumerate(self.lines[line_start:][::-1]):
            if any(key in line for key in keys):
                return len(self.lines) - ll - 1

        return LINE_NOT_FOUND

    def search_for_all(self, key: str, line_start: int = 0, line_end: int = -1):
        """Find the all times the key appears in self.lines.

        Parameters
        ----------
        key: str
            The key string to search for in self.lines
        line_start: int
            The first line to start the search from
        line_end: int
            The last line to end the search at

        Returns
        -------
        list of ints
            All times the key appears in the lines
        """
        line_index = []
        for ll, line in enumerate(self.lines[line_start:line_end]):
            if key in line:
                line_index.append(ll + line_start)
        return line_index

    def parse_scalar(self, property: str):
        """Parse a scalar property from the chunk.

        Parameters
        ----------
        property: str
            The property key to parse

        Returns
        -------
        float
            The scalar value of the property
        """
        line_start = self.reverse_search_for(scalar_property_to_line_key[property])

        if line_start == LINE_NOT_FOUND:
            return None

        line = self.lines[line_start]
        return float(line.split(":")[-1].strip().split()[0])


class AimsOutHeaderChunk(AimsOutChunk):
    """The header of the aims.out file containing general information."""

    def __init__(self, lines: list[str]):
        """Construct AimsOutHeaderChunk.

        Parameters
        ----------
        lines: list[str]
            The lines inside the aims.out header
        """
        super().__init__(lines)
        self._k_points = None
        self._k_point_weights = None

    @property
    def commit_hash(self):
        """Get the commit hash for the FHI-aims version."""
        line_start = self.reverse_search_for(["Commit number"])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError("This file does not appear to be an aims-output file")

        return self.lines[line_start].split(":")[1].strip()

    @property
    def aims_uuid(self):
        """Get the aims-uuid for the calculation."""
        line_start = self.reverse_search_for(["aims_uuid"])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError("This file does not appear to be an aims-output file")

        return self.lines[line_start].split(":")[1].strip()

    @property
    def version_number(self):
        """Get the commit hash for the FHI-aims version."""
        line_start = self.reverse_search_for(["FHI-aims version"])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError("This file does not appear to be an aims-output file")

        return self.lines[line_start].split(":")[1].strip()

    @property
    def fortran_compiler(self):
        """Get the fortran compiler used to make FHI-aims."""
        line_start = self.reverse_search_for(["Fortran compiler      :"])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError("This file does not appear to be an aims-output file")

        return self.lines[line_start].split(":")[1].split("/")[-1].strip()

    @property
    def c_compiler(self):
        """Get the C compiler used to make FHI-aims."""
        line_start = self.reverse_search_for(["C compiler            :"])
        if line_start == LINE_NOT_FOUND:
            return None

        return self.lines[line_start].split(":")[1].split("/")[-1].strip()

    @property
    def fortran_compiler_flags(self):
        """Get the fortran compiler flags used to make FHI-aims."""
        line_start = self.reverse_search_for(["Fortran compiler flags"])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError("This file does not appear to be an aims-output file")

        return self.lines[line_start].split(":")[1].strip()

    @property
    def c_compiler_flags(self):
        """Get the C compiler flags used to make FHI-aims."""
        line_start = self.reverse_search_for(["C compiler flags"])
        if line_start == LINE_NOT_FOUND:
            return None

        return self.lines[line_start].split(":")[1].strip()

    @property
    def build_type(self):
        """Get the optional build flags passed to cmake."""
        line_end = self.reverse_search_for(["Linking against:"])
        line_inds = self.search_for_all("Using", line_end=line_end)

        return [" ".join(self.lines[ind].split()[1:]).strip() for ind in line_inds]

    @property
    def linked_against(self):
        """Get all libraries used to link the FHI-aims executable."""
        line_start = self.reverse_search_for(["Linking against:"])
        if line_start == LINE_NOT_FOUND:
            return []

        linked_libs = [self.lines[line_start].split(":")[1].strip()]
        line_start += 1
        while "lib" in self.lines[line_start]:
            linked_libs.append(self.lines[line_start].strip())
            line_start += 1

        return linked_libs

    @property
    def constraints(self):
        """Parse the constraints from the aims.out file.

        Constraints for the lattice vectors are not supported.
        """
        line_inds = self.search_for_all("Found relaxation constraint for atom")
        if len(line_inds) == 0:
            return []
        fix = []
        fix_cart = []
        for ll in line_inds:
            line = self.lines[ll]
            xyz = [0, 0, 0]
            ind = int(line.split()[5][:-1]) - 1
            if "All coordinates fixed" in line and ind not in fix:
                fix.append(ind)
            if "coordinate fixed" in line:
                coord = line.split()[6]
                if coord == "x":
                    xyz[0] = 1
                elif coord == "y":
                    xyz[1] = 1
                elif coord == "z":
                    xyz[2] = 1
                keep = True
                n_mod = 0
                for n, c in enumerate(fix_cart):
                    if ind == c.index:
                        keep = False
                        n_mod = n
                        break
                if keep:
                    fix_cart.append(FixCartesian(ind, xyz))
                else:
                    fix_cart[n_mod].mask[xyz.index(1)] = 0
        if len(fix) > 0:
            fix_cart.append(FixAtoms(indices=fix))

        return fix_cart

    @property
    def initial_cell(self):
        """Parse the initial cell from the aims.out file."""
        line_start = self.reverse_search_for(["| Unit cell:"])
        if line_start == LINE_NOT_FOUND:
            return None

        return [
            [float(inp) for inp in line.split()[-3:]]
            for line in self.lines[line_start + 1 : line_start + 4]
        ]

    @property
    def initial_atoms(self):
        """Create an atoms object for the initial structure.

        Using the FHI-aims output file recreate the initial structure for
        the calculation.
        """
        line_start = self.reverse_search_for(["Atomic structure:"])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError("No information about the structure in the chunk.")

        line_start += 2

        cell = self.initial_cell
        positions = np.zeros((self.n_atoms, 3))
        symbols = [""] * self.n_atoms
        for ll, line in enumerate(self.lines[line_start : line_start + self.n_atoms]):
            inp = line.split()
            positions[ll, :] = [float(pos) for pos in inp[4:7]]
            symbols[ll] = inp[3]

        atoms = MSONableAtoms(symbols=symbols, positions=positions)

        if cell:
            atoms.set_cell(cell)
            atoms.set_pbc([True, True, True])
        atoms.set_constraint(self.constraints)

        return atoms

    @property
    def is_md(self):
        """Determine if calculation is a molecular dynamics calculation."""
        return (
            self.reverse_search_for(["Complete information for previous time-step:"])
            != LINE_NOT_FOUND
        )

    @property
    def is_relaxation(self):
        """Determine if the calculation is a geometry optimization or not."""
        return self.reverse_search_for(["Geometry relaxation:"]) != LINE_NOT_FOUND

    def _parse_k_points(self):
        """Get the list of k-points used in the calculation."""
        n_kpts = self.parse_scalar("n_kpts")
        if n_kpts is None:
            return {
                "k_points": None,
                "k_point_weights": None,
            }
        n_kpts = int(n_kpts)

        line_start = self.reverse_search_for(["| K-points in task"])
        line_end = self.reverse_search_for(["| k-point:"])
        if (
            (line_start == LINE_NOT_FOUND)
            or (line_end == LINE_NOT_FOUND)
            or (line_end - line_start != n_kpts)
        ):
            return {
                "k_points": None,
                "k_point_weights": None,
            }

        k_points = np.zeros((n_kpts, 3))
        k_point_weights = np.zeros(n_kpts)
        for kk, line in enumerate(self.lines[line_start + 1 : line_end + 1]):
            k_points[kk] = [float(inp) for inp in line.split()[4:7]]
            k_point_weights[kk] = float(line.split()[-1])

        return {
            "k_points": k_points,
            "k_point_weights": k_point_weights,
        }

    @property
    def n_atoms(self) -> int:
        """The number of atoms for the material."""
        n_atoms = self.parse_scalar("n_atoms")
        if n_atoms is None:
            raise AimsParseError(
                "No information about the number of atoms in the header."
            )
        return int(n_atoms)

    @property
    def n_bands(self):
        """The number of Kohn-Sham states for the chunk."""
        line_start = self.reverse_search_for(scalar_property_to_line_key["n_bands"])

        if line_start == LINE_NOT_FOUND:
            raise AimsParseError(
                "No information about the number of Kohn-Sham states in the header."
            )

        line = self.lines[line_start]
        if "| Number of Kohn-Sham states" in line:
            return int(line.split(":")[-1].strip().split()[0])

        return int(line.split()[-1].strip()[:-1])

    @property
    def n_electrons(self):
        """The number of electrons for the chunk."""
        line_start = self.reverse_search_for(scalar_property_to_line_key["n_electrons"])

        if line_start == LINE_NOT_FOUND:
            raise AimsParseError(
                "No information about the number of electrons in the header."
            )

        line = self.lines[line_start]
        return int(float(line.split()[-2]))

    @property
    def n_k_points(self):
        """The number of k_ppoints for the calculation."""
        n_kpts = self.parse_scalar("n_kpts")
        if n_kpts is None:
            return None

        return int(n_kpts)

    @property
    def n_spins(self):
        """The number of spin channels for the chunk."""
        n_spins = self.parse_scalar("n_spins")
        if n_spins is None:
            raise AimsParseError(
                "No information about the number of spin channels in the header."
            )
        return int(n_spins)

    @property
    def electronic_temperature(self):
        """The electronic temperature for the chunk."""
        line_start = self.reverse_search_for(
            scalar_property_to_line_key["electronic_temp"]
        )
        if line_start == LINE_NOT_FOUND:
            return 0.10

        line = self.lines[line_start]
        return float(line.split("=")[-1].strip().split()[0])

    @property
    def k_points(self):
        """All k-points listed in the calculation."""
        return self._parse_k_points()["k_points"]

    @property
    def k_point_weights(self):
        """The k-point weights for the calculation."""
        return self._parse_k_points()["k_point_weights"]

    @property
    def header_summary(self):
        """Dictionary summarizing the information inside the header."""
        return {
            "initial_atoms": self.initial_atoms,
            "initial_cell": self.initial_cell,
            "constraints": self.constraints,
            "is_relaxation": self.is_relaxation,
            "is_md": self.is_md,
            "n_atoms": self.n_atoms,
            "n_bands": self.n_bands,
            "n_electrons": self.n_electrons,
            "n_spins": self.n_spins,
            "electronic_temperature": self.electronic_temperature,
            "n_k_points": self.n_k_points,
            "k_points": self.k_points,
            "k_point_weights": self.k_point_weights,
        }

    @property
    def metadata_summary(self) -> dict[str, str]:
        """Dictionary containing all metadata for FHI-aims build."""
        return {
            "commit_hash": self.commit_hash,
            "aims_uuid": self.aims_uuid,
            "version_number": self.version_number,
            "fortran_compiler": self.fortran_compiler,
            "c_compiler": self.c_compiler,
            "fortran_compiler_flags": self.fortran_compiler_flags,
            "c_compiler_flags": self.c_compiler_flags,
            "build_type": self.build_type,
            "linked_against": self.linked_against,
        }


class AimsOutCalcChunk(AimsOutChunk):
    """A part of the aims.out file corresponding to a single structure."""

    def __init__(self, lines, header):
        """Construct the AimsOutCalcChunk.

        Parameters
        ----------
        lines: list[str]
            The lines used for the structure
        header: .AimsOutHeaderChunk
            A summary of the relevant information from the aims.out header
        """
        super().__init__(lines)
        self._header = header.header_summary

    def _parse_atoms(self):
        """Parse an atoms object from the file.

        For the given section of the aims output file generate the
        calculated structure.
        """
        start_keys = [
            "Atomic structure (and velocities) as used in the preceding time step",
            "Updated atomic structure",
            "Atomic structure that was used in the preceding time step of "
            "the wrapper",
        ]
        line_start = self.reverse_search_for(start_keys)
        if line_start == LINE_NOT_FOUND:
            return self.initial_atoms

        line_start += 1

        line_end = self.reverse_search_for(
            ['Writing the current geometry to file "geometry.in.next_step"'], line_start
        )
        if line_end == LINE_NOT_FOUND:
            line_end = len(self.lines)

        cell = []
        velocities = []
        atoms = MSONableAtoms()
        for line in self.lines[line_start:line_end]:
            if "lattice_vector   " in line:
                cell.append([float(inp) for inp in line.split()[1:]])
            elif "atom   " in line:
                line_split = line.split()
                atoms.append(
                    Atom(line_split[4], tuple([float(inp) for inp in line_split[1:4]]))
                )
            elif "velocity   " in line:
                velocities.append([float(inp) for inp in line.split()[1:]])

        if len(atoms) != self.n_atoms:
            raise AimsParseError(
                "The number of atoms is inconsistent with the initial structure"
            )

        if (len(velocities) != self.n_atoms) and (len(velocities) != 0):
            raise AimsParseError(
                "The number of velocities is inconsistent with the number of atoms"
            )

        if len(cell) == 3:
            atoms.set_cell(np.array(cell))
            atoms.set_pbc([True, True, True])
        elif len(cell) != 0:
            raise AimsParseError(
                "Parsed geometry has incorrect number of lattice vectors."
            )

        if len(velocities) > 0:
            atoms.set_velocities(np.array(velocities))
        atoms.set_constraint(self.constraints)

        return atoms

    @property
    def forces(self):
        """Parse the forces from the aims.out file."""
        line_start = self.reverse_search_for(["Total atomic forces"])
        if line_start == LINE_NOT_FOUND:
            return None

        line_start += 1

        return np.array(
            [
                [float(inp) for inp in line.split()[-3:]]
                for line in self.lines[line_start : line_start + self.n_atoms]
            ]
        )

    @property
    def stresses(self):
        """Parse the stresses from the aims.out file and convert to kbar."""
        line_start = self.reverse_search_for(
            ["Per atom stress (eV) used for heat flux calculation"]
        )
        if line_start == LINE_NOT_FOUND:
            return None
        line_start += 3
        stresses = []
        for line in self.lines[line_start : line_start + self.n_atoms]:
            xx, yy, zz, xy, xz, yz = (float(d) for d in line.split()[2:8])
            stresses.append([xx, yy, zz, yz, xz, xy])

        return np.array(stresses) * ev_per_A3_to_kbar

    @property
    def stress(self):
        """Parse the stress from the aims.out file and convert to kbar."""
        from ase.stress import full_3x3_to_voigt_6_stress

        line_start = self.reverse_search_for(
            [
                "Analytical stress tensor - Symmetrized",
                "Numerical stress tensor",
            ]
        )  # Offset to relevant lines
        if line_start == LINE_NOT_FOUND:
            return None

        stress = [
            [float(inp) for inp in line.split()[2:5]]
            for line in self.lines[line_start + 5 : line_start + 8]
        ]
        return full_3x3_to_voigt_6_stress(stress) * ev_per_A3_to_kbar

    @property
    def is_metallic(self):
        """Checks if the system is metallic."""
        line_start = self.reverse_search_for(
            [
                "material is metallic within the approximate finite "
                "broadening function (occupation_type)"
            ]
        )
        return line_start != LINE_NOT_FOUND

    @property
    def energy(self):
        """Parse the energy from the aims.out file."""
        atoms = self._parse_atoms()

        if np.all(atoms.pbc) and self.is_metallic:
            line_ind = self.reverse_search_for(["Total energy corrected"])
        else:
            line_ind = self.reverse_search_for(["Total energy uncorrected"])
        if line_ind == LINE_NOT_FOUND:
            raise AimsParseError("No energy is associated with the structure.")

        return float(self.lines[line_ind].split()[5])

    @lazyproperty
    def dipole(self):
        """Parse the electric dipole moment from the aims.out file."""
        line_start = self.reverse_search_for(["Total dipole moment [eAng]"])
        if line_start == LINE_NOT_FOUND:
            return None

        line = self.lines[line_start]
        return np.array([float(inp) for inp in line.split()[6:9]])

    @lazyproperty
    def dielectric_tensor(self):
        """Parse the dielectric tensor from the aims.out file."""
        line_start = self.reverse_search_for(["PARSE DFPT_dielectric_tensor"])
        if line_start == LINE_NOT_FOUND:
            return None

        # we should find the tensor in the next three lines:
        lines = self.lines[line_start + 1 : line_start + 4]

        # make ndarray and return
        return np.array([np.fromstring(line, sep=" ") for line in lines])

    @lazyproperty
    def polarization(self):
        """Parse the polarization vector from the aims.out file."""
        line_start = self.reverse_search_for(["| Cartesian Polarization"])
        if line_start == LINE_NOT_FOUND:
            return None
        line = self.lines[line_start]
        return np.array([float(s) for s in line.split()[-3:]])

    @lazymethod
    def _parse_homo_lumo(self):
        """Parse the HOMO/LUMO values and get band gap if periodic."""
        line_start = self.reverse_search_for(["Highest occupied state (VBM)"])
        homo = float(self.lines[line_start].split(" at ")[1].split("eV")[0].strip())

        line_start = self.reverse_search_for(["Lowest unoccupied state (CBM)"])
        lumo = float(self.lines[line_start].split(" at ")[1].split("eV")[0].strip())

        line_start = self.reverse_search_for(["verall HOMO-LUMO gap"])
        homo_lumo_gap = float(
            self.lines[line_start].split(":")[1].split("eV")[0].strip()
        )

        line_start = self.reverse_search_for(["Smallest direct gap"])
        if line_start == LINE_NOT_FOUND:
            return {
                "homo": homo,
                "lumo": lumo,
                "gap": homo_lumo_gap,
                "direct_gap": homo_lumo_gap,
            }

        direct_gap = float(self.lines[line_start].split(":")[1].split("eV")[0].strip())
        return {
            "homo": homo,
            "lumo": lumo,
            "gap": homo_lumo_gap,
            "direct_gap": direct_gap,
        }

    @lazymethod
    def _parse_hirshfeld(self):
        """Parse the Hirshfled charges volumes, and dipole moments."""
        atoms = self._parse_atoms()

        line_start = self.reverse_search_for(
            ["Performing Hirshfeld analysis of fragment charges and moments."]
        )
        if line_start == LINE_NOT_FOUND:
            return {
                "charges": None,
                "volumes": None,
                "atomic_dipoles": None,
                "dipole": None,
            }

        line_inds = self.search_for_all("Hirshfeld charge", line_start, -1)
        hirshfeld_charges = np.array(
            [float(self.lines[ind].split(":")[1]) for ind in line_inds]
        )

        line_inds = self.search_for_all("Hirshfeld volume", line_start, -1)
        hirshfeld_volumes = np.array(
            [float(self.lines[ind].split(":")[1]) for ind in line_inds]
        )

        line_inds = self.search_for_all("Hirshfeld dipole vector", line_start, -1)
        hirshfeld_atomic_dipoles = np.array(
            [
                [float(inp) for inp in self.lines[ind].split(":")[1].split()]
                for ind in line_inds
            ]
        )

        if not np.any(atoms.pbc):
            hirshfeld_dipole = np.sum(
                hirshfeld_charges.reshape((-1, 1)) * atoms.get_positions(),
                axis=1,
            )
        else:
            hirshfeld_dipole = None
        return {
            "charges": hirshfeld_charges,
            "volumes": hirshfeld_volumes,
            "atomic_dipoles": hirshfeld_atomic_dipoles,
            "dipole": hirshfeld_dipole,
        }

    @lazymethod
    def _parse_eigenvalues(self):
        """Parse the eigenvalues and occupancies of the system.

        If eigenvalue for a particular k-point is not present
        in the output file then set it to np.nan.
        """
        atoms = self._parse_atoms()

        line_start = self.reverse_search_for(["Writing Kohn-Sham eigenvalues."])
        if line_start == LINE_NOT_FOUND:
            return {"eigenvalues": None, "occupancies": None}

        line_end_1 = self.reverse_search_for(
            ["Self-consistency cycle converged."], line_start
        )
        line_end_2 = self.reverse_search_for(
            [
                "What follows are estimated values for band gap, HOMO, LUMO, etc.",
                "Current spin moment of the entire structure :",
                "Highest occupied state (VBM)",
            ],
            line_start,
        )
        if line_end_1 == LINE_NOT_FOUND:
            line_end = line_end_2
        elif line_end_2 == LINE_NOT_FOUND:
            line_end = line_end_1
        else:
            line_end = min(line_end_1, line_end_2)

        n_kpts = self.n_k_points if np.all(atoms.pbc) else 1
        if n_kpts is None:
            return {"eigenvalues": None, "occupancies": None}

        eigenvalues = np.full((n_kpts, self.n_bands, self.n_spins), np.nan)
        occupancies = np.full((n_kpts, self.n_bands, self.n_spins), np.nan)

        occupation_block_start = self.search_for_all(
            "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
            line_start,
            line_end,
        )
        kpt_def = self.search_for_all("K-point: ", line_start, line_end)

        if len(kpt_def) > 0:
            kpt_inds = [int(self.lines[ll].split()[1]) - 1 for ll in kpt_def]
        elif (self.n_k_points is None) or (self.n_k_points == 1):
            kpt_inds = [0]
        else:
            raise ParseError("Cannot find k-point definitions")

        if len(kpt_inds) != len(occupation_block_start):
            raise AimsParseError(
                "Number of k-points is not equal to the number of occupation blocks."
            )
        spins = [0] * len(occupation_block_start)

        if self.n_spins == 2:
            spin_def = self.search_for_all("Spin-", line_start, line_end)
            if len(spin_def) != len(occupation_block_start):
                raise AimsParseError(
                    "The number of spins is not equal to the number of occupations."
                )

            spins = [int("Spin-down eigenvalues:" in self.lines[ll]) for ll in spin_def]

        for occ_start, kpt_ind, spin in zip(occupation_block_start, kpt_inds, spins):
            for ll, line in enumerate(
                self.lines[occ_start + 1 : occ_start + self.n_bands + 1]
            ):
                if "***" in line:
                    warn_msg = f"The {ll+1}th eigenvalue for the "
                    "{kpt_ind+1}th k-point and {spin}th channels could "
                    "not be read (likely too large to be printed "
                    "in the output file)"
                    warnings.warn(warn_msg, stacklevel=1)
                    continue
                split_line = line.split()
                eigenvalues[kpt_ind, ll, spin] = float(split_line[3])
                occupancies[kpt_ind, ll, spin] = float(split_line[1])
        return {"eigenvalues": eigenvalues, "occupancies": occupancies}

    @lazyproperty
    def atoms(self):
        """Convert AimsOutChunk to Atoms object."""
        atoms = self._parse_atoms()

        atoms.calc = SinglePointDFTCalculator(atoms)
        atoms.calc.results = self.results
        return atoms

    @property
    def structure(self) -> Structure | Molecule:
        """The pytmagen structure of the atoms."""
        return self.atoms.pymatgen

    @property
    def results(self):
        """Convert an AimsOutChunk to a Results Dictionary."""
        results = {
            "energy": self.energy,
            "free_energy": self.free_energy,
            "forces": self.forces,
            "stress": self.stress,
            "stresses": self.stresses,
            "magmom": self.magmom,
            "dipole": self.dipole,
            "fermi_energy": self.E_f,
            "n_iter": self.n_iter,
            "hirshfeld_charges": self.hirshfeld_charges,
            "hirshfeld_dipole": self.hirshfeld_dipole,
            "hirshfeld_volumes": self.hirshfeld_volumes,
            "hirshfeld_atomic_dipoles": self.hirshfeld_atomic_dipoles,
            "dielectric_tensor": self.dielectric_tensor,
            "polarization": self.polarization,
            "homo": self.homo,
            "lumo": self.lumo,
            "gap": self.gap,
            "direct_gap": self.direct_gap,
        }

        return {key: value for key, value in results.items() if value is not None}

    # Properties from the aims.out header
    @lazyproperty
    def initial_atoms(self):
        """Return the initial structure of the calculation."""
        return self._header["initial_atoms"]

    @lazyproperty
    def initial_cell(self):
        """Return the initial lattice vectors for the structure."""
        return self._header["initial_cell"]

    @lazyproperty
    def constraints(self):
        """Return the relaxation constraints for the calculation."""
        return self._header["constraints"]

    @lazyproperty
    def n_atoms(self):
        """Return the number of atoms for the material."""
        return self._header["n_atoms"]

    @lazyproperty
    def n_bands(self):
        """Return the number of Kohn-Sham states for the chunk."""
        return self._header["n_bands"]

    @lazyproperty
    def n_electrons(self):
        """Return the number of electrons for the chunk."""
        return self._header["n_electrons"]

    @lazyproperty
    def n_spins(self):
        """Return the number of spin channels for the chunk."""
        return self._header["n_spins"]

    @lazyproperty
    def electronic_temperature(self):
        """Return the electronic temperature for the chunk."""
        return self._header["electronic_temperature"]

    @lazyproperty
    def n_k_points(self):
        """Return the number of electrons for the chunk."""
        return self._header["n_k_points"]

    @lazyproperty
    def k_points(self):
        """Return the number of spin channels for the chunk."""
        return self._header["k_points"]

    @lazyproperty
    def k_point_weights(self):
        """Return tk_point_weights electronic temperature for the chunk."""
        return self._header["k_point_weights"]

    @lazyproperty
    def free_energy(self):
        """Return the free energy for the chunk."""
        return self.parse_scalar("free_energy")

    @lazyproperty
    def n_iter(self):
        """Return the number of SCF iterations.

        The number of steps needed to converge the SCF cycle for the chunk.
        """
        return self.parse_scalar("number_of_iterations")

    @lazyproperty
    def magmom(self):
        """Return the magnetic moment for the chunk."""
        return self.parse_scalar("magnetic_moment")

    @lazyproperty
    def E_f(self):
        """Return he Fermi energy for the chunk."""
        return self.parse_scalar("fermi_energy")

    @lazyproperty
    def converged(self):
        """Return True if the chunk is a fully converged final structure."""
        return (len(self.lines) > 0) and ("Have a nice day." in self.lines[-5:])

    @lazyproperty
    def hirshfeld_charges(self):
        """Return the Hirshfeld charges for the chunk."""
        return self._parse_hirshfeld()["charges"]

    @lazyproperty
    def hirshfeld_atomic_dipoles(self):
        """Return the Hirshfeld atomic dipole moments for the chunk."""
        return self._parse_hirshfeld()["atomic_dipoles"]

    @lazyproperty
    def hirshfeld_volumes(self):
        """Return the Hirshfeld volume for the chunk."""
        return self._parse_hirshfeld()["volumes"]

    @lazyproperty
    def hirshfeld_dipole(self):
        """Return the Hirshfeld systematic dipole moment for the chunk."""
        atoms = self._parse_atoms()

        if not np.any(atoms.pbc):
            return self._parse_hirshfeld()["dipole"]

        return None

    @lazyproperty
    def eigenvalues(self):
        """Return all outputted eigenvalues for the system."""
        return self._parse_eigenvalues()["eigenvalues"]

    @lazyproperty
    def occupancies(self):
        """Return all outputted occupancies for the system."""
        return self._parse_eigenvalues()["occupancies"]

    @lazyproperty
    def homo(self):
        """Return the HOMO (CBM) of the calculation."""
        return self._parse_homo_lumo()["homo"]

    @lazyproperty
    def lumo(self):
        """Return the LUMO (VBM) of the calculation."""
        return self._parse_homo_lumo()["lumo"]

    @lazyproperty
    def gap(self):
        """Return the HOMO-LUMO gap (band gap) of the calculation."""
        return self._parse_homo_lumo()["gap"]

    @lazyproperty
    def direct_gap(self):
        """Return the direct band gap of the calculation."""
        return self._parse_homo_lumo()["direct_gap"]


def get_header_chunk(fd):
    """Return the header information from the aims.out file."""
    header = []
    line = ""

    # Stop the header once the first SCF cycle begins
    while (
        "Convergence:    q app. |  density  | eigen (eV) | Etot (eV)" not in line
        and "Begin self-consistency iteration #" not in line
    ):
        try:
            line = next(fd).strip()  # Raises StopIteration on empty file
        except StopIteration:
            raise ParseError(
                "No SCF steps present, calculation failed at setup."
            ) from None

        header.append(line)
    return AimsOutHeaderChunk(header)


def get_aims_out_chunks(fd, header_chunk):
    """Yield unprocessed chunks (header, lines) for each AimsOutChunk image."""
    try:
        line = next(fd).strip()  # Raises StopIteration on empty file
    except StopIteration:
        return

    # If the calculation is relaxation the updated structural information
    # occurs before the re-initialization
    if header_chunk.is_relaxation:
        chunk_end_line = (
            "Geometry optimization: Attempting to predict improved coordinates."
        )
    else:
        chunk_end_line = "Begin self-consistency loop: Re-initialization"

    # If SCF is not converged then do not treat the next chunk_end_line as a
    # new chunk until after the SCF is re-initialized
    ignore_chunk_end_line = False
    while True:
        try:
            line = next(fd).strip()  # Raises StopIteration on empty file
        except StopIteration:
            break

        lines = []
        while chunk_end_line not in line or ignore_chunk_end_line:
            lines.append(line)
            # If SCF cycle not converged or numerical stresses are requested,
            # don't end chunk on next Re-initialization
            patterns = [
                (
                    "Self-consistency cycle not yet converged -"
                    " restarting mixer to attempt better convergence."
                ),
                (
                    "Components of the stress tensor (for mathematical "
                    "background see comments in numerical_stress.f90)."
                ),
                "Calculation of numerical stress completed",
            ]
            if any(pattern in line for pattern in patterns):
                ignore_chunk_end_line = True
            elif "Begin self-consistency loop: Re-initialization" in line:
                ignore_chunk_end_line = False

            try:
                line = next(fd).strip()
            except StopIteration:
                break
        yield AimsOutCalcChunk(lines, header_chunk)


def check_convergence(
    chunks: list[AimsOutCalcChunk], non_convergence_ok: bool = False
) -> bool:
    """Check if the aims output file is for a converged calculation.

    Parameters
    ----------
    chunks: list[.AimsOutCalcChunk]
        The list of chunks for the aims calculations
    non_convergence_ok: bool
        True if it is okay for the calculation to not be converged

    Returns
    -------
    bool
        True if the calculation is converged
    """
    if not non_convergence_ok and not chunks[-1].converged:
        raise ParseError("The calculation did not complete successfully")
    return True


@reader
def read_aims_header_info(
    fd: str | Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Read the FHI-aims header information.

    Parameters
    ----------
    fd: str or Path
        The file to read

    Returns
    -------
    The calculation metadata and the system summary
    """
    header_chunk = get_header_chunk(fd)

    system_summary = header_chunk.header_summary
    metadata = header_chunk.metadata_summary
    return metadata, system_summary


@reader
def read_aims_output(
    fd: str | Path,
    index: int | slice = -1,
    non_convergence_ok: bool = False,
) -> MSONableAtoms | Sequence[MSONableAtoms]:
    """Import FHI-aims output files with all data available.

    Includes all structures for relaxations and MD runs with FHI-aims

    Parameters
    ----------
    fd: str or Path
        The file to read
    index: int or slice
        The index of the images to read
    non_convergence_ok: bool
        True if the calculations do not have to be converged

    Returns
    -------
    The selected atoms
    """
    header_chunk = get_header_chunk(fd)
    chunks = list(get_aims_out_chunks(fd, header_chunk))
    check_convergence(chunks, non_convergence_ok)

    # Relaxations have an additional footer chunk due to how it is split
    if header_chunk.is_relaxation:
        images = [chunk.atoms for chunk in chunks[:-1]]
    else:
        images = [chunk.atoms for chunk in chunks]
    return images[index]
