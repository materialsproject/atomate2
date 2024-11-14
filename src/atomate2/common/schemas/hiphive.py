"""Schemas for phonon documents."""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import warnings
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy as sp
from emmet.core.math import Matrix3D
from emmet.core.structure import StructureMetadata

# Hiphive packages
from hiphive import (
    ClusterSpace,
    ForceConstantPotential,
    StructureContainer,
    enforce_rotational_sum_rules,
)
from hiphive import ForceConstants as HiphiveForceConstants
from hiphive.cutoffs import estimate_maximum_cutoff, is_cutoff_allowed
from hiphive.fitting import Optimizer
from hiphive.utilities import get_displacements
from monty.json import MSONable

# Jobflow packages
# Pymatgen packages
from monty.serialization import dumpfn
from phono3py.phonon3.gruneisen import Gruneisen

# Phonopy & Phono3py
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.hiphive_interface import phonopy_atoms_to_ase
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from phonopy.units import VaspToTHz
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import (
    get_ph_bs_symm_line,
    get_ph_dos,
    get_phonopy_structure,
    get_pmg_structure,
)
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.kpath import KPathSeek
from pymatgen.transformations.standard_transformations import SupercellTransformation
from typing_extensions import Self

from atomate2.aims.utils.units import omegaToTHz

warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="ase")

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from pymatgen.core.structure import Structure
    from emmet.core.math import Matrix3D

    from atomate2.vasp.jobs.base import BaseVaspMaker

logger = logging.getLogger(__name__)

ev2j = sp.constants.elementary_charge
hbar = sp.constants.hbar # J-s
kb = sp.constants.Boltzmann # J/K


def get_factor(code: str) -> float:
    """
    Get the frequency conversion factor to THz for each code.

    Parameters
    ----------
    code: str
        The code to get the conversion factor for

    Returns
    -------
    float
        The correct conversion factor

    Raises
    ------
    ValueError
        If code is not defined
    """
    if code in ["forcefields", "vasp"]:
        return VaspToTHz
    if code == "aims":
        return omegaToTHz  # Based on CODATA 2002
    raise ValueError(f"Frequency conversion factor for code ({code}) not defined.")


def get_cutoffs(supercell_structure: Structure) -> list[list[float]]:
    """
    Determine the trial cutoffs based on a supercell structure.

    This function calculates and returns the best cutoffs for 2nd, 3rd, and 4th order
    interactions for a given supercell structure. It performs linear interpolation to
    find cutoffs for specific target degrees of freedom (DOFs), generates combinations
    of cutoffs, and filters them to retain unique DOFs.

    Args:
        supercell_structure: A structure.

    Returns
    -------
        list[list[float]]: A list of lists where each sublist contains the best cutoffs
                           for 2nd, 3rd, and 4th order interactions.
    """
    # Initialize lists and variables
    cutoffs_2nd_list = []
    cutoffs_3rd_list = []
    cutoffs_4th_list = []
    n_doffs_2nd_list = []
    n_doffs_3rd_list = []
    n_doffs_4th_list = []
    # create a list of cutoffs to check starting from 2 to 11, with a step size of 0.1
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    max_cutoff = estimate_maximum_cutoff(supercell_atoms)
    cutoffs_2nd_to_check = np.arange(3, max_cutoff, 0.3)
    cutoffs_3rd_to_check = np.arange(3, max_cutoff, 0.1)
    cutoffs_4th_to_check = np.arange(3, max_cutoff, 0.1)

    # Assume that supercell_atoms is defined before
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)

    def get_best_cutoff(
        cutoffs_to_check: np.ndarray,
        target_n_doff: int,
        order: int
    ) -> tuple[float, list[int], np.ndarray]:
        """
        Find the best cutoff value for a given order of interaction.

        This function iterates over a range of cutoff values, evaluates the DOFs, and
        finds the best cutoff that meets the target number of DOFs. It performs
        iterative refinement to narrow down the best cutoff.

        Args:
        cutoffs_to_check (numpy.ndarray): An array of cutoff values to check.
        target_n_doff (int): The target number of degrees of freedom (DOFs).
        order (int): The order of interaction (2 for 2nd order, 3 for 3rd order,
                    4 for 4th order).

        Returns
        -------
            tuple: A tuple containing:
                - best_cutoff (float): The best cutoff value.
                - n_doffs_list (list[int]): A list of DOFs corresponding to the cutoffs
                                            checked.
                - cutoffs_to_check (numpy.ndarray): The array of cutoff values checked.

        """
        n_doffs_list = []
        best_cutoff = np.inf

        if order == 2:
            increment_size = 0.2
        elif order == 3:
            increment_size = 0.25
        elif order == 4:
            increment_size = 0.05

        for cutoff in cutoffs_to_check:
            # check if order == 2 and if no element in n_doffs_list is greater than 1000
            if order == 2 and all(n_doff < 1000 for n_doff in n_doffs_list):
                with contextlib.suppress(ValueError):
                    cs = ClusterSpace(
                        supercell_atoms,
                        [cutoff],
                        symprec=1e-3,
                        acoustic_sum_rules=True
                        )
            elif order == 3 and all(n_doff < 2200 for n_doff in n_doffs_list):
                with contextlib.suppress(ValueError):
                    cs = ClusterSpace(
                        supercell_atoms,
                        [2, cutoff],
                        symprec=1e-3,
                        acoustic_sum_rules=True
                        )
            elif order == 4 and all(n_doff < 2200 for n_doff in n_doffs_list):
                with contextlib.suppress(ValueError):
                    cs = ClusterSpace(
                        supercell_atoms,
                        [2, 3, cutoff],
                        symprec=1e-3,
                        acoustic_sum_rules=True
                        )

            try:
                n_doff = cs.get_n_dofs_by_order(order)
                if (
                    (order == 2 and n_doff < 1000) or
                    (order == 3 and n_doff < 2200) or
                    (order == 4 and n_doff < 2200)
                    ):
                    logger.info(f"adding n_doff = {n_doff} to the n_doff list")
                    n_doffs_list.append(n_doff)
                elif (
                    (order == 2 and n_doff > 1000) or
                    (order == 3 and n_doff > 2200) or
                    (order == 4 and n_doff > 2200)
                    ):
                    # remove all the cutoffs from cutoffs_to_check from the current
                    # cutoff once we are inside this block
                    cutoff_index = np.where(cutoffs_to_check == cutoff)[0][0]
                    cutoffs_to_check = cutoffs_to_check[:cutoff_index]
                    # do the same for n_doffs_list
                    n_doffs_list = n_doffs_list[:cutoff_index + 1]
                    break
            except UnboundLocalError:
                logger.info(f"UnboundLocalError for cutoff = {cutoff}")
                # find the index of the cutoff in the cutoffs_to_check list
                cutoff_index = np.where(cutoffs_to_check == cutoff)[0][0]
                logger.info(cutoff_index)
                # remove only the cutoff corresponding to the index from the list
                cutoffs_to_check = np.delete(cutoffs_to_check, cutoff_index)
                continue

        # Find the closest cutoff to the target
        closest_index = np.argmin(np.abs(np.array(n_doffs_list) - target_n_doff))
        best_cutoff = cutoffs_to_check[closest_index]

        while increment_size >= 0.01:
            if order == 2:
                with contextlib.suppress(ValueError):
                    cs = ClusterSpace(
                        supercell_atoms,
                        [best_cutoff],
                        symprec=1e-3,
                        acoustic_sum_rules=True
                        )
            elif order == 3:
                with contextlib.suppress(ValueError):
                    cs = ClusterSpace(
                        supercell_atoms,
                        [3, best_cutoff],
                        symprec=1e-3,
                        acoustic_sum_rules=True
                        )
            elif order == 4:
                with contextlib.suppress(ValueError):
                    cs = ClusterSpace(
                        supercell_atoms,
                        [3, 3, best_cutoff],
                        symprec=1e-3,
                        acoustic_sum_rules=True
                        )

            n_doff = cs.get_n_dofs_by_order(order)

            if n_doff > target_n_doff:
                best_cutoff -= increment_size / 2
            else:
                best_cutoff += increment_size / 2

            increment_size /= 4

        return best_cutoff, n_doffs_list, cutoffs_to_check

    # Get best cutoffs for 2nd, 3rd, and 4th order
    logger.info("Getting best cutoffs for 2nd order")
    best_2nd_order_cutoff, n_doffs_2nd_list, cutoffs_2nd_to_check = get_best_cutoff(
        cutoffs_2nd_to_check, 100, 2)
    logger.info("Getting best cutoffs for 3rd order")
    best_3rd_order_cutoff, n_doffs_3rd_list, cutoffs_3rd_to_check = get_best_cutoff(
        cutoffs_3rd_to_check, 1000, 3)
    logger.info("Getting best cutoffs for 4th order")
    best_4th_order_cutoff, n_doffs_4th_list, cutoffs_4th_to_check = get_best_cutoff(
        cutoffs_4th_to_check, 1000, 4)

    cutoffs_2nd_list.append(best_2nd_order_cutoff)
    cutoffs_3rd_list.append(best_3rd_order_cutoff)
    cutoffs_4th_list.append(best_4th_order_cutoff)

    best_cutoff_list = [[best_2nd_order_cutoff,
                         best_3rd_order_cutoff,
                         best_4th_order_cutoff]]
    logger.info(f"best_cutoff_list = {best_cutoff_list}")

    # Linear interpolation to find cutoffs for targets
    def interpolate_and_find_cutoffs(
            cutoffs_to_check: np.ndarray,
            n_doffs_list: list[int],
            targets: list[int]) -> np.ndarray:
        """
        Perform linear interpolation to find cutoff values for specific target DOFs.

        This function interpolates the given cutoff values and their corresponding DOFs
        to find the cutoff values that achieve the specified target DOFs.

        Args:
        cutoffs_to_check (numpy.ndarray): An array of cutoff values to check.
        n_doffs_list (list[int]): A list of DOFs corresponding to the cutoffs checked.
        targets (list[int]): A list of target DOFs for which to find the cutoff values.

        Returns
        -------
            numpy.ndarray: An array of interpolated cutoff values corresponding to the
            target DOFs.
        """
        logger.info(f"cutoffs_to_check = {cutoffs_to_check}")
        logger.info(f"n_doffs_list = {n_doffs_list}")
        logger.info(f"targets = {targets}")
        logger.info(f"len(cutoffs_to_check) = {len(cutoffs_to_check)}")
        logger.info(f"len(n_doffs_list) = {len(n_doffs_list)}")
        cutoffs_for_targets = np.interp(targets, n_doffs_list, cutoffs_to_check)
        logger.info(f"cutoffs_for_targets = {cutoffs_for_targets}")
        return cutoffs_for_targets


    cutoffs_2nd_list.extend(interpolate_and_find_cutoffs(
        cutoffs_2nd_to_check, n_doffs_2nd_list, [70, 220]))
    cutoffs_3rd_list.extend(interpolate_and_find_cutoffs(
        cutoffs_3rd_to_check, n_doffs_3rd_list, [1500, 2000]))
    cutoffs_4th_list.extend(interpolate_and_find_cutoffs(
        cutoffs_4th_to_check, n_doffs_4th_list, [1500, 2000]))

    # Sort the lists
    cutoffs_2nd_list = sorted(cutoffs_2nd_list)
    cutoffs_3rd_list = sorted(cutoffs_3rd_list)
    cutoffs_4th_list = sorted(cutoffs_4th_list)

    # Generate combinations of cutoffs
    cutoffs = np.array(list(map(list, product(
        cutoffs_2nd_list, cutoffs_3rd_list, cutoffs_4th_list))))
    logger.info(f"cutoffs = {cutoffs}")
    logger.info(f"len(cutoffs) = {len(cutoffs)}")
    logger.info(f"cutoffs[0] = {cutoffs[0]}")
    dofs_list = []
    for cutoff_value in cutoffs:
        logger.info(f"cutoff inside the for loop = {cutoff_value}")
        cutoff = cutoff_value.tolist()
        logger.info(f"cutoff.tolist() = {cutoff}")
        with contextlib.suppress(ValueError):
            cs = ClusterSpace(
                supercell_atoms,
                cutoff,
                symprec=1e-3,
                acoustic_sum_rules=True
                )
            n_doff_2 = cs.get_n_dofs_by_order(2)
            n_doff_3 = cs.get_n_dofs_by_order(3)
            n_doff_4 = cs.get_n_dofs_by_order(4)
            dofs_list.append([n_doff_2, n_doff_3, n_doff_4])
            logger.info(f"dofs_list = {dofs_list}")

    # Save the plots for cutoffs vs n_doffs
    def save_plot(
            cutoffs_to_check: np.ndarray,
            n_doffs_list: list[int],
            order: int) -> None:
        """Save the plot for cutoffs vs n_doffs."""
        plt.figure()
        plt.scatter(cutoffs_to_check, n_doffs_list, color="blue", label="Data Points")
        plt.plot(cutoffs_to_check,
                 n_doffs_list, color="red", label="Linear Interpolation")
        plt.xlabel(f"Cutoffs {order} Order")
        plt.ylabel(f"n_doffs_{order}")
        plt.title(f"Linear Interpolation for n_doffs_{order} vs Cutoffs {order} Order")
        plt.legend()
        plt.savefig(f"cutoffs_{order}_vs_n_doffs_{order}.png")

    save_plot(cutoffs_2nd_to_check, n_doffs_2nd_list, 2)
    save_plot(cutoffs_3rd_to_check, n_doffs_3rd_list, 3)
    save_plot(cutoffs_4th_to_check, n_doffs_4th_list, 4)

    logger.info("We have completed the cutoffs calculation.")
    logger.info(f"cutoffs = {cutoffs}")

    max_cutoff = estimate_maximum_cutoff(AseAtomsAdaptor.get_atoms(supercell_structure))
    cutoffs[cutoffs > max_cutoff] = max_cutoff
    logger.info(f"CUTOFFS \n {cutoffs}")
    logger.info(f"MAX_CUTOFF \n {max_cutoff}")
    good_cutoffs = np.all(cutoffs < max_cutoff - 0.1, axis=1)
    logger.info(f"GOOD CUTOFFS \n{good_cutoffs}")

    cutoffs = cutoffs[good_cutoffs].tolist()
    logger.info(f"cutoffs_used = {cutoffs}")

    def filter_cutoffs(
            cutoffs: list[list[int]],
            dofs_list: list[list[int]]) -> list[list[int]]:
        """Filter cutoffs based on unique DOFs."""
        # Map cutoffs to dofs_list
        cutoffs_to_dofs = {tuple(c): tuple(d) for c, d in zip(cutoffs, dofs_list)}
        logger.info(f"Cutoffs to DOFs mapping: {cutoffs_to_dofs}")

        # Track seen dofs and keep only the first occurrence of each unique dofs
        seen_dofs = set()
        new_cutoffs = []

        for c, d in zip(cutoffs, dofs_list):
            d_tuple = tuple(d)
            if d_tuple not in seen_dofs:
                seen_dofs.add(d_tuple)
                new_cutoffs.append(c)

        logger.info(f"Unique DOFs: {seen_dofs}")
        logger.info(f"New cutoffs: {new_cutoffs}")

        return new_cutoffs

    cutoffs = filter_cutoffs(cutoffs, dofs_list)
    logger.info(f"Filtered cutoffs: {cutoffs}")

    # Round each order of the cutoff to the first decimal place
    rounded_cutoffs = [[round(order, 1) for order in cutoff] for cutoff in cutoffs]
    logger.info(f"Rounded cutoffs: {rounded_cutoffs}")

    return rounded_cutoffs


class PhononComputationalSettings(BaseModel):
    """Collection to store computational settings for the phonon computation."""

    # could be optional and implemented at a later stage?
    npoints_band: int = Field("number of points for band structure computation")
    kpath_scheme: str = Field("indicates the kpath scheme")
    kpoint_density_dos: int = Field(
        "number of points for computation of free energies and densities of states",
    )


class ThermalDisplacementData(BaseModel):
    """Collection to store information on the thermal displacement matrices."""

    freq_min_thermal_displacements: float = Field(
        "cutoff frequency in THz to avoid numerical issues in the "
        "computation of the thermal displacement parameters"
    )
    thermal_displacement_matrix_cif: Optional[list[list[Matrix3D]]] = Field(
        None, description="field including thermal displacement matrices in CIF format"
    )
    thermal_displacement_matrix: Optional[list[list[Matrix3D]]] = Field(
        None,
        description="field including thermal displacement matrices in Cartesian "
        "coordinate system",
    )
    temperatures_thermal_displacements: Optional[list[int]] = Field(
        None,
        description="temperatures at which the thermal displacement matrices"
        "have been computed",
    )


class PhononUUIDs(BaseModel):
    """Collection to save all uuids connected to the phonon run."""

    optimization_run_uuid: Optional[str] = Field(
        None, description="optimization run uuid"
    )
    displacements_uuids: Optional[list[str]] = Field(
        None, description="The uuids of the displacement jobs."
    )
    static_run_uuid: Optional[str] = Field(None, description="static run uuid")
    born_run_uuid: Optional[str] = Field(None, description="born run uuid")


class ForceConstants(MSONable):
    """A force constants class."""

    def __init__(self, force_constants: list[list[Matrix3D]]) -> None:
        self.force_constants = force_constants


class PhononJobDirs(BaseModel):
    """Collection to save all job directories relevant for the phonon run."""

    displacements_job_dirs: Optional[list[Optional[str]]] = Field(
        None, description="The directories where the displacement jobs were run."
    )
    static_run_job_dir: Optional[Optional[str]] = Field(
        None, description="Directory where static run was performed."
    )
    born_run_job_dir: Optional[str] = Field(
        None, description="Directory where born run was performed."
    )
    optimization_run_job_dir: Optional[str] = Field(
        None, description="Directory where optimization run was performed."
    )
    taskdoc_run_job_dir: Optional[str] = Field(
        None, description="Directory where task doc was generated."
    )


class PhononBSDOSDoc(StructureMetadata, extra="allow"):  # type: ignore[call-arg]
    """Collection of all data produced by the phonon workflow."""

    structure: Optional[Structure] = Field(
        None, description="Structure of Materials Project."
    )

    phonon_bandstructure: Optional[PhononBandStructureSymmLine] = Field(
        None,
        description="Phonon band structure object.",
    )

    phonon_dos: Optional[PhononDos] = Field(
        None,
        description="Phonon density of states object.",
    )

    free_energies: Optional[list[float]] = Field(
        None,
        description="vibrational part of the free energies in J/mol per "
        "formula unit for temperatures in temperature_list",
    )

    heat_capacities: Optional[list[float]] = Field(
        None,
        description="heat capacities in J/K/mol per "
        "formula unit for temperatures in temperature_list",
    )

    internal_energies: Optional[list[float]] = Field(
        None,
        description="internal energies in J/mol per "
        "formula unit for temperatures in temperature_list",
    )
    entropies: Optional[list[float]] = Field(
        None,
        description="entropies in J/(K*mol) per formula unit"
        "for temperatures in temperature_list ",
    )

    temperatures: Optional[list[int]] = Field(
        None,
        description="temperatures at which the vibrational"
        " part of the free energies"
        " and other properties have been computed",
    )

    total_dft_energy: Optional[float] = Field("total DFT energy per formula unit in eV")

    has_imaginary_modes: Optional[bool] = Field(
        None, description="if true, structure has imaginary modes"
    )

    # needed, e.g. to compute Grueneisen parameter etc
    force_constants: Optional[ForceConstants] = Field(
        None, description="Force constants between every pair of atoms in the structure"
    )

    born: Optional[list[Matrix3D]] = Field(
        None,
        description="born charges as computed from phonopy. Only for symmetrically "
        "different atoms",
    )

    epsilon_static: Optional[Matrix3D] = Field(
        None, description="The high-frequency dielectric constant"
    )

    supercell_matrix: Matrix3D = Field("matrix describing the supercell")
    primitive_matrix: Matrix3D = Field(
        "matrix describing relationship to primitive cell"
    )

    code: str = Field("String describing the code for the computation")

    phonopy_settings: PhononComputationalSettings = Field(
        "Field including settings for Phonopy"
    )

    thermal_displacement_data: Optional[ThermalDisplacementData] = Field(
        "Includes all data of the computation of the thermal displacements"
    )

    jobdirs: Optional[PhononJobDirs] = Field(
        "Field including all relevant job directories"
    )

    uuids: Optional[PhononUUIDs] = Field("Field including all relevant uuids")

    @classmethod
    def from_forces_born(
        cls,
        structure: Structure,
        supercell_matrix: np.array,
        displacement: float,
        sym_reduce: bool,
        symprec: float,
        use_symmetrized_structure: Union[str, None],
        kpath_scheme: str,
        code: str,
        displacement_data: dict[str, list],
        total_dft_energy: float,
        bulk_modulus: float,
        epsilon_static: Matrix3D = None,
        born: Matrix3D = None,
        fit_method: str | None = "rfe", # FIT_METHOD = "least-squares" #least-squares #omp #rfe #elasticnet
        disp_cut: float | None = None,
        temperature_qha: float | list | dict = [ i * 100 for i in range(21)], # Temp. for phonopy calc. of thermo. properties (free energy etc.)
        imaginary_tol: float = 0.025,  # in THz
        cutoffs: list[list[float]] | None = None,
        **kwargs,
    ) -> Self:
        """Generate collection of phonon data.

        Parameters
        ----------
        structure: Structure object
        supercell_matrix: numpy array describing the supercell
        displacement: float
            size of displacement in angstrom
        sym_reduce: bool
            if True, phonopy will use symmetry
        symprec: float
            precision to determine kpaths,
            primitive cells and symmetry in phonopy and pymatgen
        use_symmetrized_structure: str
            primitive, conventional or None
        kpath_scheme: str
            kpath scheme to generate phonon band structure
        code: str
            which code was used for computation
        displacement_data:
            output of the displacement data
        total_dft_energy: float
            total energy in eV per cell
        epsilon_static: Matrix3D
            The high-frequency dielectric constant
        born: Matrix3D
            born charges
        **kwargs:
            additional arguments
        """
        logger.info("Starting from_forces_born.")
        print("Starting from_forces_born.")
        factor = get_factor(code)
        # This opens the opportunity to add support for other codes
        # that are supported by phonopy

        cell = get_phonopy_structure(structure)

        if use_symmetrized_structure == "primitive":
            primitive_matrix: Union[np.ndarray, str] = np.eye(3)
        else:
            primitive_matrix = "auto"

        # TARP: THIS IS BAD! Including for discussions sake
        if cell.magnetic_moments is not None and primitive_matrix == "auto":
            if np.any(cell.magnetic_moments != 0.0):
                raise ValueError(
                    "For materials with magnetic moments, "
                    "use_symmetrized_structure must be 'primitive'"
                )
            cell.magnetic_moments = None

        phonon = Phonopy(
            cell,
            supercell_matrix,
            primitive_matrix=primitive_matrix,
            factor=factor,
            symprec=symprec,
            is_symmetry=sym_reduce,
        )
        phonon.generate_displacements(distance=displacement)
        set_of_forces = [np.array(forces) for forces in displacement_data["forces"]]

        if born is not None and epsilon_static is not None:
            if len(structure) == len(born):
                borns, epsilon = symmetrize_borns_and_epsilon(
                    ucell=phonon.unitcell,
                    borns=np.array(born),
                    epsilon=np.array(epsilon_static),
                    symprec=symprec,
                    primitive_matrix=phonon.primitive_matrix,
                    supercell_matrix=phonon.supercell_matrix,
                    is_symmetry=kwargs.get("symmetrize_born", True),
                )
            else:
                raise ValueError(
                    "Number of born charges does not agree with number of atoms"
                )
            if code == "vasp" and not np.all(np.isclose(borns, 0.0)):
                phonon.nac_params = {
                    "born": borns,
                    "dielectric": epsilon,
                    "factor": 14.399652,
                }
            # Other codes could be added here
        else:
            borns = None
            epsilon = None

        # Produces all force constants
        phonon.produce_force_constants(forces=set_of_forces) # add the def run_hiphive() here and get the 2nd order FCs out. then use JZ's way to add FC to the phonon object.


        logger.info("Starting run_hiphive.")
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/npj_paper_review/based_on_MACE/mp_1591/launcher_2024-07-15-03-37-51-098937/launcher_2024-07-15-03-53-52-366405" # mp-1591 -- 3 configs per displ
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/5th_Sem/MSE299/npjReview/configs_convergence/mp-2691/1ConfigsPerDispl" # mp-2691 19 Å -- 1, 2, 3, 4 configs -- VASP -- CdSe
        # 3. Hiphive Fitting of FCPs upto 4th order
        PhononBSDOSDoc.run_hiphive(
            parent_structure=structure,
            fit_method=fit_method,
            disp_cut=disp_cut,
            bulk_modulus=bulk_modulus,
            temperature_qha=temperature_qha,
            imaginary_tol=imaginary_tol,
            # prev_dir_json_saver=static_calcs.output["current_dir"],
            cutoffs=cutoffs,
            displacement_data=displacement_data,
            supercell_matrix=supercell_matrix,
            T_QHA= [i * 100 for i in range(21)]  # Temp. for phonopy calc. of thermo. properties (free energy etc.)
        )
        logger.info("Completed run_hiphive.")


        # Read the force constants from the output file of pheasy code
        force_constants = parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS_2ND") # FORCE_CONSTANTS_2ND FORCE_CONSTANTS
        phonon.force_constants = force_constants
        # symmetrize the force constants to make them physically correct based on the space group
        # symmetry of the crystal structure.
        phonon.symmetrize_force_constants()







        # with phonopy.load("phonopy.yaml") the phonopy API can be used
        phonon.save("phonopy.yaml")

        # get phonon band structure
        kpath_dict, kpath_concrete = PhononBSDOSDoc.get_kpath(
            structure=get_pmg_structure(phonon.primitive),
            kpath_scheme=kpath_scheme,
            symprec=symprec,
        )

        npoints_band = kwargs.get("npoints_band", 101)
        qpoints, connections = get_band_qpoints_and_path_connections(
            kpath_concrete, npoints=kwargs.get("npoints_band", 101)
        )

        # phonon band structures will always be computed
        filename_band_yaml = "phonon_band_structure.yaml"

        # TODO: potentially add kwargs to avoid computation of eigenvectors
        phonon.run_band_structure(
            qpoints,
            path_connections=connections,
            with_eigenvectors=kwargs.get("band_structure_eigenvectors", False),
            is_band_connection=kwargs.get("band_structure_eigenvectors", False),
        )
        phonon.write_yaml_band_structure(filename=filename_band_yaml)
        bs_symm_line = get_ph_bs_symm_line(
            filename_band_yaml, labels_dict=kpath_dict, has_nac=born is not None
        )
        new_plotter = PhononBSPlotter(bs=bs_symm_line)
        new_plotter.save_plot(
            filename=kwargs.get("filename_bs", "phonon_band_structure.pdf"),
            units=kwargs.get("units", "THz"),
        )

        # will determine if imaginary modes are present in the structure
        imaginary_modes = bs_symm_line.has_imaginary_freq(
            tol=kwargs.get("tol_imaginary_modes", 1e-5)
        )

        # gets data for visualization on website - yaml is also enough
        if kwargs.get("band_structure_eigenvectors"):
            bs_symm_line.write_phononwebsite("phonon_website.json")

        # get phonon density of states
        filename_dos_yaml = "phonon_dos.yaml"

        kpoint_density_dos = kwargs.get("kpoint_density_dos", 7_000)
        kpoint = Kpoints.automatic_density(
            structure=get_pmg_structure(phonon.primitive),
            kppa=kpoint_density_dos,
            force_gamma=True,
        )
        phonon.run_mesh(kpoint.kpts[0])
        phonon.run_total_dos()
        phonon.write_total_dos(filename=filename_dos_yaml)
        dos = get_ph_dos(filename_dos_yaml)
        new_plotter_dos = PhononDosPlotter()
        new_plotter_dos.add_dos(label="total", dos=dos)
        new_plotter_dos.save_plot(
            filename=kwargs.get("filename_dos", "phonon_dos.pdf"),
            units=kwargs.get("units", "THz"),
        )

        # compute vibrational part of free energies per formula unit
        temperature_range = np.arange(
            kwargs.get("tmin", 0), kwargs.get("tmax", 1000), kwargs.get("tstep", 10)
        )

        free_energies = [
            dos.helmholtz_free_energy(
                temp=temp, structure=get_pmg_structure(phonon.primitive)
            )
            for temp in temperature_range
        ]

        entropies = [
            dos.entropy(temp=temp, structure=get_pmg_structure(phonon.primitive))
            for temp in temperature_range
        ]

        internal_energies = [
            dos.internal_energy(
                temp=temp, structure=get_pmg_structure(phonon.primitive)
            )
            for temp in temperature_range
        ]

        heat_capacities = [
            dos.cv(temp=temp, structure=get_pmg_structure(phonon.primitive))
            for temp in temperature_range
        ]

        # will compute thermal displacement matrices
        # for the primitive cell (phonon.primitive!)
        # only this is available in phonopy
        if kwargs.get("create_thermal_displacements"):
            phonon.run_mesh(
                kpoint.kpts[0], with_eigenvectors=True, is_mesh_symmetry=False
            )
            freq_min_thermal_displacements = kwargs.get(
                "freq_min_thermal_displacements", 0.0
            )
            phonon.run_thermal_displacement_matrices(
                t_min=kwargs.get("tmin_thermal_displacements", 0),
                t_max=kwargs.get("tmax_thermal_displacements", 500),
                t_step=kwargs.get("tstep_thermal_displacements", 100),
                freq_min=freq_min_thermal_displacements,
            )

            temperature_range_thermal_displacements = np.arange(
                kwargs.get("tmin_thermal_displacements", 0),
                kwargs.get("tmax_thermal_displacements", 500),
                kwargs.get("tstep_thermal_displacements", 100),
            )
            for idx, temp in enumerate(temperature_range_thermal_displacements):
                phonon.thermal_displacement_matrices.write_cif(
                    phonon.primitive, idx, filename=f"tdispmat_{temp}K.cif"
                )
            _disp_mat = phonon._thermal_displacement_matrices  # noqa: SLF001
            tdisp_mat = _disp_mat.thermal_displacement_matrices.tolist()

            tdisp_mat_cif = _disp_mat.thermal_displacement_matrices_cif.tolist()

        else:
            tdisp_mat = None
            tdisp_mat_cif = None

        formula_units = (
            structure.composition.num_atoms
            / structure.composition.reduced_composition.num_atoms
        )

        total_dft_energy_per_formula_unit = (
            total_dft_energy / formula_units if total_dft_energy is not None else None
        )

        logger.info("Finished from_forces_born.")

        return cls.from_structure(
            structure=structure,
            meta_structure=structure,
            phonon_bandstructure=bs_symm_line,
            phonon_dos=dos,
            free_energies=free_energies,
            internal_energies=internal_energies,
            heat_capacities=heat_capacities,
            entropies=entropies,
            temperatures=temperature_range.tolist(),
            total_dft_energy=total_dft_energy_per_formula_unit,
            has_imaginary_modes=imaginary_modes,
            force_constants={"force_constants": phonon.force_constants.tolist()}
            if kwargs["store_force_constants"]
            else None,
            born=borns.tolist() if borns is not None else None,
            epsilon_static=epsilon.tolist() if epsilon is not None else None,
            supercell_matrix=phonon.supercell_matrix.tolist(),
            primitive_matrix=phonon.primitive_matrix.tolist(),
            code=code,
            thermal_displacement_data={
                "temperatures_thermal_displacements": temperature_range_thermal_displacements.tolist(),  # noqa: E501
                "thermal_displacement_matrix_cif": tdisp_mat_cif,
                "thermal_displacement_matrix": tdisp_mat,
                "freq_min_thermal_displacements": freq_min_thermal_displacements,
            }
            if kwargs.get("create_thermal_displacements")
            else None,
            jobdirs={
                "displacements_job_dirs": displacement_data["dirs"],
                "static_run_job_dir": kwargs["static_run_job_dir"],
                "born_run_job_dir": kwargs["born_run_job_dir"],
                "optimization_run_job_dir": kwargs["optimization_run_job_dir"],
                "taskdoc_run_job_dir": str(Path.cwd()),
            },
            uuids={
                "displacements_uuids": displacement_data["uuids"],
                "born_run_uuid": kwargs["born_run_uuid"],
                "optimization_run_uuid": kwargs["optimization_run_uuid"],
                "static_run_uuid": kwargs["static_run_uuid"],
            },
            phonopy_settings={
                "npoints_band": npoints_band,
                "kpath_scheme": kpath_scheme,
                "kpoint_density_dos": kpoint_density_dos,
            },
        )

    @staticmethod
    def get_kpath(
        structure: Structure, kpath_scheme: str, symprec: float, **kpath_kwargs
    ) -> tuple:
        """Get high-symmetry points in k-space in phonopy format.

        Parameters
        ----------
        structure: Structure Object
        kpath_scheme: str
            string describing kpath
        symprec: float
            precision for symmetry determination
        **kpath_kwargs:
            additional parameters that can be passed to this method as a dict
        """
        if kpath_scheme in ("setyawan_curtarolo", "latimer_munro", "hinuma"):
            high_symm_kpath = HighSymmKpath(
                structure, path_type=kpath_scheme, symprec=symprec, **kpath_kwargs
            )
            kpath = high_symm_kpath.kpath
        elif kpath_scheme == "seekpath":
            high_symm_kpath = KPathSeek(structure, symprec=symprec, **kpath_kwargs)
            kpath = high_symm_kpath._kpath  # noqa: SLF001
        else:
            raise ValueError(f"Unexpected {kpath_scheme=}")

        path = copy.deepcopy(kpath["path"])

        for set_idx, label_set in enumerate(kpath["path"]):
            for lbl_idx, label in enumerate(label_set):
                path[set_idx][lbl_idx] = kpath["kpoints"][label]
        return kpath["kpoints"], path

    @staticmethod
    def run_hiphive(
        parent_structure: Structure,
        displacement_data: dict[str, list],
        cutoffs: list[list] | None = None,
        fit_method: str | None = "rfe", # FIT_METHOD = "least-squares" #least-squares #omp #rfe #elasticnet
        disp_cut: float | None = None,
        bulk_modulus: float | None = None,
        temperature_qha: float | list | dict = None,
        imaginary_tol: float | None = None,
        prev_dir_json_saver: str | None = None,
        supercell_structure: Structure | None = None,
        supercell_matrix: np.array | None = None,
        T_QHA: ClassVar[list[int]] = [i * 100 for i in range(21)]  # Temp. for phonopy calc. of thermo. properties (free energy etc.)
    ) -> dict:
        """
        Fit force constants using hiPhive.

        Requires "perturbed_structures.json", "perturbed_forces.json", and
        "structure_data.json" files to be present in the current working directory.

        Args:
            cutoffs (Optional[list[list]]): A list of cutoffs to trial. If None,
                a set of trial cutoffs will be generated based on the structure
                (default).
            separate_fit: If True, harmonic and anharmonic force constants are fit
                separately and sequentially, harmonic first then anharmonic. If
                False, then they are all fit in one go. Default is False.
            disp_cut: if separate_fit=True, determines the mean displacement of perturbed
                structure to be included in harmonic (<) or anharmonic (>) fitting
            imaginary_tol (float): Tolerance used to decide if a phonon mode
                is imaginary, in THz.
            fit_method (str): Method used for fitting force constants. This can
                be any of the values allowed by the hiphive ``Optimizer`` class.
        """
        logger.info(f"cutoffs = {cutoffs}")
        logger.info(f"disp_cut is {disp_cut}")
        logger.info(f"fit_method is {fit_method}")

        supercell_structure = SupercellTransformation(
            scaling_matrix=supercell_matrix
            ).apply_transformation(parent_structure)

        # copy_hiphive_outputs(prev_dir_json_saver)

        # perturbed_structures = loadfn("perturbed_structures.json")
        # perturbed_forces = loadfn("perturbed_forces_new.json")
        # structure_data = loadfn("structure_data.json")

        # parent_structure = structure_data["structure"]
        # supercell_structure = structure_data["supercell_structure"]
        # supercell_matrix = np.array(structure_data["supercell_matrix"])

        if cutoffs is None:
            # cutoffs = get_cutoffs(supercell_structure)
            # cutoffs = [[5, 4, 3.5]] # Ba2DySbO6 cubic
            cutoffs = [[5, 4, 3]] # CoS2 cubic
            # cutoffs = [[9, 6, 4]] # Bi4Au
            logger.info(f"cutoffs is {cutoffs}")
        else:
            pass

        t_qha = temperature_qha if temperature_qha is not None else T_QHA
        if isinstance(t_qha, list):
            t_qha.sort()
        else:
            # Handle the case where T_qha is not a list
            # You can choose to raise an exception or handle it differently
            # For example, if T_qha is a single temperature value, you can
            # convert it to a list
            t_qha = [t_qha]

        logger.info(f"t_qha is {t_qha}")

        structures = []
        logger.info(f"supercell_structure is {supercell_structure}")

        set_of_forces = [np.array(forces) for forces in displacement_data["forces"]]
        supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
        for structure, forces in zip(displacement_data["structure"], set_of_forces):
            logger.info(f"structure is {structure}")
            atoms = AseAtomsAdaptor.get_atoms(structure)
            displacements = get_displacements(atoms, supercell_atoms)
            atoms.new_array("displacements", displacements)
            atoms.new_array("forces", forces)
            atoms.positions = supercell_atoms.get_positions()
            structures.append(atoms)

            # Calculate mean displacements
            mean_displacements = np.linalg.norm(displacements, axis=1).mean()
            logger.info(f"Mean displacements while reading individual displacements: "
                        f"{mean_displacements}")
            # Calculate standard deviation of displacements
            std_displacements = np.linalg.norm(displacements, axis=1).std()
            logger.info("Standard deviation of displacements while reading individual"
                        "displacements: "
                        f"{std_displacements}")

        all_cutoffs = cutoffs
        logger.info(f"all_cutoffs is {all_cutoffs}")

        fcs, param, cs, fitting_data, fcp = PhononBSDOSDoc.fit_force_constants(
            parent_structure=parent_structure,
            supercell_matrix=supercell_matrix,
            supercell_structure=supercell_structure,
            structures=structures,
            all_cutoffs=all_cutoffs,
            disp_cut=disp_cut,
            imaginary_tol=imaginary_tol,
            fit_method=fit_method,
        )

        logger.info("Saving Harmonic props")
        thermal_data, phonopy = PhononBSDOSDoc.harmonic_properties(
            parent_structure, supercell_matrix, fcs, t_qha, imaginary_tol
        )

        anharmonic_data = PhononBSDOSDoc.anharmonic_properties(
            phonopy,
            fcs,
            t_qha,
            thermal_data["heat_capacity"],
            thermal_data["n_imaginary"],
            bulk_modulus,
        )

        if fcs is None:
            raise RuntimeError("Could not find a force constant solution")

        if isinstance(fcs, ForceConstants):
            logger.info("Writing force_constants")
            fcs.write("force_constants.fcs")
        else:
            logger.info("fcs is not an instance of ForceConstants")

        if isinstance(fcp, ForceConstantPotential):
            logger.info("Writing force_constants_potential")
            fcp.write("force_constants_potential.fcp")

        logger.info("Saving parameters")
        np.savetxt("parameters.txt", param)

        if isinstance(cs, ClusterSpace):
            logger.info("Writing cluster_space")
            cs.write("cluster_space.cs")
            logger.info("cluster_space writing is complete")
        else:
            logger.info("cs is not an instance of ClusterSpace")

        logger.info("Saving phonopy_params")
        phonopy.save("phonopy_params.yaml")
        fitting_data["best_n_imaginary"] = thermal_data.pop("n_imaginary")
        thermal_data.update(anharmonic_data)
        logger.info("Saving fitting_data")
        dumpfn(fitting_data, "fitting_data.json")
        logger.info("Saving thermal_data")
        dumpfn(thermal_data, "thermal_data.json")

        logger.info("Writing cluster space and force_constants")
        logger.info(f"{type(fcs)}")

        logger.info(f"phonopy_atoms_to_ase(phonopy.supercell) = {phonopy_atoms_to_ase(phonopy.supercell)}")
        logger.info(f"supercell_atoms = {supercell_atoms}")
        from pymatgen.io.ase import MSONAtoms
        structure_data_phonopy_atoms = {
            "supercell_structure_phonopy_atoms": MSONAtoms(phonopy_atoms_to_ase(phonopy.supercell)),
        }
        structure_data_pymatgen_atoms = {
            "supercell_structure_pymatgen_atoms": supercell_atoms,
        }

        dumpfn(structure_data_phonopy_atoms, "structure_data_phonopy_atoms.json")
        dumpfn(structure_data_pymatgen_atoms, "structure_data_pymatgen_atoms.json")

        primitive_atoms_phonopy = phonopy_atoms_to_ase(phonopy.primitive)
        primitive_atoms_pymatgen = AseAtomsAdaptor.get_atoms(parent_structure)

        prim_structure_data_phonopy_atoms = {
            "primitive_structure_phonopy_atoms": MSONAtoms(primitive_atoms_phonopy),
        }
        prim_structure_data_pymatgen_atoms = {
            "primitive_structure_pymatgen_atoms": primitive_atoms_pymatgen,
        }
        # dumpfn primitve and supercell atoms
        dumpfn(prim_structure_data_phonopy_atoms, "prim_structure_data_phonopy_atoms.json")
        dumpfn(prim_structure_data_pymatgen_atoms, "prim_structure_data_pymatgen_atoms.json")


        primitive_structure_phonopy = AseAtomsAdaptor.get_structure(primitive_atoms_phonopy)
        primitive_structure_pymatgen = AseAtomsAdaptor.get_structure(primitive_atoms_pymatgen)

        # dumpfn primitve and supercell structures
        dumpfn(primitive_structure_phonopy, "primitive_structure_phonopy.json")
        dumpfn(primitive_structure_pymatgen, "primitive_structure_pymatgen.json")

        primitive_structure_phonopy_new = SpacegroupAnalyzer(primitive_structure_phonopy).find_primitive()
        primitive_structure_pymatgen_new = SpacegroupAnalyzer(primitive_structure_pymatgen).find_primitive()

        # dumpfn primitve and supercell structures
        dumpfn(primitive_structure_phonopy_new, "primitive_structure_phonopy_new.json")
        dumpfn(primitive_structure_pymatgen_new, "primitive_structure_pymatgen_new.json")

        if fitting_data["best_n_imaginary"] == 0:
        # if True:
            logger.info("No imaginary modes! Writing ShengBTE files")
            atoms = AseAtomsAdaptor.get_atoms(parent_structure)
            # # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_1", atoms, order=3)
            # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_from_fcs_pymatgen_struct", atoms)
            fcs.write_to_phonopy("FORCE_CONSTANTS_2ND", format="text")

            # primitive_atoms_phonopy = phonopy_atoms_to_ase(phonopy.primitive)
            # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_from_fcs_phonopy_struct", primitive_atoms_phonopy)

            HiphiveForceConstants.write_to_phonopy(fcs, "fc2.hdf5", "hdf5") # ForceConstants
            # ForceConstants.write_to_phono3py(fcs, "fc3.hdf5", "hdf5")
            HiphiveForceConstants.write_to_phono3py(fcs, "fc3.hdf5") # ForceConstants

            ### detour from hdf5
            supercell_atoms_phonopy = phonopy_atoms_to_ase(phonopy.supercell)
            supercell_atoms_pymatgen = AseAtomsAdaptor.get_atoms(supercell_structure)

            # dumpfn(supercell_atoms_phonopy, "supercell_atoms_phonopy.json")
            dumpfn(supercell_atoms_pymatgen, "supercell_atoms_pymatgen.json")

            # check if the supercell_atoms are the same
            if supercell_atoms_phonopy == supercell_atoms_pymatgen:
                logger.info("supercell_atoms are the same")
            else:
                logger.info("supercell_atoms are different")

            supercell_atoms = supercell_atoms_phonopy
            # supercell_atoms = supercell_atoms_pymatgen
            # fcs = ForceConstants.read_phono3py(supercell_atoms, "fc3.hdf5", order=3)
            fcs = HiphiveForceConstants.read_phono3py(supercell_atoms, "fc3.hdf5") # ForceConstants
            # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, order=3, fc_tol=1e-4)
            # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, fc_tol=1e-4)

            # # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_from_hdf5_phonopy_struct", atoms) # this was the original way of writing shengBTE files
            # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_from_hdf5_phonopy_struct", phonopy_atoms_to_ase(phonopy.primitive))

            supercell_atoms = supercell_atoms_pymatgen
            fcs = HiphiveForceConstants.read_phono3py(supercell_atoms, "fc3.hdf5") # ForceConstants
            fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_from_hdf5_pymatgen_struct", atoms)

        else:
            logger.info(f"best_n_imaginary = {fitting_data['best_n_imaginary']}")
            logger.info("ShengBTE files not written due to imaginary modes.")
            logger.info("You may want to perform phonon renormalization.")

        current_dir = os.getcwd()

        outputs: dict[str, list] = {
            "thermal_data": thermal_data,
            "anharmonic_data": anharmonic_data,
            "fitting_data": fitting_data,
            "param": param,
            "current_dir": current_dir
        }

        return outputs

    @staticmethod
    def fit_force_constants(
        parent_structure: Structure,
        supercell_matrix: np.ndarray,
        supercell_structure: Structure,
        structures: list[Atoms],
        all_cutoffs: list[list[float]],
        disp_cut: float | None = 0.055,
        imaginary_tol: float = 0.025,  # in THz
        fit_method: str | None = "rfe", # FIT_METHOD = "least-squares" #least-squares #omp #rfe #elasticnet
        n_jobs: int | None = -1,
        fit_kwargs: dict | None = None,
    ) -> tuple:
        """
        Fit FC using hiphive.

        Fit force constants using hiphive and select the optimum cutoff values.
        The optimum cutoffs will be determined according to:
        1. Number imaginary modes < ``max_n_imaginary``.
        2. Most negative imaginary frequency < ``max_imaginary_freq``.
        3. Least number of imaginary modes.
        4. Lowest free energy at 300 K.
        If criteria 1 and 2 are not satisfied, None will be returned as the
        force constants.
        Args:
            parent_structure: Initial input structure.
            supercell_matrix: Supercell transformation matrix.
            structures: A list of ase atoms objects with "forces" and
                "displacements" arrays added, as required by hiPhive.
            all_cutoffs: A nested list of cutoff values to trial. Each set of
                cutoffs contains the radii for different orders starting with second
                order.
            disp_cut: determines the mean displacement of perturbed
                structure to be included in harmonic (<) or anharmonic (>) fitting
            imaginary_tol: Tolerance used to decide if a phonon mode is imaginary,
                in THz.
            max_n_imaginary: Maximum number of imaginary modes allowed in the
                the final fitted force constant solution. If this criteria is not
                reached by any cutoff combination this FireTask will fizzle.
            max_imaginary_freq: Maximum allowed imaginary frequency in the
                final fitted force constant solution. If this criteria is not
                reached by any cutoff combination this FireTask will fizzle.
            fit_method: Method used for fitting force constants. This can be
                any of the values allowed by the hiphive ``Optimizer`` class.
            n_jobs: Number of processors to use for fitting coefficients. -1 means use all
                processors.
            fit_kwargs: Additional arguments passed to the hiphive force constant
                optimizer.

        Returns
        -------
            A tuple of the best fitted force constants as a hiphive
            ``SortedForceConstants`` object, array of parameters, cluster space,
            and a dictionary of information on the fitting results.
        """
        logger.info("Starting force constant fitting.")
        logger.info(f"disp_cut is {disp_cut}")
        logger.info(f"fit_method is {fit_method}")

        fitting_data: dict[str, Any] = {
            "cutoffs": [],
            "rmse_test": [],
            "imaginary": [],
            "cs_dofs": [],
            "n_imaginary": [],
            "parent_structure": parent_structure,
            "supercell_structure": supercell_structure,
            "supercell_matrix": supercell_matrix,
            "fit_method": fit_method,
            "disp_cut": disp_cut,
            "imaginary_tol": imaginary_tol,
            "best_cutoff": None,
            "best_rmse": np.inf
        }

        best_fit = {
            "n_imaginary": np.inf,
            "rmse_test": np.inf,
            "imaginary": None,
            "cs_dofs": [None, None, None],
            "cluster_space": None,
            "force_constants": None,
            "parameters": None,
            "cutoffs": None,
            "force_constants_potential": None,
        }
        # all_cutoffs = all_cutoffs[0] #later change it back to all_cutoffs
        n_cutoffs = len(all_cutoffs)
        logger.info(f"len_cutoffs={n_cutoffs}")

        fit_kwargs = fit_kwargs if fit_kwargs else {}
        if fit_method == "rfe" and n_jobs == -1:
            fit_kwargs["n_jobs"] = 1
        elif fit_method == "lasso":
            fit_kwargs["lasso"] = dict(max_iter=1000)
        elif fit_method == "elasticnet":
            fit_kwargs = {"max_iter": 100000}

        logger.info(f"CPU COUNT: {os.cpu_count()}")

        logger.info("We are starting Joblib_s parallellized jobs")

        cutoff_results = []
        for i, cutoffs in enumerate(all_cutoffs):
            result = PhononBSDOSDoc._run_cutoffs(
                                        i=i,
                                        cutoffs=cutoffs,
                                        n_cutoffs=n_cutoffs,
                                        parent_structure=parent_structure,
                                        supercell_structure=supercell_structure,
                                        structures=structures,
                                        supercell_matrix=supercell_matrix,
                                        fit_method=fit_method,
                                        disp_cut=disp_cut,
                                        imaginary_tol=imaginary_tol,
                                        fit_kwargs=fit_kwargs,
                                    )
            cutoff_results.append(result)

        for result in cutoff_results:
            if result is None:
                logger.info("result is None")
                continue
            logger.info(f"result = {result}")
            if result != {}:
                if "cutoffs" in result:
                    fitting_data["cutoffs"].append(result["cutoffs"])
                else:
                    logger.info("Cutoffs not found in result")
                    continue
                if "rmse_test" in result:
                    fitting_data["rmse_test"].append(result["rmse_test"])
                if "n_imaginary" in result:
                    fitting_data["n_imaginary"].append(result["n_imaginary"])
                if "cs_dofs" in result:
                    fitting_data["cs_dofs"].append(result["cs_dofs"])
                if "imaginary" in result:
                    fitting_data["imaginary"].append(result["imaginary"])
                if "rmse_test" in result and (result["rmse_test"] < best_fit["rmse_test"]):
                    best_fit.update(result)
                    fitting_data["best_cutoff"] = result["cutoffs"]
                    fitting_data["best_rmse"] = result["rmse_test"]
            else:
                logger.info("result is an empty dictionary")

        logger.info(f"best_fit = {best_fit}")
        logger.info(f"fitting_data = {fitting_data}")
        logger.info("Finished fitting force constants.")

        return (
            best_fit["force_constants"],
            best_fit["parameters"],
            best_fit["cluster_space"],
            fitting_data,
            best_fit["force_constants_potential"]
            )

    @staticmethod
    def _run_cutoffs(
        i: int,
        cutoffs: list[float],
        n_cutoffs: int,
        parent_structure: Structure,
        supercell_structure: Structure,
        structures: list[Atoms],
        supercell_matrix: np.ndarray,
        fit_method: str | None,
        disp_cut: float | None,
        fit_kwargs: dict[str, Any],
        imaginary_tol: float = 0.025,  # in THz
    ) -> dict[str, Any]:

        logger.info(f"Testing cutoffs {i+1} out of {n_cutoffs}: {cutoffs}")
        logger.info(f"fit_method is {fit_method}")
        supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
        supercell_atoms = structures[0]
        logger.info(f"supercell_atoms = {supercell_atoms}")

        if not is_cutoff_allowed(supercell_atoms, max(cutoffs)):
            logger.info("Skipping cutoff due as it is not commensurate with supercell size")
            return {}

        # # if you want to select specific clusters then use the following prototype code:
        # from hiphive.cutoffs import BeClusterFilter
        # be_cf = BeClusterFilter()
        # cs = ClusterSpace(
        #     supercell_atoms,
        #     cutoffs,
        #     cluster_filter=be_cf,
        #     symprec=1e-3,
        #     acoustic_sum_rules=True
        #     )
        # logger.info(f"cs orbit data = {cs.orbit_data}")

        try:
            cs = ClusterSpace(supercell_atoms, cutoffs, symprec=1e-3, acoustic_sum_rules=True)
        except ValueError:
            logger.info("ValueError encountered, moving to the next cutoff")
            return {}

        logger.debug(cs.__repr__())
        logger.info(cs)
        cs_2_dofs = cs.get_n_dofs_by_order(2)
        cs_3_dofs = cs.get_n_dofs_by_order(3)
        cs_4_dofs = cs.get_n_dofs_by_order(4) # uncomment this later
        cs_dofs = [cs_2_dofs, cs_3_dofs, cs_4_dofs] # uncomment this later
        # cs_dofs = [cs_2_dofs, cs_3_dofs]
        logger.info(cs_dofs)
        n2nd = cs.get_n_dofs_by_order(2)
        nall = cs.n_dofs

        logger.info("Fitting harmonic force constants separately")
        separate_fit = False # Change this back to true
        logger.info(f"disp_cut = {disp_cut}")

        sc = PhononBSDOSDoc.get_structure_container(
            cs, structures, separate_fit, disp_cut, ncut=n2nd, param2=None
        )
        opt = Optimizer(sc.get_fit_data(), fit_method, [0, n2nd], **fit_kwargs)
        try:
            opt.train()
        except Exception:
            logger.exception(f"Error occurred in opt.train() for cutoff: {i}, {cutoffs}")

            return {}
        param_harmonic = opt.parameters  # harmonic force constant parameters
        param_tmp = np.concatenate(
            (param_harmonic, np.zeros(cs.n_dofs - len(param_harmonic)))
        )

        parent_phonopy = get_phonopy_structure(parent_structure)
        phonopy = Phonopy(parent_phonopy, supercell_matrix=supercell_matrix)
        phonopy.primitive.get_number_of_atoms()

        # Ensure supercell_matrix is a numpy array
        supercell_matrix = np.array(supercell_matrix)
        mesh = supercell_matrix.diagonal() * 2
        mesh = [10, 10, 10] #TODO change this later

        fcp = ForceConstantPotential(cs, param_tmp)
        # supercell_atoms = phonopy_atoms_to_ase(phonopy.supercell)
        logger.info(f"supercell atoms = {supercell_atoms}")
        fcs = fcp.get_force_constants(supercell_atoms)
        logger.info("Did you get the large Condition number error?")

        phonopy.set_force_constants(fcs.get_fc_array(2))
        # phonopy.set_mesh(
        #     mesh, is_eigenvectors=False, is_mesh_symmetry=False
        # )  # run_mesh(is_gamma_center=True)
        phonopy.set_mesh(
            mesh, is_eigenvectors=True, is_mesh_symmetry=True
        )  # run_mesh(is_gamma_center=True)
        # phonopy.run_mesh(mesh, with_eigenvectors=False, is_mesh_symmetry=False)
        phonopy.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=True)
        omega = phonopy.mesh.frequencies  # THz
        omega = np.sort(omega.flatten())
        logger.info(f"omega_one_shot_fit = {omega}")
        imaginary = np.any(omega < -1e-3)
        logger.info(f"imaginary_one_shot_fit = {imaginary}")

        # # Phonopy's way of calculating phonon frequencies
        # structure_phonopy = get_phonopy_structure(parent_structure)
        # phonon = Phonopy(structure_phonopy, supercell_matrix=supercell_matrix)
        # phonon.set_force_constants(fcs.get_fc_array(2))
        # phonon.run_mesh(mesh=100.0, is_mesh_symmetry=False, is_gamma_center=True)
        # mesh = phonon.get_mesh_dict()
        # omega = mesh["frequencies"]
        # omega = np.sort(omega.flatten())
        # logger.info(f"omega_phonopy_one_shot_fitting = {omega}")
        # # imaginary = np.any(omega < -1e-3)
        # imaginary = np.any(omega < -imaginary_tol)
        # logger.info(f"imaginary_phonopy_one_shot_fitting = {imaginary}")
        n_imaginary = int(np.sum(omega < -np.abs(imaginary_tol)))

        if imaginary:
            logger.info(
                "Imaginary modes found! Fitting anharmonic force constants separately"
            )
            sc = PhononBSDOSDoc.get_structure_container(
                cs, structures, separate_fit, disp_cut, ncut=n2nd, param2=param_harmonic
            )
            opt = Optimizer(sc.get_fit_data(), fit_method, [n2nd, nall], **fit_kwargs)

            try:
                opt.train()
            except Exception:
                logger.exception(f"Error occured in opt.train() for cutoff: {i}, {cutoffs}")
                return {}
            param_anharmonic = opt.parameters  # anharmonic force constant parameters

            parameters = np.concatenate((param_harmonic, param_anharmonic))  # combine

            if nall != len(parameters):
                raise ValueError("nall is not equal to the length of parameters.")
            logger.info(f"Training complete for cutoff: {i}, {cutoffs}")


            parent_phonopy = get_phonopy_structure(parent_structure)
            phonopy = Phonopy(parent_phonopy, supercell_matrix=supercell_matrix)
            phonopy.primitive.get_number_of_atoms()
            mesh = supercell_matrix.diagonal() * 2
            mesh = [10, 10, 10] #TODO change this later

            fcp = ForceConstantPotential(cs, parameters)
            # supercell_atoms = phonopy_atoms_to_ase(phonopy.supercell)
            logger.info(f"supercell atoms = {supercell_atoms}")
            fcs = fcp.get_force_constants(supercell_atoms)
            logger.info("Did you get the large Condition number error?")

            phonopy.set_force_constants(fcs.get_fc_array(2))
            phonopy.set_mesh(
                mesh, is_eigenvectors=False, is_mesh_symmetry=False
            )  # run_mesh(is_gamma_center=True)
            phonopy.run_mesh(mesh=100.0, with_eigenvectors=False, is_mesh_symmetry=False)
            omega = phonopy.mesh.frequencies  # THz
            omega = np.sort(omega.flatten())
            logger.info(f"omega_seperate_fit = {omega}")
            imaginary = np.any(omega < -1e-3)
            logger.info(f"imaginary_seperate_fit = {imaginary}")

            # Phonopy's way of calculating phonon frequencies
            structure_phonopy = get_phonopy_structure(parent_structure)
            phonon = Phonopy(structure_phonopy, supercell_matrix=supercell_matrix)
            phonon.set_force_constants(fcs.get_fc_array(2))
            phonon.run_mesh(mesh, is_mesh_symmetry=False, is_gamma_center=True)
            mesh = phonon.get_mesh_dict()
            omega = mesh["frequencies"]
            omega = np.sort(omega.flatten())
            logger.info(f"omega_phonopy_seperate_fit = {omega}")
            # imaginary = np.any(omega < -1e-3)
            # logger.info(f"imaginary_phonopy_seperate_fit = {imaginary}")
            imaginary = np.any(omega < -imaginary_tol)
            logger.info(f"imaginary_phonopy_seperate_fit = {imaginary}")
            n_imaginary = int(np.sum(omega < -np.abs(imaginary_tol)))

        else:
            logger.info("No imaginary modes! Fitting all force constants in one shot")
            separate_fit = False
            sc = PhononBSDOSDoc.get_structure_container(
                cs, structures, separate_fit, disp_cut=None, ncut=None, param2=None
            )
            opt = Optimizer(sc.get_fit_data(), fit_method, [0, nall], **fit_kwargs)

            try:
                opt.train()
            except Exception:
                logger.exception(f"Error occured in opt.train() for cutoff: {i}, {cutoffs}")
                return {}
            parameters = opt.parameters
            logger.info(f"Training complete for cutoff: {i}, {cutoffs}")


        logger.info(f"parameters before enforcing sum rules {parameters}")
        logger.info(f"Memory use: {psutil.virtual_memory().percent} %")
        parameters = enforce_rotational_sum_rules(cs, parameters, ["Huang", "Born-Huang"])
        fcp = ForceConstantPotential(cs, parameters)
        # supercell_atoms = phonopy_atoms_to_ase(phonopy.supercell)
        fcs = fcp.get_force_constants(supercell_atoms)
        logger.info(f"FCS generated for cutoff {i}, {cutoffs}")

        return {
            "cutoffs": cutoffs,
            "rmse_test": opt.rmse_test,
            "cluster_space": sc.cluster_space,
            "parameters": parameters,
            "force_constants": fcs,
            "force_constants_potential": fcp,
            "imaginary": imaginary,
            "cs_dofs": cs_dofs,
            "n_imaginary": n_imaginary,
        }

    @staticmethod
    def get_structure_container(
        cs: ClusterSpace,
        structures: list[Atoms],
        separate_fit: bool,
        disp_cut: float,
        ncut: int,
        param2: np.ndarray,
    ) -> StructureContainer:
        """Get a hiPhive StructureContainer from cutoffs and a list of atoms objects.

        Args:
            cs: ClusterSpace
            structures: A list of ase atoms objects with the "forces" and
                "displacements" arrays included.
            separate_fit: Boolean to determine whether harmonic and anharmonic fitting
                are to be done separately (True) or in one shot (False)
            disp_cut: if separate_fit true, determines the mean displacement of perturbed
                structure to be included in harmonic (<) or anharmonic (>) fitting
            ncut: the parameter index where fitting separation occurs
            param2: previously fit parameter array (harmonic only for now, hence 2).

        Returns
        -------
            A hiPhive StructureContainer.
        """
        sc = StructureContainer(cs)
        logger.info(f"sc = {sc}")
        logger.info(f"initial shape of fit matrix = {sc.data_shape}")
        saved_structures = []
        for _, structure in enumerate(structures):
            displacements = structure.get_array("displacements")
            forces = structure.get_array("forces")
            # Calculate mean displacements
            mean_displacements = np.linalg.norm(displacements, axis=1).mean()
            logger.info(f"Mean displacements: {mean_displacements}")
            # Calculate mean forces
            mean_forces = np.linalg.norm(forces, axis=1).mean()
            logger.info(f"Mean Forces: {mean_forces}")
            # Calculate standard deviation of displacements
            std_displacements = np.linalg.norm(displacements, axis=1).std()
            logger.info(f"Standard deviation of displacements: "
                        f"{std_displacements}")
            # Calculate standard deviation of forces
            std_forces = np.linalg.norm(forces, axis=1).std()
            logger.info(f"Standard deviation of forces: "
                        f"{std_forces}")
            if not separate_fit:  # fit all
                sc.add_structure(structure)
            # for harmonic fitting
            elif separate_fit and param2 is None and mean_displacements < disp_cut:
                logger.info("We are in harmonic fitting if statement")
                # logger.info(f"mean_disp = {mean_displacements}")
                logger.info(f"mean_forces = {mean_forces}")
                sc.add_structure(structure)
            # for anharmonic fitting
            elif separate_fit and param2 is not None and mean_displacements >= disp_cut:
                logger.info("We are in anharmonic fitting if statement")
                # logger.info(f"mean_disp = {mean_displacements}")
                logger.info(f"mean_forces = {mean_forces}")
                sc.add_structure(structure)
                saved_structures.append(structure)

        logger.info("final shape of fit matrix (total # of atoms in all added"
                    f"supercells, n_dofs) = (rows, columns) = {sc.data_shape}")
        logger.info("We have completed adding structures")
        logger.info(f"sc.get_fit_data() = {sc.get_fit_data()}")

        if separate_fit and param2 is not None:  # do after anharmonic fitting
            a_mat = sc.get_fit_data()[0]  # displacement matrix
            f_vec = sc.get_fit_data()[1]  # force vector
            f_vec -= np.dot(a_mat[:, :ncut], param2)  # subtract harmonic forces
            sc.delete_all_structures()
            for i, structure in enumerate(saved_structures):
                natoms = structure.get_global_number_of_atoms()
                ndisp = natoms * 3
                structure.set_array(
                    "forces", f_vec[i * ndisp : (i + 1) * ndisp].reshape(natoms, 3)
                )
                sc.add_structure(structure)

        logger.debug(sc.__repr__())

        return sc

    @staticmethod
    def harmonic_properties(
        structure: Structure,
        supercell_matrix: np.ndarray,
        fcs: ForceConstants,
        temperature: list,
        imaginary_tol: float = 0.025,  # in THz
        mesh: list = None
    ) -> tuple[dict,Phonopy]:
        """
        Thermodynamic (harmonic) phonon props calculated using the force constants.

        Args:
            structure: The parent structure.
            supercell_matrix: The supercell transformation matrix.
            force_constants: The force constants in numpy format.
            imaginary_tol: Tolerance used to decide if a phonon mode is imaginary,
                in THz.

        Returns
        -------
            A tuple of the number of imaginary modes at Gamma, the minimum phonon
            frequency at Gamma, and the free energy, entropy, and heat capacity
        """
        logger.info("Evaluating harmonic properties...")
        fcs2 = fcs.get_fc_array(2)
        logger.info("fcs2 & fcs3 read...")
        logger.info(f"fcs2 = {fcs2}")
        parent_phonopy = get_phonopy_structure(structure)
        phonopy = Phonopy(parent_phonopy, supercell_matrix=supercell_matrix)
        natom = phonopy.primitive.get_number_of_atoms()
        # Ensure supercell_matrix is a numpy array
        supercell_matrix = np.array(supercell_matrix)
        mesh = supercell_matrix.diagonal()*2
        mesh = [10, 10, 10] #TODO change this later
        logger.info(f"Mesh: {mesh}")

        phonopy.set_force_constants(fcs2)
        phonopy.set_mesh(mesh,is_eigenvectors=True,is_mesh_symmetry=False)
        phonopy.run_thermal_properties(temperatures=temperature)
        logger.info("Thermal properties successfully run!")

        _, free_energy, entropy, heat_capacity = phonopy.get_thermal_properties()

        ## Use the following lines to convert the units to eV/atom
        # free_energy *= 1000/sp.constants.Avogadro/eV2J/natom # kJ/mol to eV/atom
        # entropy *= 1/sp.constants.Avogadro/eV2J/natom # J/K/mol to eV/K/atom
        # heat_capacity *= 1/sp.constants.Avogadro/ev2j/natom # J/K/mol to eV/K/atom

        freq = phonopy.mesh.frequencies # in THz
        logger.info(f"Frequencies: {freq}")
        logger.info(f"freq_flatten = {np.sort(freq.flatten())}")

        n_imaginary = int(np.sum(freq < -np.abs(imaginary_tol)))

        if n_imaginary == 0:
            logger.info("No imaginary modes!")
        else:
            logger.warning("Imaginary modes found!")

        ### Added from Phonon workflow
        logger.info("starting the thermal prop calculation as per the phonon workflow")
        from atomate2.common.schemas.phonons import PhononBSDOSDoc
        from pymatgen.io.phonopy import (
            get_ph_bs_symm_line,
            get_ph_dos,
            get_pmg_structure,
        )
        kpath_scheme = "seekpath"
        symprec = 1e-4
        from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
        from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
        from pymatgen.io.vasp import Kpoints

        # with phonopy.load("phonopy.yaml") the phonopy API can be used
        phonopy.save("phonopy.yaml")

        # get phonon band structure
        kpath_dict, kpath_concrete = PhononBSDOSDoc.get_kpath(
            structure=get_pmg_structure(phonopy.primitive),
            kpath_scheme=kpath_scheme,
            symprec=symprec,
        )

        npoints_band = 101
        qpoints, connections = get_band_qpoints_and_path_connections(
            kpath_concrete, npoints=101 # changed from 101
        )

        # phonon band structures will always be computed
        filename_band_yaml = "phonon_band_structure.yaml"

        # TODO: potentially add kwargs to avoid computation of eigenvectors
        phonopy.run_band_structure(
            qpoints,
            path_connections=connections,
            with_eigenvectors=False,
            is_band_connection=False,
        )
        phonopy.write_yaml_band_structure(filename=filename_band_yaml)
        bs_symm_line = get_ph_bs_symm_line(
            filename_band_yaml, labels_dict=kpath_dict, has_nac=False
        )
        new_plotter = PhononBSPlotter(bs=bs_symm_line)
        new_plotter.save_plot(
            filename="phonon_band_structure.pdf",
            units="THz",
        )

        # will determine if imaginary modes are present in the structure
        imaginary_modes = bs_symm_line.has_imaginary_freq(
            tol=1e-5
        )

        # # gets data for visualization on website - yaml is also enough
        # if kwargs.get("band_structure_eigenvectors"):
        #     bs_symm_line.write_phononwebsite("phonon_website.json")

        # get phonon density of states
        filename_dos_yaml = "phonon_dos.yaml"

        kpoint_density_dos = 7000
        kpoint = Kpoints.automatic_density(
            structure=get_pmg_structure(phonopy.primitive),
            kppa=kpoint_density_dos,
            force_gamma=True,
        )
        phonopy.run_mesh(kpoint.kpts[0])
        phonopy.run_total_dos()
        phonopy.write_total_dos(filename=filename_dos_yaml)
        dos = get_ph_dos(filename_dos_yaml)
        new_plotter_dos = PhononDosPlotter()
        new_plotter_dos.add_dos(label="total", dos=dos)
        new_plotter_dos.save_plot(
            filename="phonon_dos.pdf",
            units="THz",
        )

        # compute vibrational part of free energies per formula unit
        temperature_range = np.arange(
            0, 1000, 10
        )

        free_energies = [
            dos.helmholtz_free_energy(
                temp=temp, structure=get_pmg_structure(phonopy.primitive)
            )
            for temp in temperature_range
        ]

        entropies = [
            dos.entropy(temp=temp, structure=get_pmg_structure(phonopy.primitive))
            for temp in temperature_range
        ]

        internal_energies = [
            dos.internal_energy(
                temp=temp, structure=get_pmg_structure(phonopy.primitive)
            )
            for temp in temperature_range
        ]

        heat_capacities = [
            dos.cv(temp=temp, structure=get_pmg_structure(phonopy.primitive))
            for temp in temperature_range
        ]

        # save a json file of temperature_range, free_energies, entropies, internal_energies, heat_capacities
        # Organize data into a dictionary
        data = {
            "temperature_range": temperature_range.tolist(),  # Convert to list for JSON serialization
            "free_energies": free_energies,
            "entropies": entropies,
            "internal_energies": internal_energies,
            "heat_capacities": heat_capacities
        }

        # Define the output file path
        output_file_path = "thermodynamic_properties.json"

        # Write the dictionary to a JSON file
        with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=4)  # `indent=4` for pretty printing
        
        logger.info("ending the thermal prop calculation as per the phonon workflow")

        ### Added from Phonon workflow

        return {
            "temperature": temperature,
            "free_energy": free_energy,
            "entropy": entropy,
            "heat_capacity": heat_capacity,
            "n_imaginary": n_imaginary
            }, phonopy

    @staticmethod
    def anharmonic_properties(
        phonopy: Phonopy,
        fcs: ForceConstants,
        temperature: list,
        heat_capacity: np.ndarray,
        n_imaginary: float,
        bulk_modulus: float = None
    ) -> dict:

        if n_imaginary == 0:
            logger.info("Evaluating anharmonic properties...")
            fcs2 = fcs.get_fc_array(2)
            fcs3 = fcs.get_fc_array(3)
            grun, cte, dlfrac = PhononBSDOSDoc.gruneisen(phonopy,fcs2,fcs3,temperature,heat_capacity,
                                        bulk_modulus=bulk_modulus)
        else: # do not calculate these if imaginary modes exist
            logger.warning("Gruneisen and thermal expansion cannot be calculated with"
                        "imaginary modes. All set to 0.")
            grun = np.zeros((len(temperature),3))
            cte = np.zeros((len(temperature),3))
            dlfrac = np.zeros((len(temperature),3))

        return {
            "gruneisen": grun,
            "thermal_expansion": cte,
            "expansion_fraction": dlfrac,
            }

    @staticmethod
    def get_total_grun(
            omega: np.ndarray,
            grun: np.ndarray,
            kweight: np.ndarray,
            t: float
    ) -> np.ndarray:
        total = 0
        weight = 0
        nptk = omega.shape[0]
        nbands = omega.shape[1]
        omega = abs(omega)*1e12*2*np.pi
        if t==0:
            total = np.zeros((3,3))
            grun_total_diag = np.zeros(3)
        else:
            for i in range(nptk):
                for j in range(nbands):
                    x = hbar*omega[i,j]/(2.0*kb*t)
                    dbe = (x/np.sinh(x))**2
                    weight += dbe*kweight[i]
                    total += dbe*kweight[i]*grun[i,j]
            total = total/weight
            grun_total_diag = np.array([total[0,2],total[1,1],total[2,0]])

            def percent_diff(a: float, b: float) -> float:
                return abs((a-b)/b)
            # This process preserves cell symmetry upon thermal expansion, i.e., it prevents
            # symmetry-identical directions from inadvertently expanding by different ratios
            # when/if the Gruneisen routine returns slightly different ratios for those
            # directions
            avg012 = np.mean((grun_total_diag[0],grun_total_diag[1],grun_total_diag[2]))
            avg01 = np.mean((grun_total_diag[0],grun_total_diag[1]))
            avg02 = np.mean((grun_total_diag[0],grun_total_diag[2]))
            avg12 = np.mean((grun_total_diag[1],grun_total_diag[2]))
            if percent_diff(grun_total_diag[0],avg012) < 0.1:
                if percent_diff(grun_total_diag[1],avg012) < 0.1:
                    if percent_diff(grun_total_diag[2],avg012) < 0.1: # all similar
                        grun_total_diag[0] = avg012
                        grun_total_diag[1] = avg012
                        grun_total_diag[2] = avg012
                    elif percent_diff(grun_total_diag[2],avg02) < 0.1: # 0 and 2 similar
                        grun_total_diag[0] = avg02
                        grun_total_diag[2] = avg02
                    elif percent_diff(grun_total_diag[2],avg12) < 0.1: # 1 and 2 similar
                        grun_total_diag[1] = avg12
                        grun_total_diag[2] = avg12
                    else:
                        pass
                elif percent_diff(grun_total_diag[1],avg01) < 0.1: # 0 and 1 similar
                    grun_total_diag[0] = avg01
                    grun_total_diag[1] = avg01
                elif percent_diff(grun_total_diag[1],avg12) < 0.1: # 1 and 2 similar
                    grun_total_diag[1] = avg12
                    grun_total_diag[2] = avg12
                else:
                    pass
            elif percent_diff(grun_total_diag[0],avg01) < 0.1: # 0 and 1 similar
                grun_total_diag[0] = avg01
                grun_total_diag[1] = avg01
            elif percent_diff(grun_total_diag[0],avg02) < 0.1: # 0 and 2 similar
                grun_total_diag[0] = avg02
                grun_total_diag[2] = avg02
            else: # nothing similar
                pass

        return grun_total_diag

    @staticmethod
    def gruneisen(
            phonopy: Phonopy,
            fcs2: np.ndarray,
            fcs3: np.ndarray,
            temperature: list,
            heat_capacity: np.ndarray, # in J/K/mol
            bulk_modulus: float = None # in GPa
    ) -> tuple[list,list]:

        gruneisen = Gruneisen(fcs2,fcs3,phonopy.supercell,phonopy.primitive)
        gruneisen.set_sampling_mesh(phonopy.mesh_numbers,is_gamma_center=True)
        gruneisen.run()
        grun = gruneisen.get_gruneisen_parameters() # (nptk,nmode,3,3)
        omega = gruneisen._frequencies  # noqa: SLF001
        # qp = gruneisen._qpoints
        kweight = gruneisen._weights  # noqa: SLF001

        grun_tot = [PhononBSDOSDoc.get_total_grun(omega, grun, kweight, temp) for temp in temperature]
        grun_tot = np.nan_to_num(np.array(grun_tot))

        # linear thermal expansion coefficeint and fraction
        if bulk_modulus is None:
            cte = None
            dlfrac = None
        else:
            # heat_capacity *= eV2J*phonopy.primitive.get_number_of_atoms()
            # # eV/K/atom to J/K
            heat_capacity *= 1/sp.constants.Avogadro # J/K/mol to J/K
            # to convert from J/K/atom multiply by phonopy.primitive.get_number_of_atoms()
            # Convert heat_capacity to an array if it's a scalar
            # heat_capacity = np.array([heat_capacity])
            logger.info(f"heat capacity = {heat_capacity}")
            vol = phonopy.primitive.get_volume()

            logger.info(f"grun_tot: {grun_tot}")
            logger.info(f"grun_tot shape: {grun_tot.shape}")
            logger.info(f"heat_capacity shape: {heat_capacity.shape}")
            logger.info(f"heat_capacity: {heat_capacity}")
            logger.info(f"vol: {vol}")
            logger.info(f"bulk_modulus: {bulk_modulus}")
    #        cte = grun_tot*heat_capacity.repeat(3)/(vol/10**30)/(bulk_modulus*10**9)/3
            cte = grun_tot*heat_capacity.repeat(3).reshape(
                len(heat_capacity),3)/(vol/10**30)/(bulk_modulus*10**9)/3
            cte = np.nan_to_num(cte)
            if len(temperature)==1:
                dlfrac = cte*temperature
            else:
                dlfrac = PhononBSDOSDoc.thermal_expansion(temperature, cte)
            logger.info(f"Gruneisen: \n {grun_tot}")
            logger.info(f"Coefficient of Thermal Expansion: \n {cte}")
            logger.info(f"Linear Expansion Fraction: \n {dlfrac}")

        return grun_tot, cte, dlfrac

    @staticmethod
    def thermal_expansion(
            temperature: list,
            cte: np.array,
    ) -> np.ndarray:
        if len(temperature) != len(cte):
            raise ValueError("Length of temperature and cte lists must be equal.")
        if 0 not in temperature:
            temperature = [0, *temperature]
            cte = np.array([np.array([0, 0, 0]), *list(cte)])
        temperature = np.array(temperature)
        ind = np.argsort(temperature)
        temperature = temperature[ind]
        cte = np.array(cte)[ind]
        # linear expansion fraction
        dlfrac = copy.copy(cte)
        for t in range(len(temperature)):
            dlfrac[t,:] = np.trapz(cte[:t+1,:],temperature[:t+1],axis=0)
        dlfrac = np.nan_to_num(dlfrac)
        logger.info(f"Linear Expansion Fraction: \n {dlfrac}")
        return dlfrac