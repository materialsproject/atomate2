"""Common Flow for calculating harmonic & anharmonic props of phonon."""

# Basic Python packages
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np

# Jobflow packages
from jobflow import Flow, Maker

from atomate2.common.jobs.hiphive import (
    hiphive_static_calcs,
    # quality_control,
    run_fc_to_pdos,
    run_hiphive,
    run_hiphive_individually,
    run_hiphive_renormalization,
    run_lattice_thermal_conductivity,
)
from atomate2.forcefields.jobs import (
    CHGNetStaticMaker,
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
)

# Atomate2 packages
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

    from atomate2.vasp.flows.core import DoubleRelaxMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker

logger = logging.getLogger(__name__)

__all__ = ["BaseHiphiveMaker"]

__author__ = "Alex Ganose, Junsoo Park, Zhuoying Zhu, Hrushikesh Sahasrabuddhe"
__email__ = "aganose@lbl.gov, jsyony37@lbl.gov, zyzhu@lbl.gov, hpsahasrabuddhe@lbl.gov"


@dataclass
class BaseHiphiveMaker(Maker, ABC):
    """
    Workflow to calc. interatomic force constants and vibrational props using hiPhive.

    A summary of the workflow is as follows:
    1. Structure relaxtion
    2. Calculate a supercell transformation matrix that brings the
       structure as close as cubic as possible, with all lattice lengths
       greater than 5 nearest neighbor distances. Then, perturb the atomic sites
       for each supercell using a Fixed displacement rattle procedure. The atoms are
       perturbed roughly according to a normal deviation around the average value.
       A number of standard deviation perturbation distances are included. Multiple
       supercells may be generated for each perturbation distance. Then, run static
       VASP calculations on each perturbed supercell to calculate atomic forces.
       Then, aggregate the forces and the perturbed structures.
    3. Conduct the fit atomic force constants using the regression schemes in hiPhive.
    4. Perform phonon renormalization at finite temperature - useful when unstable
       modes exist
    5. Output the interatomic force constants, and phonon band structure and density of
       states to the database
    6. Solve the lattice thermal conductivity using ShengBTE and output to the database.

    Args
    ----------
    name : str
        Name of the flows produced by this maker.
    bulk_relax_maker (BaseVaspMaker | None):
        The VASP input generator for bulk relaxation,
        default is DoubleRelaxMaker using TightRelaxMaker.
    phonon_displacement_maker (BaseVaspMaker | None):
        The VASP input generator for phonon displacement calculations,
        default is PhononDisplacementMaker.
    ff_displacement_maker (BaseVaspMaker | None):
        The force field displacement maker, default is CHGNetStaticMaker.
    min_length (float):
        Minimum length of supercell lattice vectors in Angstroms, default is 13.0.
    prefer_90_degrees (bool):
        Whether to prefer 90 degree angles in supercell matrix,
        default is True.
    supercell_matrix_kwargs (dict):
        Keyword arguments for supercell matrix calculation, default is {}.
    IMAGINARY_TOL (float):
        Imaginary frequency tolerance in THz, default is 0.025.
    MESH_DENSITY (float):
        Mesh density for phonon calculations, default is 100.0.
    T_QHA (list):
        Temperatures for phonopy thermodynamic calculations,
        default is [0, 100, 200, ..., 2000].
    T_RENORM (list):
        Temperatures for renormalization calculations, default is [1500].
    T_KLAT (int):
        Temperature for lattice thermal conductivity calculation, default is 300.
    FIT_METHOD (str):
        Method for fitting force constants, default is "rfe".
    RENORM_METHOD (str):
        Method for renormalization, default is 'pseudoinverse'.
    RENORM_NCONFIG (int):
        Number of configurations for renormalization, default is 5.
    RENORM_CONV_THRESH (float):
        Convergence threshold for renormalization in meV/atom, default is 0.1.
    RENORM_MAX_ITER (int):
        Maximum iterations for renormalization, default is 30.
    THERM_COND_SOLVER (str):
        Solver for lattice thermal conductivity, default is "almabte". Other options
        include "shengbte" and "phono3py".
    """

    name: str = "Lattice-Dynamics"
    bulk_relax_maker: DoubleRelaxMaker | ForceFieldRelaxMaker | None = None
    phonon_displacement_maker: BaseVaspMaker | ForceFieldStaticMaker | None = field(
        default_factory=lambda:PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(auto_lreal=True)
        )
    )
    ff_displacement_maker: ForceFieldStaticMaker | None = field(
        default_factory=CHGNetStaticMaker
    )
    min_length: float | None = 13.0
    prefer_90_degrees: bool = True
    supercell_matrix_kwargs: dict = field(default_factory=dict)
    IMAGINARY_TOL = 0.025  # in THz
    MESH_DENSITY = 100.0  # should always be a float
    T_QHA: ClassVar[list[int]] = [
        i * 100 for i in range(21)
    ]  # Temp. for phonopy calc. of thermo. properties (free energy etc.)
    T_RENORM: ClassVar[list[int]] = [
        300 # 300, 500, 600, 700, 800, 900, 1000, 1500, 2500, 2700, 3000
    ]  # [i*100 for i in range(0,16)] # Temp. at which renorm. is to be performed
    # If renormalization is performed,
    # T_RENORM overrides T_KLAT for lattice thermal conductivity
    T_KLAT: ClassVar[dict] = {"min":100,"max":1000,"step":100} #[i*100 for i in range(0,11)]
    FIT_METHOD = "omp" #least-squares #omp #rfe #elasticnet
    RENORM_METHOD = "least_squares" # pseudoinverse
    RENORM_NCONFIG = 5  # Changed from 50
    RENORM_CONV_THRESH = 0.1  # meV/atom
    RENORM_MAX_ITER = 30  # Changed from 20
    THERM_COND_SOLVER: str = "almabte"

    def make(
        self,
        mpid: str,
        structure: Structure,
        bulk_modulus: float,
        supercell_matrix: Matrix3D | None = None,
        fit_method: str | None = FIT_METHOD,
        disp_cut: float | None = None,
        cutoffs: list[list[float]] | None = None,
        prev_dir: str | Path | None = None,
        calculate_lattice_thermal_conductivity: bool = True,
        renormalize: bool = True,
        renormalize_temperature: list = T_RENORM,
        renormalize_method: str = RENORM_METHOD,
        renormalize_nconfig: int = RENORM_NCONFIG,
        renormalize_conv_thresh: float = RENORM_CONV_THRESH,
        renormalize_max_iter: int = RENORM_MAX_ITER,
        renormalize_thermal_expansion_iter: bool = False,
        mesh_density: float = MESH_DENSITY,
        thermal_conductivity_temperature: list = T_KLAT,
        imaginary_tol: float = IMAGINARY_TOL,
        temperature_qha: float | list | dict = T_QHA,
        n_structures: float = 1,
        fixed_displs: float | None = None,
    ) -> Flow:
        """
        Make flow to calculate the harmonic & anharmonic properties of phonon.

        Parameters
        ----------
        mpid (str):
            The Materials Project ID (MPID) of the material.
        structure (Structure):
            The A pymatgen structure of the material.
        bulk_modulus (float):
            Bulk modulus of the material in GPa.
        supercell_matrix (Matrix3D, optional):
            Supercell transformation matrix, default is None.
        fit_method (str, optional):
            Method for fitting force constants using hiphive, default is None.
        disp_cut (float, optional):
            Cutoff distance for displacements in Angstroms, default is None.
        cutoffs (List[List[float]], optional):
            List of cutoff distances for different force constants fitting,
            default is None.
        prev_dir (str | Path | None, optional):
            Previous RELAX calculation directory to use for copying outputs.,
            default is None.
        calculate_lattice_thermal_conductivity (bool, optional):
            Calculate lattice thermal conductivity, default is True.
        renormalize (bool, optional):
            Perform renormalization, default is False.
        renormalize_temperature (float | List | Dict, optional):
            Temperatures for renormalization, default is T_RENORM.
        renormalize_method (str, optional):
            Method for renormalization, default is RENORM_METHOD.
        renormalize_nconfig (int, optional):
            Number of configurations for renormalization, default is RENORM_NCONFIG.
        renormalize_conv_thresh (float, optional):
            Convergence threshold for renormalization in meV/atom,
            default is RENORM_CONV_THRESH.
        renormalize_max_iter (int, optional):
            Maximum iterations for renormalization, default is RENORM_MAX_ITER.
        renormalize_thermal_expansion_iter (bool, optional):
            Include thermal expansion during renormalization iterations,
            default is False.
        mesh_density (float, optional):
            Mesh density for phonon calculations, default is MESH_DENSITY.
        thermal_conductivity_temperature (float | List | Dict, optional):
            Temperatures for thermal conductivity calculations, default is T_KLAT.
        imaginary_tol (float, optional):
            Imaginary frequency tolerance in THz, default is IMAGINARY_TOL.
        temperature_qha (float, optional):
            Temperatures for phonopy thermodynamic calculations, default is T_QHA.
        n_structures (float, optional):
            Number of structures to consider for calculations, default is None.
        fixed_displs (float, optional):
            Avg value of atomic displacement in Angstroms, default is None.
        """
        jobs = []
        outputs = []
        loops = 1 # revert back to 1

        # 1. Relax the structure
        if self.bulk_relax_maker is not None:
            bulk_kwargs = {}
            if self.prev_calc_dir_argname is not None:
                bulk_kwargs[self.prev_calc_dir_argname] = prev_dir
            bulk = self.bulk_relax_maker.make(structure, **bulk_kwargs)
            bulk.update_config({"manager_config": {"_fworker": "gpu_fworker"}})
            jobs.append(bulk)
            outputs.append(bulk.output)
            structure = bulk.output.structure
            prev_dir = bulk.output.dir_name

        bulk.update_metadata(
            {
                "tag": [
                    f"mp_id={mpid}",
                    f"bulk_modulus={bulk_modulus}",
                    f"relax_{loops}",
                    f"nConfigsPerStd={n_structures}",
                    f"fixedDispls={fixed_displs}",
                    f"dispCut={disp_cut}",
                    f"supercell_matrix_kwargs={self.supercell_matrix_kwargs}",
                    f"loop={loops}",
                ]
            }
        )

        # # 2. if supercell_matrix is None, supercell size will be determined after relax
        # # maker to ensure that cell lengths are really larger than threshold.
        # # then, perturbations will be generated based on the supercell size.
        # # STATIC calculations will then be run on the perturbed structures, and the
        # # forces and perturbed structures will be aggregated.
        # from pymatgen.core.structure import IStructure
        # file_path = "/pscratch/sd/h/hrushi99/atomate2/MgO_Zhuoying_LAC_NRC/block_2024-03-30-01-25-24-547097/launcher_2024-03-31-20-40-35-997684/launcher_2024-03-31-20-43-19-730610/CONTCAR"
        # prev_dir = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/bulk_relax_2_4622_96_VASP/launcher_2024-03-31-20-43-19-730610/CONTCAR"
        # structure = IStructure.from_file(file_path)
        # prev_dir = "/pscratch/sd/h/hrushi99/atomate2/MgO_Zhuoying_LAC_NRC/block_2024-03-30-01-25-24-547097/launcher_2024-03-31-20-40-35-997684/launcher_2024-03-31-20-43-19-730610"
        # prev_dir = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/tight_relax2_4622_VASP_6_phonopy_exact/launcher_2024-04-14-02-09-48-767506"
        # prev_dir = "/pscratch/sd/h/hrushi99/atomate2/MgO_Zhuoying_LAC_NRC/block_2024-04-10-16-08-15-920932/launcher_2024-04-14-01-38-32-712261/launcher_2024-04-14-02-09-48-767506"
        # prev_dir = "/pscratch/sd/h/hrushi99/atomate2/MgO_Zhuoying_LAC_NRC/block_2024-04-10-16-08-15-920932/launcher_2024-05-01-23-21-05-447819/launcher_2024-05-02-02-33-26-940509"
        # prev_dir = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/tight_relax_2_552098"
        static_calcs = hiphive_static_calcs(
                structure=structure,
                supercell_matrix=supercell_matrix,
                min_length=self.min_length,
                prefer_90_degrees=self.prefer_90_degrees,
                n_structures=n_structures,
                # fixed_displs=fixed_displs,
                loops=loops,
                prev_dir=prev_dir,
                phonon_displacement_maker=self.phonon_displacement_maker,
                ff_displacement_maker=self.ff_displacement_maker,
                supercell_matrix_kwargs=self.supercell_matrix_kwargs,
                bulk_modulus=bulk_modulus,
                mpid=mpid
        )
        jobs.append(static_calcs)

        # # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_6350"
        # # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_disp_sampling_updated/launcher_2024-04-07-12-23-44-934527"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96/launcher_2024-04-01-08-09-15-480067"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96_fix_1/launcher_2024-04-01-08-09-15-480067"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96_fix_2/launcher_2024-04-01-08-09-15-480067"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96_fix_3/launcher_2024-04-01-08-09-15-480067"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96_fix_4/launcher_2024-04-01-08-09-15-480067"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96_fix_5/launcher_2024-04-01-08-09-15-480067"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_50_disp_sampling_updated_only_harmonic/launcher_2024-04-08-10-11-41-526328"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96_fix_3/launcher_2024-04-01-08-09-15-480067"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96_fix_6/launcher_2024-04-01-08-09-15-480067"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96_fix_7/launcher_2024-04-01-08-09-15-480067"
        # # prev_dir_json_saver = "/pscratch/sd/h/hrushi99/atomate2/hiphive_4622_VASP_96_fix_3/launcher_2024-04-01-08-09-15-480067"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/Rohith_Mo2Zr_6715_VASP/launcher_2024-04-10-03-20-44-201391/launcher_2024-04-10-03-24-30-451833"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_5_less_than_0,01/launcher_2024-04-11-11-54-49-707881"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96_fix_8/launcher_2024-04-01-08-09-15-480067"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96_fix_9/launcher_2024-04-01-08-09-15-480067"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_6_phonopy_INCAR/launcher_2024-04-13-21-57-43-344701"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96_fix_10/launcher_2024-04-01-08-09-15-480067"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_6_phonopy_exact_INCAR/launcher_2024-04-14-19-15-11-794304"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_12_phonopy_exact_INCAR/launcher_2024-04-15-17-38-04-444124"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_41_VASP/launcher_2024-04-05-15-03-16-098315"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_1565_VASP"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_131_VASP/launcher_2024-04-06-01-48-46-676261"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_1265_VASP/launcher_2024-03-23-20-19-18-215601"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_2605_VASP"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-02-18-15-12-605420-37451"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_552098/launcher_2024-05-03-14-29-46-370557"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-04-05-19-20-297426-32094"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_552098_combined_8_displs/launcher_2024-05-03-14-29-46-370557" # incorrect because the two sets of unperturbed supercells are not correct
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_552098_more_displs/launcher_2024-05-04-21-20-42-358408"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/mp_552098_24_supercell_2_displs"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_10200/launcher_2024-05-05-01-42-41-754182"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_552098_24_supercell/launcher_2024-05-05-01-56-31-721610"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_552098_config1&2/launcher_2024-05-05-06-26-45-539326"
        # # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_552098_mc_rattle_2configs_per_displ/launcher_2024-05-05-09-13-06-826623"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_1479/launcher_2024-05-07-09-30-15-846278"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_1078250/launcher_2024-05-08-09-12-40-347220"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_30530_fixed_displ/5ConfigsPerDisp/launcher_2024-05-17-17-15-12-950067"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_30530_fixed_displ/4ConfigsPerDisp/launcher_2024-05-17-17-15-12-950067"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_30530_fixed_displ/3ConfigsPerDisp/launcher_2024-05-17-17-15-12-950067"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_30530_fixed_displ/2ConfigsPerDisp/launcher_2024-05-17-17-15-12-950067"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_23339_fixed_displ/1ConfigsPerDisp/launcher_2024-05-18-04-26-03-314022"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_23339_mc_rattle/0,002_std_dev/modified_structure_file"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_23339_mc_rattle_max_disp_0,01/launcher_2024-05-19-09-01-41-194193"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-21-02-41-58-958977-44099"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-21-15-43-36-117945-61868"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_23339_new_displ_sampling_fixed_displ/launcher_2024-05-22-08-12-18-202746"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-22-17-04-06-407141-16525"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_23339_new_displ_sampling_fixed_displ_V2/launcher_2024-05-22-08-03-19-338222"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-22-21-14-36-631909-87559"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-22-21-37-04-484228-12098"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-22-23-49-02-538291-81754"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-23-02-43-21-051328-27029"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-23-03-11-21-132926-87658"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-23-03-41-28-678925-80078"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_23339_fixed_displ_fixed_force_distribution/launcher_2024-05-23-05-42-33-120701"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_23339_fixed_displ_fixed_force_distribution/prefinal_chgnet/launcher_2024-05-23-05-37-57-125152"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_4622_fixed_displ_fixed_force_distribution/launcher_2024-05-23-07-50-44-015298"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_4622_fixed_displ_fixed_force_distribution/prefinal_chgnet"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-23-20-13-18-146041-72144"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_1479_fixed_displ_fixed_force_distribution/launcher_2024-05-23-18-22-12-087175"
        prev_dir_json_saver = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-28-22-29-33-614395-68155"
        # prev_dir_json_saver = "/Users/HPSahasrabuddhe/Downloads/supporting_data/runtime_costs/GaP/3_pre_fc"
        # 3. Hiphive Fitting of FCPs upto 4th order
        if n_structures >= 10:
            fit_force_constant = run_hiphive_individually(
                mpid = mpid,
                cutoffs = cutoffs,
                fit_method = fit_method,
                disp_cut = disp_cut,
                bulk_modulus = bulk_modulus,
                temperature_qha = temperature_qha,
                imaginary_tol = imaginary_tol,
                prev_dir_json_saver = static_calcs.output["current_dir"],
                # prev_dir_json_saver = prev_dir_json_saver,
                loop = loops,
            )
        else:
            fit_force_constant = run_hiphive(
                fit_method=fit_method,
                disp_cut=disp_cut,
                bulk_modulus=bulk_modulus,
                temperature_qha=temperature_qha,
                # mesh_density=mesh_density,
                imaginary_tol=imaginary_tol,
                prev_dir_json_saver=static_calcs.output["current_dir"],
                # prev_dir_json_saver=prev_dir_json_saver,
                loop=loops,
                cutoffs=cutoffs
            )
        fit_force_constant.name += f" {loops}"
        fit_force_constant.update_config({"manager_config": {"_fworker": "cpu_reg_fworker"}})
        jobs.append(fit_force_constant)
        outputs.append(fit_force_constant.output)
        fit_force_constant.metadata.update(
            {
                "tag": [
                    f"mp_id={mpid}",
                    f"bulk_modulus={bulk_modulus}",
                    f"cutoffs={cutoffs}",
                    f"fit_force_constant_{loops}",
                    f"nConfigsPerStd={n_structures}",
                    f"fixedDispls={fixed_displs}",
                    f"dispCut={disp_cut}",
                    f"supercell_matrix={supercell_matrix}",
                    f"loop={loops}",
                ]
            }
        )

        # TODO: Implement Quality Control Job
        # # # # # 7. Quality Control Job to check if the desired Test RMSE is achieved,
        # # # # # if not, then increase the number of structures --
        # # # # # Using "addintion" feature of jobflow
        # # # # loops += 1
        # # # # n_structures += 1
        # # # # logger.info(f"Number of structures increased to {n_structures}")
        # # # # logger.info(f"loop = {loops}")
        # # # # error_check_job = quality_control(
        # # # #     rmse_test=fit_force_constant.output[5],
        # # # #     n_structures=n_structures,
        # # # #     fixedDispls=fixed_displs,
        # # # #     loop=loops,
        # # # #     fit_method=fit_method,
        # # # #     disp_cut=disp_cut,
        # # # #     bulk_modulus=bulk_modulus,
        # # # #     temperature_qha=temperature_qha,
        # # # #     mesh_density=mesh_density,
        # # # #     imaginary_tol=imaginary_tol,
        # # # #     prev_dir_json_saver=static_calcs.output["current_dir"],
        # # # #     prev_dir=prev_dir,
        # # # #     supercell_matrix=supercell_matrix,
        # # # #     # supercell_matrix_kwargs=supercell_matrix_kwargs,
        # # # # )
        # # # # error_check_job.name += f" {loops}"
        # # # # jobs.append(error_check_job)
        # # # # outputs.append(error_check_job.output)
        # # # # error_check_job.metadata.update(
        # # # #     {
        # # # #         "tag": [
        # # # #             f"error_check_job_{loops}",
        # # # #             f"nConfigsPerStd={n_structures}",
        # # # #             f"fixedDispls={fixed_displs}",
        # # # #             f"dispCut={disp_cut}",
        # # # #             # f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
        # # # #             f"supercell_matrix={supercell_matrix}",
        # # # #             f"loop={loops}",
        # # # #         ]
        # # # #     }
        # # # # )

        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_1479_displ_sampling"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-07-20-59-31-454856-98976" # fix 3
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-08-05-05-27-911205-37950" # fix 5
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-10-19-33-51-707117-99373"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-10-22-08-52-739207-65556"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-16-03-39-27-546384-72231"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-16-05-08-16-334659-40420"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-16-20-30-11-696345-68541"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-16-20-19-30-722449-33890"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-16-20-19-30-722449-33890"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-02-18-45-24-612532-70205"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-03-15-43-37-821524-48496"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/hiphive_paper_review/hiphive_23339_fixed_displ/1ConfigsPerDisp/10_5_3,6/job_2024-05-18-07-29-03-325724-14004"
        # 4. Perform phonon renormalization to obtain temperature-dependent
        # force constants using hiPhive
        outputs_renorm = []
        if renormalize:
            for temperature in renormalize_temperature:
                nconfig = renormalize_nconfig * (1 + temperature // 100)
                renormalization = run_hiphive_renormalization(
                    temperature=temperature,
                    renorm_method=renormalize_method,
                    nconfig=nconfig,
                    renorm_TE_iter=renormalize_thermal_expansion_iter,
                    bulk_modulus=bulk_modulus,
                    prev_dir_hiphive=fit_force_constant.output["current_dir"],
                    # prev_dir_hiphive=prev_dir_hiphive,
                    loop=loops,
                )
                renormalization.name += f" {temperature} {loops}"
                renormalization.update_config({"manager_config": {"_fworker": "cpu_reg_fworker"}})
                jobs.append(renormalization)
                outputs_renorm.append(renormalization.output)
                outputs.append(renormalization.output)
                renormalization.metadata.update(
                    {
                        "tag": [
                            f"mp_id={mpid}",
                            f"bulk_modulus={bulk_modulus}",
                            f"run_renormalization_{loops}",
                            f"nConfigsPerStd={n_structures}",
                            f"fixedDispls={fixed_displs}",
                            f"dispCut={disp_cut}",
                            f"supercell_matrix={supercell_matrix}",
                            f"loop={loops}",
                        ]
                    }
                )

        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-06-04-29-02-534935-41708"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-07-16-14-51-625183-73398"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-07-18-38-06-547834-98127"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-07-19-14-17-420858-98262"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-07-20-24-38-567687-98068"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-07-20-59-31-454856-98976"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-07-22-03-35-538120-20851"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-08-05-05-27-911205-37950"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-08-14-52-07-269546-30380"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-08-16-51-44-504507-24744"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-08-18-24-57-922525-28171"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-04-15-01-03-47-788448-49252"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-01-23-02-39-742808-59642"
        # prev_dir_hiphive = "/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/HPS_hiphive/job_2024-05-28-19-03-30-410287-51438"
        # 5. Extract Phonon Band structure & DOS from FC
        # for 0K
        fc_pdos_pb_to_db = run_fc_to_pdos(
                renormalized=renormalize,
                mesh_density=mesh_density,
                prev_dir_json_saver=fit_force_constant.output["current_dir"],
                # prev_dir_json_saver=prev_dir_hiphive,
                loop=loops,
            )
        fc_pdos_pb_to_db.name += f" {loops} 0K"
        jobs.append(fc_pdos_pb_to_db)
        outputs.append(fc_pdos_pb_to_db.output)
        fc_pdos_pb_to_db.metadata.update(
            {
                "tag": [
                    f"mp_id={mpid}",
                    f"cutoffs={cutoffs}",
                    f"bulk_modulus={bulk_modulus}",
                    f"temperature=0K"
                    f"fc_pdos_pb_to_db_{loops}",
                    f"nConfigsPerStd={n_structures}",
                    f"fixedDispls={fixed_displs}",
                    f"dispCut={disp_cut}",
                    f"supercell_matrix={supercell_matrix}",
                    f"loop={loops}",
                ]
            }
        )
        # for finite temperatures
        if renormalize:
            for i, temperature in enumerate(renormalize_temperature):
                fc_pdos_pb_to_db = run_fc_to_pdos(
                    renormalized=renormalize,
                    mesh_density=mesh_density,
                    prev_dir_json_saver=outputs_renorm[i][0],
                    loop=loops,
                )
                fc_pdos_pb_to_db.name += f" {loops} {temperature}K"
                jobs.append(fc_pdos_pb_to_db)
                outputs.append(fc_pdos_pb_to_db.output)
                fc_pdos_pb_to_db.metadata.update(
                    {
                        "tag": [
                            f"mp_id={mpid}",
                            f"bulk_modulus={bulk_modulus}",
                            f"temperature={temperature}K"
                            f"fc_pdos_pb_to_db_{loops}",
                            f"nConfigsPerStd={n_structures}",
                            f"fixedDispls={fixed_displs}",
                            f"dispCut={disp_cut}",
                            f"supercell_matrix={supercell_matrix}",
                            f"loop={loops}",
                        ]
                    }
                )


        # # 6. Lattice thermal conductivity calculation using Sheng BTE
        # if calculate_lattice_thermal_conductivity:
        #     if renormalize:
        #         temperatures = renormalize_temperature
        #     else:
        #         temperatures = thermal_conductivity_temperature
        #     # Because of the way ShengBTE works, a temperature array that is not
        #     # evenly spaced out (T_step) requires submission for each temperature
        #     if not renormalize:
        #         if isinstance(temperatures, dict):
        #             pass
        #         elif type(temperatures) in [list, np.ndarray]:
        #             assert all(np.diff(temperatures) == np.diff(temperatures)[0])
        #         lattice_thermal_conductivity = run_lattice_thermal_conductivity(
        #             renormalized=renormalize,
        #             temperature=temperatures,
        #             loop=loops,
        #             prev_dir_hiphive=fit_force_constant.output["current_dir"],
        #             therm_cond_solver= self.THERM_COND_SOLVER
        #         )
        #         lattice_thermal_conductivity.name += f" {temperatures} {loops}"
        #         lattice_thermal_conductivity.update_config({"manager_config": {"_fworker": "gpu_fworker"}})
        #         jobs.append(lattice_thermal_conductivity)
        #         outputs.append(lattice_thermal_conductivity.output)
        #         lattice_thermal_conductivity.metadata.update(
        #             {
        #                 "tag": [
        #                     f"mp_id={mpid}",
        #                     f"bulk_modulus={bulk_modulus}",
        #                     f"run_lattice_thermal_conductivity_{loops}",
        #                     f"nConfigsPerStd={n_structures}",
        #                     f"fixedDispls={fixed_displs}",
        #                     f"dispCut={disp_cut}",
        #                     f"supercell_matrix_kwargs={self.supercell_matrix_kwargs}",
        #                     f"supercell_matrix={supercell_matrix}",
        #                     f"loop={loops}",
        #                 ]
        #             }
        #         )
        #     else:
        #         for t, T in enumerate(temperatures):
        #             if T == 0:
        #                 continue
        #             lattice_thermal_conductivity = run_lattice_thermal_conductivity(
        #                 renormalized=renormalize,
        #                 temperature=T,
        #                 loop=loops,
        #                 prev_dir_hiphive=outputs_renorm[t][0],
        #                 therm_cond_solver= self.THERM_COND_SOLVER
        #             )

        #             lattice_thermal_conductivity.name += f" {T} {loops}"
        #             lattice_thermal_conductivity.update_config({"manager_config": {"_fworker": "gpu_fworker"}})
        #             jobs.append(lattice_thermal_conductivity)
        #             outputs.append(lattice_thermal_conductivity.output)
        #             lattice_thermal_conductivity.metadata.update(
        #                 {
        #                     "tag": [
        #                         f"mp_id={mpid}",
        #                         f"bulk_modulus={bulk_modulus}",
        #                         f"run_lattice_thermal_conductivity_{loops}",
        #                         f"nConfigsPerStd={n_structures}",
        #                         f"fixedDispls={fixed_displs}",
        #                         f"dispCut={disp_cut}",
        #                         f"supercell_matrix_kwargs={self.supercell_matrix_kwargs}",
        #                         f"supercell_matrix={supercell_matrix}",
        #                         f"loop={loops}",
        #                     ]
        #                 }
        #             )

        return Flow(jobs=jobs, output=outputs, name=f"{mpid}_{self.THERM_COND_SOLVER}_"
                                                    f"{disp_cut}_"
                                                    f"{cutoffs}_"
                                                    f"seperate_fit_24_harm_0.03_12_anharm_0.08"
                                                    f"{self.name}")

    @property
    @abstractmethod
    def prev_calc_dir_argname(self) -> str | None:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
