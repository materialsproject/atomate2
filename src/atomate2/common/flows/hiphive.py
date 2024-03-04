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
    run_hiphive_renormalization,
    run_lattice_thermal_conductivity,
)

# Atomate2 packages
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

    from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
    from atomate2.vasp.flows.core import DoubleRelaxMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker
from emmet.core.math import Matrix3D

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
    T_THERMAL_CONDUCTIVITY (list):
        Temperatures for thermal conductivity calculations,
        default is [0, 100, 200, 300].
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
    min_length: float | None = 13.0
    prefer_90_degrees: bool = True
    supercell_matrix_kwargs: dict = field(default_factory=dict)
    IMAGINARY_TOL = 0.025  # in THz
    MESH_DENSITY = 100.0  # should always be a float
    T_QHA: ClassVar[list[int]] = [
        i * 100 for i in range(21)
    ]  # Temp. for phonopy calc. of thermo. properties (free energy etc.)
    T_RENORM: ClassVar[list[int]] = [
        400 # 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
    ]  # [i*100 for i in range(0,16)] # Temp. at which renorm. is to be performed
    # Temperature at which lattice thermal conductivity is calculated
    # If renormalization is performed,
    # T_RENORM overrides T_KLAT for lattice thermal conductivity
    T_KLAT: ClassVar[list[int]] = [100, 200, 300, 400]  # [i*100 for i in range(0,11)]
    T_THERMAL_CONDUCTIVITY: ClassVar[list[int]] = [
        0,
        100,
        200,
        300,
    ]  # [i*100 for i in range(0,16)]
    FIT_METHOD = "rfe"
    RENORM_METHOD = "least_squares"
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
        loops = 1

        # 1. Relax the structure
        if self.bulk_relax_maker is not None:
            bulk_kwargs = {}
            if self.prev_calc_dir_argname is not None:
                bulk_kwargs[self.prev_calc_dir_argname] = prev_dir
            bulk = self.bulk_relax_maker.make(structure, **bulk_kwargs)
            jobs.append(bulk)
            outputs.append(bulk.output)
            structure = bulk.output.structure
            prev_dir = bulk.output.dir_name

        bulk.update_metadata(
            {
                "tag": [
                    f"mp_id={mpid}",
                    f"relax_{loops}",
                    f"nConfigsPerStd={n_structures}",
                    f"fixedDispls={fixed_displs}",
                    f"dispCut={disp_cut}",
                    f"supercell_matrix_kwargs={self.supercell_matrix_kwargs}",
                    f"loop={loops}",
                ]
            }
        )

        # 2. if supercell_matrix is None, supercell size will be determined after relax
        # maker to ensure that cell lengths are really larger than threshold.
        # then, perturbations will be generated based on the supercell size.
        # STATIC calculations will then be run on the perturbed structures, and the
        # forces and perturbed structures will be aggregated.
        static_calcs = hiphive_static_calcs(
                structure=structure,
                supercell_matrix=supercell_matrix,
                min_length=self.min_length,
                prefer_90_degrees=self.prefer_90_degrees,
                n_structures=n_structures,
                fixed_displs=fixed_displs,
                loops=loops,
                prev_dir=prev_dir,
                phonon_displacement_maker=self.phonon_displacement_maker,
                supercell_matrix_kwargs=self.supercell_matrix_kwargs,
        )
        jobs.append(static_calcs)


        # 3. Hiphive Fitting of FCPs upto 4th order
        fit_force_constant = run_hiphive(
            fit_method=fit_method,
            disp_cut=disp_cut,
            bulk_modulus=bulk_modulus,
            temperature_qha=temperature_qha,
            mesh_density=mesh_density,
            imaginary_tol=imaginary_tol,
            prev_dir_json_saver=static_calcs.output["current_dir"],
            loop=loops,
            cutoffs=cutoffs
        )
        fit_force_constant.name += f" {loops}"
        jobs.append(fit_force_constant)
        outputs.append(fit_force_constant.output)
        fit_force_constant.metadata.update(
            {
                "tag": [
                    f"mp_id={mpid}",
                    f"fit_force_constant_{loops}",
                    f"nConfigsPerStd={n_structures}",
                    f"fixedDispls={fixed_displs}",
                    f"dispCut={disp_cut}",
                    f"supercell_matrix={supercell_matrix}",
                    f"loop={loops}",
                ]
            }
        )

        # # 7. Quality Control Job to check if the desired Test RMSE is achieved,
        # # if not, then increase the number of structures --
        # # Using "addintion" feature of jobflow
        # loops += 1
        # n_structures += 1
        # logger.info(f"Number of structures increased to {n_structures}")
        # logger.info(f"loop = {loops}")
        # error_check_job = quality_control(
        #     rmse_test=fit_force_constant.output[5],
        #     n_structures=n_structures,
        #     fixedDispls=fixed_displs,
        #     loop=loops,
        #     fit_method=fit_method,
        #     disp_cut=disp_cut,
        #     bulk_modulus=bulk_modulus,
        #     temperature_qha=temperature_qha,
        #     mesh_density=mesh_density,
        #     imaginary_tol=imaginary_tol,
        #     prev_dir_json_saver=static_calcs.output["current_dir"],
        #     prev_dir=prev_dir,
        #     supercell_matrix=supercell_matrix,
        #     # supercell_matrix_kwargs=supercell_matrix_kwargs,
        # )
        # error_check_job.name += f" {loops}"
        # jobs.append(error_check_job)
        # outputs.append(error_check_job.output)
        # error_check_job.metadata.update(
        #     {
        #         "tag": [
        #             f"error_check_job_{loops}",
        #             f"nConfigsPerStd={n_structures}",
        #             f"fixedDispls={fixed_displs}",
        #             f"dispCut={disp_cut}",
        #             # f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
        #             f"supercell_matrix={supercell_matrix}",
        #             f"loop={loops}",
        #         ]
        #     }
        # )

        # 4. Perform phonon renormalization to obtain temperature-dependent
        # force constants using hiPhive
        if renormalize:
            for temperature in renormalize_temperature:
                nconfig = renormalize_nconfig * (1 + temperature // 100)
                renormalization = run_hiphive_renormalization(
                    temperature=temperature,
                    renorm_method=renormalize_method,
                    nconfig=nconfig,
                    conv_thresh=renormalize_conv_thresh,
                    max_iter=renormalize_max_iter,
                    renorm_TE_iter=renormalize_thermal_expansion_iter,
                    bulk_modulus=bulk_modulus,
                    mesh_density=mesh_density,
                    prev_dir_hiphive=fit_force_constant.output[4],
                    loop=loops,
                )
                renormalization.name += f" {temperature} {loops}"
                jobs.append(renormalization)
                outputs.append(renormalization.output)
                renormalization.metadata.update(
                    {
                        "tag": [
                            f"run_renormalization_{loops}",
                            f"nConfigsPerStd={n_structures}",
                            f"fixedDispls={fixed_displs}",
                            f"dispCut={disp_cut}",
                            f"supercell_matrix={supercell_matrix}",
                            f"loop={loops}",
                        ]
                    }
                )

        # 5. Extract Phonon Band structure & DOS from FC
        if renormalize:
            fc_pdos_pb_to_db = run_fc_to_pdos(
                renormalized=renormalize,
                renorm_temperature=renormalize_temperature,
                mesh_density=mesh_density,
                prev_dir_json_saver=renormalization.output[0],
                loop=loops,
            )
        else:
            fc_pdos_pb_to_db = run_fc_to_pdos(
                renormalized=renormalize,
                renorm_temperature=renormalize_temperature,
                mesh_density=mesh_density,
                prev_dir_json_saver=fit_force_constant.output[4],
                loop=loops,
            )
        fc_pdos_pb_to_db.name += f" {loops}"
        jobs.append(fc_pdos_pb_to_db)
        outputs.append(fc_pdos_pb_to_db.output)
        fc_pdos_pb_to_db.metadata.update(
            {
                "tag": [
                    f"mp_id={mpid}",
                    f"fc_pdos_pb_to_db_{loops}",
                    f"nConfigsPerStd={n_structures}",
                    f"fixedDispls={fixed_displs}",
                    f"dispCut={disp_cut}",
                    f"supercell_matrix={supercell_matrix}",
                    f"loop={loops}",
                ]
            }
        )

        # 6. Lattice thermal conductivity calculation using Sheng BTE
        if calculate_lattice_thermal_conductivity:
            if renormalize:
                temperatures = renormalize_temperature
            else:
                temperatures = thermal_conductivity_temperature
            # Because of the way ShengBTE works, a temperature array that is not
            # evenly spaced out (T_step) requires submission for each temperature
            if not renormalize:
                if isinstance(temperatures, dict):
                    pass
                elif type(temperatures) in [list, np.ndarray]:
                    assert all(np.diff(temperatures) == np.diff(temperatures)[0])
                lattice_thermal_conductivity = run_lattice_thermal_conductivity(
                    renormalized=renormalize,
                    temperature=temperatures,
                    loop=loops,
                    prev_dir_hiphive=fit_force_constant.output[4],
                    therm_cond_solver= self.THERM_COND_SOLVER
                )
            else:
                for _, T in enumerate(temperatures):
                    if T == 0:
                        continue
                    lattice_thermal_conductivity = run_lattice_thermal_conductivity(
                        renormalized=renormalize,
                        temperature=T,
                        loop=loops,
                        prev_dir_hiphive=fit_force_constant.output[4],
                        therm_cond_solver= self.THERM_COND_SOLVER
                    )

        lattice_thermal_conductivity.name += f" {loops}"
        jobs.append(lattice_thermal_conductivity)
        outputs.append(lattice_thermal_conductivity.output)
        lattice_thermal_conductivity.metadata.update(
            {
                "tag": [
                    f"run_lattice_thermal_conductivity_{loops}",
                    f"nConfigsPerStd={n_structures}",
                    f"fixedDispls={fixed_displs}",
                    f"dispCut={disp_cut}",
                    # f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
                    f"supercell_matrix={supercell_matrix}",
                    f"loop={loops}",
                ]
            }
        )

        return Flow(jobs=jobs, output=outputs, name=f"{mpid}_ShengBTE_"
                                                    f"{fixed_displs}_"
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
