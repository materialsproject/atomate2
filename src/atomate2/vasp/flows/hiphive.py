"""Flow for calculating harmonic & anharmonic props of phonon using hiPhive."""

# Basic Python packages
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np

# Jobflow packages
from jobflow import Flow, Maker

# Pymatgen packages
from atomate2.vasp.flows.core import DoubleRelaxMaker

# Atomate2 packages
# from atomate2.vasp.sets.core import MPStaticSetGenerator
from atomate2.vasp.jobs.core import (
    StaticMaker,
    # MPStaticMaker,
    TightRelaxMaker,
)
from atomate2.common.jobs.phonons import (
    PhononDisplacementMaker,
    generate_frequencies_eigenvectors,
    generate_phonon_displacements,
    get_supercell_size,
    get_total_energy_per_cell,
    run_phonon_displacements,
)
from atomate2.vasp.jobs.hiphive import (
    generate_hiphive_displacements,
    collect_perturbed_structures,
    quality_control,
    run_fc_to_pdos,
    run_hiphive,
    run_hiphive_renormalization,
    run_lattice_thermal_conductivity,
    run_static_calculations,
)
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker

logger = logging.getLogger(__name__)

__all__ = ["HiphiveMaker"]

__author__ = "Alex Ganose, Junsoo Park, Zhuoying Zhu, Hrushikesh Sahasrabuddhe"
__email__ = "aganose@lbl.gov, jsyony37@lbl.gov, zyzhu@lbl.gov, hpsahasrabuddhe@lbl.gov"


@dataclass
class HiphiveMaker(Maker):
    """
    Workflow to calc. interatomic force constants and vibrational props using hiPhive.

    A summary of the workflow is as follows:
    1. Structure relaxtion
    2. Calculate a supercell transformation matrix that brings the
       structure as close as cubic as possible, with all lattice lengths
       greater than 5 nearest neighbor distances.
    3. Perturb the atomic sites for each supercell using a Monte Carlo
       rattle procedure. The atoms are perturbed roughly according to a
       normal deviation. A number of standard deviation perturbation distances
       are included. Multiple supercells may be generated for each perturbation
       distance.
    4. Run static VASP calculations on each perturbed supercell to calculate
       atomic forces.
    5. Aggregate the forces and conduct the fit atomic force constants using
       the minimization schemes in hiPhive.
    6. Output the interatomic force constants, and phonon band structure and
       density of states to the database.
    7. Optional: Perform phonon renormalization at finite temperature - useful
       when unstable modes exist
    8. Optional: Solve the lattice thermal conductivity using ShengBTE and
       output to the database.

    Args
    ----------
    name : str
        Name of the flows produced by this maker.
    task_document_kwargs (dict):
        Keyword arguments for task document, default is {"task_label": "dummy_label"}.
    MPstatic_maker (BaseVaspMaker):
        The VASP input generator for static calculations, default is StaticMaker.
    bulk_relax_maker (BaseVaspMaker | None):
        The VASP input generator for bulk relaxation,
        default is DoubleRelaxMaker using TightRelaxMaker.
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
    SHENGBTE_CMD (str):
        Command for executing ShengBTE,
        default is "srun -n 32 ./ShengBTE 2>BTE.err >BTE.out".
    PHONO3PY_CMD (str):
        Command for executing Phono3py, default is
        "phono3py --mesh 19 19 19 --fc3 --fc2 --br --dim 5 5 5".
    """

    name: str = "Lattice-Dynamics"
    task_document_kwargs: dict = field(
        default_factory=lambda: {"task_label": "dummy_label"}
    )
    static_energy_maker: BaseVaspMaker | None = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(auto_ispin=True)
        )
    )
    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=PhononDisplacementMaker
    )
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
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
        1500
    ]  # [i*100 for i in range(0,16)] # Temp. at which renorm. is to be performed
    # Temperature at which lattice thermal conductivity is calculated
    # If renormalization is performed,
    # T_RENORM overrides T_KLAT for lattice thermal conductivity
    # T_KLAT = {"t_min":100,"t_max":1500,"t_step":100} #[i*100 for i in range(0,11)]
    T_KLAT: ClassVar[list[int]] = [100, 200, 300, 400]  # [i*100 for i in range(0,11)]
    T_THERMAL_CONDUCTIVITY: ClassVar[list[int]] = [
        0,
        100,
        200,
        300,
    ]  # [i*100 for i in range(0,16)]
    FIT_METHOD = "rfe"
    RENORM_METHOD = "pseudoinverse"
    RENORM_NCONFIG = 5  # Changed from 50
    RENORM_CONV_THRESH = 0.1  # meV/atom
    RENORM_MAX_ITER = 30  # Changed from 20
    SHENGBTE_CMD = "srun -n 32 ./ShengBTE 2>BTE.err >BTE.out"
    PHONO3PY_CMD = "phono3py --mesh 19 19 19 --fc3 --fc2 --br --dim 5 5 5"

    def make(
        self,
        mpid: str,
        structure: Structure,
        bulk_modulus: float,
        supercell_matrix: Matrix3D | None = None,
        total_dft_energy_per_formula_unit: float | None = None,
        fit_method: str | None = None,
        disp_cut: float | None = None,
        cutoffs: list[list[float]] | None = None,
        prev_vasp_dir: str | Path | None = None,
        num_supercell_kwargs: dict | None = None,
        perturbed_structure_kwargs: dict | None = None,
        calculate_lattice_thermal_conductivity: bool = True,
        # thermal_conductivity_temperature: Union[float, Dict] = T_KLAT,
        renormalize: bool = False,
        renormalize_temperature: list = T_RENORM,
        renormalize_method: str = RENORM_METHOD,
        renormalize_nconfig: int = RENORM_NCONFIG,
        renormalize_conv_thresh: float = RENORM_CONV_THRESH,
        renormalize_max_iter: int = RENORM_MAX_ITER,
        renormalize_thermal_expansion_iter: bool = False,
        mesh_density: float = MESH_DENSITY,
        shengbte_cmd: str = SHENGBTE_CMD,
        phono3py_cmd: str = PHONO3PY_CMD,
        thermal_conductivity_temperature: list = T_KLAT,
        imaginary_tol: float = IMAGINARY_TOL,
        temperature_qha: float | list | dict = T_QHA,
        n_structures: float = None,
        rattle_std: float = None,
        prev_dir_hiphive: str | Path | None = None,
    ):
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
        fit_method (str, optional):
            Method for fitting force constants using hiphive, default is None.
        disp_cut (float, optional):
            Cutoff distance for displacements in Angstroms, default is None.
        cutoffs (List[List[float]], optional):
            List of cutoff distances for different force constants fitting,
            default is None.
        prev_vasp_dir (str | Path | None, optional):
            Previous vasp calculation directory to use for copying outputs.,
            default is None.
        supercell_matrix_kwargs (dict, optional):
            Keyword arguments for supercell matrix generation, default is None.
        num_supercell_kwargs (dict, optional):
            Keyword arguments for supercell enumeration, default is None.
        perturbed_structure_kwargs (dict, optional):
            Keyword arguments for perturbed structure generation, default is None.
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
        shengbte_cmd (str, optional):
            Command for executing ShengBTE, default is SHENGBTE_CMD.
        phono3py_cmd (str, optional):
            Command for executing Phono3py, default is PHONO3PY_CMD.
        thermal_conductivity_temperature (float | List | Dict, optional):
            Temperatures for thermal conductivity calculations, default is T_KLAT.
        imaginary_tol (float, optional):
            Imaginary frequency tolerance in THz, default is IMAGINARY_TOL.
        temperature_qha (float, optional):
            Temperatures for phonopy thermodynamic calculations, default is T_QHA.
        n_structures (float, optional):
            Number of structures to consider for calculations, default is None.
        rattle_std (float, optional):
            Standard deviation for atomic displacement in Angstroms, default is None.
        prev_dir_hiphive (str | Path | None, optional):
            Previous hiphive calculation directory to use for copying outputs,
            default is None.
        """
        jobs = []
        outputs = []
        loops = 1

        # 1. Relax the structure
        if isinstance(self.bulk_relax_maker, ForceFieldRelaxMaker):
            bulk = self.bulk_relax_maker.make(structure)
        else:
            bulk = self.bulk_relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
        jobs.append(bulk)
        outputs.append(bulk.output)
        print(bulk.output.structure)
        structure = bulk.output.structure
        prev_vasp_dir = bulk.output.dir_name
        bulk.update_metadata(
            {
                "tag": [
                    f"mp_id={mpid}",
                    f"relax_{loops}",
                    f"nConfigsPerStd={n_structures}",
                    f"rattleStds={rattle_std}",
                    f"dispCut={disp_cut}",
                    f"supercell_matrix_kwargs={self.supercell_matrix_kwargs}",
                    f"loop={loops}",
                ]
            }
        )

        # if supercell_matrix is None, supercell size will be determined after relax
        # maker to ensure that cell lengths are really larger than threshold
        if supercell_matrix is None:
            supercell_job = get_supercell_size(
                structure,
                self.min_length,
                self.prefer_90_degrees,
                **self.supercell_matrix_kwargs,
            )
            jobs.append(supercell_job)
            supercell_matrix = supercell_job.output
        
        # Computation of static energy
        total_dft_energy = None
        static_run_job_dir = None
        static_run_uuid = None
        if (self.static_energy_maker is not None) and (
            total_dft_energy_per_formula_unit is None
        ):
            if isinstance(self.static_energy_maker, ForceFieldRelaxMaker):
                static_job = self.static_energy_maker.make(structure)
            else:
                static_job = self.static_energy_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
            jobs.append(static_job)
            total_dft_energy = static_job.output.output.energy
            static_run_job_dir = static_job.output.dir_name
            static_run_uuid = static_job.output.uuid
            prev_vasp_dir = static_job.output.dir_name
        elif total_dft_energy_per_formula_unit is not None:
            # to make sure that one can reuse results from Doc
            compute_total_energy_job = get_total_energy_per_cell(
                total_dft_energy_per_formula_unit, structure
            )
            jobs.append(compute_total_energy_job)
            total_dft_energy = compute_total_energy_job.output
        
        # get a phonon object from phonopy
        displacements = generate_hiphive_displacements(
            structure=structure,
            supercell_matrix=supercell_matrix,
            n_structures=n_structures,
            rattle_std=rattle_std,
            loop=loops,
        )
        jobs.append(displacements)

        # perform the phonon displacement calculations
        vasp_displacement_calcs = run_phonon_displacements(
            displacements=displacements.output,
            structure=structure,
            supercell_matrix=supercell_matrix,
            phonon_maker=self.phonon_displacement_maker,
            prev_vasp_dir=prev_vasp_dir,
        )
        jobs.append(vasp_displacement_calcs)

        # # 2. Generates supercell, performs fixed displ rattling and
        # # runs static calculations
        # vasp_static_calcs = run_static_calculations(
        #     structure=structure,
        #     supercell_matrix_kwargs=supercell_matrix_kwargs,
        #     n_structures=n_structures,
        #     rattle_std=rattle_std,
        #     loop=loops,
        #     prev_vasp_dir=prev_vasp_dir,
        #     MPstatic_maker=self.phonon_displacement_maker,
        # )
        # jobs.append(vasp_static_calcs)
        # outputs.append(vasp_static_calcs.output)
        # vasp_static_calcs.metadata.update(
        #     {
        #         "tag": [
        #             f"mp_id={mpid}",
        #             f"vasp_static_calcs_{loops}",
        #             f"nConfigsPerStd={n_structures}",
        #             f"rattleStds={rattle_std}",
        #             f"dispCut={disp_cut}",
        #             f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
        #             f"loop={loops}",
        #         ]
        #     }
        # )

        # 3.  Save "structure_data_{loop}.json", "relaxed_structures.json",
        # "perturbed_forces_{loop}.json" and "perturbed_structures_{loop}.json"
        # files locally
        json_saver = collect_perturbed_structures(
            structure=structure,
            supercell_matrix_kwargs=supercell_matrix_kwargs,
            rattled_structures=vasp_static_calcs.output,
            forces=vasp_static_calcs.output,
            loop=loops,
        )
        json_saver.name += f" {loops}"
        jobs.append(json_saver)
        outputs.append(json_saver.output)
        json_saver.metadata.update(
            {
                "tag": [
                    f"mp_id={mpid}",
                    f"json_saver_{loops}",
                    f"nConfigsPerStd={n_structures}",
                    f"rattleStds={rattle_std}",
                    f"dispCut={disp_cut}",
                    f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
                    f"loop={loops}",
                ]
            }
        )

        # 4. Hiphive Fitting of FCPs upto 4th order
        fit_force_constant = run_hiphive(
            fit_method=fit_method,
            disp_cut=disp_cut,
            bulk_modulus=bulk_modulus,
            temperature_qha=temperature_qha,
            mesh_density=mesh_density,
            imaginary_tol=imaginary_tol,
            prev_dir_json_saver=json_saver.output[3],
            loop=loops,
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
                    f"rattleStds={rattle_std}",
                    f"dispCut={disp_cut}",
                    f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
                    f"loop={loops}",
                ]
            }
        )

        # 5. Extract Phonon Band structure & DOS from FC
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
                    f"rattleStds={rattle_std}",
                    f"dispCut={disp_cut}",
                    f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
                    f"loop={loops}",
                ]
            }
        )

        # 6. Quality Control Job to check if the desired Test RMSE is achieved,
        # if not, then increase the number of structures --
        # Using "addintion" feature of jobflow
        loops += 1
        n_structures += 1
        logger.info(f"Number of structures increased to {n_structures}")
        logger.info(f"loop = {loops}")
        error_check_job = quality_control(
            rmse_test=fit_force_constant.output[5],
            n_structures=n_structures,
            rattle_std=rattle_std,
            loop=loops,
            fit_method=fit_method,
            disp_cut=disp_cut,
            bulk_modulus=bulk_modulus,
            temperature_qha=temperature_qha,
            mesh_density=mesh_density,
            imaginary_tol=imaginary_tol,
            prev_dir_json_saver=json_saver.output[3],
            prev_vasp_dir=prev_vasp_dir,
            supercell_matrix_kwargs=supercell_matrix_kwargs,
        )
        error_check_job.name += f" {loops}"
        jobs.append(error_check_job)
        outputs.append(error_check_job.output)
        error_check_job.metadata.update(
            {
                "tag": [
                    f"error_check_job_{loops}",
                    f"nConfigsPerStd={n_structures}",
                    f"rattleStds={rattle_std}",
                    f"dispCut={disp_cut}",
                    f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
                    f"loop={loops}",
                ]
            }
        )

        # 7. Perform phonon renormalization to obtain temperature-dependent
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
                            f"rattleStds={rattle_std}",
                            f"dispCut={disp_cut}",
                            f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
                            f"loop={loops}",
                        ]
                    }
                )

        # ##### 8. Lattice thermal conductivity calculation using phono3py
        # if calculate_lattice_thermal_conductivity:
        #     fw_lattice_conductivity = RunPhono3py(
        #         # phono3py_cmd=phono3py_cmd,
        #         renormalized=renormalize,
        #         loop=loops,
        #         # prev_dir_hiphive=prev_dir_hiphive,
        #         prev_dir_hiphive=fw_fit_force_constant.output[4],
        #         )
        # else:
        #     pass

        # fw_lattice_conductivity.name += f" {loops}"
        # jobs.append(fw_lattice_conductivity)
        # outputs.append(fw_lattice_conductivity.output)
        # fw_lattice_conductivity.metadata.update({"tag": [f"mp_id={mpid}",
        # f"fw_lattice_conductivity_{loops}", f"nConfigsPerStd={n_structures}",
        # f"rattleStds={rattle_std}", f"dispCut={disp_cut}",
        # f"supercell_matrix_kwargs={supercell_matrix_kwargs}", f"loop={loops}"]})

        # 8. Lattice thermal conductivity calculation using Sheng BTE
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
                    # if len(temperatures) == 0:
                    #     # Handle the case when temperatures is empty
                    #     pass
                    # elif type(temperatures) in [list, np.ndarray]:
                    #     assert all(np.diff(temperatures) == np.diff(temperatures)[0])

                lattice_thermal_conductivity = run_lattice_thermal_conductivity(
                    shengbte_cmd=shengbte_cmd,
                    renormalized=renormalize,
                    temperature=temperatures,
                    loop=loops,
                    prev_dir_hiphive=fit_force_constant.output[4],
                )
            else:
                for _t, T in enumerate(temperatures):
                    if T == 0:
                        continue
                    lattice_thermal_conductivity = run_lattice_thermal_conductivity(
                        shengbte_cmd=shengbte_cmd,
                        renormalized=renormalize,
                        temperature=T,
                        loop=loops,
                        prev_dir_hiphive=fit_force_constant.output[4],
                    )

        lattice_thermal_conductivity.name += f" {loops}"
        jobs.append(lattice_thermal_conductivity)
        outputs.append(lattice_thermal_conductivity.output)
        lattice_thermal_conductivity.metadata.update(
            {
                "tag": [
                    f"run_lattice_thermal_conductivity_{loops}",
                    f"nConfigsPerStd={n_structures}",
                    f"rattleStds={rattle_std}",
                    f"dispCut={disp_cut}",
                    f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
                    f"loop={loops}",
                ]
            }
        )

        return Flow(jobs=jobs, output=outputs, name=f"{mpid}_ShengBTE")
