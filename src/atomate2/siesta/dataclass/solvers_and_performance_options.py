"""
Module defining base SIESTA input set and generator.

class SolversAndPerformanceOptions

Based on User's Guide Siesta 5.4.0
Section: 6.13 The ELSI solver family
         6.14 The CheSS solver
         6.14.1 Input parameters
         6.15 The PEXSI solver (native interface)
         6.15.1 Pole handling
         6.15.2 Parallel environment and control options
         6.15.3 Electron tolerance and the PEXSI solver
         6.15.4 Inertia-counting
         6.15.5 Reuse of µ information across iterations
         6.15.6 Calculation of the density of states by inertia-counting
         6.15.7 Calculation of the LDOS by selected-inversion
"""

# Metadata

__all__ = ["SolversAndPerformanceOptions"]

import logging
from dataclasses import dataclass, field
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass

logger = logging.getLogger(__name__)


@dataclass
class SolversAndPerformanceOptions(FDFDataclass):
    """Data class to manage solver and performance options for SIESTA input."""

    # ---------------------------
    # 6.13 The ELSI solver family
    # ---------------------------
    # elsi_solver: str = "ELSI"  # ELSI.Solver Solver type to be used
    #   ('ELSI', 'PEXSI', 'CheSS', 'default')
    # elsi_solver: List[str] = field(default_factory=lambda:
    #   ["ELPA", "OMM","PEXSI",'NTPOLY',"SIPS","EIGENEXA","MAGMA"])
    #   # List of ELSI solvers to use if 'ELSI' is selected
    # elsi_broadening_method: str = "fermi" # ELSI.Broadening.Method ”fermi”
    # elsi_output_level: int = 0 # ELSI.Output.Level 0
    # elsi_output_json: int = 1 # ELSI.Output.Json 1
    # elsi_broadeing_mp_order: int = 1 # ELSI.Broadening.MPOrder 1
    # elsi_ill_condition_check: int = 0 # ELSI.Ill-Condition.Check 0
    # elsi_ill_condition_tolerance: float = 1e-5 # ELSI.Ill-Condition.Tolerance 10-5
    # elsi_elpa_flavor: int = 2 # ELSI.ELPA.Flavor 2
    # elsi_elpa_n_signle_precision: int = 0 # ELSI.ELPA.NSinglePrecision 0
    # elsi_elpa_autotune: int = 0 # ELSI.ELPA.Autotune 0
    # elsi_elpa_gpu: int = 0 # ELSI.ELPA.GPU 0
    # elsi_omm_flavor: int = 0 # ELSI.OMM.Flavor 0
    # elsi_omm_elpa_steps: int = 3 # ELSI.OMM.ELPA.Steps 3
    # elsi_omm_tolerance: float = 1e-9 # ELSI.OMM.Tolerance 10-9
    # elsi_pexsi_method: int = 3 # ELSI.PEXSI.Method 3
    # elsi_pexsi_tasks_per_pole: int = None # ELSI.PEXSI.TasksPerPole no default
    # elsi_pexsi_tasks_symbolic: int = 1 # ELSI.PEXSI.TasksSymbolic 1
    # elsi_pexsi_number_of_poles: int = 20 # ELSI.PEXSI.Number-of-Poles 20
    # elsi_pexsi_number_of_mu_points: int = 2 # ELSI.PEXSI.Number-of-Mu-Points 2
    # elsi_pexsi_inertia_tolerance: float = 0.05 # ELSI.PEXSI.Inertia-Tolerance 0.05
    # elsi_pexsi_initial_mu_min: float = -1.0 # ELSI.PEXSI.Initial-Mu-Min -1.0 Ry
    # elsi_pexsi_initial_mu_max: float = 0.0 # ELSI.PEXSI.Initial-Mu-Max 0.0 Ry
    # elsi_nt_poly_method: int = 2 # ELSI.NTPoly.Method 2
    # elsi_nt_poly_filter: float = 1e-9 # ELSI.NTPoly.Filter 10-9
    # elsi_nt_poly_tolerance: float = 1e-6 # ELSI.NTPoly.Tolerance 10-6
    # elsi_nt_poly_slices: int = None # ELSI.SIPS.Slices no default
    # elsi_sips_elpa_steps: int = 2 # ELSI.SIPS.ELPA.Steps 2
    # elsi_eigen_exa_method: int =2 # ELSI.EigenExa.Method 2
    # elsi_magma_solver_method: int =1 # ELSI.MAGMA.Solver-Method 1
    elsi_solver: list[str] = field(
        default_factory=lambda: [
            "ELPA",
            "OMM",
            "PEXSI",
            "NTPOLY",
            "SIPS",
            "EIGENEXA",
            "MAGMA",
        ],
        metadata={
            "description": (
                "A wrapper-level parameter to select the specific solver to be used "
                "through the ELSI interface. The 'SolutionMethod' must be set to "
                "'elsi'."
            ),
            "SIESTA keyword": None,
        },
    )

    elsi_broadening_method: str = field(
        default="fermi",
        metadata={
            "description": (
                "Sets the broadening/smearing method for electronic occupations within "
                "ELSI."
            ),
            "SIESTA keyword": "ELSI.Broadening.Method",
        },
    )

    elsi_output_level: int = field(
        default=0,
        metadata={
            "description": (
                "Controls the verbosity level of the output from the ELSI library."
            ),
            "SIESTA keyword": "ELSI.Output.Level",
        },
    )

    elsi_output_json: int = field(
        default=1,
        metadata={
            "description": (
                "Controls whether ELSI writes performance data in JSON format."
            ),
            "SIESTA keyword": "ELSI.Output.Json",
        },
    )

    elsi_broadening_mp_order: int = field(
        default=1,
        metadata={
            "description": (
                "Sets the order of the Hermite polynomial for Methfessel-Paxton "
                "smearing if it is the chosen broadening method."
            ),
            "SIESTA keyword": "ELSI.Broadening.MPOrder",
        },
    )

    elsi_ill_condition_check: int = field(
        default=0,
        metadata={
            "description": (
                "Controls whether ELSI performs a check for ill-conditioning of the "
                "overlap matrix."
            ),
            "SIESTA keyword": "ELSI.Ill-Condition.Check",
        },
    )

    elsi_ill_condition_tolerance: float = field(
        default=1e-5,
        metadata={
            "description": (
                "The tolerance for the ill-conditioning check of the overlap matrix."
            ),
            "SIESTA keyword": "ELSI.Ill-Condition.Tolerance",
        },
    )

    elsi_elpa_flavor: int = field(
        default=2,
        metadata={
            "description": (
                "Selects the specific flavor or version of the ELPA algorithm to use "
                "(e.g., one-stage vs. two-stage)."
            ),
            "SIESTA keyword": "ELSI.ELPA.Flavor",
        },
    )

    elsi_elpa_n_single_precision: int = field(
        default=0,
        metadata={
            "description": (
                "The number of single-precision steps to perform before switching to "
                "double precision in the two-stage ELPA algorithm."
            ),
            "SIESTA keyword": "ELSI.ELPA.NSinglePrecision",
        },
    )

    elsi_elpa_autotune: int = field(
        default=0,
        metadata={
            "description": (
                "Enables or disables the autotuning feature within the ELPA library."
            ),
            "SIESTA keyword": "ELSI.ELPA.Autotune",
        },
    )

    elsi_elpa_gpu: int = field(
        default=0,
        metadata={
            "description": "Enables or disables GPU acceleration for the ELPA solver.",
            "SIESTA keyword": "ELSI.ELPA.GPU",
        },
    )

    elsi_omm_flavor: int = field(
        default=0,
        metadata={
            "description": (
                "Selects the specific flavor of the OMM (Order-N) method provided by "
                "the ELSI library."
            ),
            "SIESTA keyword": "ELSI.OMM.Flavor",
        },
    )

    elsi_omm_elpa_steps: int = field(
        default=3,
        metadata={
            "description": (
                "The number of ELPA diagonalization steps to perform within the OMM "
                "algorithm."
            ),
            "SIESTA keyword": "ELSI.OMM.ELPA.Steps",
        },
    )

    elsi_omm_tolerance: float = field(
        default=1e-9,
        metadata={
            "description": "The convergence tolerance for the OMM method within ELSI.",
            "SIESTA keyword": "ELSI.OMM.Tolerance",
        },
    )

    elsi_pexsi_method: int = field(
        default=3,
        metadata={
            "description": "Selects the method used by the PEXSI solver.",
            "SIESTA keyword": "ELSI.PEXSI.Method",
        },
    )

    elsi_pexsi_tasks_per_pole: int = field(
        default=None,
        metadata={
            "description": (
                "The number of MPI tasks assigned to each pole in the PEXSI pole "
                "expansion."
            ),
            "SIESTA keyword": "ELSI.PEXSI.TasksPerPole",
        },
    )

    elsi_pexsi_tasks_symbolic: int = field(
        default=1,
        metadata={
            "description": (
                "The number of MPI tasks used for the symbolic factorization step in "
                "PEXSI."
            ),
            "SIESTA keyword": "ELSI.PEXSI.TasksSymbolic",
        },
    )

    elsi_pexsi_number_of_poles: int = field(
        default=20,
        metadata={
            "description": (
                "The number of poles used in the PEXSI pole expansion technique."
            ),
            "SIESTA keyword": "ELSI.PEXSI.Number-of-Poles",
        },
    )

    elsi_pexsi_number_of_mu_points: int = field(
        default=2,
        metadata={
            "description": (
                "The number of chemical potential (mu) points used in the PEXSI "
                "calculation."
            ),
            "SIESTA keyword": "ELSI.PEXSI.Number-of-Mu-Points",
        },
    )

    elsi_pexsi_inertia_tolerance: float = field(
        default=0.05,
        metadata={
            "description": (
                "The tolerance for the matrix inertia count in PEXSI, which relates to "
                "finding the number of eigenvalues below a given energy."
            ),
            "SIESTA keyword": "ELSI.PEXSI.Inertia-Tolerance",
        },
    )

    elsi_pexsi_initial_mu_min: float = field(
        default=-1.0,
        metadata={
            "description": (
                "The initial lower bound (in Rydberg) for the chemical potential "
                "search in PEXSI."
            ),
            "SIESTA keyword": "ELSI.PEXSI.Initial-Mu-Min",
        },
    )

    elsi_pexsi_initial_mu_max: float = field(
        default=0.0,
        metadata={
            "description": (
                "The initial upper bound (in Rydberg) for the chemical potential "
                "search in PEXSI."
            ),
            "SIESTA keyword": "ELSI.PEXSI.Initial-Mu-Max",
        },
    )

    elsi_nt_poly_method: int = field(
        default=2,
        metadata={
            "description": "Selects the method used by the NTPoly solver.",
            "SIESTA keyword": "ELSI.NTPoly.Method",
        },
    )

    elsi_nt_poly_filter: float = field(
        default=1e-9,
        metadata={
            "description": (
                "The threshold for filtering matrix elements in the NTPoly sparse "
                "matrix library."
            ),
            "SIESTA keyword": "ELSI.NTPoly.Filter",
        },
    )

    elsi_nt_poly_tolerance: float = field(
        default=1e-6,
        metadata={
            "description": "The convergence tolerance for the NTPoly solver.",
            "SIESTA keyword": "ELSI.NTPoly.Tolerance",
        },
    )

    elsi_nt_poly_slices: int = field(
        default=None,
        metadata={
            "description": (
                "The number of slices used in the SIPS (Spectrum-slicing) method."
            ),
            "SIESTA keyword": "ELSI.SIPS.Slices",
        },
    )

    elsi_sips_elpa_steps: int = field(
        default=2,
        metadata={
            "description": "The number of ELPA steps to use within the SIPS algorithm.",
            "SIESTA keyword": "ELSI.SIPS.ELPA.Steps",
        },
    )

    elsi_eigen_exa_method: int = field(
        default=2,
        metadata={
            "description": "Selects the method used by the EigenExa solver.",
            "SIESTA keyword": "ELSI.EigenExa.Method",
        },
    )

    elsi_magma_solver_method: int = field(
        default=1,
        metadata={
            "description": (
                "Selects the solver method within the MAGMA (GPU-accelerated) library."
            ),
            "SIESTA keyword": "ELSI.MAGMA.Solver-Method",
        },
    )

    # -----------------------
    # 6.14 The CheSS solver
    # 6.14.1 Input parameters
    # -----------------------
    # chess_buffer_kernel: float = 4.0 # CheSS.Buffer.Kernel 4.0 Boh
    # chess_buffer_mult: float = 6.0 # CheSS.Buffer.Mult 6.0 Bohr
    # chess_f_scale: float = 1e-1 # CheSS.Fscale 10-1 Ry
    # chess_f_scale_lowerbound: float = 1e-2 # CheSS.FscaleLowerbound 10-2 Ry
    # chess_f_scale_upperbound: float = 1e-1 # CheSS.FscaleUpperbound 10-1 Ry
    # chess_evlow_h: float = -2.0 # CheSS.evlowH -2.0 Ry
    # chess_evhigh_h: float = 2.0 # CheSS.evhighH 2.0 Ry
    # chess_evlow_s: float = 0.5 # CheSS.evlowS 0.5
    # chess_evhigh_s: float = 1.5 # CheSS.evhighS 1.5
    chess_buffer_kernel: float = field(
        default=4.0,
        metadata={
            "description": (
                "The size (in Bohr) of the real-space buffer region for the kernel "
                "(density matrix) in the CheSS solver."
            ),
            "SIESTA keyword": "CheSS.Buffer.Kernel",
        },
    )

    chess_buffer_mult: float = field(
        default=6.0,
        metadata={
            "description": (
                "The size (in Bohr) of the real-space buffer region for the multiplier "
                "in the CheSS solver."
            ),
            "SIESTA keyword": "CheSS.Buffer.Mult",
        },
    )

    chess_f_scale: float = field(
        default=1e-1,
        metadata={
            "description": (
                "An energy scaling factor (in Rydberg) used in the Chebyshev expansion."
            ),
            "SIESTA keyword": "CheSS.Fscale",
        },
    )

    chess_f_scale_lowerbound: float = field(
        default=1e-2,
        metadata={
            "description": "A lower bound for the energy scaling factor (in Rydberg).",
            "SIESTA keyword": "CheSS.FscaleLowerbound",
        },
    )

    chess_f_scale_upperbound: float = field(
        default=1e-1,
        metadata={
            "description": "An upper bound for the energy scaling factor (in Rydberg).",
            "SIESTA keyword": "CheSS.FscaleUpperbound",
        },
    )

    chess_evlow_h: float = field(
        default=-2.0,
        metadata={
            "description": (
                "The lower bound of the estimated eigenvalue spectrum of the "
                "Hamiltonian (H) in Rydberg."
            ),
            "SIESTA keyword": "CheSS.evlowH",
        },
    )

    chess_evhigh_h: float = field(
        default=2.0,
        metadata={
            "description": (
                "The upper bound of the estimated eigenvalue spectrum of the "
                "Hamiltonian (H) in Rydberg."
            ),
            "SIESTA keyword": "CheSS.evhighH",
        },
    )

    chess_evlow_s: float = field(
        default=0.5,
        metadata={
            "description": (
                "The lower bound of the estimated eigenvalue spectrum of the Overlap "
                "matrix (S)."
            ),
            "SIESTA keyword": "CheSS.evlowS",
        },
    )

    chess_evhigh_s: float = field(
        default=1.5,
        metadata={
            "description": (
                "The upper bound of the estimated eigenvalue spectrum of the Overlap "
                "matrix (S)."
            ),
            "SIESTA keyword": "CheSS.evhighS",
        },
    )

    # ----------------------------------------
    # 6.15 The PEXSI solver (native interface)
    # 6.15.1 Pole handling
    # ----------------------------------------
    # pexsi_num_poles: int = 40 # PEXSI.NumPoles 40
    # pexsi_delta_e: float = 3.0  # PEXSI.deltaE 3 Ry
    # pexsi_gap: float = 0.0 # PEXSI.Gap 0 Ry
    pexsi_num_poles: int = field(
        default=40,
        metadata={
            "description": (
                "Sets the number of poles to be used in the PEXSI pole expansion of "
                "the Fermi-Dirac function."
            ),
            "SIESTA keyword": "PEXSI.NumPoles",
        },
    )

    pexsi_delta_e: float = field(
        default=3.0,
        metadata={
            "description": (
                "The energy range (in Rydberg) around the chemical potential that is "
                "covered by the PEXSI pole expansion."
            ),
            "SIESTA keyword": "PEXSI.deltaE",
        },
    )

    pexsi_gap: float = field(
        default=0.0,
        metadata={
            "description": (
                "The estimated electronic band gap (in Rydberg) of the system, used to "
                "optimize the PEXSI algorithm."
            ),
            "SIESTA keyword": "PEXSI.Gap",
        },
    )

    # -----------------------------------------------
    # 6.15.2 Parallel environment and control options
    # -----------------------------------------------
    # mpi_n_proc_siesta: int = None # MPI.Nprocs.SIESTA 〈total processors〉
    # pexsi_np_per_pole: int = 4 # PEXSI.NP-per-pole 4
    # pexsi_ordering: int = 1 # PEXSI.Ordering 1
    # pexsi_np_symbfact: int = 1 # PEXSI.NP-symbfact 1
    # pexsi_verbosity: int = 1 # PEXSI.Verbosity 1
    mpi_n_proc_siesta: int = field(
        default=None,
        metadata={
            "description": (
                "Manually specifies the total number of MPI processors to be used by "
                "SIESTA. Defaults to all available processors if not set."
            ),
            "SIESTA keyword": "MPI.Nprocs.SIESTA",
        },
    )

    pexsi_np_per_pole: int = field(
        default=4,
        metadata={
            "description": (
                "The number of processors (MPI tasks) assigned to work on each pole in "
                "the PEXSI expansion."
            ),
            "SIESTA keyword": "PEXSI.NP-per-pole",
        },
    )

    pexsi_ordering: int = field(
        default=1,
        metadata={
            "description": (
                "Selects the matrix reordering algorithm used within PEXSI to improve "
                "sparsity and performance."
            ),
            "SIESTA keyword": "PEXSI.Ordering",
        },
    )

    pexsi_np_symbfact: int = field(
        default=1,
        metadata={
            "description": (
                "The number of processors assigned to perform the symbolic "
                "factorization step in the PEXSI algorithm."
            ),
            "SIESTA keyword": "PEXSI.NP-symbfact",
        },
    )

    pexsi_verbosity: int = field(
        default=1,
        metadata={
            "description": (
                "Sets the verbosity level for the PEXSI solver's output. Higher values "
                "produce more detailed logging."
            ),
            "SIESTA keyword": "PEXSI.Verbosity",
        },
    )

    # ----------------------------------------------
    # 6.15.3 Electron tolerance and the PEXSI solver
    # ----------------------------------------------
    # pexsi_num_electron_tolerance: float = 1e-4 # PEXSI.num-electron-tolerance 10-4
    # pexsi_num_electron_tolerance_lower_bound: float = 1e-2
    #   # PEXSI.num-electron-tolerance-lower-bound 10-2
    # pexsi_num_electron_tolerance_upper_bound: float = 0.5
    #   # PEXSI.num-electron-tolerance-upper-bound 0.5
    # pexsi_mu_max_iter: int = 10 # PEXSI.mu-max-iter 10
    # pexsi_mu: float = -0.6 # PEXSI.mu -0.6 Ry
    # pexsi_mu_pexsi_safeguard: float = 0.05 # PEXSI.mu-pexsi-safeguard 0.05 Ry
    pexsi_num_electron_tolerance: float = field(
        default=1e-4,
        metadata={
            "description": (
                "The tolerance for the difference between the calculated and true "
                "number of electrons during the chemical potential search."
            ),
            "SIESTA keyword": "PEXSI.num-electron-tolerance",
        },
    )

    pexsi_num_electron_tolerance_lower_bound: float = field(
        default=1e-2,
        metadata={
            "description": (
                "The lower bound for the electron number tolerance in the chemical "
                "potential search."
            ),
            "SIESTA keyword": "PEXSI.num-electron-tolerance-lower-bound",
        },
    )

    pexsi_num_electron_tolerance_upper_bound: float = field(
        default=0.5,
        metadata={
            "description": (
                "The upper bound for the electron number tolerance in the chemical "
                "potential search."
            ),
            "SIESTA keyword": "PEXSI.num-electron-tolerance-upper-bound",
        },
    )

    pexsi_mu_max_iter: int = field(
        default=10,
        metadata={
            "description": (
                "The maximum number of iterations for the chemical potential (mu) "
                "search algorithm."
            ),
            "SIESTA keyword": "PEXSI.mu-max-iter",
        },
    )

    pexsi_mu: float = field(
        default=-0.6,
        metadata={
            "description": (
                "An initial guess for the chemical potential (mu) in Rydberg."
            ),
            "SIESTA keyword": "PEXSI.mu",
        },
    )

    pexsi_mu_pexsi_safeguard: float = field(
        default=0.05,
        metadata={
            "description": (
                "A safeguard or buffer energy (in Rydberg) around the chemical "
                "potential to ensure stability in the PEXSI algorithm."
            ),
            "SIESTA keyword": "PEXSI.mu-pexsi-safeguard",
        },
    )

    # -----------------------
    # 6.15.4 Inertia-counting
    # -----------------------
    # pexsi_inertia_counts: int = 3 # PEXSI.Inertia-Counts 3
    # pexsi_mu_min: float = 1.0 # PEXSI.mu-min -1 Ry
    # pexsi_mu_max: float = 0.0 # PEXSI.mu-max 0 Ry
    # pexsi_safe_ddmax_no_inertia: float = 0.05 # PEXSI.safe-dDmax-no-inertia 0.05
    # pexsi_lateral_expansion_inertia: float = 3.0
    #   # PEXSI.lateral-expansion-inertia 3 eV
    # pexsi_inertia_mu_tolerance: float = 0.05 # PEXSI.Inertia-mu-tolerance 0.05 Ry
    # pexsi_inertia_max_iter: int = 5 # PEXSI.Inertia-max-iter 5
    # pexsi_inertia_min_num_shifts: int = 10 # PEXSI.Inertia-min-num-shifts 10
    # pexsi_inertia_energy_width_tolerance: float = None
    #   # PEXSI.Inertia-energy-width-tolerance 〈PEXSI.Inertia-mu-tolerance〉
    pexsi_inertia_counts: int = field(
        default=3,
        metadata={
            "description": (
                "A parameter controlling the matrix inertia counting procedure used to "
                "locate the chemical potential."
            ),
            "SIESTA keyword": "PEXSI.Inertia-Counts",
        },
    )

    pexsi_mu_min: float = field(
        default=1.0,
        metadata={
            "description": (
                "The lower bound (in Rydberg) for the chemical potential (mu) search "
                "window."
            ),
            "SIESTA keyword": "PEXSI.mu-min",
        },
    )

    pexsi_mu_max: float = field(
        default=0.0,
        metadata={
            "description": (
                "The upper bound (in Rydberg) for the chemical potential (mu) search "
                "window."
            ),
            "SIESTA keyword": "PEXSI.mu-max",
        },
    )

    pexsi_safe_ddmax_no_inertia: float = field(
        default=0.05,
        metadata={
            "description": (
                "A safeguard parameter related to the maximum allowed change in the "
                "density matrix when inertia counting is disabled."
            ),
            "SIESTA keyword": "PEXSI.safe-dDmax-no-inertia",
        },
    )

    pexsi_lateral_expansion_inertia: float = field(
        default=3.0,
        metadata={
            "description": (
                "An energy value (in eV) that controls the lateral expansion of the "
                "search window during the inertia counting procedure."
            ),
            "SIESTA keyword": "PEXSI.lateral-expansion-inertia",
        },
    )

    pexsi_inertia_mu_tolerance: float = field(
        default=0.05,
        metadata={
            "description": (
                "The convergence tolerance (in Rydberg) for the chemical potential "
                "when using the inertia counting method."
            ),
            "SIESTA keyword": "PEXSI.Inertia-mu-tolerance",
        },
    )

    pexsi_inertia_max_iter: int = field(
        default=5,
        metadata={
            "description": (
                "The maximum number of iterations for the inertia counting procedure "
                "to converge on the chemical potential."
            ),
            "SIESTA keyword": "PEXSI.Inertia-max-iter",
        },
    )

    pexsi_inertia_min_num_shifts: int = field(
        default=10,
        metadata={
            "description": (
                "The minimum number of energy shifts to be used in the inertia "
                "counting algorithm."
            ),
            "SIESTA keyword": "PEXSI.Inertia-min-num-shifts",
        },
    )

    pexsi_inertia_energy_width_tolerance: float = field(
        default=None,
        metadata={
            "description": (
                "A tolerance for the energy width of the search interval during "
                "inertia counting. Defaults to the value of "
                "'PEXSI.Inertia-mu-tolerance'."
            ),
            "SIESTA keyword": "PEXSI.Inertia-energy-width-tolerance",
        },
    )

    # -------------------------------------------------
    # 6.15.5 Reuse of µ information across iterations
    # -------------------------------------------------
    # pexsi_safe_width_ic_bracket: float = 4.0 # PEXSI.safe-width-ic-bracket 4 eV
    # pexsi_safe_ddmax_ef_iniertia: float = 0.1 # PEXSI.safe-dDmax-ef-inertia 0.1
    # pexsi_safe_ddmax_ef_solver: float = 0.05 # PEXSI.safe-dDmax-ef-solver 0.05
    # pexsi_safe_width_solver_bracker: float = 4.0
    #   # PEXSI.safe-width-solver-bracket 4 eV
    pexsi_safe_width_ic_bracket: float = field(
        default=4.0,
        metadata={
            "description": (
                "A safe energy width (in eV) for the bracketing interval used during "
                "the inertia counting (ic) procedure."
            ),
            "SIESTA keyword": "PEXSI.safe-width-ic-bracket",
        },
    )

    pexsi_safe_ddmax_ef_inertia: float = field(
        default=0.1,
        metadata={
            "description": (
                "A safeguard limit for the maximum change in the density matrix "
                "(dDmax) during the inertia-counting stage of the Fermi energy (ef) "
                "search."
            ),
            "SIESTA keyword": "PEXSI.safe-dDmax-ef-inertia",
        },
    )

    pexsi_safe_ddmax_ef_solver: float = field(
        default=0.05,
        metadata={
            "description": (
                "A safeguard limit for the maximum change in the density matrix "
                "(dDmax) during the main solver stage of the Fermi energy (ef) search."
            ),
            "SIESTA keyword": "PEXSI.safe-dDmax-ef-solver",
        },
    )

    pexsi_safe_width_solver_bracket: float = field(
        default=4.0,
        metadata={
            "description": (
                "A safe energy width (in eV) for the bracketing interval used by the "
                "main PEXSI solver."
            ),
            "SIESTA keyword": "PEXSI.safe-width-solver-bracket",
        },
    )
    # ---------------------------------------------------------------
    # 6.15.6 Calculation of the density of states by inertia-counting
    # ---------------------------------------------------------------
    # pexsi_dos: bool = False # PEXSI.DOS false
    # pexsi_dos_emin: float = -1.0 # PEXSI.DOS.Emin -1 Ry
    # pexsi_dos_emax: float = 1.0 # PEXSI.DOS.Emax 1 Ry
    # pexsi_dos_ef_reference: bool = True # PEXSI.DOS.Ef.Reference true
    # pexsi_dos_n_points: int = 200 # PEXSI.DOS.NPoints 200
    pexsi_dos: bool = field(
        default=False,
        metadata={
            "description": (
                "A master flag to enable the calculation of the Density of States "
                "(DOS) using the PEXSI solver."
            ),
            "SIESTA keyword": "PEXSI.DOS",
        },
    )

    pexsi_dos_emin: float = field(
        default=-1.0,
        metadata={
            "description": (
                "The minimum energy (in Rydberg) for the Density of States calculation "
                "window."
            ),
            "SIESTA keyword": "PEXSI.DOS.Emin",
        },
    )

    pexsi_dos_emax: float = field(
        default=1.0,
        metadata={
            "description": (
                "The maximum energy (in Rydberg) for the Density of States calculation "
                "window."
            ),
            "SIESTA keyword": "PEXSI.DOS.Emax",
        },
    )

    pexsi_dos_ef_reference: bool = field(
        default=True,
        metadata={
            "description": (
                "If true, the energy range for the DOS calculation (Emin, Emax) is set "
                "relative to the calculated Fermi level."
            ),
            "SIESTA keyword": "PEXSI.DOS.Ef.Reference",
        },
    )

    pexsi_dos_n_points: int = field(
        default=200,
        metadata={
            "description": (
                "The number of energy points to be calculated within the specified DOS "
                "window."
            ),
            "SIESTA keyword": "PEXSI.DOS.NPoints",
        },
    )

    # ----------------------------------------------------
    # 6.15.7 Calculation of the LDOS by selected-inversion
    # ----------------------------------------------------
    # pexsi_ldso: bool = False # PEXSI.LDOS false
    # pexsi_ldos_energy: float = 0.0 # PEXSI.LDOS.Energy 0 Ry
    # pexsi_ldos_broadening: float = 0.01 # PEXSI.LDOS.Broadening 0.01 Ry
    # pexsi_ldos_np_per_pole: int = None # PEXSI.LDOS.NP-per-pole 〈PEXSI.NP-per-pole〉
    pexsi_ldos: bool = field(
        default=False,
        metadata={
            "description": (
                "A master flag to enable the calculation of the Local Density of "
                "States (LDOS) using the PEXSI solver."
            ),
            "SIESTA keyword": "PEXSI.LDOS",
        },
    )

    pexsi_ldos_energy: float = field(
        default=0.0,
        metadata={
            "description": (
                "The energy (in Rydberg) at which the Local Density of States will be "
                "computed."
            ),
            "SIESTA keyword": "PEXSI.LDOS.Energy",
        },
    )

    pexsi_ldos_broadening: float = field(
        default=0.01,
        metadata={
            "description": (
                "The energy broadening (in Rydberg) applied to the LDOS calculation."
            ),
            "SIESTA keyword": "PEXSI.LDOS.Broadening",
        },
    )

    pexsi_ldos_np_per_pole: int = field(
        default=None,
        metadata={
            "description": (
                "The number of processors per pole for the LDOS calculation. Defaults "
                "to the value of 'PEXSI.NP-per-pole' if not specified."
            ),
            "SIESTA keyword": "PEXSI.LDOS.NP-per-pole",
        },
    )

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            # Register all 73 ELSI, CheSS, and PEXSI solver parameters
            self.register_fdf_params(
                # ELSI parameters
                "ELSI.Broadening.Method",
                "ELSI.Output.Level",
                "ELSI.Output.Json",
                "ELSI.Broadening.MPOrder",
                "ELSI.Ill-Condition.Check",
                "ELSI.Ill-Condition.Tolerance",
                "ELSI.ELPA.Flavor",
                "ELSI.ELPA.NSinglePrecision",
                "ELSI.ELPA.Autotune",
                "ELSI.ELPA.GPU",
                "ELSI.OMM.Flavor",
                "ELSI.OMM.ELPA.Steps",
                "ELSI.OMM.Tolerance",
                "ELSI.PEXSI.Method",
                "ELSI.PEXSI.TasksPerPole",
                "ELSI.PEXSI.TasksSymbolic",
                "ELSI.PEXSI.Number-of-Poles",
                "ELSI.PEXSI.Number-of-Mu-Points",
                "ELSI.PEXSI.Inertia-Tolerance",
                "ELSI.PEXSI.Initial-Mu-Min",
                "ELSI.PEXSI.Initial-Mu-Max",
                "ELSI.NTPoly.Method",
                "ELSI.NTPoly.Filter",
                "ELSI.NTPoly.Tolerance",
                "ELSI.SIPS.Slices",
                "ELSI.SIPS.ELPA.Steps",
                "ELSI.EigenExa.Method",
                "ELSI.MAGMA.Solver-Method",
                # CheSS parameters
                "CheSS.Buffer.Kernel",
                "CheSS.Buffer.Mult",
                "CheSS.Fscale",
                "CheSS.FscaleLowerbound",
                "CheSS.FscaleUpperbound",
                "CheSS.evlowH",
                "CheSS.evhighH",
                "CheSS.evlowS",
                "CheSS.evhighS",
                # PEXSI parameters
                "PEXSI.NumPoles",
                "PEXSI.deltaE",
                "PEXSI.Gap",
                "MPI.Nprocs.SIESTA",
                "PEXSI.NP-per-pole",
                "PEXSI.Ordering",
                "PEXSI.NP-symbfact",
                "PEXSI.Verbosity",
                "PEXSI.num-electron-tolerance",
                "PEXSI.num-electron-tolerance-lower-bound",
                "PEXSI.num-electron-tolerance-upper-bound",
                "PEXSI.mu-max-iter",
                "PEXSI.mu",
                "PEXSI.mu-pexsi-safeguard",
                "PEXSI.Inertia-Counts",
                "PEXSI.mu-min",
                "PEXSI.mu-max",
                "PEXSI.safe-dDmax-no-inertia",
                "PEXSI.lateral-expansion-inertia",
                "PEXSI.Inertia-mu-tolerance",
                "PEXSI.Inertia-max-iter",
                "PEXSI.Inertia-min-num-shifts",
                "PEXSI.Inertia-energy-width-tolerance",
                "PEXSI.safe-width-ic-bracket",
                "PEXSI.safe-dDmax-ef-inertia",
                "PEXSI.safe-dDmax-ef-solver",
                "PEXSI.safe-width-solver-bracket",
                "PEXSI.DOS",
                "PEXSI.DOS.Emin",
                "PEXSI.DOS.Emax",
                "PEXSI.DOS.Ef.Reference",
                "PEXSI.DOS.NPoints",
                "PEXSI.LDOS",
                "PEXSI.LDOS.Energy",
                "PEXSI.LDOS.Broadening",
                "PEXSI.LDOS.NP-per-pole",
            )
            self.__class__._registered = True  # noqa: SLF001 class-level registration guard

    def validate(self) -> None:
        """Validate the solver and performance options."""
        logger.info("SolversAndPerformanceOptions.validate()")
        allowed_elsi_solver = [
            "ELPA",
            "OMM",
            "PEXSI",
            "NTPOLY",
            "SIPS",
            "EIGENEXA",
            "MAGMA",
        ]
        if self.elsi_solver not in allowed_elsi_solver:
            raise ValueError(
                f"Invalid solver type '{self.elsi_solver}'. "
                f"Allowed values are: {allowed_elsi_solver}"
            )

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)

        Note:
            Due to the large number of parameters (73), this implementation
            uses a simplified approach. Full parameter-by-parameter mapping
            can be added as needed for specific use cases.
        """
        # This is a simplified implementation for the 73 solver parameters.
        # Parameters are typically set programmatically rather than from user FDF.

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters

        Note:
            Due to the large number of parameters (73), this returns an empty dict.
            Solver parameters are typically written by specialized solver setup methods.
            Full FDF generation can be added as needed for specific solver
            configurations.
        """
        # Solver parameters are typically handled by specialized configuration methods
        # rather than direct FDF output. Return empty dict for base implementation.
        return {}

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have advanced solver parameters
        # These are SIESTA-specific performance/solver options
        return {}

    def generate_solver_block(self) -> None:
        """Generate the solver and performance options block for the FDF file."""
        logger.info("SolversAndPerformanceOptions.generate_solver_block()")

    @classmethod
    def setup_solver_settings(
        cls, user_params: dict[str, Any] | None = None
    ) -> "SolversAndPerformanceOptions":
        """
        Create and configure a SolversAndPerformanceOptions instance.

        TODO: Implement full parameter parsing like other setup methods.
        Currently returns instance with default values. Users can still configure
        via user_params in the parent input generator.

        Args:
            user_params: Dictionary of user-defined parameters (optional)

        Returns
        -------
            SolversAndPerformanceOptions: Configured instance
        """
        logger.info("SolversAndPerformanceOptions.setup_solver_settings()")
        instance = cls()

        # Simple parameter assignment if provided
        if user_params:
            for key, value in user_params.items():
                key_normalized = key.lower().replace(".", "_")
                if hasattr(instance, key_normalized):
                    setattr(instance, key_normalized, value)

        return instance
