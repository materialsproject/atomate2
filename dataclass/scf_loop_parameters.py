"""
Module defining base SIESTA input set and generator.

class SCFLoopParameters

Based on User's Guide Siesta 5.4.0
Section: 6.9 The self-consistent field loop
"""

# Metadata

__all__ = ["SCFLoopParameters"]

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.dataclass.units import parse_energy

import logging

logger = logging.getLogger(__name__)


@dataclass
class SCFLoopParameters(FDFDataclass):
    """
    Data class to manage self-consistent field (SCF) loop parameters for SIESTA input.
    """

    # ----------------------------------
    # 6.9 The self-consistent-field loop
    # ----------------------------------
    # mix_scf_iterations: int = 0     #   MinSCFIterations
    # max_scf_iterations: int = 200   #  MaxSCFIterations  Maximum number of SCF iterations
    # scf_must_converge: bool = True  # SCF.MustConverge
    mix_scf_iterations: int = field(
        default=0,
        metadata={
            "description": "The minimum number of SCF iterations to be performed, even if convergence is achieved earlier.",
            "SIESTA keyword": "MinSCFIterations",
        },
    )

    max_scf_iterations: int = field(
        default=200,
        metadata={
            "description": "The maximum number of allowed iterations in the self-consistent field (SCF) cycle.",
            "SIESTA keyword": "MaxSCFIterations",
        },
    )

    scf_must_converge: bool = field(
        default=True,
        metadata={
            "description": "If true, the program will stop with an error if the SCF cycle does not converge within the maximum number of iterations.",
            "SIESTA keyword": "SCF.MustConverge",
        },
    )
    # -----------------------
    # 6.9.1 Harris functional
    # -----------------------
    # harris_functional: bool = False # Harris.Functional
    harris_functional: bool = field(
        default=False,
        metadata={
            "description": "If true, performs a single-iteration, non-self-consistent calculation using the Harris functional approximation for the total energy.",
            "SIESTA keyword": "Harris.Functional",
        },
    )

    # --------------------
    # 6.9.2 Mixing options
    # --------------------
    # scf_mix: str = "Hamiltonian"  # SCF.Mix Hamiltonian|density|charge
    # scf_mix_spin: str = " " # SCF.Mix.Spin all|spinor|sum|sum+diff
    # scf_mix_first: bool = True  #SCF.Mix.First
    # scf_mix_first_force: bool = False #SCF.Mix.First.Force
    # scf_mixer_method: str = " " # SCF.Mixer.Method Pulay|Broyden|Linear
    # scf_mixer_variant: str = "Pulay"  # SCF.Mixer.Variant  Mixing scheme for the SCF loop (e.g., 'original' 'Pulay', 'Simple')
    # scf_mixer_weight: float = 0.25  # SCF.Mixer.Weight Mixing parameter (typically between 0 and 1)
    # scf_mixer_history: int = 2 # SCF.Mixer.History 2
    # scf_mixer_kick: int = 0 # SCF.Mixer.Kick 0
    # scf_mixer_kick_weight: float = 0.2  # SCF.Mixer.Kick.Weight 〈SCF.Mixer.Weight〉
    # scf_mixer_restart: int = 0 # SCF.Mixer.Restart
    # scf_mixer_restart_save: int = 1 # SCF.Mixer.Restart.Save
    # scf_mixer_linear_after: int = -1 #SCF.Mixer.Linear.After
    # scf_mixer_linear_after_weight: float = 0.0 # SCF.Mixer.Linear.After.Weight 〈SCF.Mixer.Weight〉
    # scf_mixers_block:  Dict[str, float] = field(default_factory=dict) # %block SCF.Mixers 〈None〉
    # compat_pre_v4_dm_h: bool = False  #Compat.Pre-v4-DM-H false
    # scf_mix_after_convergence: bool = False # SCF.Mix.AfterConvergence false

    scf_mix: str = field(
        default="Hamiltonian",
        metadata={
            "description": "Specifies which quantity is mixed during the SCF cycle. Options: 'Hamiltonian', 'density', 'charge'.",
            "SIESTA keyword": "SCF.Mix",
        },
    )

    scf_mix_spin: str = field(
        default=" ",
        metadata={
            "description": "In spin-polarized calculations, specifies how spin components are mixed. Options: 'all', 'spinor', 'sum', 'sum+diff'.",
            "SIESTA keyword": "SCF.Mix.Spin",
        },
    )

    scf_mix_first: bool = field(
        default=True,
        metadata={
            "description": "A flag to control whether mixing is applied from the very first SCF iteration.",
            "SIESTA keyword": "SCF.Mix.First",
        },
    )

    scf_mix_first_force: bool = field(
        default=False,
        metadata={
            "description": "A flag to force mixing on the first iteration, even when starting from a converged density matrix from a previous run.",
            "SIESTA keyword": "SCF.Mix.First.Force",
        },
    )

    scf_mixer_method: str = field(
        default=" ",
        metadata={
            "description": "Selects the primary algorithm for charge mixing. Options: 'Pulay', 'Broyden', 'Linear'.",
            "SIESTA keyword": "SCF.Mixer.Method",
        },
    )

    scf_mixer_variant: str = field(
        default="Pulay",
        metadata={
            "description": "Specifies a particular variant or implementation of the chosen mixer method (e.g., 'original', 'Pulay', 'Simple').",
            "SIESTA keyword": "SCF.Mixer.Variant",
        },
    )

    scf_mixer_weight: float = field(
        default=0.25,
        metadata={
            "description": "The linear mixing parameter, controlling the weight of the new density/Hamiltonian mixed with the previous one.",
            "SIESTA keyword": "SCF.Mixer.Weight",
        },
    )

    scf_mixer_history: int = field(
        default=2,
        metadata={
            "description": "The number of previous SCF steps to use in history-based mixing schemes like Pulay or Broyden.",
            "SIESTA keyword": "SCF.Mixer.History",
        },
    )

    scf_mixer_kick: int = field(
        default=0,
        metadata={
            "description": "The number of initial iterations to apply a special 'kick' to the density to move away from local minima.",
            "SIESTA keyword": "SCF.Mixer.Kick",
        },
    )

    scf_mixer_kick_weight: float = field(
        default=0.2,
        metadata={
            "description": "The mixing weight to be used during the initial 'kick' iterations. Defaults to SCF.Mixer.Weight if not set.",
            "SIESTA keyword": "SCF.Mixer.Kick.Weight",
        },
    )

    scf_mixer_restart: int = field(
        default=0,
        metadata={
            "description": "The number of iterations after which the mixer's history is reset.",
            "SIESTA keyword": "SCF.Mixer.Restart",
        },
    )

    scf_mixer_restart_save: int = field(
        default=1,
        metadata={
            "description": "The number of previous steps to save and reuse after a mixer restart.",
            "SIESTA keyword": "SCF.Mixer.Restart.Save",
        },
    )

    scf_mixer_linear_after: int = field(
        default=-1,
        metadata={
            "description": "The number of iterations after which the mixer should switch to simple linear mixing. A value of -1 disables this.",
            "SIESTA keyword": "SCF.Mixer.Linear.After",
        },
    )

    scf_mixer_linear_after_weight: float = field(
        default=0.0,
        metadata={
            "description": "The mixing weight to use after switching to linear mixing. Defaults to SCF.Mixer.Weight if not set.",
            "SIESTA keyword": "SCF.Mixer.Linear.After.Weight",
        },
    )

    scf_mixers_block: Dict[str, float] = field(
        default_factory=dict,
        metadata={
            "description": "A block to define a sequence of different mixers to be used at different stages of the SCF cycle.",
            "SIESTA keyword": "%block SCF.Mixers",
        },
    )

    compat_pre_v4_dm_h: bool = field(
        default=False,
        metadata={
            "description": "A compatibility flag to reproduce the behavior of the DM/H mixing from SIESTA versions before 4.0.",
            "SIESTA keyword": "Compat.Pre-v4-DM-H",
        },
    )

    scf_mix_after_convergence: bool = field(
        default=False,
        metadata={
            "description": "A flag to perform one final mixing step after the SCF cycle has already converged.",
            "SIESTA keyword": "SCF.Mix.AfterConvergence",
        },
    )

    # ----------------------------------
    # 6.9.3 Mixing of the Charge Density
    # ----------------------------------
    # scf_kerker_q0sq: float = 0.0 # SCF.Kerker.q0sq 0 Ry
    # scf_rho_g_mixing_cutoff: float = 9 # SCF.RhoGMixingCutoff 9 Ry
    # scf_rho_g_diis_depth: int = 0 # SCF.RhoG.DIIS.Depth 0
    # scf_rho_g_metric_preconditioner_cutoff: float = None # SCF.RhoG.Metric.Preconditioner.Cutoff 〈None〉
    # scf_debug_rho_g_mixing: bool = False #SCF.DebugRhoGMixing false
    # debug_diis: bool = False # Debug.DIIS false
    # scf_mix_charge_scf1: bool = False # SCF.MixCharge.SCF1 false

    scf_kerker_q0sq: float = field(
        default=0.0,
        metadata={
            "description": "The squared wave-vector parameter (q0^2) for Kerker preconditioning, used to improve SCF convergence by damping charge fluctuations.",
            "SIESTA keyword": "SCF.Kerker.q0sq",
            "unit": "Ry",
        },
    )

    scf_rho_g_mixing_cutoff: float = field(
        default=9.0,
        metadata={
            "description": "A reciprocal-space cutoff (in Rydberg) for real-space density (Rho(G)) mixing schemes.",
            "SIESTA keyword": "SCF.RhoGMixingCutoff",
            "unit": "Ry",
        },
    )

    scf_rho_g_diis_depth: int = field(
        default=0,
        metadata={
            "description": "The history depth for the DIIS (Direct Inversion in the Iterative Subspace) mixer when using real-space density mixing.",
            "SIESTA keyword": "SCF.RhoG.DIIS.Depth",
        },
    )

    scf_rho_g_metric_preconditioner_cutoff: float = field(
        default=None,
        metadata={
            "description": "A cutoff for the preconditioner in the metric used for real-space density mixing.",
            "SIESTA keyword": "SCF.RhoG.Metric.Preconditioner.Cutoff",
        },
    )

    scf_debug_rho_g_mixing: bool = field(
        default=False,
        metadata={
            "description": "A debugging flag to print detailed information about the real-space density (Rho(G)) mixing process.",
            "SIESTA keyword": "SCF.DebugRhoGMixing",
        },
    )

    debug_diis: bool = field(
        default=False,
        metadata={
            "description": "A general debugging flag to print detailed information from any DIIS (Direct Inversion in the Iterative Subspace) algorithm used.",
            "SIESTA keyword": "Debug.DIIS",
        },
    )

    scf_mix_charge_scf1: bool = field(
        default=False,
        metadata={
            "description": "A flag to enforce charge mixing on the very first SCF iteration, even if the primary mixing scheme is different.",
            "SIESTA keyword": "SCF.MixCharge.SCF1",
        },
    )

    # ------------------------------------------
    # 6.9.4 Initialization of the density-matrix
    # ------------------------------------------
    # dm_use_save_dm: bool = True  #DM.UseSaveDM true
    # dm_init_unfold: bool = True # DM.Init.Unfold true
    # dm_formatted_files: bool = True  # DM.FormattedFiles false
    # dm_formatted_input: bool = True # DM.FormattedInput false
    # dm_formatted_output: bool = False # DM.FormattedOutput false
    # dm_init_random_states: int = 0 # DM.Init.RandomStates 0
    # dm_allow_reuse: bool = True # DM.AllowReuse true
    # dm_allow_extrapolation: bool = True # DM.AllowExtrapolation true
    # dm_history_depth: int = 1 #DM.History.Depth 1

    dm_use_save_dm: bool = field(
        default=True,
        metadata={
            "description": "A flag to enable reading a previously saved Density Matrix from a file to use as the initial guess for the SCF cycle.",
            "SIESTA keyword": "DM.UseSaveDM",
        },
    )

    dm_init_unfold: bool = field(
        default=True,
        metadata={
            "description": "If true, allows unfolding a Density Matrix from a smaller unit cell calculation to initialize a larger supercell calculation.",
            "SIESTA keyword": "DM.Init.Unfold",
        },
    )

    dm_formatted_files: bool = field(
        default=True,
        metadata={
            "description": "A global flag to use formatted (human-readable) files for all Density Matrix I/O. Can be overridden by specific input/output flags.",
            "SIESTA keyword": "DM.FormattedFiles",
        },
    )

    dm_formatted_input: bool = field(
        default=True,
        metadata={
            "description": "Specifies that the input Density Matrix file to be read is in a formatted (ASCII, human-readable) format.",
            "SIESTA keyword": "DM.FormattedInput",
        },
    )

    dm_formatted_output: bool = field(
        default=False,
        metadata={
            "description": "Specifies that the output Density Matrix file should be written in a formatted (ASCII, human-readable) format.",
            "SIESTA keyword": "DM.FormattedOutput",
        },
    )

    dm_init_random_states: int = field(
        default=0,
        metadata={
            "description": "The number of random states to introduce during the initialization of the Density Matrix, useful for symmetry breaking.",
            "SIESTA keyword": "DM.Init.RandomStates",
        },
    )

    dm_allow_reuse: bool = field(
        default=True,
        metadata={
            "description": "During a geometry optimization or MD run, allows reusing the converged DM from the previous step as the initial guess for the current step.",
            "SIESTA keyword": "DM.AllowReuse",
        },
    )

    dm_allow_extrapolation: bool = field(
        default=True,
        metadata={
            "description": "During a geometry optimization or MD run, allows extrapolating the DM from previous steps to provide a better initial guess for the current step.",
            "SIESTA keyword": "DM.AllowExtrapolation",
        },
    )

    dm_history_depth: int = field(
        default=1,
        metadata={
            "description": "The number of previous geometry steps to use in the history for Density Matrix extrapolation schemes.",
            "SIESTA keyword": "DM.History.Depth",
        },
    )

    # -----------------------------------------------------------
    # 6.9.5 Initialization of the SCF cycle with charge densities
    # -----------------------------------------------------------
    # scf_read_charge_netcdf: bool = False # SCF.Read.Charge.NetCDF
    # scf_read_deformation_charge_netcdf: bool = False # SCF.Read.Deformation.Charge.NetCDF false

    scf_read_charge_netcdf: bool = field(
        default=False,
        metadata={
            "description": "A flag to read the initial total charge density from a file in NetCDF format.",
            "SIESTA keyword": "SCF.Read.Charge.NetCDF",
        },
    )

    scf_read_deformation_charge_netcdf: bool = field(
        default=False,
        metadata={
            "description": "A flag to read the initial deformation charge density from a file in NetCDF format.",
            "SIESTA keyword": "SCF.Read.Deformation.Charge.NetCDF",
        },
    )
    # ----------------------------------------------
    # 6.9.6 Output of density matrix and Hamiltonian
    # ----------------------------------------------
    # use_blocked_write_mat: bool =  False # Use.Blocked.WriteMat
    # write_dm: bool = True # Write.DM true
    # write_dm_end_of_cycle: bool = None  #Write.DM.end.of.cycle 〈Write.DM〉
    # write_h: bool = False # Write.H false
    # write_h_end_of_cycle: bool = None # Write.H.end.of.cycle 〈Write.H〉
    # write_dm_netcdf: bool = True # Write.DM.NetCDF true
    # write_dmhs_netcdf: bool = True # Write.DMHS.NetCDF true
    # write_dm_history_netcdf: bool = False  # Write.DM.History.NetCDF false
    # write_dmhs_history_netcdf: bool = False  # Write.DMHS.History.NetCDF false
    # write_tshs_history: bool = False # Write.TSHS.History false
    use_blocked_write_mat: bool = field(
        default=False,
        metadata={
            "description": "A technical flag to use a blocked I/O algorithm when writing matrices, which can be more efficient for very large systems.",
            "SIESTA keyword": "Use.Blocked.WriteMat",
        },
    )

    write_dm: bool = field(
        default=True,
        metadata={
            "description": "A flag to enable writing the converged Density Matrix (DM) to a file at the end of the calculation.",
            "SIESTA keyword": "Write.DM",
        },
    )

    write_dm_end_of_cycle: bool = field(
        default=None,
        metadata={
            "description": "If true, writes the Density Matrix at the end of every SCF cycle. By default, it follows the 'Write.DM' setting.",
            "SIESTA keyword": "Write.DM.end.of.cycle",
        },
    )

    write_h: bool = field(
        default=False,
        metadata={
            "description": "A flag to enable writing the converged Hamiltonian matrix (H) to a file.",
            "SIESTA keyword": "Write.H",
        },
    )

    write_h_end_of_cycle: bool = field(
        default=None,
        metadata={
            "description": "If true, writes the Hamiltonian matrix at the end of every SCF cycle. By default, it follows the 'Write.H' setting.",
            "SIESTA keyword": "Write.H.end.of.cycle",
        },
    )

    write_dm_netcdf: bool = field(
        default=True,
        metadata={
            "description": "A flag to enable writing the Density Matrix in the portable NetCDF format.",
            "SIESTA keyword": "Write.DM.NetCDF",
        },
    )

    write_dmhs_netcdf: bool = field(
        default=True,
        metadata={
            "description": "A flag to enable writing the Density Matrix, Hamiltonian, and Overlap matrix (DMHS) into a single NetCDF file.",
            "SIESTA keyword": "Write.DMHS.NetCDF",
        },
    )

    write_dm_history_netcdf: bool = field(
        default=False,
        metadata={
            "description": "During a geometry optimization or MD run, writes the history of Density Matrices from previous steps to a NetCDF file.",
            "SIESTA keyword": "Write.DM.History.NetCDF",
        },
    )

    write_dmhs_history_netcdf: bool = field(
        default=False,
        metadata={
            "description": "During a geometry optimization or MD run, writes the history of DM, H, and S matrices from previous steps to a NetCDF file.",
            "SIESTA keyword": "Write.DMHS.History.NetCDF",
        },
    )

    write_tshs_history: bool = field(
        default=False,
        metadata={
            "description": "In a TranSIESTA calculation, writes the history of the Hamiltonian and Self-energy matrices for the electrodes.",
            "SIESTA keyword": "Write.TSHS.History",
        },
    )

    # --------------------------
    # 6.9.7 Convergence criteria
    # --------------------------
    # scf_dm_converge: bool = True # SCF.DM.Converge true
    # scf_dm_tolerance: float = 1e-5  # SCF.DM.Tolerance 10−4  Convergence threshold for the SCF loop
    # dm_normalization_tolerance: float = 1e-5 # DM.Normalization.Tolerance 10−5
    # scf_h_converge: bool = True # SCF.H.Converge true
    # scf_h_tolerance: float = 1e-3 # SCF.H.Tolerance 10−3 eV
    # scf_edm_converge: bool = True # SCF.EDM.Converge true
    # scf_edm_tolerance: float = 1e-3 # SCF.EDM.Tolerance 10−3 eV
    # scf_free_e_converge: bool = False # SCF.FreeE.Converge false
    # scf_free_e_tolerance: float = 1e-4 # SCF.FreeE.Tolerance 10−4 eV
    # scf_harris_converge: bool = False # SCF.Harris.Converge false
    # scf_harris_tolerance: float = 1e-4 # SCF.Harris.Tolerance 10−4 eV
    scf_dm_converge: bool = field(
        default=True,
        metadata={
            "description": "A flag to enable convergence checking based on the change in the Density Matrix (DM).",
            "SIESTA keyword": "SCF.DM.Converge",
        },
    )

    scf_dm_tolerance: float = field(
        default=1e-5,
        metadata={
            "description": "The convergence threshold for the maximum absolute difference between elements of the Density Matrix in consecutive SCF steps.",
            "SIESTA keyword": "SCF.DM.Tolerance",
        },
    )

    dm_normalization_tolerance: float = field(
        default=1e-5,
        metadata={
            "description": "The tolerance for the deviation of the Density Matrix trace from the total number of electrons.",
            "SIESTA keyword": "DM.Normalization.Tolerance",
        },
    )

    scf_h_converge: bool = field(
        default=True,
        metadata={
            "description": "A flag to enable convergence checking based on the change in the Hamiltonian matrix (H).",
            "SIESTA keyword": "SCF.H.Converge",
        },
    )

    scf_h_tolerance: float = field(
        default=1e-3,
        metadata={
            "description": "The convergence threshold (in eV) for the maximum difference between Hamiltonian matrix elements in consecutive SCF steps.",
            "SIESTA keyword": "SCF.H.Tolerance",
            "unit": "eV",
        },
    )

    scf_edm_converge: bool = field(
        default=True,
        metadata={
            "description": "A flag to enable convergence checking based on the change in the total energy calculated from the previous step's DM.",
            "SIESTA keyword": "SCF.EDM.Converge",
        },
    )

    scf_edm_tolerance: float = field(
        default=1e-3,
        metadata={
            "description": "The convergence threshold (in eV) for the change in the total energy between SCF steps.",
            "SIESTA keyword": "SCF.EDM.Tolerance",
            "unit": "eV",
        },
    )

    scf_free_e_converge: bool = field(
        default=False,
        metadata={
            "description": "A flag to enable convergence checking based on the change in the free energy (for finite electronic temperatures).",
            "SIESTA keyword": "SCF.FreeE.Converge",
        },
    )

    scf_free_e_tolerance: float = field(
        default=1e-4,
        metadata={
            "description": "The convergence threshold (in eV) for the change in free energy between SCF steps.",
            "SIESTA keyword": "SCF.FreeE.Tolerance",
            "unit": "eV",
        },
    )

    scf_harris_converge: bool = field(
        default=False,
        metadata={
            "description": "A flag to enable convergence checking based on the change in the Harris functional energy.",
            "SIESTA keyword": "SCF.Harris.Converge",
        },
    )

    scf_harris_tolerance: float = field(
        default=1e-4,
        metadata={
            "description": "The convergence threshold (in eV) for the change in the Harris energy between SCF steps.",
            "SIESTA keyword": "SCF.Harris.Tolerance",
            "unit": "eV",
        },
    )

    scf_fdf_arguments: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "A dictionary for any additional or arbitrary FDF flags related to SCF loop. This allows for using keywords not explicitly defined elsewhere.",
            "SIESTA keyword": None,
        },
    )

    comments: str = field(
        default="SCFLoopParameters",
        metadata={
            "description": "User-provided comments to be included as a comment block in the FDF file.",
            "SIESTA keyword": None,
        },
    )

    def __post_init__(self):
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "MinSCFIterations",
                "MaxSCFIterations",
                "SCF.MustConverge",
                "Harris.Functional",
                "SCF.Mix",
                "SCF.Mix.Spin",
                "SCF.Mix.First",
                "SCF.Mix.First.Force",
                "SCF.Mix.AfterConvergence",
                "SCF.MixCharge.SCF1",
                "SCF.Mixer.Method",
                "SCF.Mixer.Variant",
                "SCF.Mixer.Weight",
                "SCF.MixingWeight",  # Alias for SCF.Mixer.Weight
                "SCF.Mixer.History",
                "SCF.WriteExtra",
                "SCF.Write.Extra",
                "SCF.Mixer.Kick",
                "SCF.Mixer.Kick.Weight",
                "SCF.Mixer.Restart",
                "SCF.Mixer.Restart.Save",
                "SCF.Mixer.Linear.After",
                "SCF.Mixer.Linear.After.Weight",
                "%block SCF.Mixers",
                "SCF.DM.Tolerance",
                "DM.Tolerance",  # Legacy alias for SCF.DM.Tolerance (older SIESTA versions)
                "SCF.DM.Converge",
                "SCF.H.Tolerance",
                "SCF.H.Converge",
                "SCF.EDM.Tolerance",
                "SCF.EDM.Converge",
                "SCF.FreeE.Tolerance",
                "SCF.FreeE.Converge",
                "SCF.Harris.Tolerance",
                "SCF.Harris.Converge",
                "DM.UseSaveDM",
                "DM.History.Depth",
                "DM.AllowReuse",
                "DM.AllowExtrapolation",
                "DM.Normalization.Tolerance",
                "DM.FormattedFiles",
                "DM.FormattedInput",
                "DM.FormattedOutput",
                "DM.Init.RandomStates",
                "DM.Init.Unfold",
                "Write.DM",
                "Write.DM.end.of.cycle",
                "Write.DM.NetCDF",
                "Write.DM.History.NetCDF",
                "Write.H",
                "Write.H.end.of.cycle",
                "Write.DMHS.NetCDF",
                "Write.DMHS.History.NetCDF",
                "Write.TSHS.History",
                "SCF.Read.Charge.NetCDF",
                "SCF.Read.Deformation.Charge.NetCDF",
                "Use.Blocked.WriteMat",
                "Compat.Pre-v4-DM-H",
                "SCF.Kerker.q0sq",
                "SCF.RhoG.DIIS.Depth",
                "SCF.RhoGMixingCutoff",
                "SCF.RhoG.Metric.Preconditioner.Cutoff",
                "SCF.DebugRhoGMixing",
                "Debug.DIIS",
            )
            self.__class__._registered = True

    @classmethod
    def setup_scf_settings(
        cls, user_params: Optional[Dict[str, Any]] = None
    ) -> "SCFLoopParameters":
        """
        Create and configure SCFLoopParameters instance based on user parameters.

        Args:
            user_params (dict, optional): Dictionary of user-defined parameters.

        Returns:
            SCFLoopParameters: Configured instance with FDF arguments.
        """
        from dataclasses import fields

        # Initialize with defaults
        scf_instance = cls()

        # Process user parameters if provided
        if user_params:
            for key, value in user_params.items():
                # Normalize key: lowercase and replace dots with underscores
                key_normalized = key.lower().replace(".", "_")

                # Match by comparing without underscores (handles CamelCase -> snake_case)
                key_no_underscores = key_normalized.replace("_", "")
                matching_field = None
                for f in fields(cls):
                    field_no_underscores = f.name.lower().replace("_", "")
                    if field_no_underscores == key_no_underscores:
                        matching_field = f.name
                        break

                if matching_field:
                    original_key = matching_field

                    # Handle type conversion
                    if "bool" in str(type(getattr(scf_instance, original_key))):
                        if isinstance(value, str):
                            value = value.lower() in ("true", "t", "1", "yes")
                        value = bool(value)
                    elif "int" in str(type(getattr(scf_instance, original_key))):
                        value = int(value)
                    elif "float" in str(type(getattr(scf_instance, original_key))):
                        value = float(value)

                    setattr(scf_instance, original_key, value)

        # Validate and generate FDF block
        scf_instance.validate()
        scf_instance.generate_scf_block()

        return scf_instance

    def validate(self):
        """
        Validates the SCF loop parameters.
        """
        logger.info("SCFLoopParameters.validate()")
        allowed_scf_mixer_variant = ["Pulay", "Simple", "kresse", "GR"]
        if self.scf_mixer_variant not in allowed_scf_mixer_variant:
            raise ValueError(
                f"Invalid mixing scheme '{self.scf_mixer_variant}'. Allowed values are: {allowed_scf_mixer_variant}"
            )
        if not (0 <= self.scf_mixer_weight <= 1):
            raise ValueError("Mixing parameter must be between 0 and 1.")

    def update_from_fdf(self, fdf_dict: Dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Note: Given the large number of SCF parameters (60+), this implementation
        handles the most commonly used ones. Full implementation would handle all 60.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            # Common SCF parameters
            if key_lower in ["maxscfiterations", "max_scf_iterations"]:
                self.max_scf_iterations = int(value)
            elif key_lower in ["minscfiterations", "mix_scf_iterations"]:
                self.mix_scf_iterations = int(value)
            elif key_lower in ["scf.mixer.weight", "scf_mixer_weight"]:
                self.scf_mixer_weight = (
                    float(value) if isinstance(value, str) else value
                )
            elif key_lower in ["scf.mixer.method", "scf_mixer_method"]:
                self.scf_mixer_method = str(value)
            elif key_lower in ["scf.mixer.history", "scf_mixer_history"]:
                self.scf_mixer_history = int(value)
            elif key_lower in ["scf.mixer.variant", "scf_mixer_variant"]:
                self.scf_mixer_variant = str(value)
            elif key_lower in [
                "scf.dm.tolerance",
                "scf_dm_tolerance",
                "dm.tolerance",
                "dm_tolerance",
            ]:
                # Supports both SCF.DM.Tolerance (newer) and DM.Tolerance (legacy)
                self.scf_dm_tolerance = parse_energy(value, target_unit="eV")
            # Additional parameters can be added following the same pattern

    def generate_fdf(self) -> Dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Note: Outputs only non-default values for commonly used SCF parameters.
        Full implementation would output all 60 parameters.

        Returns:
            Dictionary of FDF parameters
        """
        from collections import OrderedDict

        fdf = OrderedDict()
        fdf["#SCFLoopParameters"] = "SCFLoopParameters"

        # MaxSCFIterations - always write with default marker
        if self.max_scf_iterations == 200:
            fdf[
                "MaxSCFIterations"
            ] = f"{self.max_scf_iterations}  # SIESTA DEFAULT VALUE"
        else:
            fdf["MaxSCFIterations"] = str(self.max_scf_iterations)

        # MinSCFIterations - only write if non-zero
        if self.mix_scf_iterations != 0:
            fdf["MinSCFIterations"] = str(self.mix_scf_iterations)

        # SCF.MustConverge - always write with default marker
        if self.scf_must_converge:
            fdf["SCF.MustConverge"] = "True  # SIESTA DEFAULT VALUE"
        else:
            fdf["SCF.MustConverge"] = "false"

        # Mixer method - write if set
        if self.scf_mixer_method and self.scf_mixer_method.strip():
            fdf["SCF.Mixer.Method"] = self.scf_mixer_method

        # SCF.Mixer.Weight - always write with default marker
        if self.scf_mixer_weight == 0.25:
            fdf["SCF.Mixer.Weight"] = f"{self.scf_mixer_weight}  # SIESTA DEFAULT VALUE"
        else:
            fdf["SCF.Mixer.Weight"] = str(self.scf_mixer_weight)

        # SCF.Mixer.History - always write with default marker
        if self.scf_mixer_history == 2:
            fdf[
                "SCF.Mixer.History"
            ] = f"{self.scf_mixer_history}  # SIESTA DEFAULT VALUE"
        else:
            fdf["SCF.Mixer.History"] = str(self.scf_mixer_history)

        # SCF.Mixer.Variant - write if not default
        if self.scf_mixer_variant != "Pulay":
            fdf["SCF.Mixer.Variant"] = self.scf_mixer_variant

        # SCF.DM.Tolerance - always write with default marker
        if self.scf_dm_tolerance == 1.0e-5:
            fdf[
                "SCF.DM.Tolerance"
            ] = f"{self.scf_dm_tolerance} eV  # SIESTA DEFAULT VALUE"
        else:
            fdf["SCF.DM.Tolerance"] = f"{self.scf_dm_tolerance} eV"

        # DM.UseSaveDM - always write with default marker
        if self.dm_use_save_dm:
            fdf["DM.UseSaveDM"] = f"{self.dm_use_save_dm}  # SIESTA DEFAULT VALUE"
        else:
            fdf["DM.UseSaveDM"] = f"{self.dm_use_save_dm}"

        return fdf

    def to_ase(self) -> Dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns:
            Dictionary of ASE parameters
        """
        # ASE doesn't have direct SCF parameter equivalents
        # Most SCF settings are SIESTA-specific
        return {}

    @staticmethod
    def should_use_save_dm(system_changes: Optional[list] = None) -> bool:
        """
        Determine if saved DM should be reused based on system changes.

        This encapsulates the conditional logic for DM.UseSaveDM:
        - If system hasn't changed (numbers, magmoms, charges), use saved DM (True)
        - If system has changed, don't reuse incompatible DM (False)

        Args:
            system_changes: List of system properties that have changed
                          (from ASE calculator context)

        Returns:
            bool: True if saved DM should be used, False otherwise
        """
        if system_changes is None:
            return True  # No changes, safe to reuse DM

        # Check if critical properties changed
        if (
            "numbers" in system_changes
            or "initial_magmoms" in system_changes
            or "initial_charges" in system_changes
        ):
            return False  # System changed - don't reuse old DM

        return True  # No critical changes, safe to reuse DM

    def generate_scf_block(self):
        """
        Generates the SCF loop parameters block for the FDF file.

        This is a wrapper around generate_fdf() to maintain backward compatibility
        with code that calls this method directly (e.g., setup_scf_loop_parameters()).

        By calling generate_fdf(), we ensure:
        - Single source of truth for FDF generation
        - Proper "# SIESTA DEFAULT VALUE" markers on default parameters
        - Consistency with user_params, powerups, and tier presets
        - DRY principle (no parameter duplication)
        - Values updated via update_from_fdf() are properly reflected
        """
        logger.info("SCFLoopParameters.generate_scf_block()")

        from collections import OrderedDict

        # Call generate_fdf() which uses the current dataclass attributes
        # (these have been updated from user_params/powerups/tiers via update_from_fdf())
        fdf = self.generate_fdf()

        # Add comment header
        fdf_with_header = OrderedDict()
        if self.comments:
            fdf_with_header["#SCFLoopParameters"] = self.comments
        fdf_with_header.update(fdf)

        self.scf_fdf_arguments = fdf_with_header
