"""
Module defining base SIESTA input set and generator.

class MolecularDynamicsAndRelaxation

Based on User's Guide Siesta 5.4.0
Section:  7 STRUCTURAL RELAXATION, AND MOLECULAR DYNAMICS
          7.2 Structural relaxation
          7.2.1 Conjugate-gradients optimization
          7.2.2 Broyden optimization
          7.2.3 FIRE relaxation
          7.4 Molecular dynamics
          7.5 Output options for dynamics
"""

# Metadata

__all__ = ["MolecularDynamicsAndRelaxation"]

from dataclasses import dataclass, field
from typing import Dict
from typing import Any
from typing import List
from typing import Optional
from collections import OrderedDict


from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.dataclass.units import parse_force
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

import logging

logger = logging.getLogger(__name__)


@dataclass
class MolecularDynamicsAndRelaxation(FDFDataclass):
    """
    Data class to manage molecular dynamics (MD) and structural relaxation options for SIESTA input.
    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.ERROR
    )  # Default to show info & errors messages

    # --------------------------------------------------------
    # 7 STRUCTURAL RELAXATION, AND MOLECULAR DYNAMICS
    # --------------------------------------------------------
    # md_type_of_run: str = "CG"  # Relaxation method ('CG' for conjugate gradients, 'BFGS', etc.)
    md_type_of_run: str = field(
        default="CG",
        metadata={
            "description": "Selects the algorithm for a molecular dynamics or geometry optimization run. Common options are 'CG' (conjugate gradients), 'BFGS', 'Verlet', and 'Nose'.",
            "SIESTA keyword": "MD.TypeOfRun",
        },
    )

    # -------------------------
    # 7.2 Structural relaxation
    # -------------------------
    # md_variable_cell: bool = False # MD.VariableCell false
    # md_constant_volume: bool = False # Constant.Volume false
    # md_relax_cell_only: bool = False # MD.RelaxCellOnly false
    # md_max_force_tol: float = 0.01  # MD.MaxForceTol 0.04 eV/Ang Tolerance for relaxation convergence in eV/Angstrom
    # md_max_stress_tol: float = 0.01  # MD.MaxStressTol 1 GPa Tolerance for relaxation convergence in GPa
    # md_steps: int = 200  # Maximum number of steps for relaxation
    # md_max_displ: float = 0.2  # MD.MaxDispl 0.2 Bohr
    # md_precondition_variable_cell: float = 5.0 # MD.PreconditionVariableCell 5 Ang
    # zm_force_tol_length: float = 0.0155574 # ZM.ForceTolLength 0.00155574 Ry/Bohr
    # zm_force_tol_angle: float = 0.00356549 # ZM.ForceTolAngle 0.00356549 Ry/rad
    # zm_max_displ_length: float = 0.2 # ZM.MaxDisplLength 0.2 Bohr
    # zm_max_displ_angle: float = 0.003 # ZM.MaxDisplAngle 0.003 rad
    md_variable_cell: bool = field(
        default=False,
        metadata={
            "description": "If true, allows the lattice vectors (the simulation cell) to change during a geometry optimization.",
            "SIESTA keyword": "MD.VariableCell",
        },
    )

    md_constant_volume: bool = field(
        default=False,
        metadata={
            "description": "If true, keeps the volume of the simulation cell constant during a variable-cell geometry optimization.",
            "SIESTA keyword": "MD.ConstantVolume",
        },
    )

    md_relax_cell_only: bool = field(
        default=False,
        metadata={
            "description": "If true, only the lattice vectors are relaxed, while the fractional coordinates of the atoms are kept fixed.",
            "SIESTA keyword": "MD.RelaxCellOnly",
        },
    )

    md_max_force_tol: float = field(
        default=0.01,
        metadata={
            "description": "The convergence threshold (in eV/Angstrom) for the maximum force on any atom during a geometry optimization.",
            "SIESTA keyword": "MD.MaxForceTol",
            "unit": "eV/Ang",
        },
    )

    md_max_stress_tol: float = field(
        default=0.01,
        metadata={
            "description": "The convergence threshold (in GPa) for the maximum component of the stress tensor during a variable-cell geometry optimization.",
            "SIESTA keyword": "MD.MaxStressTol",
            "unit": "GPa",
        },
    )

    md_steps: int = field(
        default=200,
        metadata={
            "description": "The maximum number of molecular dynamics or geometry optimization steps to be performed.",
            "SIESTA keyword": "MD.NumCGsteps",
        },
    )

    md_max_displ: float = field(
        default=0.2,
        metadata={
            "description": "The maximum allowed atomic displacement (in Bohr) in a single geometry optimization step.",
            "SIESTA keyword": "MD.MaxDispl",
            "unit": "Bohr",
        },
    )

    md_precondition_variable_cell: float = field(
        default=5.0,
        metadata={
            "description": "A preconditioning factor (in Angstrom) for variable-cell relaxation, related to the bulk modulus.",
            "SIESTA keyword": "MD.PreconditionVariableCell",
            "unit": "Ang",
        },
    )

    zm_force_tol_length: float = field(
        default=0.0155574,
        metadata={
            "description": "The force tolerance (in Ry/Bohr) for bond lengths in a Z-matrix (internal coordinates) optimization.",
            "SIESTA keyword": "ZM.ForceTolLength",
            "unit": "Ry/Bohr",
        },
    )

    zm_force_tol_angle: float = field(
        default=0.00356549,
        metadata={
            "description": "The force tolerance (in Ry/radian) for angles in a Z-matrix (internal coordinates) optimization.",
            "SIESTA keyword": "ZM.ForceTolAngle",
            "unit": "Ry/rad",
        },
    )

    zm_max_displ_length: float = field(
        default=0.2,
        metadata={
            "description": "The maximum allowed displacement (in Bohr) for bond lengths in a Z-matrix optimization step.",
            "SIESTA keyword": "ZM.MaxDisplLength",
            "unit": "Bohr",
        },
    )

    zm_max_displ_angle: float = field(
        default=0.003,
        metadata={
            "description": "The maximum allowed displacement (in radians) for angles in a Z-matrix optimization step.",
            "SIESTA keyword": "ZM.MaxDisplAngle",
            "unit": "rad",
        },
    )

    # --------------------------------------
    # 7.2.1 Conjugate-gradients optimization
    # --------------------------------------
    # md_use_save_cg: bool = False # MD.UseSaveCG false
    md_use_save_cg: bool = field(
        default=False,
        metadata={
            "description": "If true, reads the conjugate-gradients history from a previous run to allow for an efficient restart of a geometry optimization.",
            "SIESTA keyword": "MD.UseSaveCG",
        },
    )

    # --------------------------
    # 7.2.2 Broyden optimization
    # --------------------------
    # md_broyden_history_steps: int = 5 # MD.Broyden.History.Steps 5
    # md_broyden_cycle_on_maxit: bool = True # MD.Broyden.Cycle.On.Maxit true
    # md_broyden_initial_inverse_jacobian: int = 1 # MD.Broyden.Initial.Inverse.Jacobian 1
    md_broyden_history_steps: int = field(
        default=5,
        metadata={
            "description": "Sets the number of previous steps (history depth) used to construct the approximate Hessian matrix in the Broyden geometry optimization method.",
            "SIESTA keyword": "MD.Broyden.History.Steps",
        },
    )

    md_broyden_cycle_on_maxit: bool = field(
        default=True,
        metadata={
            "description": "A flag that controls the behavior of the Broyden algorithm when the maximum number of iterations for an inner loop is reached, potentially restarting the cycle.",
            "SIESTA keyword": "MD.Broyden.Cycle.On.Maxit",
        },
    )

    md_broyden_initial_inverse_jacobian: int = field(
        default=1,
        metadata={
            "description": "Selects the scheme for constructing the initial guess for the inverse Jacobian (or Hessian) matrix at the start of a Broyden geometry optimization.",
            "SIESTA keyword": "MD.Broyden.Initial.Inverse.Jacobian",
        },
    )

    # ---------------------
    # 7.2.3 FIRE relaxation
    # ---------------------
    # target_pressure: float = 0.0 # Target.Pressure 0 GPa
    # block_targ_stress_voigt: Optional[List[int]] = field(default_factory=list)  # %block Target.Stress.Voigt −1 −1 −1 0 0 0
    # block_md_target_stress: Optional[List[int]] = field(default_factory=list) # %block MD.TargetStress −1 −1 −1 0 0 0
    # md_remote_intramolecular_pressure: bool = False # MD.RemoveIntramolecularPressure false
    target_pressure: float = field(
        default=0.0,
        metadata={
            "description": "The target external pressure (in GPa) for a variable-cell geometry optimization.",
            "SIESTA keyword": "Target.Pressure",
            "unit": "GPa",
        },
    )

    block_targ_stress_voigt: Optional[List[int]] = field(
        default_factory=list,
        metadata={
            "description": "A block to specify the target stress tensor in Voigt notation. A value of -1 for a component means it is determined by the Target.Pressure.",
            "SIESTA keyword": "%block Target.Stress.Voigt",
        },
    )

    block_md_target_stress: Optional[List[int]] = field(
        default_factory=list,
        metadata={
            "description": "A block to specify the target stress tensor for a molecular dynamics run under constant pressure.",
            "SIESTA keyword": "%block MD.TargetStress",
        },
    )

    md_remote_intramolecular_pressure: bool = field(
        default=False,
        metadata={
            "description": "If true, attempts to remove the intramolecular contribution to the pressure, which is useful for soft materials or molecules in a box.",
            "SIESTA keyword": "MD.RemoveIntramolecularPressure",
        },
    )

    # ----------------------
    # 7.4 Molecular dynamics
    # ----------------------
    # md_initial_time_step: int = 1 # MD.InitialTimeStep 1
    # md_final_time_step: int = None # MD.FinalTimeStep 〈MD.Steps〉
    # md_length_time_step: int = 1 # MD.LengthTimeStep 1 fs
    # md_initial_temperature: float = 0.0 # MD.InitialTemperature 0 K
    # md_target_temperature: float = 0.0 # MD.TargetTemperature 0 K
    # md_nose_mass: float = 100.0 # MD.NoseMass 100 Ry fs2
    # md_parrinello_rahmans_mass: float = 100.0 # MD.ParrinelloRahmanMass 100 Ry fs2
    # md_anneal_option: str = "TemperatureAndPressure" # MD.AnnealOption TemperatureAndPressure
    # md_tau_relax: int = 100 # MD.TauRelax 100 fs
    # md_bulk_modulus: float = 100.0  # MD.BulkModulus 100 Ry/Bohr3
    md_initial_time_step: int = field(
        default=1,
        metadata={
            "description": "The step number at which the MD simulation will start.",
            "SIESTA keyword": "MD.InitialTimeStep",
        },
    )

    md_final_time_step: Optional[int] = field(
        default=None,
        metadata={
            "description": "The step number at which the MD simulation will end. Defaults to running for a total of MD.NumCGsteps.",
            "SIESTA keyword": "MD.FinalTimeStep",
        },
    )

    md_length_time_step: float = field(
        default=1.0,
        metadata={
            "description": "The size of the time step (in femtoseconds) used for the molecular dynamics integration.",
            "SIESTA keyword": "MD.LengthTimeStep",
            "unit": "fs",
        },
    )

    md_initial_temperature: float = field(
        default=0.0,
        metadata={
            "description": "The initial temperature (in Kelvin) for an MD simulation, used to set the initial atomic velocities.",
            "SIESTA keyword": "MD.InitialTemperature",
            "unit": "K",
        },
    )

    md_target_temperature: float = field(
        default=0.0,
        metadata={
            "description": "The target temperature (in Kelvin) to be maintained by a thermostat in an NVT or NPT simulation.",
            "SIESTA keyword": "MD.TargetTemperature",
            "unit": "K",
        },
    )

    md_nose_mass: float = field(
        default=100.0,
        metadata={
            "description": "The fictitious mass parameter (in Ry*fs^2) for the Nosé-Hoover thermostat, controlling the coupling to the heat bath.",
            "SIESTA keyword": "MD.NoseMass",
            "unit": "Ry*fs^2",
        },
    )

    md_parrinello_rahmans_mass: float = field(
        default=100.0,
        metadata={
            "description": "The fictitious mass parameter (in Ry*fs^2) for the Parrinello-Rahman barostat, controlling the dynamics of the simulation cell.",
            "SIESTA keyword": "MD.ParrinelloRahmanMass",
            "unit": "Ry*fs^2",
        },
    )

    md_anneal_option: str = field(
        default="TemperatureAndPressure",
        metadata={
            "description": "In a simulated annealing run, this specifies whether to anneal the 'Temperature', 'Pressure', or both 'TemperatureAndPressure'.",
            "SIESTA keyword": "MD.AnnealOption",
        },
    )

    md_tau_relax: float = field(
        default=100.0,
        metadata={
            "description": "The relaxation time (in femtoseconds) for the thermostat or barostat, offering an alternative way to set the coupling strength.",
            "SIESTA keyword": "MD.TauRelax",
            "unit": "fs",
        },
    )

    md_bulk_modulus: float = field(
        default=100.0,
        metadata={
            "description": "An estimate of the system's bulk modulus (in Ry/Bohr^3), used as a preconditioner in constant-pressure simulations.",
            "SIESTA keyword": "MD.BulkModulus",
            "unit": "Ry/Bohr^3",
        },
    )

    # -------------------------------
    # 7.5 Output options for dynamics
    # -------------------------------
    # write_coor_initial: bool = True # WriteCoorInitial true
    # write_coor_step: bool = False # WriteCoorStep false
    # write_fores: bool = False # WriteForces false
    # write_md_history: bool = False # WriteMDHistory false
    # write_orbital_index: bool = True # Write.OrbitalIndex true

    # md_use_save_XV: bool = True # instructs SIESTA to read the atomic positions and velocities stored in file SystemLabel.XV by a previous run

    # # For TDED
    # perform_tded: bool = False

    # # For MD
    # md_ensemble: str = "NVT"  # Ensemble type for MD ('NVE', 'NVT', 'NPT', etc.)
    # md_temperature: float = 300.0  # Target temperature for MD in Kelvin
    # #md_steps: int = 1000  # Number of MD steps to perform
    # md_time_step: float = 1.0  # Time step for MD in femtoseconds

    # perform_md: bool = False  # Flag to indicate if molecular dynamics should be performed
    # md_fdf_arguments : Dict[str, Any] = field(default_factory=dict) # Optional fdf_arguments to return md_fdf_arguments
    # perform_relaxation: bool = False  # Flag to indicate if structural relaxation should be performed
    # relaxation_fdf_arguments : Dict[str, Any] = field(default_factory=dict) # Optional fdf_arguments to return relaxation_fdf_arguments

    write_coor_initial: bool = field(
        default=True,
        metadata={
            "description": "If true, writes the initial atomic coordinates to a file.",
            "SIESTA keyword": "WriteCoorInitial",
        },
    )

    write_coor_step: bool = field(
        default=False,
        metadata={
            "description": "If true, writes the atomic coordinates at every step of a geometry optimization or MD run.",
            "SIESTA keyword": "WriteCoorStep",
        },
    )

    write_forces: bool = field(
        default=True,  # Atomate2 default (SIESTA default is False)
        metadata={
            "description": "If true, writes the atomic forces at every step of a geometry optimization or MD run. Atomate2 sets this to True by default (SIESTA default is False) because forces are needed for workflow analysis.",
            "SIESTA keyword": "WriteForces",
        },
    )

    write_md_history: bool = field(
        default=False,
        metadata={
            "description": "If true, writes the MD history file (.MD), which includes velocities and other thermodynamic data.",
            "SIESTA keyword": "WriteMDHistory",
        },
    )

    write_orbital_index: bool = field(
        default=True,
        metadata={
            "description": "If true, writes the .OI file, which maps the internal orbital numbering to the quantum numbers (n, l, m, zeta).",
            "SIESTA keyword": "Write.OrbitalIndex",
        },
    )

    # MD Restart and Wrapper-level Controls
    md_use_save_XV: bool = field(
        default=True,
        metadata={
            "description": "If true, reads atomic positions, velocities, and cell vectors from a previous run's .XV file to restart an MD or geometry optimization.",
            "SIESTA keyword": "MD.UseSaveXV",
        },
    )

    perform_tded: bool = field(
        default=False,
        metadata={
            "description": "A wrapper-level flag to enable a Time-Dependent Density Functional Theory (TDDFT) calculation for simulating dynamics under a time-varying potential.",
            "SIESTA keyword": None,
        },
    )

    perform_md: bool = field(
        default=False,
        metadata={
            "description": "A high-level wrapper flag to enable a molecular dynamics simulation. Sets MD.TypeOfRun and related parameters.",
            "SIESTA keyword": None,
        },
    )

    perform_relaxation: bool = field(
        default=False,
        metadata={
            "description": "A high-level wrapper flag to enable a geometry optimization (structural relaxation). Sets MD.TypeOfRun to 'CG' or 'BFGS'.",
            "SIESTA keyword": None,
        },
    )

    md_ensemble: str = field(
        default="NVT",
        metadata={
            "description": "A high-level wrapper flag to select the thermodynamic ensemble ('NVE', 'NVT', 'NPT') for an MD run. This controls which thermostat/barostat is used.",
            "SIESTA keyword": None,
        },
    )

    md_temperature: float = field(
        default=300.0,
        metadata={
            "description": "A wrapper-level parameter for the target temperature (in Kelvin) for an MD simulation.",
            "SIESTA keyword": "MD.TargetTemperature",
            "unit": "K",
        },
    )

    md_time_step: float = field(
        default=1.0,
        metadata={
            "description": "A wrapper-level parameter for the molecular dynamics time step (in femtoseconds).",
            "SIESTA keyword": "MD.LengthTimeStep",
            "unit": "fs",
        },
    )

    md_fdf_arguments: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "A dictionary for any additional or arbitrary FDF flags related to molecular dynamics.",
            "SIESTA keyword": None,
        },
    )

    relaxation_fdf_arguments: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "A dictionary for any additional or arbitrary FDF flags related to geometry optimization (relaxation).",
            "SIESTA keyword": None,
        },
    )

    comments: str = field(
        default="",
        metadata={
            "description": "User-provided comments to be included as a comment block in the FDF file.",
            "SIESTA keyword": None,
        },
    )

    def __post_init__(self):
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "MD.TypeOfRun",
                "MD.VariableCell",
                "MD.ConstantVolume",
                "MD.NumCGsteps",
                "MD.MaxCGDispl",
                "MD.MaxForceTol",
                "MD.MaxStressTol",
                "MD.InitialTimeStep",
                "MD.FinalTimeStep",
                "MD.LengthTimeStep",
                "MD.InitialTemperature",
                "MD.TargetTemperature",
                "MD.TargetPressure",
                "Target.Pressure",
                "%block MD.TargetStress",
                "%block Target.Stress.Voigt",
                "MD.NoseMass",
                "MD.ParrinelloRahmanMass",
                "MD.TauRelax",
                "MD.BulkModulus",
                "MD.AnnealOption",
                "MD.UseSaveXV",
                "MD.UseSaveCG",
                "MD.RelaxCellOnly",
                "MD.RemoveIntramolecularPressure",
                "MD.PreconditionVariableCell",
                "MD.Broyden.History.Steps",
                "MD.Broyden.Initial.Inverse.Jacobian",
                "MD.Broyden.Cycle.On.Maxit",
                "ZM.ForceTolLength",
                "ZM.ForceTolAngle",
                "ZM.MaxDisplLength",
                "ZM.MaxDisplAngle",
                "WriteCoorInitial",
                "WriteCoorStep",
                "WriteForces",
                "WriteMDHistory",
                "Write.OrbitalIndex",
            )
            self.__class__._registered = True

    def validate(self):
        """
        Validates the molecular dynamics and relaxation options.
        """
        # console = Console()
        logger.info("MolecularDynamicsAndRelaxation.validate()")

        # Allowed TDED
        allowed_md_type_of_run_tded = ["TDED"]
        if self.perform_tded and self.md_type_of_run not in allowed_md_type_of_run_tded:
            raise ValueError(
                f"Invalid MD FC  '{self.md_type_of_run}'. Allowed values are: {allowed_md_type_of_run_tded}"
            )

        # Allowed Ensembles
        allowed_md_type_of_run_molecular_dynamics = [
            "Verlet",
            "Nose",
            "ParinelloRahman",
            "NoseParinelloRahman",
            "Anneal",
        ]  # ['NVE', 'NVT', 'NPT', 'NPH']
        if (
            self.perform_md
            and self.md_ensemble not in allowed_md_type_of_run_molecular_dynamics
        ):
            raise ValueError(
                f"Invalid MD ensemble '{self.md_ensemble}'. Allowed values are: {allowed_md_type_of_run_molecular_dynamics}"
            )

        # Allowed Relaxation Method
        allowed_md_type_of_run_relaxation = ["CG", "BFGS", "FIRE", "LUA"]
        if (
            self.perform_relaxation
            and self.md_type_of_run not in allowed_md_type_of_run_relaxation
        ):
            raise ValueError(
                f"Invalid relaxation method '{self.md_type_of_run}'. Allowed values are: {allowed_md_type_of_run_relaxation}"
            )

        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]Validation & Generation: [yellow]MolecularDynamicsAndRelaxation[/yellow] Successful![/green]"
            )
            # console.print(f"[green]Validation: [yellow]MolecularDynamicsAndRelaxation[/yellow] Successful![/green]")

    def update_from_fdf(self, fdf_dict: Dict[str, Any]) -> None:
        """Update this dataclass from FDF parameters."""
        logger.info(
            f"MolecularDynamicsAndRelaxation.update_from_fdf() called with {len(fdf_dict)} parameters"
        )
        for key, value in fdf_dict.items():
            key_lower = key.lower()
            if key_lower in ["md.typeofrun", "md_type_of_run"]:
                logger.info(f"  Setting md_type_of_run = {value}")
                self.md_type_of_run = str(value)
            elif key_lower in ["md.maxforcetol", "md_max_force_tol"]:
                logger.info(f"  Setting md_max_force_tol = {value}")
                self.md_max_force_tol = parse_force(value, target_unit="eV/Ang")
            elif key_lower in ["md.numcgsteps", "md_num_cg_steps", "md_steps"]:
                logger.info(f"  Setting md_steps = {value}")
                self.md_steps = int(value)
            elif key_lower in ["md.maxstresstol", "md_max_stress_tol"]:
                # Parse stress value (GPa)
                self.md_max_stress_tol = (
                    float(value.split()[0]) if isinstance(value, str) else float(value)
                )
            elif key_lower in ["md.variablecell", "md_variable_cell"]:
                self.md_variable_cell = (
                    value
                    if isinstance(value, bool)
                    else str(value).lower() in ["true", "t", "yes", "1"]
                )
            elif key_lower in ["md.usesavexv", "md_use_save_xv"]:
                self.md_use_save_XV = (
                    value
                    if isinstance(value, bool)
                    else str(value).lower() in ["true", "t", "yes", "1"]
                )

    def generate_fdf(self) -> Dict[str, Any]:
        """Generate SIESTA FDF format parameters.

        This generates the same parameters as generate_relaxation_block() to ensure
        consistency whether called from _initialize_modules() or core.py.
        Uses dataclass attributes which have been updated from user_params/powerups/tiers.
        """
        fdf: Dict[str, Any] = OrderedDict()

        # Generate if relaxation is enabled OR if user explicitly set MD.TypeOfRun
        # This handles force constants (FC), MD runs, etc. that aren't "relaxation"
        user_set_md_type = self.md_type_of_run != "CG"  # CG is default
        if not self.perform_relaxation and not user_set_md_type:
            return fdf

        # Add section header
        fdf["#MolecularDynamicsAndRelaxation"] = (
            self.comments
            if self.comments
            else "MolecularDynamicsAndRelaxation SETTINGS"
        )

        # Write all relaxation parameters with default markers
        # MD.TypeOfRun
        if self.md_type_of_run == "CG":
            fdf["MD.TypeOfRun"] = f"{self.md_type_of_run}  # SIESTA DEFAULT VALUE"
        else:
            fdf["MD.TypeOfRun"] = f"{self.md_type_of_run}"

        # MD.NumCGsteps
        if self.md_steps == 200:
            fdf["MD.NumCGsteps"] = f"{self.md_steps}  # SIESTA DEFAULT VALUE"
        else:
            fdf["MD.NumCGsteps"] = f"{self.md_steps}"

        # MD.MaxForceTol
        if self.md_max_force_tol == 0.01:
            fdf[
                "MD.MaxForceTol"
            ] = f"{self.md_max_force_tol} eV/Ang  # SIESTA DEFAULT VALUE"
        else:
            fdf["MD.MaxForceTol"] = f"{self.md_max_force_tol} eV/Ang"

        # MD.MaxStressTol
        if self.md_max_stress_tol == 0.01:
            fdf[
                "MD.MaxStressTol"
            ] = f"{self.md_max_stress_tol} GPa  # SIESTA DEFAULT VALUE"
        else:
            fdf["MD.MaxStressTol"] = f"{self.md_max_stress_tol} GPa"

        # MD.VariableCell
        if not self.md_variable_cell:
            fdf["MD.VariableCell"] = f"{self.md_variable_cell}  # SIESTA DEFAULT VALUE"
        else:
            fdf["MD.VariableCell"] = f"{self.md_variable_cell}"

        # MD.UseSaveXV
        if self.md_use_save_XV:
            fdf["MD.UseSaveXV"] = f"{self.md_use_save_XV}  # SIESTA DEFAULT VALUE"
        else:
            fdf["MD.UseSaveXV"] = f"{self.md_use_save_XV}"

        # WriteForces - always write (atomate2 default is True, SIESTA default is False)
        if self.write_forces:
            fdf[
                "WriteForces"
            ] = f"{self.write_forces}  # ATOMATE2 DEFAULT (SIESTA default: False)"
        else:
            fdf["WriteForces"] = f"{self.write_forces}"

        return fdf

    def to_ase(self) -> Dict[str, Any]:
        """Generate ASE-format parameters."""
        return {}

    def generate_md_block(self):
        """
        Generates the molecular dynamics options block for the FDF file.
        """
        logger.info("MolecularDynamicsAndRelaxation.generate_md_block()")
        if not self.perform_md:
            return ""
        self.md_fdf_arguments = {
            "MD.Use": "True",
            "MD.Ensemble": f"{self.md_ensemble}",
            "MD.Temperature": f"{self.md_temperature} K",
            "MD.Steps": f"{self.md_steps}",
            "MD.TimeStep": f"{self.md_time_step} fs",
        }

    def generate_relaxation_block(self):
        """
        Generates the structural relaxation options block for the FDF file.

        This is a wrapper around generate_fdf() to maintain backward compatibility
        with code that calls this method directly (e.g., core.py).

        By calling generate_fdf(), we ensure:
        - Single source of truth for FDF generation
        - Consistency with user_params, powerups, and tier presets
        - DRY principle (no parameter duplication)
        - Values updated via update_from_fdf() are properly reflected
        """
        logger.info("MolecularDynamicsAndRelaxation.generate_relaxation_block()")
        if not self.perform_relaxation:
            return ""

        # Call generate_fdf() which uses the current dataclass attributes
        # (these have been updated from user_params/powerups/tiers via update_from_fdf())
        self.relaxation_fdf_arguments = self.generate_fdf()

    @classmethod
    def setup_md_relax_settings(
        cls, user_params: Optional[Dict[str, Any]] = None
    ) -> "MolecularDynamicsAndRelaxation":
        """
        Create and configure a MolecularDynamicsAndRelaxation instance from user parameters.

        This classmethod provides a convenient way to initialize MD/relaxation settings
        for the tier-based input system, processing user-provided parameters and
        generating appropriate FDF arguments.

        Parameters
        ----------
        user_params : dict, optional
            Dictionary of user-defined parameters (case-insensitive).
            If None or empty, all default values are used.

        Returns
        -------
        MolecularDynamicsAndRelaxation
            Configured instance with FDF arguments populated
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]MolecularDynamicsAndRelaxation.setup_md_relax_settings()[/green]"
            )

        # Initialize with defaults
        instance = cls()

        # Handle empty user_params
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using default MD/relaxation settings.[/blue]"
                )
        else:
            # Process user_params - simple attribute setting for now
            # More sophisticated parameter processing can be added here if needed
            from dataclasses import fields as dc_fields

            valid_fields = {f.name.lower() for f in dc_fields(cls)}

            for key, value in user_params.items():
                key_normalized = key.lower().replace(".", "_")
                if key_normalized in valid_fields:
                    # Find original field name (preserving case)
                    original_key = next(
                        f.name
                        for f in dc_fields(cls)
                        if f.name.lower() == key_normalized
                    )
                    setattr(instance, original_key, value)

        # Validate settings
        try:
            instance.validate()
        except ValueError as e:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.ERROR.value:
                console.print(f"[red]MD/Relaxation validation failed: {e}[/red]")
            raise

        # Generate FDF blocks based on what's enabled
        if instance.perform_md:
            instance.generate_md_block()
        if instance.perform_relaxation:
            instance.generate_relaxation_block()

        return instance
