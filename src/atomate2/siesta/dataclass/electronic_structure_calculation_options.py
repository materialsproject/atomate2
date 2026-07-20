"""
Module defining base SIESTA input set and generator.

class ElectronicStructureCalculationOptions

Based on User's Guide Siesta 5.4.0
Section: 6.12 Calculation of the electronic structure
         6.12.1 Diagonalization options
         6.12.2 Output of eigenvalues and wavefunctions
         6.12.3 Occupation of electronic states and Fermi level
         6.12.4 Orbital minimization method (OMM)
         6.12.5 Order(N) calculations
"""

# Metadata

__all__ = ["ElectronicStructureCalculationOptions"]

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

from atomate2.siesta.dataclass.base import FDFDataclass

logger = logging.getLogger(__name__)


@dataclass
class ElectronicStructureCalculationOptions(FDFDataclass):
    """Manage electronic structure calculation options for SIESTA input."""

    # --------------------------------------------
    # 6.12 Calculation of the electronic structure
    # --------------------------------------------
    # solution_method: str = "" # SolutionMethod
    solution_method: str = field(
        default="",
        metadata={
            "description": (
                "Selects the algorithm to solve the Kohn-Sham equations. Options "
                "include 'diagon' (standard diagonalization), 'OMM' (Order-N "
                "linear scaling), or 'PEXSI' (pole expansion)."
            ),
            "SIESTA keyword": "SolutionMethod",
        },
    )
    # ------------------------------
    # 6.12.1 Diagonalization options
    # ------------------------------
    # number_of_eigen_states: int = None  #NumberOfEigenStates 〈all orbitals〉
    # diag_wfs_cache: str = None # Diag.WFS.Cache none|cdf
    # diag_use_2d: bool = True # Diag.Use2D true
    # diag_processor_y: int = None # Diag.ProcessorY
    # diag_block_size: int = None  #Diag.BlockSize 〈BlockSize〉
    # diag_algorithm: str = 'divide-and-Conquer'
    # diag_elpa_gpu: bool = False  #Diag.ELPA.GPU false
    # diag_elpa_gpu_string: str = 'nvidia-gpu' #Diag.ELPA.GPU.String nvidia-gpu
    # diag_parallel_over_k: bool = False  #Diag.ParallelOverK false
    # diag_abs_tol: float = 1e-16 #Diag.AbsTol 10−16  # noqa: RUF003
    # diag_or_fac: float = 1e-3 #Diag.OrFac 10−3  # noqa: RUF003
    # diag_memory: int = 1  #Diag.Memory 1
    # diag_upper_lower: str = 'lower' #Diag.UpperLower lower|upper
    number_of_eigen_states: int = field(
        default=None,
        metadata={
            "description": (
                "Specifies the total number of electronic eigenstates (orbitals) "
                "to be computed. Defaults to all available orbitals if not set."
            ),
            "SIESTA keyword": "NumberOfEigenStates",
        },
    )

    diag_wfs_cache: str = field(
        default=None,
        metadata={
            "description": (
                "A flag to cache wavefunctions to disk to reduce memory usage. "
                "Options are 'none' or 'cdf'."
            ),
            "SIESTA keyword": "Diag.WFS.Cache",
        },
    )

    diag_use_2d: bool = field(
        default=True,
        metadata={
            "description": (
                "Enables the use of a 2D block-cyclic distribution for matrices, "
                "required for ScaLAPACK-based diagonalizers."
            ),
            "SIESTA keyword": "Diag.Use2D",
        },
    )

    diag_processor_y: int = field(
        default=None,
        metadata={
            "description": (
                "Manually sets the number of processors in the 'Y' dimension of "
                "the 2D processor grid for diagonalization."
            ),
            "SIESTA keyword": "Diag.ProcessorY",
        },
    )

    diag_block_size: int = field(
        default=None,
        metadata={
            "description": (
                "The block size for the 2D block-cyclic data distribution of "
                "matrices. A key performance tuning parameter."
            ),
            "SIESTA keyword": "Diag.BlockSize",
        },
    )

    diag_algorithm: str = field(
        default="divide-and-Conquer",
        metadata={
            "description": (
                "Selects the specific algorithm used by the underlying "
                "diagonalization library (e.g., ScaLAPACK's 'DC' for "
                "Divide-and-Conquer)."
            ),
            "SIESTA keyword": "Diag.Algorithm",
        },
    )

    diag_elpa_gpu: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to enable GPU offloading when using the ELPA "
                "diagonalization library."
            ),
            "SIESTA keyword": "Diag.ELPA.GPU",
        },
    )

    diag_elpa_gpu_string: str = field(
        default="nvidia-gpu",
        metadata={
            "description": (
                "A string passed to the ELPA library to specify the GPU type "
                "(e.g., 'nvidia-gpu')."
            ),
            "SIESTA keyword": "Diag.ELPA.GPU.String",
        },
    )

    diag_parallel_over_k: bool = field(
        default=False,
        metadata={
            "description": (
                "Enables an additional level of parallelism by distributing "
                "k-points across different processor groups."
            ),
            "SIESTA keyword": "Diag.ParallelOverK",
        },
    )

    diag_abs_tol: float = field(
        default=1e-16,
        metadata={
            "description": (
                "An absolute tolerance for the diagonalization process, "
                "affecting the precision of the computed eigenvalues and "
                "eigenvectors."
            ),
            "SIESTA keyword": "Diag.AbsTol",
        },
    )

    diag_or_fac: float = field(
        default=1e-3,
        metadata={
            "description": (
                "The orthogonalization factor, a tolerance parameter used in "
                "some iterative diagonalization schemes."
            ),
            "SIESTA keyword": "Diag.OrFac",
        },
    )

    diag_memory: int = field(
        default=1,
        metadata={
            "description": (
                "The amount of memory (in MBytes) to be allocated per processor "
                "for the diagonalization workspace."
            ),
            "SIESTA keyword": "Diag.Memory",
        },
    )

    diag_upper_lower: str = field(
        default="lower",
        metadata={
            "description": (
                "Specifies whether the solver should use the 'lower' or 'upper' "
                "triangle of the symmetric matrices."
            ),
            "SIESTA keyword": "Diag.UpperLower",
        },
    )

    # -----------------------------------------
    # !!! Deprecated diagonalization options
    # diag_mrrr: bool = False  # Diag.MRRR false
    # diag_divide_and_conquer: bool = True # Diag.DivideAndConquer true
    # diag_elpa: bool = False # Diag.ELPA false
    # diag_no_expert: bool = False # Diag.NoExpert false
    diag_mrrr: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to explicitly request the "
                "'Multiple-Relatively-Robust-Representations' (MRRR) algorithm "
                "for diagonalization, known for its speed."
            ),
            "SIESTA keyword": "Diag.MRRR",
        },
    )

    diag_divide_and_conquer: bool = field(
        default=True,
        metadata={
            "description": (
                "A flag to explicitly request the 'Divide and Conquer' algorithm "
                "for diagonalization."
            ),
            "SIESTA keyword": "Diag.DivideAndConquer",
        },
    )

    diag_elpa: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to enable the use of the ELPA (Eigenvalue Solvers for "
                "Petaflop Applications) library for diagonalization, which is "
                "highly efficient on parallel machines."
            ),
            "SIESTA keyword": "Diag.ELPA",
        },
    )

    diag_no_expert: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to use the basic, non-expert drivers from ScaLAPACK or "
                "ELPA, which can be useful for debugging."
            ),
            "SIESTA keyword": "Diag.NoExpert",
        },
    )
    # -----------------------------------------------
    # 6.12.2 Output of eigenvalues and wavefunctions
    # -----------------------------------------------
    # write_eigenvalues: bool = False # WriteEigenvalues false
    write_eigenvalues: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to control whether the calculated Kohn-Sham eigenvalues "
                "are written to a file (.EIG), which is useful for plotting band "
                "structures."
            ),
            "SIESTA keyword": "WriteEigenvalues",
        },
    )
    # ------------------------------------------------------
    # 6.12.3 Occupation of electronic states and Fermi level
    # ------------------------------------------------------
    # occupation_function: str = "FD"  #OccupationFunction FD  Method for occupation
    # ('FD' for Fermi-Dirac, 'MP' for Methfessel-Paxton)
    # occupation_mp_order: int = 1 #OccupationMPOrder 1
    # electronic_temperature: float = 300    # ElectronicTemperature 300 K
    occupation_function: str = field(
        default="FD",
        metadata={
            "description": (
                "Selects the function for determining the occupation of "
                "electronic states. "
                "Common options are 'FD' (Fermi-Dirac) and 'MP' "
                "(Methfessel-Paxton)."
            ),
            "SIESTA keyword": "OccupationFunction",
        },
    )

    occupation_mp_order: int = field(
        default=1,
        metadata={
            "description": (
                "Sets the order of the Hermite polynomial for Methfessel-Paxton "
                "smearing ('OccupationFunction MP')."
            ),
            "SIESTA keyword": "OccupationMPOrder",
        },
    )

    electronic_temperature: float = field(
        default=300.0,
        metadata={
            "description": (
                "Sets the electronic temperature (in Kelvin) which defines the "
                "broadening/smearing of the occupation function around the Fermi "
                "level."
            ),
            "SIESTA keyword": "ElectronicTemperature",
            "unit": "K",
        },
    )
    # ----------------------------------------
    # 6.12.4 Orbital minimization method (OMM)
    # ----------------------------------------
    # omm_use_cholesky: bool = True #OMM.UseCholesky true
    # omm_use_2d: bool = True  #OMM.Use2D true
    # omm_user_sparse: bool = False #OMM.UseSparse false
    # omm_precon: int = -1 # OMM.Precon -1
    # omm_precon_first_step: int = None  # OMM.PreconFirstStep 〈OMM.Precon〉
    # omm_diagon: int = 0 #OMM.Diagon 0
    # omm_diagon_frist_step: int = None   #OMM.DiagonFirstStep 〈OMM.Diagon〉
    # omm_block_size: int = None #OMM.BlockSize 〈BlockSize〉
    # omm_t_prcon_scale: float = 10 # OMM.TPreconScale 10 Ry
    # omm_rel_tol: float = 1e-9 #OMM.RelTol 10−9  # noqa: RUF003
    # omm_eigenvalues: bool = True # OMM.Eigenvalues false
    # omm_write_coeffs: bool = True # OMM.WriteCoeffs false
    # omm_read_coeffs: bool = False # OMM.ReadCoeffs false
    # omm_long_output: bool = False # OMM.LongOutput false
    omm_use_cholesky: bool = field(
        default=True,
        metadata={
            "description": (
                "A flag to enable the use of Cholesky decomposition for matrix "
                "inversion within the OMM (Order-N) solver."
            ),
            "SIESTA keyword": "OMM.UseCholesky",
        },
    )

    omm_use_2d: bool = field(
        default=True,
        metadata={
            "description": (
                "Enables the use of a 2D block-cyclic distribution for matrices, "
                "required for parallel OMM calculations."
            ),
            "SIESTA keyword": "OMM.Use2D",
        },
    )

    omm_user_sparse: bool = field(
        default=False,
        metadata={
            "description": (
                "Enables the use of sparse matrix algebra libraries within the "
                "OMM solver, which is efficient for very large systems."
            ),
            "SIESTA keyword": "OMM.UseSparse",
        },
    )

    omm_precon: int = field(
        default=-1,
        metadata={
            "description": (
                "Controls the use and type of preconditioner to accelerate the "
                "OMM minimization."
            ),
            "SIESTA keyword": "OMM.Precon",
        },
    )

    omm_precon_first_step: int = field(
        default=None,
        metadata={
            "description": (
                "Specifies the preconditioner scheme for the first geometry/MD "
                "step. Defaults to the value of 'OMM.Precon'."
            ),
            "SIESTA keyword": "OMM.PreconFirstStep",
        },
    )

    omm_diagon: int = field(
        default=0,
        metadata={
            "description": (
                "The number of OMM steps after which a full diagonalization is "
                "performed to purify the density matrix. A value of 0 disables it."
            ),
            "SIESTA keyword": "OMM.Diagon",
        },
    )

    omm_diagon_first_step: int = field(
        default=None,
        metadata={
            "description": (
                "The diagonalization frequency for the first geometry/MD step. "
                "Defaults to the value of 'OMM.Diagon'."
            ),
            "SIESTA keyword": "OMM.DiagonFirstStep",
        },
    )

    omm_block_size: int = field(
        default=None,
        metadata={
            "description": (
                "The block size for the 2D block-cyclic matrix distribution in "
                "parallel OMM calculations."
            ),
            "SIESTA keyword": "OMM.BlockSize",
        },
    )

    omm_t_prcon_scale: float = field(
        default=10.0,
        metadata={
            "description": "An energy scale (in Rydberg) for the OMM preconditioner.",
            "SIESTA keyword": "OMM.TPreconScale",
            "unit": "Ry",
        },
    )

    omm_rel_tol: float = field(
        default=1e-9,
        metadata={
            "description": (
                "A relative tolerance criterion for the convergence of the OMM "
                "minimization algorithm."
            ),
            "SIESTA keyword": "OMM.RelTol",
        },
    )

    omm_eigenvalues: bool = field(
        default=True,
        metadata={
            "description": (
                "A flag to compute and write the band-structure eigenvalues, "
                "even when using the OMM method."
            ),
            "SIESTA keyword": "OMM.Eigenvalues",
        },
    )

    omm_write_coeffs: bool = field(
        default=True,
        metadata={
            "description": (
                "If true, writes the OMM coefficients (density matrix "
                "information) to a file for restarting calculations."
            ),
            "SIESTA keyword": "OMM.WriteCoeffs",
        },
    )

    omm_read_coeffs: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, reads the OMM coefficients from a file to initialize "
                "the calculation."
            ),
            "SIESTA keyword": "OMM.ReadCoeffs",
        },
    )

    omm_long_output: bool = field(
        default=False,
        metadata={
            "description": (
                "A debugging flag to enable long, detailed output from the OMM solver."
            ),
            "SIESTA keyword": "OMM.LongOutput",
        },
    )

    # ----------------------------
    # 6.12.5 Order(N) calculations
    # ----------------------------
    # on_funcional: str = "Kim" #ON.functional Kim
    # on_max_num_iter: int = 1000 # ON.MaxNumIter 1000
    # on_etol: int = 1e-8  #ON.Etol 10−8  # noqa: RUF003
    # on_eta: int = 0 # ON.eta 0 eV
    # on_eta_alpha: int = 0    # ON.eta.alpha 0 eV
    # on_eta_beta: int = 0 #  ON.eta.beta 0 eV
    # on_rc_lwf: float = 9.5 # ON.RcLWF 9.5 Bohr
    # on_chemical_potential: bool = False # ON.ChemicalPotential false
    # on_chemical_potential_use: bool = False # ON.ChemicalPotential.Use false
    # on_chemical_potential_rc: float = 9.5 # ON.ChemicalPotential.Rc 9.5 Bohr
    # on_chemical_potential_temperature: float = 0.05
    # ON.ChemicalPotential.Temperature 0.05 Ry
    # on_chemical_potential_order: int = 100 # ON.ChemicalPotential.Order 100
    # on_lower_meomory: bool = False # ON.LowerMemory false
    # on_use_save_lwf:bool = False # ON.UseSaveLWF false
    on_funcional: str = field(
        default="Kim",
        metadata={
            "description": (
                "Selects the specific Order-N functional to be used, for "
                "example, 'Kim' for the Kim-Mauri-Galli functional."
            ),
            "SIESTA keyword": "ON.functional",
        },
    )

    on_max_num_iter: int = field(
        default=1000,
        metadata={
            "description": (
                "The maximum number of iterations for the Order-N minimization loop."
            ),
            "SIESTA keyword": "ON.MaxNumIter",
        },
    )

    on_etol: float = field(
        default=1e-8,
        metadata={
            "description": (
                "The energy convergence tolerance for the Order-N minimization process."
            ),
            "SIESTA keyword": "ON.Etol",
        },
    )

    on_eta: float = field(
        default=0.0,
        metadata={
            "description": (
                "The value of the electronic chemical potential (Fermi Level) in eV."
            ),
            "SIESTA keyword": "ON.eta",
            "unit": "eV",
        },
    )

    on_eta_alpha: float = field(
        default=0.0,
        metadata={
            "description": (
                "The chemical potential (in eV) for the alpha (spin-up) channel "
                "in spin-polarized calculations."
            ),
            "SIESTA keyword": "ON.eta.alpha",
            "unit": "eV",
        },
    )

    on_eta_beta: float = field(
        default=0.0,
        metadata={
            "description": (
                "The chemical potential (in eV) for the beta (spin-down) channel "
                "in spin-polarized calculations."
            ),
            "SIESTA keyword": "ON.eta.beta",
            "unit": "eV",
        },
    )

    on_rc_lwf: float = field(
        default=9.5,
        metadata={
            "description": (
                "The cutoff radius (in Bohr) for the localized Wannier functions (LWF)."
            ),
            "SIESTA keyword": "ON.RcLWF",
            "unit": "Bohr",
        },
    )

    on_chemical_potential: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to enable the automatic determination of the chemical "
                "potential."
            ),
            "SIESTA keyword": "ON.ChemicalPotential",
        },
    )

    on_chemical_potential_use: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, use the automatically determined chemical potential in "
                "the calculation."
            ),
            "SIESTA keyword": "ON.ChemicalPotential.Use",
        },
    )

    on_chemical_potential_rc: float = field(
        default=9.5,
        metadata={
            "description": (
                "A cutoff radius (in Bohr) used in the algorithm for determining "
                "the chemical potential."
            ),
            "SIESTA keyword": "ON.ChemicalPotential.Rc",
            "unit": "Bohr",
        },
    )

    on_chemical_potential_temperature: float = field(
        default=0.05,
        metadata={
            "description": (
                "An electronic temperature (in Rydberg) used in the chemical "
                "potential determination algorithm."
            ),
            "SIESTA keyword": "ON.ChemicalPotential.Temperature",
            "unit": "Ry",
        },
    )

    on_chemical_potential_order: int = field(
        default=100,
        metadata={
            "description": (
                "The order of the polynomial expansion used in the chemical "
                "potential search algorithm."
            ),
            "SIESTA keyword": "ON.ChemicalPotential.Order",
        },
    )

    on_lower_memory: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, attempts to use a lower-memory algorithm, possibly at "
                "the cost of increased computation time."
            ),
            "SIESTA keyword": "ON.LowerMemory",
        },
    )

    on_use_save_lwf: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, reads previously saved localized Wannier functions "
                "(LWF) to initialize the calculation."
            ),
            "SIESTA keyword": "ON.UseSaveLWF",
        },
    )

    electronic_structure_fdf_arguments: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "A dictionary for any additional or arbitrary FDF flags related "
                "to electronic structure. This allows for using keywords not "
                "explicitly defined elsewhere."
            ),
            "SIESTA keyword": None,
        },
    )

    comments: str = field(
        default="ElectronicStructureCalculationOptions",
        metadata={
            "description": (
                "User-provided comments to be included as a comment block in the "
                "FDF file."
            ),
            "SIESTA keyword": None,
        },
    )

    _registered: ClassVar[bool]

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                # Solution method
                "SolutionMethod",
                # Diagonalization options
                "NumberOfEigenStates",
                "Diag.WFS.Cache",
                "Diag.Use2D",
                "Diag.ProcessorY",
                "Diag.BlockSize",
                "Diag.Algorithm",
                "Diag.ELPA.GPU",
                "Diag.ELPA.GPU.String",
                "Diag.ParallelOverK",
                "Diag.AbsTol",
                "Diag.OrFac",
                "Diag.Memory",
                "Diag.UpperLower",
                "Diag.MRRR",
                "Diag.DivideAndConquer",
                "Diag.ELPA",
                "Diag.NoExpert",
                # Eigenvalues/wavefunctions
                "WriteEigenvalues",
                # Occupation
                "OccupationFunction",
                "OccupationMPOrder",
                "ElectronicTemperature",
                # OMM (Order-N) options
                "OMM.UseCholesky",
                "OMM.Use2D",
                "OMM.UseSparse",
                "OMM.Precon",
                "OMM.PreconFirstStep",
                "OMM.Diagon",
                "OMM.DiagonFirstStep",
                "OMM.BlockSize",
                "OMM.TPreconScale",
                "OMM.RelTol",
                "OMM.Eigenvalues",
                "OMM.WriteCoeffs",
                "OMM.ReadCoeffs",
                "OMM.LongOutput",
                # ON (Order-N) options
                "ON.functional",
                "ON.MaxNumIter",
                "ON.Etol",
                "ON.eta",
                "ON.eta.alpha",
                "ON.eta.beta",
                "ON.RcLWF",
                "ON.ChemicalPotential",
                "ON.ChemicalPotential.Use",
                "ON.ChemicalPotential.Rc",
                "ON.ChemicalPotential.Temperature",
                "ON.ChemicalPotential.Order",
                "ON.LowerMemory",
                "ON.UseSaveLWF",
            )
            self.__class__._registered = True  # noqa: SLF001 class-level registration guard

    @classmethod
    def setup_electronic_structure_settings(
        cls, user_params: dict[str, Any] | None = None
    ) -> "ElectronicStructureCalculationOptions":
        """
        Create and configure an ElectronicStructureCalculationOptions instance.

        Configures the instance based on the provided user parameters.

        Args:
            user_params (dict, optional): Dictionary of user-defined parameters.

        Returns
        -------
            ElectronicStructureCalculationOptions: Configured instance with FDF
            arguments.
        """
        from dataclasses import fields

        # Initialize with defaults
        electronic_instance = cls()

        # Process user parameters if provided
        if user_params:
            for key, value in user_params.items():
                # Normalize key: lowercase and replace dots with underscores
                key_normalized = key.lower().replace(".", "_")

                # Match by comparing without underscores
                # (handles CamelCase -> snake_case)
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
                    # Special case: ElectronicTemperature can be float or string
                    # with units
                    converted_value = value
                    if original_key == "electronic_temperature":
                        # Keep as-is (string with units like "1000 K" or float)
                        pass
                    elif "bool" in str(
                        type(getattr(electronic_instance, original_key))
                    ):
                        if isinstance(value, str):
                            converted_value = value.lower() in ("true", "t", "1", "yes")
                        converted_value = bool(converted_value)
                    elif "int" in str(type(getattr(electronic_instance, original_key))):
                        converted_value = int(value)
                    elif "float" in str(
                        type(getattr(electronic_instance, original_key))
                    ):
                        converted_value = float(value)

                    setattr(electronic_instance, original_key, converted_value)

        # Validate and generate FDF block
        electronic_instance.validate()
        electronic_instance.generate_electronic_structure_block()

        return electronic_instance

    def validate(self) -> None:
        """Validate the electronic structure calculation options."""
        logger.info("ElectronicStructureCalculationOptions.validate()")
        allowed_solution_method = ["diagon", "OMM", "OrderN", "PEXSI", "ELSI", "CheSS"]  # noqa: F841
        allowed_diag_algorithm = [  # noqa: F841
            "divide-and-Conquer",
            "divide-and-Conquer-2stage",
            "MRRR",
            "MRRR-2stage",
            "expert",
            "expert-2stage",
            "noexpert|QR",
            "noexpert-2stage|QR-2stage",
            "ELPA-1stage",
            "ELPA|ELPA-2stage",
        ]
        allowed_occupation_function = ["FD", "MP", "Cold"]
        allowed_on_funcional = ["Kim", "Ordejon-Mauri", "files"]  # noqa: F841

        if self.occupation_function not in allowed_occupation_function:
            raise ValueError(
                f"Invalid occupation method '{self.occupation_function}'. "
                f"Allowed values are: {allowed_occupation_function}"
            )

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)

        Note:
            Simplified implementation focusing on commonly used parameters.
            Full implementation would handle all 49 electronic structure parameters.
        """
        for key, value in fdf_dict.items():
            key_lower = key.lower()

            # Solution method
            if key_lower in ["solutionmethod", "solution_method"]:
                self.solution_method = str(value)

            # Diagonalization
            elif key_lower in ["numberofeigenstates", "number_of_eigen_states"]:
                self.number_of_eigen_states = int(value) if value else None
            elif key_lower in ["diag.algorithm", "diag_algorithm"]:
                self.diag_algorithm = str(value)

            # Occupation
            elif key_lower in ["occupationfunction", "occupation_function"]:
                self.occupation_function = str(value)
            elif key_lower in ["occupationmporder", "occupation_mp_order"]:
                self.occupation_mp_order = int(value)
            elif key_lower in ["electronictemperature", "electronic_temperature"]:
                # Parse temperature (can be in K)
                temp_str = str(value).split()[0]
                self.electronic_temperature = float(temp_str)

            # Boolean flags
            elif key_lower in ["writeeigenvalues", "write_eigenvalues"]:
                self.write_eigenvalues = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters

        Note:
            Simplified implementation focusing on commonly used parameters.
            Full implementation would handle all 49 electronic structure parameters.
        """
        from collections import OrderedDict

        fdf = OrderedDict()
        fdf["#ElectronicStructure"] = "Electronic Structure Settings"

        # Solution method - always write if set
        if self.solution_method:
            fdf["SolutionMethod"] = self.solution_method

        # Diagonalization - always write
        if self.number_of_eigen_states is not None:
            fdf["NumberOfEigenStates"] = str(self.number_of_eigen_states)

        # Diag algorithm - always write with default marker
        if self.diag_algorithm == "divide-and-Conquer":
            fdf["Diag.Algorithm"] = f"{self.diag_algorithm}  # SIESTA DEFAULT VALUE"
        else:
            fdf["Diag.Algorithm"] = self.diag_algorithm

        # Occupation function - always write with default marker
        if self.occupation_function == "FD":
            fdf["OccupationFunction"] = (
                f"{self.occupation_function}  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["OccupationFunction"] = self.occupation_function

        # OccupationMPOrder - write when using MP with default marker
        if self.occupation_function == "MP":
            if self.occupation_mp_order == 1:
                fdf["OccupationMPOrder"] = (
                    f"{self.occupation_mp_order}  # SIESTA DEFAULT VALUE"
                )
            else:
                fdf["OccupationMPOrder"] = str(self.occupation_mp_order)

        # Electronic temperature - always write with default marker
        if self.electronic_temperature == 300.0:
            fdf["ElectronicTemperature"] = (
                f"{self.electronic_temperature} K  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["ElectronicTemperature"] = f"{self.electronic_temperature} K"

        # Boolean flags - always write
        if self.write_eigenvalues:
            fdf["WriteEigenvalues"] = "true"

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters.

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE doesn't have detailed electronic structure options
        # Most of these are SIESTA-specific
        return {}

    def generate_electronic_structure_block(self) -> None:
        """
        Generate the electronic structure calculation options block for the FDF file.

        This is a wrapper around generate_fdf() to maintain backward compatibility
        with code that calls this method directly
        (e.g., setup_electronic_structure_calculation_options()).

        By calling generate_fdf(), we ensure:
        - Single source of truth for FDF generation
        - Proper "# SIESTA DEFAULT VALUE" markers on default parameters
        - Consistency with user_params, powerups, and tier presets
        - DRY principle (no parameter duplication)
        - Values updated via update_from_fdf() are properly reflected
        """
        logger.info(
            "ElectronicStructureCalculationOptions."
            "generate_electronic_structure_block()"
        )

        from collections import OrderedDict

        # Call generate_fdf() which uses the current dataclass attributes
        # (these have been updated from user_params/powerups/tiers via
        # update_from_fdf())
        fdf = self.generate_fdf()

        # Add comment header
        fdf_with_header = OrderedDict()
        if self.comments:
            fdf_with_header["#ElectronicStructureCalculationOptions"] = self.comments
        fdf_with_header.update(fdf)

        self.electronic_structure_fdf_arguments = fdf_with_header
