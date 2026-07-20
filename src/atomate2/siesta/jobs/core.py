"""Define all Core SIESTA jobs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any  # noqa: F401 - Needed for Sphinx autodoc

from jobflow import Maker, Response, job
from monty.serialization import dumpfn

from atomate2.siesta import SETTINGS
from atomate2.siesta.files import (
    cleanup_siesta_outputs,
    gzip_output_folder,  # SIESTA-specific variant (exclude_files + subfolder)
    write_siesta_input_set,
)
from atomate2.siesta.jobs.base import _FILES_TO_EXCLUDE, _FILES_TO_ZIP, BaseSiestaMaker
from atomate2.siesta.run import run_siesta_socket, should_stop_children
from atomate2.siesta.schemas.task import SiestaTaskDoc
from atomate2.siesta.sets.core import (
    BandStructureSetGenerator,
    DOSSetGenerator,
    LuaSetGenerator,
    OpticalSetGenerator,
    PDOSSetGenerator,
    PhononSetGenerator,
    RelaxSetGenerator,  # , RelaxSetGeneratorTest
    SocketIOSetGenerator,
    StaticSetGenerator,
)
from atomate2.siesta.sets.parser import read_siesta_output

if TYPE_CHECKING:
    from jobflow import Flow
    from pymatgen.core import Molecule, Structure

    # from atomate2.siesta.sets.base_first import SiestaInputGenerator
    from atomate2.siesta.sets.base import SiestaInputGenerator

logger = logging.getLogger(__name__)


@dataclass
class StaticMaker(BaseSiestaMaker):
    """
    SIESTA Static (Single-Point) Energy Calculation.

    Performs a self-consistent field (SCF) calculation to determine the electronic
    structure and total energy of a fixed geometry. This is the foundation for most
    DFT calculations and provides essential properties without geometry optimization.

    Workflow Steps:
    ---------------
    1. Initialize electronic density (from atomic superposition or previous calculation)
    2. Solve Kohn-Sham equations self-consistently until convergence
    3. Calculate final properties: total energy, forces, stress tensor
    4. Generate output: density matrix, wave functions, Hamiltonian

    Key Results:
    ------------
    • Total Energy: Ground-state DFT energy (eV)
    • Forces: Atomic forces (eV/Å) - useful for checking if structure is relaxed
    • Stress Tensor: Cell stress (GPa) - for pressure analysis
    • Electron Density: Charge distribution for post-processing
    • Band Structure Data: For subsequent band structure calculations
    • Density of States: Electronic DOS at Fermi level

    Applications:
    -------------
    • Energy calculations at fixed geometry
    • Force and stress evaluation without relaxation
    • Reference calculations for more complex workflows
    • Post-processing: charge density, LDOS, COOP analysis
    • Checkpoint for continuation calculations (phonons, band structure)

    Parameters
    ----------
    calc_type : str
        The type key for the calculation (default: "scf")
    name : str
        The job name (default: "SCF Calculation")
    input_set_generator : SiestaInputGenerator
        The InputGenerator for the calculation (default: StaticSetGenerator)

    Examples
    --------
    >>> from atomate2.siesta.jobs.core import StaticMaker
    >>> from pymatgen.core import Structure
    >>> structure = Structure.from_file("structure.cif")
    >>> maker = StaticMaker.scf(user_params={"PAO.BasisSize": "DZP"})
    >>> job = maker.make(structure)

    Notes
    -----
    Static calculations are faster than relaxations and useful when:
    - You have an already relaxed structure
    - You need forces/stresses at a specific geometry
    - Setting up for band structure or DOS calculations
    - Testing convergence parameters (k-points, basis, cutoff)
    """

    input_set_generator: SiestaInputGenerator = field(
        default_factory=StaticSetGenerator
    )
    calc_type: str = "scf"
    name: str = "SCF Calculation"

    @classmethod
    def scf(cls, *args, **kwargs) -> StaticMaker:
        """
        Create a SCF maker.

        Parameters are split into two groups:
        - Maker parameters: dry_run, dry_run_output_dir, dry_run_format,
          dry_run_label, etc.
        - InputSetGenerator parameters: user_params, etc.
        """
        logger.info("StaticMaker.scf()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "strict_convergence",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
            "manager_config",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        return cls(
            input_set_generator=StaticSetGenerator(*args, **input_gen_kwargs),
            name=cls.name,
            **maker_kwargs,
        )


@dataclass
class RelaxMaker(BaseSiestaMaker):
    """
    SIESTA Structure Relaxation (Geometry Optimization).

    Optimizes atomic positions and optionally lattice parameters to find the
    minimum energy configuration. Uses conjugate gradient or similar algorithms
    to iteratively reduce forces and stresses until convergence criteria are met.

    Workflow Steps:
    ---------------
    1. Read initial structure and setup calculation
    2. Calculate forces and stresses via SCF at current geometry
    3. Update atomic positions (and lattice if variable-cell) using
       optimization algorithm
    4. Repeat steps 2-3 until convergence:
       - Maximum force < MD.MaxForceTol (default: 0.04 eV/Å)
       - Maximum stress < MD.MaxStressTol (default: 1.0 GPa, variable-cell only)
    5. Output relaxed structure and final properties

    Relaxation Modes:
    -----------------
    • Fixed-Cell: Optimize atomic positions only (lattice constants frozen)
      - Use when lattice parameters are known/experimental
      - Faster than variable-cell (fewer degrees of freedom)
      - Common for surface slabs, molecules, defects

    • Variable-Cell: Optimize both atomic positions AND lattice parameters
      - Find equilibrium lattice constants from scratch
      - Essential for unknown materials, high-pressure studies
      - Slower but provides complete equilibrium geometry

    Key Results:
    ------------
    • Relaxed Structure: Optimized atomic positions and lattice
    • Final Energy: Minimum energy configuration (eV)
    • Final Forces: Residual atomic forces (should be < tolerance)
    • Final Stress: Residual cell stress (variable-cell only)
    • Optimization History: Energy/force evolution during relaxation

    Applications:
    -------------
    • Find equilibrium structures for new materials
    • Relax experimental structures (remove thermal effects)
    • Pre-processing for phonons, band structure, NEB
    • Geometry optimization for molecules and surfaces
    • High-pressure structure prediction (with external pressure)

    Parameters
    ----------
    calc_type : str
        The type key for the calculation (default: "relax")
    name : str
        The job name (default: "Relaxation calculation")
    input_set_generator : SiestaInputGenerator
        The InputGenerator for the calculation (default: RelaxSetGenerator)

    Examples
    --------
    >>> from atomate2.siesta.jobs.core import RelaxMaker
    >>> from pymatgen.core import Structure
    >>> structure = Structure.from_file("unrelaxed.cif")
    >>>
    >>> # Fixed-cell relaxation (positions only)
    >>> maker = RelaxMaker.fixed_cell_relaxation(
    ...     user_params={"MD.MaxForceTol": "0.01 eV/Ang"}
    ... )
    >>> job = maker.make(structure)
    >>>
    >>> # Variable-cell relaxation (positions + lattice)
    >>> maker = RelaxMaker.variable_cell_relaxation(
    ...     user_params={"MD.MaxStressTol": "0.5 GPa"}
    ... )
    >>> job = maker.make(structure)

    Notes
    -----
    Convergence tips:
    - Start with looser tolerances (0.04 eV/Å) then tighten if needed
    - Use fixed-cell for surfaces/2D materials (prevent vacuum collapse)
    - Variable-cell essential for bulk materials without experimental data
    - For difficult convergence, try different MD.VariableCell options
    """

    input_set_generator: SiestaInputGenerator = field(default_factory=RelaxSetGenerator)
    calc_type: str = "relax"
    name: str = "Relaxation calculation"

    def __post_init__(self) -> None:
        """Initialize RelaxMaker with appropriate custodian handlers."""
        # If custodian is enabled but no handlers specified, use
        # relaxation-specific handlers
        if self.use_custodian and self.custodian_handlers is None:
            from atomate2.siesta.custodian.handlers import DEFAULT_RELAXATION_HANDLERS

            # Use DEFAULT_RELAXATION_HANDLERS which includes SCFRelaxationHandler
            # This handler removes DM file and increases SCF.MaxIter for better
            # convergence
            object.__setattr__(
                self, "custodian_handlers", DEFAULT_RELAXATION_HANDLERS.copy()
            )
            logger.info(
                "RelaxMaker: Using DEFAULT_RELAXATION_HANDLERS "
                "(includes SCFRelaxationHandler for DM removal + SCF.MaxIter increase)"
            )

    @classmethod
    def fixed_cell_relaxation(cls, *args, **kwargs) -> RelaxMaker:
        """
        Create a fixed cell relaxation maker.

        Parameters are split into two groups:
        - Maker parameters: use_custodian, custodian_handlers,
          custodian_max_errors, etc.
        - InputSetGenerator parameters: user_params, etc.
        """
        logger.info("RelaxMaker.fixed_cell_relaxation()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "strict_convergence",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
            "manager_config",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        return cls(
            input_set_generator=RelaxSetGenerator(
                *args, relax_cell=False, **input_gen_kwargs
            ),
            name=cls.name + "-fixed-cell",
            **maker_kwargs,
        )

    @classmethod
    def variable_cell_relaxation(cls, *args, **kwargs) -> RelaxMaker:
        """
        Create a variable cell relaxation maker.

        Parameters are split into two groups:
        - Maker parameters: use_custodian, custodian_handlers,
          custodian_max_errors, etc.
        - InputSetGenerator parameters: user_params, etc.
        """
        logger.info("RelaxMaker.variable_cell_relaxation()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "strict_convergence",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
            "manager_config",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        return cls(
            input_set_generator=RelaxSetGenerator(
                *args, relax_cell=True, **input_gen_kwargs
            ),
            name=cls.name + "-variable-cell",
            **maker_kwargs,
        )


@dataclass
class LuaMaker(BaseSiestaMaker):
    """
    SIESTA Lua Scripting Interface for Advanced Dynamics and NEB Calculations.

    Enables custom optimization algorithms and advanced molecular dynamics through
    SIESTA's Lua scripting interface. This provides fine-grained control over ionic
    movement, relaxation algorithms, and reaction pathway calculations beyond the
    built-in MD options.

    Workflow Steps:
    ---------------
    1. Load Lua script (user-provided or default: relax_geometry_lbfgs.lua)
    2. Initialize atomic positions and forces
    3. Execute Lua script controlling ionic steps:
       - Calculate forces via SCF at each geometry
       - Update positions using Lua-defined algorithm (LBFGS, FIRE, custom)
       - Apply constraints, convergence checks
    4. Output final structure and trajectory

    Key Results:
    ------------
    • Relaxed Structure: Optimized geometry from Lua-controlled algorithm
    • Trajectory: Step-by-step ionic movement (.MD, .XV files)
    • Final Energy: Converged total energy (eV)
    • Force History: Evolution of atomic forces during optimization
    • Algorithm-Specific Data: Custom outputs from Lua script

    Applications:
    -------------
    • Advanced relaxation algorithms (LBFGS, FIRE, custom optimizers)
    • Nudged Elastic Band (NEB) transition state searches
    • Custom molecular dynamics protocols
    • Constrained optimization (selective relaxation, reaction coordinates)
    • Method development and algorithm testing

    Parameters
    ----------
    calc_type : str
        The type key for the calculation (default: "relax")
    name : str
        The job name (default: "Lua calculation")
    input_set_generator : SiestaInputGenerator
        The InputGenerator for the calculation (default: LuaSetGenerator)

    Examples
    --------
    >>> from atomate2.siesta.jobs.core import LuaMaker
    >>> from pymatgen.core import Structure
    >>>
    >>> # Advanced LBFGS fixed-cell relaxation
    >>> structure = Structure.from_file("initial.cif")
    >>> maker = LuaMaker.fixed_cell_relaxation(
    ...     user_params={"Lua.Script": "relax_geometry_lbfgs.lua"}
    ... )
    >>> job = maker.make(structure)
    >>>
    >>> # Variable-cell relaxation (lattice optimization)
    >>> maker = LuaMaker.variable_cell_relaxation(
    ...     user_params={"MD.VariableCell": "true"}
    ... )
    >>> job = maker.make(structure)
    >>>
    >>> # NEB transition state search
    >>> maker = LuaMaker.neb(
    ...     user_params={"Lua.Script": "neb.lua", "neb.nimages": 7, "neb.spring": 0.02}
    ... )
    >>> job = maker.make(structure)

    Notes
    -----
    Lua scripting best practices:
    - Use LBFGS for faster convergence vs CG (3-5x speedup typical)
    - Default script (relax_geometry_lbfgs.lua) works for most cases
    - Custom scripts require SIESTA Lua API knowledge
    - NEB requires initial and final structures (see flows/neb.py)
    - Check SIESTA manual section 4.2 for Lua API reference
    """

    input_set_generator: SiestaInputGenerator = field(default_factory=LuaSetGenerator)
    calc_type: str = "relax"
    name: str = "Lua calculation"

    @classmethod
    def fixed_cell_relaxation(cls, *args, **kwargs) -> LuaMaker:
        """
        Create a fixed cell relaxation maker.

        Parameters are split into two groups:
        - Maker parameters: dry_run, dry_run_output_dir, dry_run_format,
          dry_run_label, etc.
        - InputSetGenerator parameters: user_params, etc.
        """
        logger.info("LuaMaker.fixed_cell_relaxation()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "strict_convergence",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
            "manager_config",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        # Automatically add default Lua.Script if not provided by user
        if "user_params" not in input_gen_kwargs:
            input_gen_kwargs["user_params"] = {}
        if input_gen_kwargs["user_params"] is None:
            input_gen_kwargs["user_params"] = {}

        # Only set default Lua.Script if user hasn't provided it
        user_params = input_gen_kwargs["user_params"]
        if "Lua.Script" not in user_params and "lua.script" not in {
            k.lower() for k in user_params
        }:
            user_params["Lua.Script"] = "relax_geometry_lbfgs.lua"
            input_gen_kwargs["user_params"] = user_params

        return cls(
            input_set_generator=LuaSetGenerator(
                *args, lua_type="lua_relaxation", relax_cell=False, **input_gen_kwargs
            ),
            name=cls.name + "-fixed-cell",
            **maker_kwargs,
        )

    @classmethod
    def variable_cell_relaxation(cls, *args, **kwargs) -> LuaMaker:
        """
        Create a variable cell relaxation maker.

        Parameters are split into two groups:
        - Maker parameters: dry_run, dry_run_output_dir, dry_run_format,
          dry_run_label, etc.
        - InputSetGenerator parameters: user_params, etc.
        """
        logger.info("LuaMaker.variable_cell_relaxation()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "strict_convergence",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
            "manager_config",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        # Automatically add default Lua.Script if not provided by user
        if "user_params" not in input_gen_kwargs:
            input_gen_kwargs["user_params"] = {}
        if input_gen_kwargs["user_params"] is None:
            input_gen_kwargs["user_params"] = {}

        # Only set default Lua.Script if user hasn't provided it
        user_params = input_gen_kwargs["user_params"]
        if "Lua.Script" not in user_params and "lua.script" not in {
            k.lower() for k in user_params
        }:
            user_params["Lua.Script"] = "relax_cell_geometry.lua"
            input_gen_kwargs["user_params"] = user_params

        return cls(
            input_set_generator=LuaSetGenerator(
                *args, lua_type="lua_relaxation", relax_cell=True, **input_gen_kwargs
            ),
            name=cls.name + "-variable-cell",
            **maker_kwargs,
        )

    @classmethod
    def neb(cls, *args, **kwargs) -> LuaMaker:
        """
        Create a fixed cell neb maker.

        Parameters are split into two groups:
        - Maker parameters: dry_run, dry_run_output_dir, dry_run_format,
          dry_run_label, etc.
        - InputSetGenerator parameters: user_params, etc.
        """
        logger.info("LuaMaker.neb()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "strict_convergence",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
            "manager_config",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        # Automatically add default Lua.Script if not provided by user
        if "user_params" not in input_gen_kwargs:
            input_gen_kwargs["user_params"] = {}
        if input_gen_kwargs["user_params"] is None:
            input_gen_kwargs["user_params"] = {}

        # Only set default Lua.Script if user hasn't provided it
        user_params = input_gen_kwargs["user_params"]
        if "Lua.Script" not in user_params and "lua.script" not in {
            k.lower() for k in user_params
        }:
            user_params["Lua.Script"] = "neb.lua"
            input_gen_kwargs["user_params"] = user_params

        return cls(
            input_set_generator=LuaSetGenerator(
                *args, lua_type="lua_neb", relax_cell=False, **input_gen_kwargs
            ),
            name=cls.name + "-neb-with-fix-cell",
            **maker_kwargs,
        )


@dataclass
class SocketIOStaticMaker(BaseSiestaMaker):
    """
    SIESTA Socket Communication for High-Throughput Energy Calculations.

    Accelerates batch calculations by maintaining a persistent SIESTA process
    connected via socket protocol. Eliminates repeated initialization overhead
    (basis generation, pseudopotential loading) when computing properties for
    multiple structures with identical calculation parameters.

    Workflow Steps:
    ---------------
    1. Start SIESTA socket server on specified host:port
    2. Initialize basis sets, pseudopotentials, grid parameters (once only)
    3. For each structure in the batch:
       - Send atomic positions via socket
       - SIESTA computes SCF energy and forces
       - Receive results without restarting
    4. Close socket connection after all structures processed

    Key Results:
    ------------
    • Multiple Structure Energies: Total energies for entire batch (eV)
    • Forces: Atomic forces for each structure (eV/Å)
    • Stress Tensors: Cell stresses if applicable (GPa)
    • Speedup: 5-50x faster than individual jobs (depends on SCF cost vs setup cost)
    • Batch Task Document: Aggregated results from all structures

    Applications:
    -------------
    • Molecular dynamics trajectory analysis (energy/force evaluation)
    • High-throughput screening (same parameters, many structures)
    • Adsorption site scanning (multiple adsorbate positions)
    • Structure interpolation (NEB images, reaction paths)
    • Machine learning dataset generation

    Parameters
    ----------
    calc_type : str
        The type key for the calculation (default: "multi_scf")
    name : str
        The job name (default: "SCF Calculations Socket")
    host : str
        The hostname for socket server (default: "localhost")
    port : int
        Socket port number (default: 12345)
    input_set_generator : SiestaInputGenerator
        The InputGenerator for the calculation (default: SocketIOSetGenerator)

    Examples
    --------
    >>> from atomate2.siesta.jobs.core import SocketIOStaticMaker
    >>> from pymatgen.core import Structure
    >>>
    >>> # Batch energy calculation for 100 structures
    >>> structures = [Structure.from_file(f"struct_{i}.cif") for i in range(100)]
    >>> maker = SocketIOStaticMaker(
    ...     host="localhost",
    ...     port=12345,
    ...     user_params={"PAO.BasisSize": "DZP", "kpts": [4, 4, 4]},
    ... )
    >>> job = maker.make(structures)
    >>>
    >>> # MD trajectory analysis
    >>> from pymatgen.io.ase import AseAtomsAdaptor
    >>> traj_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in trajectory]
    >>> job = maker.make(traj_structures)

    Notes
    -----
    Performance considerations:
    - Most efficient when setup cost >> SCF cost (small systems, many structures)
    - Typical speedup: 10-20x for 100+ structures
    - Ensure port is available and not blocked by firewall
    - All structures must use same pseudopotentials and basis sets
    - Socket errors automatically handled with reconnection attempts
    - Not suitable for single structure calculations (use StaticMaker instead)
    """

    calc_type: str = "multi_scf"
    name: str = "SCF Calculations Socket"
    host: str = "localhost"
    port: int = 12345
    input_set_generator: SiestaInputGenerator = field(
        default_factory=SocketIOSetGenerator
    )

    @job
    def make(
        self,
        structure: list[Structure | Molecule],
        prev_dir: str | Path | None = None,
    ) -> Response:
        """
        Run multiple SIESTA calculation with the socket.

        Calculate the properties for multiple structures using the same parameters
        using socket communication to speed up the calculations.

        Parameters
        ----------
        structure : list[Molecule | Structure]
            The list of structure objects to run SIESTA on
        prev_dir : str or Path or None
            A previous SIESTA calculation directory to copy output files from.

        Returns
        -------
        The output response for the calculations
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)
        logger.info("SocketIOStaticMaker.make()")
        # copy previous inputs
        if not isinstance(structure, list):
            structure = [structure]

        from_prev = prev_dir is not None
        if from_prev:
            hostless_prev_dir = str(prev_dir).split(":")[1]
            images = read_siesta_output(f"{hostless_prev_dir}/siesta.out")
            if not isinstance(images, Sequence):
                images = [images]

            for img in images:
                img.calc = None

            for ii in range(-1 * len(structure), 0, -1):
                if structure[ii] in images:
                    del structure[ii]

        # write aims input files
        self.write_input_set_kwargs["prev_dir"] = prev_dir
        write_siesta_input_set(
            structure[0], self.input_set_generator, **self.write_input_set_kwargs
        )

        # write any additional data
        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))

        # run SIESTA
        run_siesta_socket(structure, **self.run_aims_kwargs)

        # parse SIESTA outputs
        task_doc = SiestaTaskDoc.from_directory(Path.cwd(), **self.task_document_kwargs)
        task_doc.task_label = self.name

        # decide whether child jobs should proceed
        stop_children = should_stop_children(task_doc, **self.stop_children_kwargs)

        # cleanup files to save disk space
        cleanup_siesta_outputs(directory=Path.cwd())

        # gzip folder
        gzip_output_folder(
            directory=Path.cwd(),
            setting=SETTINGS.SIESTA_ZIP_FILES,
            files_list=_FILES_TO_ZIP,
            exclude_files=_FILES_TO_EXCLUDE,
        )

        return Response(
            stop_children=stop_children,
            output=task_doc if self.store_output_data else None,
        )


@dataclass
class BandStructureMaker(BaseSiestaMaker):
    """
    SIESTA Electronic Band Structure Calculation.

    Computes electronic eigenvalues along high-symmetry k-paths in the Brillouin
    zone to visualize band dispersion, band gaps, and electronic character. Requires
    a prior SCF calculation to generate converged Hamiltonian and density matrix.

    Workflow Steps:
    ---------------
    1. Read converged Hamiltonian from previous SCF calculation
    2. Define k-path through high-symmetry points (automatic from structure symmetry)
    3. Diagonalize Hamiltonian at each k-point along path (non-self-consistent)
    4. Output band eigenvalues (eV) and k-point coordinates
    5. Generate band structure data for plotting (pymatgen BandStructure object)

    Key Results:
    ------------
    • Band Structure: Eigenvalues E(k) along high-symmetry path
    • Band Gap: Direct/indirect gap and VBM/CBM positions (eV)
    • Effective Masses: Curvature near band extrema (optional)
    • Symmetry Labels: Γ, X, M, K, etc. marking special k-points
    • Pymatgen Object: Plottable BandStructureSymmLine for visualization

    Applications:
    -------------
    • Semiconductor band gap determination (direct vs indirect)
    • Metal vs insulator classification
    • Topological material characterization (band inversions)
    • Optical transition analysis (allowed transitions)
    • Validation against experimental ARPES data

    Parameters
    ----------
    calc_type : str
        The type key for the calculation (default: "band_structure")
    name : str
        The job name (default: "bands")
    input_set_generator : BandStructureSetGenerator
        The InputGenerator for the calculation

    Examples
    --------
    >>> from atomate2.siesta.jobs.core import BandStructureMaker, StaticMaker
    >>> from atomate2.siesta.flows.bands import BandStructureFlowMaker
    >>> from pymatgen.core import Structure
    >>>
    >>> # Complete band structure workflow (SCF + bands)
    >>> structure = Structure.from_file("Si.cif")
    >>> flow_maker = BandStructureFlowMaker()
    >>> flow = flow_maker.make(structure)
    >>>
    >>> # Standalone (requires prev_dir from SCF)
    >>> bands_maker = BandStructureMaker.bandstructure_calculation(
    ...     user_params={"BandLinesScale": "pi/a"}
    ... )
    >>> job = bands_maker.make(structure, prev_dir="scf_calculation/")

    Notes
    -----
    Critical requirements:
    - MUST run SCF calculation first to generate Hamiltonian
    - Use same basis set, mesh cutoff, and pseudopotentials as SCF
    - K-path automatically determined from structure symmetry (seekpath)
    - For accurate gaps, ensure dense k-grid in SCF calculation
    - Non-self-consistent: fast but requires good initial Hamiltonian
    - For spin-polarized systems, bands plotted separately for each spin channel
    """

    input_set_generator: BandStructureSetGenerator = field(
        default_factory=BandStructureSetGenerator
    )
    calc_type = "band_structure"
    name: str = "bands"
    # structure: list[Structure | Molecule]

    @classmethod
    def bandstructure_calculation(cls, *args, **kwargs) -> BandStructureMaker:
        """
        Create a bandstructure calculation maker.

        Parameters are split into two groups:
        - Maker parameters: dry_run, use_custodian, custodian_handlers, etc.
        - InputSetGenerator parameters: user_params, etc.
        """
        logger.info("BandStructureMaker.bandstructure_calculation()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "strict_convergence",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
            "manager_config",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        return cls(
            input_set_generator=BandStructureSetGenerator(*args, **input_gen_kwargs),
            name=cls.name + " (Calculation)",
            **maker_kwargs,
        )


@dataclass
class DOSMaker(BaseSiestaMaker):
    """
    SIESTA Total Density of States (DOS) Calculation.

    Computes the total electronic density of states by sampling eigenvalues on a
    dense k-point mesh and applying Gaussian/Lorentzian broadening. Provides
    energy-resolved information about electronic structure complementary to band
    structure calculations.

    Workflow Steps:
    ---------------
    1. Read converged Hamiltonian from previous SCF calculation
    2. Sample eigenvalues on dense k-point mesh (typically 10x SCF density)
    3. Apply broadening (Gaussian width ~0.1 eV) to discrete levels
    4. Integrate over Brillouin zone with k-point weights
    5. Output total DOS vs energy (states/eV)

    Key Results:
    ------------
    • Total DOS: Electronic states per energy (states/eV)
    • Fermi Level: Chemical potential at specified temperature (eV)
    • Band Gap: Energy gap between VBM and CBM (eV)
    • DOS at Fermi Level: N(E_F) - indicator of metallic character
    • Integration: Total electrons from DOS integration (validation check)

    Applications:
    -------------
    • Metal vs insulator classification (N(E_F) analysis)
    • Band gap estimation (faster than band structure)
    • Thermoelectric properties (Seebeck coefficient estimation)
    • Magnetic moment analysis (spin-polarized DOS difference)
    • Comparison with experimental photoemission spectroscopy (PES)

    Parameters
    ----------
    calc_type : str
        The type key for the calculation (default: "dos")
    name : str
        The job name (default: "DOS calculation")
    input_set_generator : DOSSetGenerator
        The InputGenerator for the DOS calculation

    Examples
    --------
    >>> from atomate2.siesta.jobs.core import DOSMaker, StaticMaker
    >>> from pymatgen.core import Structure
    >>>
    >>> # Basic DOS calculation (requires SCF first)
    >>> structure = Structure.from_file("Fe.cif")
    >>> dos_maker = DOSMaker.dos_calculation(
    ...     user_params={
    ...         "kpts": [12, 12, 12],  # Dense k-mesh for accurate DOS
    ...         "%block ProjectedDensityOfStates": [
    ...             "-10.0 5.0 0.1 500 eV"  # Energy range and resolution
    ...         ],
    ...     }
    ... )
    >>> job = dos_maker.make(structure, prev_dir="scf_calculation/")
    >>>
    >>> # Spin-polarized DOS
    >>> dos_maker = DOSMaker.dos_calculation(
    ...     user_params={"Spin": "polarized", "kpts": [16, 16, 16]}
    ... )
    >>> job = dos_maker.make(structure)

    Notes
    -----
    Convergence and accuracy:
    - K-point mesh density critical (test convergence!)
    - Typical requirement: 2-5x denser than SCF mesh
    - Energy range: cover all relevant states (VBM-10 eV to CBM+5 eV)
    - Broadening width: 0.05-0.2 eV (balance resolution vs noise)
    - Spin-polarized systems: separate DOS for up/down spins
    - Output file: siesta.DOS (energy, total DOS, integrated DOS)
    """

    input_set_generator: DOSSetGenerator = field(default_factory=DOSSetGenerator)
    calc_type: str = "dos"
    name: str = "DOS calculation"

    @classmethod
    def dos_calculation(cls, *args, **kwargs) -> DOSMaker:
        """
        Create a DOS calculation maker.

        Parameters are split into two groups:
        - Maker parameters: dry_run, use_custodian, custodian_handlers, etc.
        - InputSetGenerator parameters: user_params, etc.

        Parameters
        ----------
        *args
            Positional arguments passed to DOSSetGenerator
        dry_run : bool, optional
            Enable dry-run mode (generate input files without running SIESTA)
        **kwargs
            Keyword arguments (user_params, etc.)

        Returns
        -------
        DOSMaker
            Configured DOS calculation maker

        Examples
        --------
        >>> dos_maker = DOSMaker.dos_calculation(
        ...     dry_run=True, user_params={"Mesh.Cutoff": "400 Ry"}
        ... )
        """
        logger.info("DOSMaker.dos_calculation()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "strict_convergence",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
            "manager_config",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        return cls(
            input_set_generator=DOSSetGenerator(*args, **input_gen_kwargs),
            name="DOS calculation",
            **maker_kwargs,
        )


@dataclass
class PDOSMaker(BaseSiestaMaker):
    """
    SIESTA Projected Density of States (PDOS) Calculation.

    Decomposes total DOS into orbital-resolved contributions from individual atoms,
    revealing which atomic orbitals contribute to specific energy regions. Essential
    for understanding chemical bonding, orbital interactions, and atom-specific
    electronic character.

    Workflow Steps:
    ---------------
    1. Read converged Hamiltonian from previous SCF calculation
    2. Sample eigenvalues on dense k-point mesh with orbital projections
    3. Project wavefunctions onto atomic orbitals (s, p, d, f shells)
    4. Apply broadening and integrate over Brillouin zone
    5. Output total DOS + per-atom orbital-resolved PDOS

    Key Results:
    ------------
    • Total DOS: Overall electronic density of states (states/eV)
    • Orbital-Resolved PDOS: Contributions from s, p, d, f orbitals per atom
    • Atom-Specific DOS: Electronic states localized on each atom
    • Bonding Analysis: Orbital overlap and hybridization patterns
    • XML Output: Machine-readable PDOS data (siesta.PDOS.xml)

    Applications:
    -------------
    • Chemical bonding analysis (σ/π bonding, hybridization)
    • Identify orbital contributions to band edges (VBM/CBM character)
    • Magnetic moment decomposition (spin-polarized PDOS)
    • Ligand field splitting in transition metal complexes
    • Charge transfer analysis (donor/acceptor orbital populations)
    • Comparison with X-ray photoelectron spectroscopy (XPS)

    Parameters
    ----------
    calc_type : str
        The type key for the calculation (default: "pdos")
    name : str
        The job name (default: "PDOS calculation")
    input_set_generator : PDOSSetGenerator
        The InputGenerator for the PDOS calculation

    Examples
    --------
    >>> from atomate2.siesta.jobs.core import PDOSMaker
    >>> from pymatgen.core import Structure
    >>>
    >>> # PDOS for transition metal oxide
    >>> structure = Structure.from_file("TiO2.cif")
    >>> pdos_maker = PDOSMaker.pdos_calculation(
    ...     user_params={
    ...         "kpts": [12, 12, 8],
    ...         "%block ProjectedDensityOfStates": ["-10.0 5.0 0.1 500 eV"],
    ...     }
    ... )
    >>> job = pdos_maker.make(structure, prev_dir="scf_calculation/")
    >>>
    >>> # Spin-polarized PDOS for magnetic system
    >>> pdos_maker = PDOSMaker.pdos_calculation(
    ...     user_params={
    ...         "Spin": "polarized",
    ...         "kpts": [16, 16, 16],
    ...         "%block ProjectedDensityOfStates": ["-5.0 2.0 0.05 300 eV"],
    ...     }
    ... )
    >>> job = pdos_maker.make(structure)

    Notes
    -----
    PDOS output files:
    - siesta.DOS: Total DOS (same as DOSMaker)
    - siesta.PDOS: Human-readable PDOS (atom, orbital, energy, DOS)
    - siesta.PDOS.xml: Machine-readable format for plotting tools

    Analysis tips:
    - Compare O-2p and Ti-3d PDOS to identify band gap character
    - Spin-polarized PDOS shows magnetic moment origin (3d vs 4f)
    - Overlap between atom PDOS indicates bonding/antibonding states
    - Peak positions reveal orbital energies and ligand field splittings
    - Use dense k-mesh (same as DOS requirements)
    """  # noqa: RUF002

    input_set_generator: PDOSSetGenerator = field(default_factory=PDOSSetGenerator)
    calc_type: str = "pdos"
    name: str = "PDOS calculation"

    @classmethod
    def pdos_calculation(cls, *args, **kwargs) -> PDOSMaker:
        """
        Create a PDOS calculation maker.

        Parameters are split into two groups:
        - Maker parameters: dry_run, use_custodian, custodian_handlers, etc.
        - InputSetGenerator parameters: user_params, etc.

        Parameters
        ----------
        *args
            Positional arguments passed to PDOSSetGenerator
        dry_run : bool, optional
            Enable dry-run mode (generate input files without running SIESTA)
        **kwargs
            Keyword arguments (user_params, etc.)

        Returns
        -------
        PDOSMaker
            Configured PDOS calculation maker

        Examples
        --------
        >>> # PDOS for all atoms (default behavior)
        >>> pdos_maker = PDOSMaker.pdos_calculation(dry_run=True)
        >>>
        >>> # PDOS with custom parameters
        >>> pdos_maker = PDOSMaker.pdos_calculation(
        ...     dry_run=True, user_params={"Mesh.Cutoff": "400 Ry"}
        ... )
        """
        logger.info("PDOSMaker.pdos_calculation()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "strict_convergence",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
            "manager_config",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        return cls(
            input_set_generator=PDOSSetGenerator(*args, **input_gen_kwargs),
            name="PDOS calculation",
            **maker_kwargs,
        )


@dataclass
class PhononMaker(BaseSiestaMaker):
    """
    SIESTA Phonon Force Constants Calculation.

    Computes interatomic force constants for vibrational analysis by calculating
    forces on atoms displaced from equilibrium positions in a supercell. This is
    the core calculation step for phonopy-based phonon workflows, providing the
    force data needed to build the dynamical matrix.

    Workflow Steps:
    ---------------
    1. Generate supercell with symmetry-reduced atomic displacements (phonopy)
    2. For each displaced configuration:
       - Read structure with displacement applied
       - Run static SCF calculation
       - Extract forces on all atoms
    3. Collect force sets from all displacements
    4. Output force constants for phonopy post-processing

    Key Results:
    ------------
    • Force Sets: Atomic forces for each displacement configuration (eV/Å)
    • Force Constants: Second derivatives of energy wrt atomic positions
    • Displacement Data: Mapping of displacements to force responses
    • Supercell Information: Supercell matrix and symmetry operations

    Applications:
    -------------
    • Phonon dispersion calculation (input for phonopy)
    • Thermal properties (heat capacity, thermal expansion)
    • Thermodynamic stability (phonon free energy)
    • Infrared/Raman spectroscopy prediction
    • Grüneisen parameter calculation (volume dependence)

    Parameters
    ----------
    calc_type : str
        The type key for the calculation (default: "relax")
    name : str
        The job name (default: "Relaxation calculation")
    input_set_generator : SiestaInputGenerator
        The InputGenerator for the calculation (default: PhononSetGenerator)

    Examples
    --------
    >>> from atomate2.siesta.jobs.core import PhononMaker
    >>> from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker
    >>> from pymatgen.core import Structure
    >>>
    >>> # Complete phonon workflow (recommended)
    >>> structure = Structure.from_file("Si.cif")
    >>> phonon_flow = PhononFlowMaker(
    ...     min_length=12.0,  # Supercell size
    ...     displacement=0.01,  # Displacement magnitude (Angstroms)
    ... )
    >>> flow = phonon_flow.make(structure)
    >>>
    >>> # Standalone force constants calculation
    >>> fc_maker = PhononMaker.fc_calculations(
    ...     user_params={
    ...         "PAO.BasisSize": "DZP",
    ...         "kpts": [8, 8, 8],
    ...         "MeshCutoff": "400 Ry",
    ...     }
    ... )
    >>> # Note: Requires pre-generated displaced structures from phonopy

    Notes
    -----
    Best practices:
    - Use well-converged SCF parameters (forces must be accurate!)
    - Displacement: 0.01 Å typical (larger for soft modes, smaller for hard)
    - Supercell: min_length ≥ 12 Å to avoid spurious interactions
    - K-point mesh: scale inversely with supercell size (Γ-point often sufficient)
    - CRITICAL: Relax structure first (residual forces < 0.01 eV/Å)
    - For anharmonic systems, consider finite-temperature effects
    - Output used by phonopy to compute phonon frequencies and eigenvectors
    """

    input_set_generator: SiestaInputGenerator = field(
        default_factory=PhononSetGenerator
    )
    calc_type: str = "relax"
    name: str = "Relaxation calculation"
    # md_last: int = None

    @classmethod
    def fc_calculations(cls, *args, **kwargs) -> PhononMaker:
        """
        Create a phonon force constants calculation maker.

        Parameters are split into two groups:
        - Maker parameters: dry_run, use_custodian, custodian_handlers, etc.
        - InputSetGenerator parameters: user_params, etc.
        """
        logger.info("PhononMaker.fc_calculations()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "strict_convergence",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
            "manager_config",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        return cls(
            input_set_generator=PhononSetGenerator(
                *args, md_type_of_run="FC", **input_gen_kwargs
            ),
            name=cls.name + "-phonon",
            **maker_kwargs,
        )


@dataclass
class OpticalMaker(BaseSiestaMaker):
    """
    SIESTA Optical Properties Calculation.

    Computes optical response functions (dielectric function, absorption coefficient,
    refractive index) by calculating momentum matrix elements between occupied and
    unoccupied states. Uses the random phase approximation (RPA) to determine how
    the material interacts with electromagnetic radiation.

    Workflow Steps:
    ---------------
    1. Read converged Hamiltonian from previous SCF calculation
    2. Calculate momentum matrix elements <i|p|j> between band pairs
    3. Sum contributions from all optical transitions (valence → conduction)
    4. Apply energy conservation and occupation factors
    5. Output frequency-dependent optical properties (ε(ω), α(ω), n(ω))

    Key Results:
    ------------
    • Dielectric Function: ε(ω) = ε₁(ω) + iε₂(ω) (real and imaginary parts)
    • Absorption Coefficient: α(ω) - photon absorption vs wavelength
    • Refractive Index: n(ω) - real part of complex refractive index
    • Reflectivity: R(ω) - fraction of light reflected at normal incidence
    • Energy Loss Function: Im(-1/ε) - electron energy loss spectroscopy
    • Optical Gap: First absorption onset (may differ from electronic gap)

    Applications:
    -------------
    • Solar cell materials (absorption spectrum, optimal band gap)
    • Transparent conductors (optical gap vs electrical conductivity)
    • Photocatalysis (light absorption efficiency)
    • Optical coatings (refractive index engineering)
    • Comparison with UV-Vis spectroscopy experiments
    • Optoelectronic device design (LEDs, photodetectors)

    Parameters
    ----------
    calc_type : str
        The type key for the calculation (default: "Optical")
    name : str
        The job name (default: "Optical calculation")
    input_set_generator : SiestaInputGenerator
        The InputGenerator for the calculation (default: OpticalSetGenerator)

    Examples
    --------
    >>> from atomate2.siesta.jobs.core import OpticalMaker
    >>> from pymatgen.core import Structure
    >>>
    >>> # Basic optical properties calculation
    >>> structure = Structure.from_file("TiO2.cif")
    >>> optical_maker = OpticalMaker.optical_calculations(
    ...     user_params={
    ...         "Optical.Mesh.Cutoff": "300 Ry",
    ...         "Optical.Broaden": "0.1 eV",
    ...         "Optical.EnergyMinimum": "0.0 eV",
    ...         "Optical.EnergyMaximum": "10.0 eV",
    ...     }
    ... )
    >>> job = optical_maker.make(structure, prev_dir="scf_calculation/")
    >>>
    >>> # High-resolution spectrum
    >>> optical_maker = OpticalMaker.optical_calculations(
    ...     user_params={
    ...         "kpts": [16, 16, 16],  # Dense k-mesh for accurate transitions
    ...         "Optical.NumberOfBands": 200,  # Include many conduction bands
    ...         "Optical.Broaden": "0.05 eV",  # Sharp features
    ...     }
    ... )
    >>> job = optical_maker.make(structure)

    Notes
    -----
    Convergence requirements:
    - Dense k-point mesh essential (optical transitions sample entire BZ)
    - Include enough unoccupied bands (NumberOfBands >> N_electrons/2)
    - Broadening: 0.05-0.2 eV (experimental linewidth guidance)
    - Energy range: cover experimental measurement range
    - Scissors correction: adjust band gap to match experiment if needed

    Limitations:
    - RPA approximation: no excitonic effects (underestimates absorption)
    - Independent particle picture: electron-hole interactions neglected
    - For accurate exciton binding, use GW-BSE (not in SIESTA)
    - Good for trends and qualitative predictions

    Output files:
    - siesta.EPSIMG: Imaginary dielectric function ε₂(ω)
    - siesta.EPSREAL: Real dielectric function ε₁(ω)
    - Additional optical data in standard output
    """  # noqa: RUF002

    input_set_generator: SiestaInputGenerator = field(
        default_factory=OpticalSetGenerator
    )
    calc_type: str = "Optical"
    name: str = "Optical calculation"

    @classmethod
    def optical_calculations(cls, *args, **kwargs) -> OpticalMaker:
        """
        Create an optical properties calculation maker.

        Parameters are split into two groups:
        - Maker parameters: dry_run, use_custodian, custodian_handlers, etc.
        - InputSetGenerator parameters: user_params, etc.

        Parameters
        ----------
        *args
            Positional arguments passed to OpticalSetGenerator
        **kwargs
            Keyword arguments passed to OpticalSetGenerator

        Returns
        -------
        OpticalMaker
            Configured optical calculation maker
        """
        logger.info("OpticalMaker.optical_calculations()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "strict_convergence",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
            "manager_config",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        return cls(
            input_set_generator=OpticalSetGenerator(
                *args, optical_calculation=True, **input_gen_kwargs
            ),
            name=cls.name,
            **maker_kwargs,
        )


@dataclass
class SiestaPhononMaker(Maker):
    """
    SIESTA Phonon Calculation with Automatic Plotting and Analysis.

    High-level wrapper for complete phonon workflows combining structure relaxation,
    force constants calculation via phonopy, and automatic generation of publication-
    quality plots and comprehensive text summaries. Provides fine-grained control over
    relaxation and force calculation parameters for maximum accuracy.

    Workflow Steps:
    ---------------
    1. Optional structure relaxation (variable-cell or fixed-cell)
    2. Generate symmetry-reduced displaced supercells (phonopy)
    3. Calculate forces for each displacement (static SCF calculations)
    4. Build force constants matrix and dynamical matrix (phonopy)
    5. Compute phonon dispersion along high-symmetry paths
    6. Calculate phonon DOS and thermal properties (Cv, S, F)
    7. Generate automatic plots: band structure, DOS, thermal properties
    8. Write comprehensive text summary with analysis

    Key Results:
    ------------
    • Phonon Dispersion: ω(q) along high-symmetry paths (THz or cm⁻¹)
    • Phonon DOS: Vibrational density of states
    • Thermal Properties: Heat capacity Cv(T), entropy S(T), free energy F(T)
    • Zero-Point Energy: Quantum vibrational contribution at 0 K
    • Plots: phonon_bands.png, phonon_dos.png, thermal_properties.png
    • Summary: Detailed text report with frequencies, symmetries, analysis

    Applications:
    -------------
    • Thermodynamic stability (imaginary frequencies → instability)
    • Thermal expansion coefficient via quasi-harmonic approximation
    • Heat capacity prediction for materials design
    • Infrared/Raman spectroscopy peak assignment
    • Isotope effects on vibrational properties
    • Comparison with inelastic neutron scattering experiments

    Parameters
    ----------
    name : str
        Name of the workflow (default: "siesta phonopy")
    relax_maker : RelaxMaker | None
        Maker for structure relaxation with custom parameters (default: None)
    static_maker : StaticMaker
        Maker for force calculations with custom parameters
    min_length : float
        Minimum supercell length in Angstroms (default: 12.0)
    displacement : float
        Atomic displacement in Angstroms (default: 0.01)
    supercell_matrix : list[list[int]] | None
        Manual supercell matrix (overrides min_length if provided)
    prefer_90_degrees : bool
        Prefer supercells with ~90° angles (default: True)
    symprec : float
        Symmetry precision (default: 1e-5)
    use_symmetry : bool
        Use symmetry to reduce calculations (default: True)
    mesh : tuple[int, int, int]
        Q-point mesh for DOS/thermal properties (default: (50, 50, 50))
    create_thermal_properties : bool
        Calculate thermal properties (default: True)
    t_min : float
        Minimum temperature in K (default: 0)
    t_max : float
        Maximum temperature in K (default: 1000)
    t_step : float
        Temperature step in K (default: 10)
    generate_plots : bool
        Generate plots and analysis files (default: True)
    plot_band_structure : bool
        Plot phonon band structure (default: True)
    plot_dos : bool
        Plot phonon DOS (default: True)
    plot_thermal : bool
        Plot thermal properties (default: True)
    write_summary : bool
        Write comprehensive text summary (default: True)

    Examples
    --------
    >>> from atomate2.siesta.jobs.core import SiestaPhononMaker, RelaxMaker, StaticMaker
    >>> from atomate2.siesta.powerups import update_user_siesta_settings
    >>> from pymatgen.core import Structure
    >>>
    >>> # Separate parameters for relaxation and forces
    >>> structure = Structure.from_file("Si.cif")
    >>> relax_params = {
    ...     "PAO.BasisSize": "DZP",
    ...     "kpts": [6, 6, 6],
    ...     "MeshCutoff": "300 Ry",
    ... }
    >>> force_params = {
    ...     "PAO.BasisSize": "DZP",
    ...     "kpts": [8, 8, 8],
    ...     "MeshCutoff": "400 Ry",
    ... }
    >>>
    >>> relax_maker = update_user_siesta_settings(
    ...     RelaxMaker.variable_cell_relaxation(), relax_params
    ... )
    >>> static_maker = update_user_siesta_settings(StaticMaker(), force_params)
    >>>
    >>> # Complete phonon workflow with automatic plotting
    >>> phonon_maker = SiestaPhononMaker(
    ...     relax_maker=relax_maker,
    ...     static_maker=static_maker,
    ...     min_length=12.0,
    ...     displacement=0.01,
    ...     t_max=500,  # Thermal properties up to 500 K
    ... )
    >>> flow = phonon_maker.make(structure)
    >>>
    >>> # Skip relaxation (for already relaxed structures)
    >>> phonon_maker = SiestaPhononMaker(
    ...     relax_maker=None,  # No relaxation
    ...     static_maker=static_maker,
    ...     min_length=15.0,  # Larger supercell for higher accuracy
    ... )
    >>> flow = phonon_maker.make(structure)

    Notes
    -----
    Best practices:
    - CRITICAL: Use tighter force tolerances for force calculations vs relaxation
    - Typical: relax with 0.04 eV/Å, forces with 0.01 eV/Å or better
    - Supercell size: min_length ≥ 12 Å (15 Å for soft materials)
    - Displacement: 0.01 Å standard (0.005 Å for very stiff systems)
    - K-points for forces: can use Γ-point for large supercells (>100 atoms)
    - Symmetry exploitation: use_symmetry=True reduces calculations by 10-100x
    - For publication: increase mesh to (100,100,100) for smooth DOS

    Output files:
    - phonon_bands.png: Phonon dispersion plot
    - phonon_dos.png: Phonon density of states
    - thermal_properties.png: Cv(T), S(T), F(T) plots
    - phonon_summary.txt: Comprehensive analysis report
    """

    name: str = "siesta phonopy"
    relax_maker: RelaxMaker | None = None
    static_maker: StaticMaker = field(default_factory=StaticMaker)
    min_length: float = 12.0
    displacement: float = 0.01
    supercell_matrix: list[list[int]] | None = None
    prefer_90_degrees: bool = True
    symprec: float = 1e-5
    use_symmetry: bool = True
    mesh: tuple[int, int, int] = (50, 50, 50)
    create_thermal_properties: bool = True
    t_min: float = 0
    t_max: float = 1000
    t_step: float = 10
    generate_plots: bool = True
    plot_band_structure: bool = True
    plot_dos: bool = True
    plot_thermal: bool = True
    write_summary: bool = True
    dry_run: bool = False

    def make(
        self, structure: Structure, prev_dir: str | Path | None = None
    ) -> Flow:
        """
        Create phonon calculation workflow with optional plotting.

        Parameters
        ----------
        structure : Structure
            Input structure
        prev_dir : str | Path | None
            Previous directory

        Returns
        -------
        Flow
            Phonon calculation workflow with plotting jobs
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)
        from dataclasses import replace

        from jobflow import Flow

        from atomate2.siesta.jobs.phonon.phonopy import PhonopyMaker

        jobs = []

        # Apply dry_run to static_maker if specified
        static_maker = self.static_maker
        if hasattr(static_maker, "dry_run"):
            static_maker = replace(static_maker, dry_run=self.dry_run)  # type: ignore[call-arg]

        # Apply dry_run to relax_maker if specified and relax_maker exists
        relax_maker = self.relax_maker
        if relax_maker is not None and hasattr(relax_maker, "dry_run"):
            relax_maker = replace(relax_maker, dry_run=self.dry_run)  # type: ignore[call-arg]

        # Create PhonopyMaker with all parameters
        phonopy_maker = PhonopyMaker(
            name=self.name,
            relax_maker=relax_maker,
            static_maker=static_maker,
            min_length=self.min_length,
            displacement=self.displacement,
            supercell_matrix=self.supercell_matrix,
            prefer_90_degrees=self.prefer_90_degrees,
            symprec=self.symprec,
            use_symmetry=self.use_symmetry,
            mesh=self.mesh,
            create_thermal_properties=self.create_thermal_properties,
            t_min=self.t_min,
            t_max=self.t_max,
            t_step=self.t_step,
        )

        # Main phonon calculation flow
        phonon_flow = phonopy_maker.make(structure, prev_dir=prev_dir)
        jobs.append(phonon_flow)

        # Add plotting and analysis jobs if requested
        if self.generate_plots:
            from atomate2.siesta.jobs.phonon.plotting import (
                plot_phonon_band_structure,
                plot_phonon_dos,
                plot_thermal_properties,
                write_phonon_summary,
            )

            # Get phonon results from the flow
            # The last job in phonon_flow should be the analysis job
            phonon_output = phonon_flow.output

            if self.plot_band_structure:
                band_plot = plot_phonon_band_structure(phonon_doc=phonon_output)
                band_plot.name = f"{self.name}_band_structure_plot"
                jobs.append(band_plot)

            if self.plot_dos:
                dos_plot = plot_phonon_dos(phonon_doc=phonon_output)
                dos_plot.name = f"{self.name}_DOS_plot"
                jobs.append(dos_plot)

            if self.plot_thermal and self.create_thermal_properties:
                thermal_plot = plot_thermal_properties(phonon_doc=phonon_output)
                thermal_plot.name = f"{self.name}_thermal_properties_plot"
                jobs.append(thermal_plot)

            if self.write_summary:
                summary = write_phonon_summary(phonon_doc=phonon_output)
                summary.name = f"{self.name}_write_summary"
                jobs.append(summary)

        return Flow(jobs, output=phonon_flow.output, name=self.name)
