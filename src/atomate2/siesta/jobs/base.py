"""Defines the base SIESTA Maker."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jobflow import Maker, Response, job
from monty.serialization import dumpfn

from atomate2.siesta import SETTINGS
from atomate2.siesta.files import (
    cleanup_siesta_outputs,
    copy_siesta_outputs,
    gzip_output_folder,
    write_siesta_input_set,
)
from atomate2.siesta.run import (
    run_optical,
    run_optical_input,
    run_siesta,
    run_vibra,
    should_stop_children,
)
from atomate2.siesta.schemas.task import SiestaTaskDoc
from atomate2.siesta.sets.base import SiestaInputGenerator
from atomate2.siesta.sets.core import OpticalSetGenerator, PhononSetGenerator
from atomate2.siesta.utils.common import print_in_box_rich
from atomate2.siesta.utils.logo import print_fancy_logo


def display_welcome_banner():
    """
    Display the atomate2siesta welcome banner and configuration settings.

    This function prints the logo and current SIESTA configuration. It must be
    called explicitly (importing the package has no stdout side effects); gate
    the call on ``SETTINGS.SIESTA_SHOW_BANNER`` where a banner is wanted.
    """
    print_fancy_logo()

    text_dict = {
        "SIESTA_CMD": SETTINGS.SIESTA_CMD,
        "SIESTA_ZIP_FILES": SETTINGS.SIESTA_ZIP_FILES,
        "CONFIG_FILE": SETTINGS.CONFIG_FILE,
        "FLOS_PATH": SETTINGS.FLOS_PATH,
        "SIESTA_PP_PATH": SETTINGS.SIESTA_PP_PATH,
    }

    # Try to read database configuration from ~/.jobflow.yaml
    jobflow_config_path = Path.home() / ".jobflow.yaml"
    if jobflow_config_path.exists():
        try:
            import yaml

            with open(jobflow_config_path) as f:
                config = yaml.safe_load(f)

            # Extract database information from JOB_STORE.docs_store
            if config and "JOB_STORE" in config:
                docs_store = config["JOB_STORE"].get("docs_store", {})
                if docs_store:
                    host = docs_store.get("host", "localhost")
                    port = docs_store.get("port", 27017)
                    database = docs_store.get("database", "N/A")
                    collection = docs_store.get("collection_name", "tasks")

                    text_dict["DATABASE"] = f"{database} @ {host}:{port}"
                    text_dict["COLLECTION"] = collection
        except Exception:
            # If we can't read the config, silently skip database info
            pass

    print_in_box_rich(text_dict)


# NOTE: the welcome banner is intentionally NOT displayed at import time.
# Importing a library must have no stdout side effects. Call
# display_welcome_banner() explicitly (e.g. from a CLI entry point) and gate
# it on SETTINGS.SIESTA_SHOW_BANNER where a banner is actually wanted.


if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

logger = logging.getLogger(__name__)

# Input files.
# Files to EXCLUDE from compression (keep uncompressed for easy access)
_FILES_TO_EXCLUDE = [
    "siesta.fdf",  # Main input file - keep readable
    "siesta.out",  # Main output file - keep readable
    "siesta.XV",  # Structure file - keep readable
]

# Compress all SIESTA files except those in _FILES_TO_EXCLUDE
# Using wildcard patterns to match all SIESTA-generated files
_FILES_TO_ZIP = [
    "*.DM",  # Density matrix files
    "*.BONDS",  # Bond information
    "*.EIG",  # Eigenvalues
    "*.FA",  # Forces and stress
    "*.FC",  # Force constants
    "*.KP",  # K-points
    "*.ORB_INDX",  # Orbital indices
    "*.STRUCT_*",  # Structure files (except XV)
    "*.ANI",  # Animation files
    "*.MDE",  # MD energy files
    "*.MDF",  # MD force files
    "*.ion",  # Ion files
    "*.ion.xml",  # Ion XML files
    "*.bands",  # Band structure
    "*.PDOS",  # Projected DOS
    "*.LDOS",  # Local DOS
    "*.RHO",  # Charge density
    "*.VH",  # Hartree potential
    "*.VT",  # Total potential
    "*.RHOINIT",  # Initial charge density
    "FORCE_STRESS",  # Force and stress output
    "MESSAGES",  # Messages file
    "BASIS_ENTHALPY",  # Basis enthalpy
    "BASIS_HARRIS_ENTHALPY",  # Harris enthalpy
    "CLOCK",  # Clock/timing file
    "fdf.*.log",  # FDF log files
    "INPUT_TMP.*",  # Temporary input files
    "NON_TRIMMED_KP_LIST",  # K-point list
    "PARALLEL_DIST",  # Parallel distribution info
    "siesta.alloc",  # Memory allocation info
    "siesta.HSX",  # Hamiltonian matrix
    "siesta.BONDS_FINAL",  # Final bonds
    "siesta.bib",  # Bibliography/citation info
    "siesta.CG",  # Conjugate gradient optimization history
    "OUTVARS.yml",  # Output variables YAML
    "0_NORMAL_EXIT",  # Exit status marker
]


@dataclass
class BaseSiestaMaker(Maker):
    """
    Base Siesta job maker.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .SiestaInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.write_siesta_input_set`.
    copy_siesta_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.copy_siesta_outputs`.
    run_siesta_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.run_siesta`.
    task_document_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict[str, Any]
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    store_output_data: bool
        Whether the job output (TaskDoc) should be stored in the JobStore through
        the response.
    use_custodian : bool
        Whether to use custodian for automatic error detection and recovery.
        When enabled, SIESTA calculations will be automatically retried with
        corrected parameters if errors are detected.
    custodian_handlers : list
        List of error handlers to use with custodian. If None, uses default
        handlers (SCF, Memory, Time, Numerical). Only used if use_custodian=True.
    custodian_max_errors : int
        Maximum number of errors to correct before giving up. Only used if
        use_custodian=True.
    strict_convergence : bool
        Whether to enforce strict convergence checking for relaxation calculations.
        If True, validator will fail if geometry does not converge (forces above
        tolerance). If False (default), non-converged relaxations are allowed
        (useful for dirty/fast calculations). This is independent of use_custodian:
        - use_custodian=False, strict_convergence=False: Fast, may not converge (dirty mode)
        - use_custodian=True, strict_convergence=False: Auto-recovery, lenient (default)
        - use_custodian=False, strict_convergence=True: Must converge or fail (strict)
        - use_custodian=True, strict_convergence=True: Auto-recovery + strict check (paranoid)
    dry_run : bool
        If True, skip SIESTA calculation and generate all input files without running.
        This generates complete SIESTA input files (siesta.fdf, structure.fdf, pseudopotentials)
        and saves the structure, allowing full validation of calculation setup before running
        expensive calculations.
    dry_run_output_dir : str
        Base directory to save dry-run outputs. A subdirectory will be created for each
        calculation using the label. Only used if dry_run=True.
    dry_run_format : str
        Output format for structure file (e.g., "cif", "xsf", "json").
        Only used if dry_run=True.
    dry_run_label : str or None
        Custom label for dry-run output directory and structure file. If None, automatically
        generated from maker name and formula. Only used if dry_run=True.
    manager_config : dict[str, Any] or None
        Configuration for jobflow-remote resource management. When set, this dict is
        propagated to jobs via ``job.update_config()`` to control HPC resources.
        Format: ``{"resources": {"ntasks_per_node": 4, "time": "02:00:00", ...}}``.
        Useful for heterogeneous workflows where different jobs need different resources
        (e.g., small molecule reference vs large supercell). When None (default),
        jobs use the default jobflow-remote worker resources.
    """

    name: str = "base"
    input_set_generator: SiestaInputGenerator = field(
        default_factory=SiestaInputGenerator
    )
    # input_set_generator: SiestaInputGenerator = field(default_factory=SiestaInputGenerator(Structure))
    # input_set_generator: SiestaInputGenerator | None = None  # Allow None
    write_input_set_kwargs: dict[str, Any] = field(default_factory=dict)
    copy_siesta_kwargs: dict[str, Any] = field(default_factory=dict)
    run_siesta_kwargs: dict[str, Any] = field(default_factory=dict)
    task_document_kwargs: dict[str, Any] = field(default_factory=dict)
    stop_children_kwargs: dict[str, Any] = field(default_factory=dict)
    write_additional_data: dict[str, Any] = field(default_factory=dict)
    store_output_data: bool = True
    use_custodian: bool = False
    custodian_handlers: list | None = None
    custodian_max_errors: int = 5
    strict_convergence: bool = False
    dry_run: bool = False
    dry_run_output_dir: str = "dry_run_output"
    dry_run_format: str = "cif"
    dry_run_label: str | None = None
    manager_config: dict[str, Any] | None = None

    @job
    def make(
        self,
        structure: Structure | Molecule,
        prev_dir: str | Path | None = None,
        extra_dir: str | Path | None = None,
    ) -> Response:
        """Run an SIESTA calculation or create dry-run preview.

        Parameters
        ----------
        structure : Structure or Molecule
            A pymatgen Structure object to create the calculation for.
        prev_dir : str or Path or None
            A previous SIESTA calculation directory to copy output files from.
        extra_dir : str or Path or None
            An additional directory to copy files from.

        Returns
        -------
        Response
            A jobflow Response containing the calculation results or dry-run output.
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)
        logger.info("BaseSiestaMaker.make()")

        if self.dry_run:
            return self._make_dry_run(structure, prev_dir)
        return self._make_calculation(structure, prev_dir, extra_dir)

    def _make_dry_run(
        self,
        structure: Structure | Molecule,
        prev_dir: str | Path | None = None,
    ) -> Response:
        """Create dry-run job (saves structure and generates SIESTA input files).

        Parameters
        ----------
        structure : Structure or Molecule
            A pymatgen Structure object to save.
        prev_dir : str or Path or None
            Previous directory (included in metadata).

        Returns
        -------
        Response
            A jobflow Response containing dry-run output information.
        """
        # Generate label if not provided
        label = self.dry_run_label
        if label is None:
            label = self._generate_dry_run_label(structure)

        # Create output directory for this specific calculation
        output_dir = Path(self.dry_run_output_dir) / label
        output_dir.mkdir(parents=True, exist_ok=True)
        # Convert to absolute path for use as dir_name/prev_dir
        output_dir = output_dir.resolve()

        # Save structure to file in output directory
        filename = output_dir / f"{label}.{self.dry_run_format}"
        structure.to(filename=str(filename), fmt=self.dry_run_format)

        logger.info(f"Dry-run: Saved structure to {filename}")

        # Generate and write SIESTA input files
        try:
            # Save current directory and change to output directory
            # (needed because write_siesta_input_set writes to cwd)
            import os

            original_dir = Path.cwd()
            os.chdir(output_dir)

            try:
                write_siesta_input_set(
                    structure,
                    self.input_set_generator,
                    directory=".",  # Write to current directory (which is already output_dir)
                    prev_dir=prev_dir,
                    **self.write_input_set_kwargs,
                )
                logger.info(f"Dry-run: Generated SIESTA input files in {output_dir}")

                # List generated input files
                input_files = []
                for pattern in ["*.fdf", "*.psf", "*.vps", "*.lua", "*.psml", "*.json"]:
                    input_files.extend([f.name for f in output_dir.glob(pattern)])

                input_files_generated = True
            finally:
                # Always restore original directory
                os.chdir(original_dir)

        except Exception as e:
            logger.warning(f"Dry-run: Failed to generate input files: {e}")
            input_files = []
            input_files_generated = False

        # Extract lattice parameters
        lattice = structure.lattice
        lattice_info = {
            "a": lattice.a,
            "b": lattice.b,
            "c": lattice.c,
            "alpha": lattice.alpha,
            "beta": lattice.beta,
            "gamma": lattice.gamma,
            "volume": lattice.volume,
        }

        # Create metadata
        metadata = {
            "maker_name": self.name,
            "maker_type": self.__class__.__name__,
            "tier": getattr(self.input_set_generator, "tier", None),
            "prev_dir": str(prev_dir) if prev_dir else None,
            "input_structure": True,
        }

        # Create output dictionary
        # Note: For EOS and other analysis flows, we include dummy energy=0.0
        # so the flow can complete without errors. The plots/analysis will show
        # flat lines at E=0, indicating dry-run mode.

        # Create dummy forces array (n_atoms, 3) for phonon workflows
        import numpy as np

        dummy_forces = np.zeros((len(structure), 3)).tolist()

        output = {
            "dry_run": True,
            "label": label,
            "output_directory": str(output_dir),
            "structure_file": str(filename),
            "input_files": input_files,
            "input_files_generated": input_files_generated,
            "formula": structure.composition.reduced_formula,
            "num_atoms": len(structure),
            "lattice": lattice_info,
            "metadata": metadata,
            "structure": structure,  # Include structure for jobflow reference resolution
            "dir_name": str(output_dir),  # Include dir_name for EOS/other flows
            "energy": 0.0,  # Dummy energy for compatibility with analysis flows
            "stress": [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],  # Dummy stress tensor
            "forces": dummy_forces,  # Dummy forces (n_atoms, 3) for phonon workflows
            "output": {  # Nested output dict for compatibility with some flow patterns
                "structure": structure,
                "energy": 0.0,
                "stress": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "forces": dummy_forces,  # Dummy forces for phonon workflows
            },
        }

        return Response(output=output)

    def _make_calculation(
        self,
        structure: Structure | Molecule,
        prev_dir: str | Path | None = None,
        extra_dir: str | Path | None = None,
    ) -> Response:
        """Normal SIESTA calculation (full calculation workflow).

        Parameters
        ----------
        structure : Structure or Molecule
            A pymatgen Structure object to create the calculation for.
        prev_dir : str or Path or None
            A previous SIESTA calculation directory to copy output files from.
        extra_dir : str or Path or None
            An additional directory to copy files from.

        Returns
        -------
        Response
            A jobflow Response containing the SIESTA calculation results.
        """
        # copy previous inputs if needed (governed by self.copy_aims_kwargs)
        if prev_dir is not None:
            copy_siesta_outputs(prev_dir, **self.copy_siesta_kwargs)

        if extra_dir is not None:
            copy_siesta_outputs(
                extra_dir,
                additional_siesta_files=["*.xyz", "*.lua"],
                **self.copy_siesta_kwargs,
            )

        # write SIESTA input files
        self.write_input_set_kwargs["prev_dir"] = prev_dir

        write_siesta_input_set(
            structure, self.input_set_generator, **self.write_input_set_kwargs
        )

        # write any additional data (useful for LUA stuff or images)
        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))

        # Force SCF.MustConverge=True when using custodian
        # This ensures SIESTA fails (non-zero exit) when SCF doesn't converge,
        # which triggers error handlers to apply corrections (remove DM, adjust mixer, etc.)
        # Without this, dirty presets set SCF.MustConverge=False → SIESTA exits successfully
        # → handlers never trigger → no error recovery
        if self.use_custodian:
            from atomate2.siesta.custodian.fdf_utils import update_fdf_file

            fdf_file = Path.cwd() / "siesta.fdf"
            if fdf_file.exists():
                update_fdf_file(fdf_file, {"SCF.MustConverge": True})
                logger.info(
                    "Custodian enabled: Forcing SCF.MustConverge=True for error recovery"
                )

        # run SIESTA (with or without custodian)
        if self.use_custodian:
            from custodian import Custodian

            from atomate2.siesta.custodian import DEFAULT_HANDLERS
            from atomate2.siesta.custodian.jobs import SiestaJob
            from atomate2.siesta.custodian.validators import get_validator

            # Use provided handlers or defaults
            if self.custodian_handlers is not None:
                handlers = self.custodian_handlers
                logger.info(
                    f"Using custom handlers: {[type(h).__name__ for h in handlers]}"
                )
                for h in handlers:
                    logger.info(
                        f"  {type(h).__name__}: max_num_corrections={h.max_num_corrections}"
                    )
            else:
                handlers = DEFAULT_HANDLERS.copy()
                logger.info("Using default handlers")

            # Create SIESTA job
            siesta_job = SiestaJob(
                siesta_cmd=SETTINGS.SIESTA_CMD, output_file="siesta.out"
            )

            # Create validators (use appropriate validator for calc_type)
            validator = get_validator(
                self.calc_type,
                strict_convergence=self.strict_convergence,
            )
            validators = [validator]
            logger.info(
                f"Using {type(validator).__name__} for {self.calc_type} calculation "
                f"(strict_convergence={self.strict_convergence})"
            )

            # Create Custodian orchestrator
            custodian = Custodian(
                handlers=handlers,
                jobs=[siesta_job],
                validators=validators,
                max_errors=self.custodian_max_errors,
            )

            # Run with custodian framework
            # Custodian will automatically:
            # - Run SIESTA
            # - Check for errors
            # - Apply corrections
            # - Retry as needed
            # - Validate output
            custodian.run()

            logger.info("Custodian execution completed")
        else:
            # Standard run without custodian
            run_siesta(**self.run_siesta_kwargs)

        # Now run different codes depending on postprocessing
        self.run_post_siesta()

        # parse SIESTA outputs
        task_doc = SiestaTaskDoc.from_directory(Path.cwd(), **self.task_document_kwargs)
        task_doc.task_label = self.name

        # Validate output if strict_convergence enabled (without custodian)
        if self.strict_convergence and not self.use_custodian:
            from atomate2.siesta.custodian.validators import get_validator

            validator = get_validator(
                self.calc_type,
                strict_convergence=self.strict_convergence,
            )
            logger.info(
                f"Running {type(validator).__name__} in strict mode "
                f"(strict_convergence={self.strict_convergence})"
            )

            # Validate the calculation (check() returns True if validation FAILS)
            validation_failed = validator.check(str(Path.cwd()))
            if validation_failed:
                errors = validator._get_validation_errors(Path.cwd())
                error_msg = "\n".join(errors)
                raise ValueError(
                    f"Validation failed in strict convergence mode:\n{error_msg}"
                )

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

    def run_post_siesta(self):
        """
        Run post-processing steps after the SIESTA calculation, specifically for phonon calculations.

        This method checks if the `input_set_generator` is an instance of `PhononSetGenerator`,
        which indicates that the job is related to phonon calculations. If it is a phonon job,
        the method will proceed to run the `vibra` command to perform vibrational analysis.

        PhononSetGenerator is typically used for generating inputs for phonon-related calculations
        in SIESTA. After the main SIESTA calculation is completed, vibrational modes can be
        analyzed using the `vibra` binary.

        The `vibra` command reads the SIESTA input and output files (e.g., `siesta.fdf`) and
        produces vibrational analysis results (e.g., `siesta.vibra.out`).

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method does not return any values but performs actions that involve running
            post-SIESTA calculations for phonon jobs.

        Notes
        -----
        - This method should be called after the SIESTA calculation has finished.
        - The `run_vibra` function will use the same keyword arguments (`run_siesta_kwargs`)
          passed to the main `run_siesta` method.

        """
        logger.info("BaseSiestaMaker.run_post_siesta()")
        # Check if the input set generator corresponds to PhononSetGenerator
        if isinstance(self.input_set_generator, PhononSetGenerator):
            logger.info("Running phonon calculation with VIBRA post-processing")

            # Run VIBRA post-processing for phonon analysis
            run_vibra(**self.run_siesta_kwargs)

        if isinstance(self.input_set_generator, OpticalSetGenerator):
            logger.info("Running optical calculation with post-processing")
            # Run optical_input post-processing for phonon analysis
            run_optical_input(**self.run_siesta_kwargs)
            run_optical(**self.run_siesta_kwargs)

    def _generate_dry_run_label(self, structure: Structure | Molecule) -> str:
        """Generate automatic label for dry-run output.

        Parameters
        ----------
        structure : Structure or Molecule
            Structure to generate label from.

        Returns
        -------
        str
            Label in format: {maker_name}_{formula}
            Special characters unsafe for filenames are replaced with safe alternatives.
        """
        formula = structure.composition.reduced_formula
        # Sanitize maker name for filesystem use
        safe_name = self.name.replace("/", "-").replace("\\", "-").replace(":", "-")
        return f"{safe_name}_{formula}"
