"""Define the ASE interface to SIESTA.

Written by Mads Engelund (see www.espeem.com)
Home of the SIESTA package:
http://www.uam.es/departamentos/ciencias/fismateriac/siesta
2017.04 - Pedro Brandimarte: changes for python 2-3 compatible
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.siesta.import_ion_xml import get_ion
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.utils import deprecated

from atomate2.siesta.sets.ase.parameters import PAOBasisBlock, SiestaParameters
from atomate2.siesta.sets.ase.siesta_input import SiestaInput
from atomate2.siesta.sets.ase.utils import (
    _nonpolarized_alias,
    bandpath2bandpoints,
    block,
    comment_in_box,
    generate_atomic_coordinates,
    var,
)
from atomate2.siesta.sets.siesta_structure_fdf import generate_structure_fdf

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable, Iterable, Iterator
    from typing import TextIO

    from ase import Atoms

logger = logging.getLogger(__name__)


class Siesta(FileIOCalculator):
    """ASE Calculator interface for the SIESTA DFT code."""

    allowed_xc: ClassVar[dict[str, list[str]]] = {
        "LDA": ["PZ", "CA", "PW92"],
        "GGA": [
            "PW91",
            "PBE",
            "revPBE",
            "RPBE",
            "WC",
            "AM05",
            "PBEsol",
            "PBEJsJrLO",
            "PBEGcGxLO",
            "PBEGcGxHEG",
            "BLYP",
        ],
        "VDW": ["DRSLL", "LMKLL", "KBM", "C09", "BH", "VV"],
    }
    name = "siesta"
    _legacy_default_command = "siesta < PREFIX.fdf > PREFIX.out"
    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "dipole",
        "eigenvalues",
        "density",
        "fermi_energy",
    ]
    default_parameters = SiestaParameters()
    accepts_bandpath_keyword = True
    fileio_rules = FileIOCalculator.ruleset(
        configspec=dict(pseudo_path=None),
        stdin_name="{prefix}.fdf",
        stdout_name="{prefix}.out",
    )

    def __init__(
        self,
        command: str | None = None,
        profile: Any = None,
        directory: str | Path = ".",
        **kwargs,
    ) -> None:
        """Initialize the SIESTA calculator with specified parameters."""
        logger.info("Siesta.__init__()")
        parameters = self.default_parameters.__class__(**kwargs)
        FileIOCalculator.__init__(
            self, command=command, profile=profile, directory=directory, **parameters
        )

    def __getitem__(self, key: str) -> Any:
        """Return the parameter value for the given key."""
        logger.info("Siesta.__getitem__()")
        return self.parameters[key]

    def species(self, atoms: Atoms) -> tuple:
        """Return the species and species numbers for the given atoms."""
        logger.info("Siesta.species()")
        return SiestaInput.get_species(atoms, list(self["species"]), self["basis_set"])

    @deprecated(
        "The keyword 'UNPOLARIZED' has been deprecated, and replaced by "
        "'non-polarized'",
        category=FutureWarning,
        callback=_nonpolarized_alias,
    )
    def set(self, **kwargs) -> None:
        """Set parameters on the calculator, validating and normalizing them."""
        logger.info("Siesta.set()")
        current = self.parameters.copy()
        current.update(kwargs)
        kwargs = current
        default_keys = list(self.__class__.default_parameters)
        offending_keys = set(kwargs) - set(default_keys)
        if len(offending_keys) > 0:
            raise ValueError(
                f"'set' does not take the keywords: {list(offending_keys)}"
            )
        parameters = self.__class__.default_parameters.copy()
        parameters.update(kwargs)
        for arg in ["mesh_cutoff", "energy_shift"]:
            value = kwargs.get(arg)
            if value is not None and not (
                isinstance(value, (float, int)) and value > 0
            ):
                raise ValueError(
                    f"'{arg}' must be a positive number (in eV), got '{value}'"
                )

        xc = kwargs.get("xc", "LDA")
        if isinstance(xc, (tuple, list)) and len(xc) == 2:
            functional, authors = xc
            if functional.lower() not in [k.lower() for k in self.allowed_xc]:
                raise ValueError(f"Unrecognized functional keyword: '{functional}'")
            lsauthorslower = [a.lower() for a in self.allowed_xc[functional]]
            if authors.lower() not in lsauthorslower:
                raise ValueError(
                    f"Unrecognized authors keyword for {functional}: '{authors}'"
                )
        elif xc in self.allowed_xc:
            functional = xc
            authors = self.allowed_xc[xc][0]
        else:
            found = False
            for key, value in self.allowed_xc.items():
                if xc in value:
                    found = True
                    functional = key
                    authors = xc
                    break
            if not found:
                raise ValueError(f"Unrecognized 'xc' keyword: '{xc}'")
        kwargs["xc"] = (functional, authors)
        if kwargs["fdf_arguments"] is None:
            kwargs["fdf_arguments"] = {}
        if not isinstance(kwargs["fdf_arguments"], dict):
            raise TypeError("fdf_arguments must be a dictionary.")
        FileIOCalculator.set(self, **kwargs)

    def set_fdf_arguments(self, fdf_arguments: dict | None) -> None:
        """Set the FDF arguments after validating them."""
        logger.info("Siesta.set_fdf_arguments()")
        self.validate_fdf_arguments(fdf_arguments)
        FileIOCalculator.set(self, fdf_arguments=fdf_arguments)

    def validate_fdf_arguments(self, fdf_arguments: dict | None) -> None:
        """Validate that the FDF arguments are a dictionary or None."""
        logger.info("Siesta.validate_fdf_arguments()")
        if fdf_arguments is not None and not isinstance(fdf_arguments, dict):
            raise TypeError("fdf_arguments must be a dictionary.")

    def write_input(
        self,
        atoms: Atoms,
        properties: list | None = None,
        system_changes: list | None = None,
    ) -> None:
        """Write the SIESTA FDF input file for the given atoms."""
        logger.info("Siesta.write_input()")
        super().write_input(
            atoms=atoms, properties=properties, system_changes=system_changes
        )
        filename = self.getpath(ext="fdf")

        # DM.UseSaveDM: Use dataclass method to determine value based on system_changes
        # Update fdf_arguments directly so it appears in SCFLoopParameters section
        from atomate2.siesta.dataclass.scf_loop_parameters import SCFLoopParameters

        dm_use_save_dm_value = SCFLoopParameters.should_use_save_dm(system_changes)
        self["fdf_arguments"]["DM.UseSaveDM"] = (
            f"{dm_use_save_dm_value}  # SIESTA DEFAULT VALUE"
            if dm_use_save_dm_value
            else f"{dm_use_save_dm_value}"
        )

        more_fdf_args = {}
        if "density" in properties:
            more_fdf_args["SaveRho"] = True
        species, species_numbers = self.species(atoms)
        pseudo_path = (
            self["pseudo_path"]
            or self.profile.configvars.get("pseudo_path")
            or self.cfg.get("SIESTA_PP_PATH")
        )
        if not pseudo_path:
            raise Exception(  # noqa: TRY002 - preserve original exception type
                "Please configure pseudo_path or SIESTA_PP_PATH envvar"
            )
        structure_fdf = self["structure_fdf"] or atoms.info.get("structure_fdf")
        if structure_fdf:
            structure_fdf_path = Path(structure_fdf)
            if not structure_fdf_path.is_absolute():
                structure_fdf_path = Path(self.directory) / structure_fdf_path
            if not structure_fdf_path.exists():
                logger.info(f"Generating {structure_fdf_path}")
                generate_structure_fdf(
                    atoms=atoms,
                    output_file=str(structure_fdf_path),
                    input_file=self["restart"] or None,
                    xv=self["restart"] is not None,
                )
            atoms.info["structure_fdf"] = str(structure_fdf_path)
            atoms.info["directory"] = str(self.directory)
        species_info = SpeciesInfo(
            atoms=atoms,
            pseudo_path=Path(pseudo_path),
            pseudo_qualifier=self.pseudo_qualifier(),
            species=species,
            use_structure_fdf=bool(structure_fdf),
            fdf_user_args=self["fdf_arguments"],
        )

        writer = FDFWriter(
            name=self.prefix,
            xc=self["xc"],
            fdf_user_args=self["fdf_arguments"],
            more_fdf_args=more_fdf_args,
            spin=self["spin"],
            species_numbers=species_numbers,
            atomic_coord_format=self["atomic_coord_format"],
            kpts=self["kpts"],
            bandpath=self["bandpath"],
            species_info=species_info,
            use_structure_fdf=bool(structure_fdf),
        )
        with open(filename, "w") as fd:
            writer.write(fd)
        writer.link_pseudos_into_directory(
            symlink_pseudos=self["symlink_pseudos"], directory=Path(self.directory)
        )

    def read(self, filename: str) -> None:
        """Read a SIESTA restart file and load the resulting atoms."""
        logger.info("Siesta.read()")
        fname = self.getpath(filename)
        if not fname.exists():
            raise ReadError(f"The restart file '{fname}' does not exist")
        with fname.open() as fd:
            self.atoms = read_siesta_xv(fd)
        self.read_results()

    def getpath(self, fname: str | None = None, ext: str | None = None) -> Path:
        """Return the path to a file in the calculation directory."""
        logger.info("Siesta.getpath()")
        if fname is None:
            fname = self.prefix
        if ext is not None:
            fname = f"{fname}.{ext}"
        return Path(self.directory) / fname

    def pseudo_qualifier(self) -> str:
        """Return the pseudopotential qualifier."""
        logger.info("Siesta.pseudo_qualifier()")
        if self["pseudo_qualifier"] is None:
            return self["xc"][0].lower()
        return self["pseudo_qualifier"]

    def read_results(self) -> None:
        """Read the SIESTA output results into the results dictionary."""
        logger.info("Siesta.read_results()")
        from ase.io.siesta_output import OutputReader

        reader = OutputReader(
            prefix=self.prefix,
            directory=Path(self.directory),
            bandpath=self["bandpath"],
        )
        results = reader.read_results()
        self.results.update(results)
        self.results["ion"] = self.read_ion(self.atoms)

    def read_ion(self, atoms: Atoms) -> dict:
        """Read the ion.xml data for each species."""
        logger.info("Siesta.read_ion()")
        species, _ = self.species(atoms)
        ion_results = {}
        for species_number, spec in enumerate(species, start=1):
            symbol = spec["symbol"]
            atomic_number = atomic_numbers[symbol]
            if spec["pseudopotential"] is None:
                label = symbol if self.pseudo_qualifier() == "" else f"{symbol}"
                pseudopotential = self.getpath(label, "psf")
            else:
                pseudopotential = Path(spec["pseudopotential"])
                label = pseudopotential.stem
            name = f"{label}.{species_number}"
            if spec["ghost"]:
                name = f"{name}.ghost"
                atomic_number = -atomic_number
            if name not in ion_results:
                fname = self.getpath(name, "ion.xml")
                if fname.is_file():
                    ion_results[name] = get_ion(str(fname))
        return ion_results

    def band_structure(self) -> Any:
        """Return the calculated band structure."""
        logger.info("Siesta.band_structure()")
        return self.results["bandstructure"]

    def get_fermi_level(self) -> float:
        """Return the Fermi energy."""
        logger.info("Siesta.get_fermi_level()")
        return self.results["fermi_energy"]

    def get_k_point_weights(self) -> np.ndarray:
        """Return the k-point weights."""
        logger.info("Siesta.get_k_point_weights()")
        return self.results["kpoint_weights"]

    def get_ibz_k_points(self) -> np.ndarray:
        """Return the irreducible Brillouin zone k-points."""
        logger.info("Siesta.get_ibz_k_points()")
        return self.results["kpoints"]

    def get_eigenvalues(self, kpt: int = 0, spin: int = 0) -> np.ndarray:
        """Return the eigenvalues for a given k-point and spin."""
        logger.info("Siesta.get_eigenvalues()")
        return self.results["eigenvalues"][spin, kpt]

    def get_number_of_spins(self) -> int:
        """Return the number of spin channels."""
        logger.info("Siesta.get_number_of_spins()")
        return self.results["eigenvalues"].shape[0]


@dataclass
class SpeciesInfo:
    """Data class to manage species-related information for SIESTA."""

    atoms: Atoms
    pseudo_path: Path
    pseudo_qualifier: str
    species: list
    use_structure_fdf: bool
    fdf_user_args: dict | None = None

    def __post_init__(self) -> None:
        """Build file instructions and basis information after init."""
        pao_basis = []
        basis_sizes = []
        file_instructions = []
        chemical_labels = []
        # Get species_Z_dict from atoms.info if available
        species_z_dict = self.atoms.info.get("species_Z_dict", {})
        for species_number, spec in enumerate(self.species, start=1):
            symbol = spec["symbol"]
            atomic_number = atomic_numbers[symbol]
            tag = spec["tag"]
            is_ghost = spec["ghost"]
            logger.debug(
                f"symbol={symbol}, atomic_number={atomic_number}, tag={tag}, "
                f"ghost={is_ghost}, pseudo_qualifier={self.pseudo_qualifier}"
            )

            # Use tag as label if available, otherwise use symbol
            label = tag if tag is not None else symbol

            # Determine pseudopotential file based on atomic number
            if species_z_dict and species_number in species_z_dict:
                z_value = species_z_dict[species_number]
                base_symbol = next(
                    s for s, z in atomic_numbers.items() if z == abs(z_value)
                )
            else:
                base_symbol = symbol
                z_value = -atomic_number if is_ghost else atomic_number

            if spec["pseudopotential"] is None:
                src_path = self.pseudo_path / f"{base_symbol}.psf"
                logger.debug("No custom pseudopotential specified")
            else:
                src_path = Path(spec["pseudopotential"])
                logger.debug(f"Custom pseudopotential: {src_path}")
            if not src_path.is_absolute():
                src_path = self.pseudo_path / src_path
            if not src_path.exists():
                src_path = self.pseudo_path / f"{base_symbol}.psml"
                logger.debug(f"Pseudopotential not found, trying PSML: {src_path}")

            # Generate target name with correct extension, using label for file name
            extension = src_path.suffix  # .psf or .psml
            # Use label for target file name (e.g., O_ghosst.psml)
            name = f"{label}{extension}"
            logger.debug(f"Generated pseudopotential name: {name}")
            instr = FileInstruction(src_path, name)
            file_instructions.append(instr)

            if not self.use_structure_fdf:
                pseudo_name = name
                string = f"    {species_number} {atomic_number} {label} {pseudo_name}"
                chemical_labels.append(string)
                self.chemical_labels = chemical_labels

            # Basis set
            if isinstance(spec["basis_set"], PAOBasisBlock):
                pao_basis.append(spec["basis_set"].script(label))
            else:
                basis_sizes.append((f"    {label}", spec["basis_set"]))

        self.file_instructions = file_instructions
        self.pao_basis = pao_basis
        self.basis_sizes = basis_sizes
        logger.info("SpeciesInfo.__post_init__()")

    def generate_text(self) -> Iterator[str]:
        """Yield the FDF text lines for this species information."""
        logger.info("SpeciesInfo.generate_text()")
        # BASIS SPECIFICATION MOVED TO DATACLASS (BasisSetsAndProjectors)
        # ASE no longer writes PAO.Basis or PAO.BasisSizes blocks directly.
        # The dataclass handles ALL basis specifications with proper priority logic:
        #   1. %block PAO.Basis (custom basis - highest priority)
        #   2. %block PAO.BasisSizes (per-species sizes - medium priority)
        #   3. PAO.BasisSize (global scalar - lowest priority/fallback)
        #
        # This ensures no duplication and maintains single source of truth.
        #
        # NOTE: self.pao_basis and self.basis_sizes are still generated in __post_init__
        # for potential future use (e.g., populating dataclass fields), but are NOT
        # written to FDF here.

        yield "\n"
        if not self.use_structure_fdf:
            yield var("ChemicalSpecieslabel", self.chemical_labels)
            yield "\n"


@dataclass
class FileInstruction:
    """Data class to handle file operations for pseudopotentials."""

    src_path: Path
    targetname: str

    def copy_to(self, directory: Path) -> None:
        """Copy the pseudopotential file into the directory."""
        logger.info("FileInstruction.copy_to()")
        self._link(shutil.copy, directory)

    def symlink_to(self, directory: Path) -> None:
        """Symlink the pseudopotential file into the directory."""
        logger.info("FileInstruction.symlink_to()")
        self._link(os.symlink, directory)

    def _link(self, file_operation: Callable, directory: Path) -> None:
        logger.info("FileInstruction._link()")
        dst_path = directory / self.targetname
        if self.src_path == dst_path:
            return
        dst_path.unlink(missing_ok=True)
        file_operation(self.src_path, dst_path)


@dataclass
class FDFWriter:
    """Generate SIESTA FDF input file content.

    Optionally uses ``%include structure.fdf``.
    """

    name: str
    xc: tuple
    fdf_user_args: OrderedDict
    more_fdf_args: dict
    spin: str
    species_numbers: object
    atomic_coord_format: str
    kpts: object
    bandpath: object
    species_info: SpeciesInfo
    use_structure_fdf: bool

    def write(self, fd: TextIO) -> None:
        """Write FDF content to a file descriptor, handling nested generators."""
        logger.info("FDFWriter.write()")

        def flatten_generator(gen: Iterable) -> Iterator:
            """Recursively yield strings, flattening any nested generators."""
            for item in gen:
                if isinstance(item, (str, bytes)):
                    yield item
                else:
                    yield from flatten_generator(item)

        # Stream chunks one at a time rather than using writelines.
        for chunk in flatten_generator(self.generate_text()):  # noqa: FURB122
            fd.write(chunk)

    def generate_text(self) -> Iterator:
        """Yield the full FDF text content for the calculation."""
        logger.info("FDFWriter.generate_text()")
        yield comment_in_box(["Atomate2-Siesta Generated FDF"])

        yield var("SystemName", self.name)
        yield var("SystemLabel", self.name)
        yield "\n"
        if self.use_structure_fdf:
            yield comment_in_box(["Structure Definition"])
            structure_fdf = Path(
                self.species_info.atoms.info.get("structure_fdf", "structure.fdf")
            )
            if not structure_fdf.is_absolute():
                structure_fdf = (
                    Path(self.species_info.atoms.info.get("directory", "."))
                    / structure_fdf
                )
            if not structure_fdf.exists():
                raise FileNotFoundError(
                    f"structure.fdf file not found at {structure_fdf}"
                )
            yield f"%include {structure_fdf.name}\n"
        else:
            cell = self.species_info.atoms.cell
            if cell.rank in [1, 2]:
                raise ValueError(
                    "Expected 3D unit cell or no unit cell. You may wish to "
                    "add vacuum along some directions."
                )
            if np.any(cell):
                yield comment_in_box(["Structure Definition"])
                yield var("LatticeConstant", "1.0 Ang")
                yield block("LatticeVectors", cell)
            yield from generate_atomic_coordinates(
                self.species_info.atoms, self.species_numbers, self.atomic_coord_format
            )

        yield "\n"
        yield from self.species_info.generate_text()

        # Get magnetic moments to check if we need to auto-override Spin setting
        atoms = self.species_info.atoms
        magmoms = atoms.get_initial_magnetic_moments()
        logger.debug(f"magmoms={magmoms}")

        # Check if there are non-zero magnetic moments
        has_nonzero_magmoms = False
        if len(magmoms) != 0:
            if isinstance(magmoms[0], np.ndarray):
                has_nonzero_magmoms = any(M[0] != 0 for M in magmoms)
            elif isinstance(magmoms[0], float):
                has_nonzero_magmoms = any(M != 0 for M in magmoms)

        # Auto-override Spin from non-polarized to polarized when magnetic
        # moments present
        # Only if user did NOT explicitly set Spin (checked via _user_set_spin flag)
        fdf_arguments = self.fdf_user_args
        user_explicitly_set_spin = fdf_arguments.get("_user_set_spin", False)

        if (
            has_nonzero_magmoms
            and "Spin" in fdf_arguments
            and not user_explicitly_set_spin
        ):
            spin_setting = (
                fdf_arguments["Spin"].lower()
                if isinstance(fdf_arguments["Spin"], str)
                else str(fdf_arguments["Spin"]).lower()
            )
            if spin_setting in ["non-polarized", "unpolarized"]:
                # Override to polarized when magnetic moments are present
                fdf_arguments = fdf_arguments.copy()  # Don't modify original
                fdf_arguments["Spin"] = "polarized"
                logger.info(
                    "Automatically changed Spin from 'non-polarized' to "
                    "'polarized' due to non-zero magnetic moments"
                )
                logger.info(
                    "User can override by explicitly setting Spin in "
                    "user_params, fdf_arguments, or tier presets"
                )

        # Remove internal control flags before writing FDF
        fdf_arguments_to_write = {
            k: v for k, v in fdf_arguments.items() if not k.startswith("_")
        }

        logger.info(
            f"FDFWriter.generate_text: About to write "
            f"{len(fdf_arguments_to_write)} FDF arguments"
        )
        logger.debug(
            f"FDFWriter.generate_text: All FDF keys: "
            f"{list(fdf_arguments_to_write.keys())}"
        )

        for key, value in fdf_arguments_to_write.items():
            yield var(key, value)

        # Note: SCFMustConverge is now handled by SCFLoopParameters dataclass
        # (removed hardcoded default here to maintain single source of truth)
        yield "\n"
        yield "\n"
        yield "\n"
        for key, value in self.more_fdf_args.items():
            yield var(key, value)

        # NOTE: DM.InitSpin block is handled by SpinSettings dataclass
        # SpinSettings auto-generates it from structure.magmom and adds it
        # to fdf_arguments
        # This ensures single source of truth and consistent behavior
        yield "\n"
        if self.kpts is not None:
            # Note: k-point generation is handled elsewhere
            pass
        if self.bandpath is not None:
            lines = bandpath2bandpoints(self.bandpath)
            yield lines
        yield "\n"

    def link_pseudos_into_directory(
        self, *, symlink_pseudos: bool | None = None, directory: Path
    ) -> None:
        """Link or copy the pseudopotential files into the directory."""
        logger.info("FDFWriter.link_pseudos_into_directory()")
        if symlink_pseudos is None:
            symlink_pseudos = os.name != "nt"
        for instruction in self.species_info.file_instructions:
            if symlink_pseudos:
                instruction.symlink_to(directory)
            else:
                instruction.copy_to(directory)
